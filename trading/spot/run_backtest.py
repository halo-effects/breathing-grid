"""CLI runner for the Spot DCA Backtest Engine.

Usage:
    python -m trading.spot.run_backtest --symbol BTC/USDT --timeframe 5m --days 30 --capital 10000 --profile medium
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading.spot.backtest_engine import SpotBacktestEngine, PROFILES


def fetch_data(symbol: str, timeframe: str, days: int, exchange: str) -> pd.DataFrame:
    """Fetch historical OHLCV data via CCXT."""
    import ccxt

    # Map exchange names
    exchange_map = {"aster": "aster", "hyperliquid": "hyperliquid"}
    exch_id = exchange_map.get(exchange, exchange)

    try:
        exch = getattr(ccxt, exch_id)({"enableRateLimit": True})
    except AttributeError:
        print(f"Exchange '{exch_id}' not found in CCXT, falling back to binance for data")
        exch = ccxt.binance({"enableRateLimit": True})

    exch.load_markets()

    # Resolve symbol format
    if symbol in exch.markets:
        ccxt_symbol = symbol
    elif symbol.replace("/", "") in [m.replace("/", "") for m in exch.markets]:
        # Try to find matching market
        for m in exch.markets:
            if m.replace("/", "") == symbol.replace("/", ""):
                ccxt_symbol = m
                break
        else:
            ccxt_symbol = symbol
    else:
        ccxt_symbol = symbol

    # Timeframe to ms
    tf_ms = {
        "1m": 60000, "5m": 300000, "15m": 900000,
        "1h": 3600000, "4h": 14400000, "1d": 86400000,
    }
    candle_ms = tf_ms.get(timeframe, 300000)
    total_candles = int(days * 86400000 / candle_ms)

    print(f"Fetching {total_candles} candles of {ccxt_symbol} {timeframe} from {exch_id}...")

    all_candles = []
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    batch = min(1000, total_candles)

    while len(all_candles) < total_candles:
        candles = exch.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=batch)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + candle_ms
        if len(candles) < batch:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Deduplicate
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Fetched {len(df)} candles ({df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
    return df


def print_results(result):
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print(f"  SPOT DCA BACKTEST RESULTS — {result.symbol}")
    print(f"  Profile: {result.profile.upper()} | Exchange: {result.exchange}")
    print("=" * 60)
    print(f"  Initial Capital:      ${result.initial_capital:,.2f}")
    print(f"  Final Equity:         ${result.final_equity:,.2f}")
    print(f"  Total Return:         {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:         {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:         {result.sharpe_ratio:.2f}")
    print("-" * 60)
    print(f"  Deals Completed:      {result.total_deals_completed}")
    print(f"  Deals/Day:            {result.deals_per_day:.2f}")
    print(f"  Win Rate:             {result.win_rate:.1f}%")
    print(f"  Avg Profit/Deal:      ${result.avg_profit_per_deal_usd:+.2f} ({result.avg_profit_per_deal_pct:+.2f}%)")
    print(f"  Avg Hold Time:        {result.avg_hold_time_hours:.1f} hours")
    print(f"  Largest Loss:         ${result.largest_single_loss:.2f}")
    print(f"  Total Fees:           ${result.total_fees_paid:.2f}")
    print(f"  Capital Utilization:  {result.capital_utilization_pct:.1f}%")
    if result.total_deals_open > 0:
        print(f"  Open Deals (forced):  {result.total_deals_open}")

    if result.per_lot_stats:
        print("-" * 60)
        print("  Per-Lot Breakdown:")
        print(f"  {'Level':<8} {'Count':<8} {'Avg PnL':<12} {'Avg Ret%':<10} {'Win%':<8}")
        for lid in sorted(result.per_lot_stats.keys()):
            s = result.per_lot_stats[lid]
            label = "Base" if lid == 0 else f"SO{lid}"
            print(f"  {label:<8} {int(s['count']):<8} ${s['avg_pnl']:<11.2f} {s['avg_return_pct']:<10.2f} {s['win_rate']:<8.1f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Spot DCA Backtest Runner")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital USD")
    parser.add_argument("--profile", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--exchange", default="aster", help="Exchange for fee schedule")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Fetch data
    df = fetch_data(args.symbol, args.timeframe, args.days, args.exchange)

    # Run backtest
    engine = SpotBacktestEngine(
        profile=args.profile,
        capital=args.capital,
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    result = engine.run(df)

    # Print results
    print_results(result)

    # Save results
    results_dir = Path(__file__).parent / "backtest_results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{args.symbol.replace('/', '')}_{args.profile}_{ts}.json"
    out_file.write_text(result.to_json())
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
