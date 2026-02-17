"""Run 15m and 1h timeframe backtests by resampling 5m data."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from trading.spot.backtest_engine import SpotBacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)

COINS = [
    ("aster", "BTC/USDT", "aster_BTC_USDT_5m.csv"),
    ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
    ("aster", "BNB/USDT", "aster_BNB_USDT_5m.csv"),
    ("aster", "ASTER/USDT", "aster_ASTER_USDT_5m.csv"),
    ("hyperliquid", "HYPE/USDC", "hyperliquid_HYPE_USDC_5m.csv"),
]

PROFILES = ["low", "medium", "high"]
TIMEFRAMES = {"15m": 3, "1h": 12}
CAPITAL = 10000.0
DATE_STR = datetime.now().strftime("%Y%m%d")


def load_csv(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def resample(df, factor):
    n = len(df)
    trim = n - (n % factor)
    df2 = df.iloc[:trim].copy()
    groups = np.arange(trim) // factor
    result = df2.groupby(groups).agg(
        timestamp=("timestamp", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index(drop=True)
    return result


def run_backtest(df, exchange, symbol, profile, timeframe):
    engine = SpotBacktestEngine(
        profile=profile, capital=CAPITAL, exchange=exchange,
        symbol=symbol, timeframe=timeframe,
    )
    return engine.run(df)


def main():
    all_results = []

    for exchange, symbol, csv_file in COINS:
        csv_path = DATA_DIR / csv_file
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            continue

        df_5m = load_csv(csv_path)
        print(f"\nLoaded {csv_file}: {len(df_5m)} candles")

        for tf_name, factor in TIMEFRAMES.items():
            df_tf = resample(df_5m, factor)
            print(f"  Resampled to {tf_name}: {len(df_tf)} candles")

            for profile in PROFILES:
                print(f"    Running {exchange} {symbol} {profile} {tf_name}...", end=" ", flush=True)
                try:
                    result = run_backtest(df_tf, exchange, symbol, profile, tf_name)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                sym_clean = symbol.replace("/", "_")
                fname = f"{exchange}_{sym_clean}_{profile}_{tf_name}_{DATE_STR}.json"
                (RESULTS_DIR / fname).write_text(result.to_json())

                print(f"Return: {result.total_return_pct:+.2f}%, Deals: {result.total_deals_completed}, Sharpe: {result.sharpe_ratio}")
                all_results.append({
                    "exchange": exchange, "symbol": symbol, "profile": profile,
                    "timeframe": tf_name, "file": fname,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "deals_completed": result.total_deals_completed,
                    "win_rate": result.win_rate,
                    "avg_profit_per_deal_pct": result.avg_profit_per_deal_pct,
                    "avg_hold_time_hours": result.avg_hold_time_hours,
                    "total_fees_paid": result.total_fees_paid,
                    "final_equity": result.final_equity,
                    "capital_utilization_pct": result.capital_utilization_pct,
                })

    # Now load existing 5m results
    print("\n\nLoading existing 5m results...")
    for f in RESULTS_DIR.glob("*_5m_*.json"):
        try:
            data = json.loads(f.read_text())
            all_results.append({
                "exchange": data.get("exchange", ""),
                "symbol": data.get("symbol", ""),
                "profile": data.get("profile", ""),
                "timeframe": "5m",
                "file": f.name,
                "total_return_pct": data.get("total_return_pct", 0),
                "max_drawdown_pct": data.get("max_drawdown_pct", 0),
                "sharpe_ratio": data.get("sharpe_ratio", 0),
                "deals_completed": data.get("total_deals_completed", 0),
                "win_rate": data.get("win_rate", 0),
                "avg_profit_per_deal_pct": data.get("avg_profit_per_deal_pct", 0),
                "avg_hold_time_hours": data.get("avg_hold_time_hours", 0),
                "total_fees_paid": data.get("total_fees_paid", 0),
                "final_equity": data.get("final_equity", 0),
                "capital_utilization_pct": data.get("capital_utilization_pct", 0),
            })
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")

    # Sort by return
    all_results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    # Save comparison report
    (RESULTS_DIR / "comparison_report.json").write_text(json.dumps(all_results, indent=2))

    # Print comparison table
    print("\n" + "=" * 130)
    print("FULL COMPARISON TABLE — ALL TIMEFRAMES")
    print("=" * 130)
    print(f"{'#':<4} {'Exchange':<12} {'Symbol':<12} {'Profile':<8} {'TF':<5} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'Deals':>6} {'Win%':>6} {'AvgPnL%':>9} {'Fees$':>8}")
    print("-" * 130)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<4} {r['exchange']:<12} {r['symbol']:<12} {r['profile']:<8} {r['timeframe']:<5} {r['total_return_pct']:>+8.2f}% {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f} {r['deals_completed']:>6} {r['win_rate']:>5.1f}% {r['avg_profit_per_deal_pct']:>+8.2f}% {r['total_fees_paid']:>8.2f}")

    # Generate SUMMARY.md
    generate_summary(all_results)
    print(f"\nResults saved to {RESULTS_DIR}")


def generate_summary(all_results):
    lines = ["# Spot DCA Backtest — Multi-Timeframe Summary", ""]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Capital: ${CAPITAL:,.0f} per run")
    lines.append("")

    # Full table
    lines.append("## Full Comparison (sorted by return)")
    lines.append("")
    lines.append("| # | Exchange | Symbol | Profile | TF | Return% | MaxDD% | Sharpe | Deals | Win% | Avg PnL% | Fees$ |")
    lines.append("|---|----------|--------|---------|-----|---------|--------|--------|-------|------|----------|-------|")
    for i, r in enumerate(all_results, 1):
        lines.append(f"| {i} | {r['exchange']} | {r['symbol']} | {r['profile']} | {r['timeframe']} | {r['total_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | {r['sharpe_ratio']:.2f} | {r['deals_completed']} | {r['win_rate']:.1f}% | {r['avg_profit_per_deal_pct']:+.2f}% | {r['total_fees_paid']:.2f} |")

    # Top 10
    lines.append("")
    lines.append("## Top 10 Combinations")
    lines.append("")
    for i, r in enumerate(all_results[:10], 1):
        lines.append(f"{i}. **{r['exchange']} {r['symbol']} {r['profile']} {r['timeframe']}** — {r['total_return_pct']:+.2f}% return, {r['sharpe_ratio']:.2f} Sharpe")

    # Best timeframe per coin
    lines.append("")
    lines.append("## Best Timeframe per Coin")
    lines.append("")
    coins = {}
    for r in all_results:
        key = f"{r['exchange']} {r['symbol']}"
        if key not in coins or r['total_return_pct'] > coins[key]['total_return_pct']:
            coins[key] = r
    for key, r in sorted(coins.items()):
        lines.append(f"- **{key}**: {r['timeframe']} ({r['profile']}) — {r['total_return_pct']:+.2f}%")

    # Best timeframe per profile
    lines.append("")
    lines.append("## Best Timeframe per Profile")
    lines.append("")
    for profile in ["low", "medium", "high"]:
        prof_results = [r for r in all_results if r['profile'] == profile]
        if prof_results:
            best = prof_results[0]  # already sorted
            lines.append(f"- **{profile}**: {best['exchange']} {best['symbol']} {best['timeframe']} — {best['total_return_pct']:+.2f}%")

    # Timeframe averages
    lines.append("")
    lines.append("## Average Return by Timeframe")
    lines.append("")
    for tf in ["5m", "15m", "1h"]:
        tf_results = [r for r in all_results if r['timeframe'] == tf]
        if tf_results:
            avg_ret = np.mean([r['total_return_pct'] for r in tf_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in tf_results])
            lines.append(f"- **{tf}**: avg return {avg_ret:+.2f}%, avg Sharpe {avg_sharpe:.2f} (n={len(tf_results)})")

    # Key patterns
    lines.append("")
    lines.append("## Key Patterns")
    lines.append("")
    for tf in ["5m", "15m", "1h"]:
        tf_results = [r for r in all_results if r['timeframe'] == tf]
        if tf_results:
            avg = np.mean([r['total_return_pct'] for r in tf_results])
            lines.append(f"- {tf} average return: {avg:+.2f}%")

    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print("SUMMARY.md generated")


if __name__ == "__main__":
    main()
