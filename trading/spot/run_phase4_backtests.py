"""Phase 4: Batch backtest runner — all coins × all profiles from local CSVs.
Optimized: pre-computes regimes/ATR once per CSV, patches engine to skip recomputation.
"""
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading.spot.backtest_engine import SpotBacktestEngine, PROFILES as ENGINE_PROFILES
from trading.regime_detector import classify_regime_v2
from trading.indicators import atr_pct as compute_atr_pct

CONFIGS = [
    ("aster", "BTC/USDT", "aster_BTC_USDT_5m.csv"),
    ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
    ("aster", "BNB/USDT", "aster_BNB_USDT_5m.csv"),
    ("aster", "ASTER/USDT", "aster_ASTER_USDT_5m.csv"),
    ("hyperliquid", "HYPE/USDC", "hyperliquid_HYPE_USDC_5m.csv"),
    ("hyperliquid", "BTC/USDC", "hyperliquid_BTC_USDC_5m.csv"),
    ("hyperliquid", "ETH/USDC", "hyperliquid_ETH_USDC_5m.csv"),
]
PROFILE_NAMES = ["low", "medium", "high"]
CAPITAL = 10000.0
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "backtest_results"


def load_csv(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def patched_run(engine, df, regimes, atr_pct_series):
    """Run backtest with pre-computed regimes and ATR to avoid redundant computation."""
    if len(df) < 100:
        from trading.spot.backtest_engine import BacktestResult
        return BacktestResult()

    sma50 = df["close"].rolling(50).mean()
    peak_equity = engine.initial_capital

    for i in range(100, len(df)):
        row = df.iloc[i]
        ts = str(row["timestamp"])
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        regime = regimes.iloc[i] if i < len(regimes) else "UNKNOWN"
        engine._current_regime = regime
        engine._current_atr_pct = float(atr_pct_series.iloc[i]) if not pd.isna(atr_pct_series.iloc[i]) else 0.0
        engine._trend_bullish = price >= float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else True

        tp_pct = engine._adaptive_tp(regime, engine._current_atr_pct)
        dev_pct = engine._adaptive_deviation(regime, engine._current_atr_pct, tp_pct)

        if engine._halted:
            engine._check_exits(high, low, price, ts, regime)
        else:
            engine._check_safety_order_fills(low, price, ts, regime, dev_pct, tp_pct)
            engine._check_exits(high, low, price, ts, regime)
            if not engine.deals and regime not in {"EXTREME"}:
                engine._open_deal(price, ts, regime, tp_pct)

        equity = engine._equity(price)
        engine.equity_snapshots.append({"timestamp": ts, "equity": equity, "cash": engine.cash, "price": price})
        deployed = sum(d.capital_deployed for d in engine.deals)
        engine._utilization_samples.append(deployed / engine.initial_capital * 100 if engine.initial_capital > 0 else 0)

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd >= engine.profile.max_drawdown_pct and not engine._halted:
            engine._halted = True

    last_price = float(df.iloc[-1]["close"])
    last_ts = str(df.iloc[-1]["timestamp"])
    for deal in list(engine.deals):
        engine._force_close_deal(deal, last_price, last_ts)

    return engine._compile_results(df)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    all_results = []
    failures = []
    total_start = time.time()

    for exchange, symbol, csv_file in CONFIGS:
        csv_path = DATA_DIR / csv_file
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            continue

        df = load_csv(csv_path)
        print(f"\nLoaded {csv_file}: {len(df)} candles")

        # Pre-compute expensive indicators ONCE per coin
        print(f"  Computing regimes & ATR...", end=" ", flush=True)
        t0 = time.time()
        regimes = classify_regime_v2(df, "5m")
        atr_pct_series = compute_atr_pct(df, 14)
        print(f"done in {time.time()-t0:.1f}s")

        for profile in PROFILE_NAMES:
            label = f"{exchange}_{symbol.replace('/', '_')}_{profile}"
            print(f"  Running {label}...", end=" ", flush=True)
            t0 = time.time()

            try:
                engine = SpotBacktestEngine(
                    profile=profile, capital=CAPITAL,
                    exchange=exchange, symbol=symbol, timeframe="5m",
                )
                result = patched_run(engine, df, regimes, atr_pct_series)
                elapsed = time.time() - t0
                print(f"done in {elapsed:.1f}s — return={result.total_return_pct:+.2f}%")

                out_file = RESULTS_DIR / f"{exchange}_{symbol.replace('/', '_')}_{profile}_{date_str}.json"
                out_file.write_text(result.to_json())

                all_results.append({
                    "exchange": exchange,
                    "symbol": symbol,
                    "profile": profile,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "deals_per_day": result.deals_per_day,
                    "avg_profit_per_deal": result.avg_profit_per_deal_usd,
                    "avg_hold_hours": result.avg_hold_time_hours,
                    "capital_utilization_pct": result.capital_utilization_pct,
                    "win_rate": result.win_rate,
                    "total_fees": result.total_fees_paid,
                    "final_equity": result.final_equity,
                    "deals_completed": result.total_deals_completed,
                    "runtime_sec": round(elapsed, 1),
                })
            except Exception as e:
                import traceback
                elapsed = time.time() - t0
                print(f"FAILED in {elapsed:.1f}s — {e}")
                traceback.print_exc()
                failures.append({"label": label, "error": str(e)})

    total_elapsed = time.time() - total_start
    all_results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    report = {"generated": datetime.now().isoformat(), "results": all_results, "failures": failures}
    (RESULTS_DIR / "comparison_report.json").write_text(json.dumps(report, indent=2))

    # Print table
    print("\n" + "=" * 140)
    print(f"{'Exchange':<14} {'Symbol':<14} {'Profile':<8} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'Deals/d':>8} {'AvgPnL$':>9} {'HoldHrs':>8} {'Util%':>7} {'Win%':>6} {'Fees$':>8} {'Time':>6}")
    print("-" * 140)
    for r in all_results:
        print(f"{r['exchange']:<14} {r['symbol']:<14} {r['profile']:<8} {r['total_return_pct']:>+8.2f}% {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f} {r['deals_per_day']:>8.2f} {r['avg_profit_per_deal']:>+8.2f}$ {r['avg_hold_hours']:>7.1f}h {r['capital_utilization_pct']:>6.1f}% {r['win_rate']:>5.1f}% {r['total_fees']:>7.2f}$ {r['runtime_sec']:>5.1f}s")
    print("=" * 140)

    print(f"\nTop 5:")
    for i, r in enumerate(all_results[:5], 1):
        print(f"  {i}. {r['exchange']} {r['symbol']} [{r['profile']}] → {r['total_return_pct']:+.2f}% (Sharpe={r['sharpe_ratio']:.2f})")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f['label']}: {f['error']}")

    print(f"\nTotal runtime: {total_elapsed:.1f}s ({len(all_results)} successful, {len(failures)} failed)")
    generate_summary(all_results, failures, total_elapsed)


def generate_summary(results, failures, total_runtime):
    if not results:
        return
    lines = [
        "# Phase 4 Backtest Results Summary\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total runs:** {len(results)} successful, {len(failures)} failed | **Runtime:** {total_runtime:.0f}s\n",
    ]

    lines.append("## Top 5 Performers\n")
    lines.append("| Rank | Exchange | Symbol | Profile | Return% | Sharpe | MaxDD% | Win% |")
    lines.append("|------|----------|--------|---------|---------|--------|--------|------|")
    for i, r in enumerate(results[:5], 1):
        lines.append(f"| {i} | {r['exchange']} | {r['symbol']} | {r['profile']} | {r['total_return_pct']:+.2f}% | {r['sharpe_ratio']:.2f} | {r['max_drawdown_pct']:.2f}% | {r['win_rate']:.1f}% |")

    lines.append("\n## Best Coin per Profile\n")
    for p in PROFILE_NAMES:
        pr = [r for r in results if r["profile"] == p]
        if pr:
            best = pr[0]
            lines.append(f"- **{p.upper()}:** {best['exchange']} {best['symbol']} → {best['total_return_pct']:+.2f}% (Sharpe={best['sharpe_ratio']:.2f})")

    lines.append("\n## Best Profile per Coin\n")
    coins = list(dict.fromkeys((r["exchange"], r["symbol"]) for r in results))
    for exch, sym in coins:
        cr = sorted([r for r in results if r["exchange"] == exch and r["symbol"] == sym], key=lambda x: x["total_return_pct"], reverse=True)
        best = cr[0]
        lines.append(f"- **{exch} {sym}:** {best['profile'].upper()} → {best['total_return_pct']:+.2f}% (Sharpe={best['sharpe_ratio']:.2f})")

    lines.append("\n## Risk-Adjusted Returns by Profile\n")
    lines.append("| Profile | Avg Return% | Avg Sharpe | Avg MaxDD% | Avg Win% |")
    lines.append("|---------|-------------|------------|------------|----------|")
    for p in PROFILE_NAMES:
        pr = [r for r in results if r["profile"] == p]
        if pr:
            lines.append(f"| {p.upper()} | {sum(r['total_return_pct'] for r in pr)/len(pr):+.2f}% | {sum(r['sharpe_ratio'] for r in pr)/len(pr):.2f} | {sum(r['max_drawdown_pct'] for r in pr)/len(pr):.2f}% | {sum(r['win_rate'] for r in pr)/len(pr):.1f}% |")

    lines.append("\n## Key Observations\n")
    profile_wins = {}
    for exch, sym in coins:
        cr = sorted([r for r in results if r["exchange"] == exch and r["symbol"] == sym], key=lambda x: x["total_return_pct"], reverse=True)
        profile_wins[cr[0]["profile"]] = profile_wins.get(cr[0]["profile"], 0) + 1
    lines.append(f"- **Profile win count** (best return per coin): {', '.join(f'{k.upper()}={v}' for k,v in sorted(profile_wins.items(), key=lambda x:-x[1]))}")

    for exch_name in ["aster", "hyperliquid"]:
        er = [r for r in results if r["exchange"] == exch_name]
        if er:
            avg = sum(r["total_return_pct"] for r in er) / len(er)
            days = "90" if exch_name == "aster" else "17"
            lines.append(f"- **{exch_name.title()}** ({days}-day) avg return: {avg:+.2f}%")

    # Best risk-adjusted
    best_sharpe = max(results, key=lambda x: x["sharpe_ratio"])
    lines.append(f"- **Best risk-adjusted:** {best_sharpe['exchange']} {best_sharpe['symbol']} [{best_sharpe['profile']}] Sharpe={best_sharpe['sharpe_ratio']:.2f}")

    lines.append("\n## Recommendations for Phase 5\n")
    lines.append("- Review coins with negative returns — consider excluding or adjusting parameters")
    lines.append("- Compare Sharpe ratios across profiles to find optimal risk-adjusted strategy")
    lines.append("- Hyperliquid results based on only 17 days — gather more data before production")
    lines.append("- Consider per-coin parameter optimization (deviation, TP ranges) by volatility")
    lines.append("- Test different ATR baseline values for coins with unusual volatility profiles")
    lines.append("- If all returns are negative, the DCA strategy may need fundamental changes (entry signals, trend filters)")

    if failures:
        lines.append("\n## Failures\n")
        for f in failures:
            lines.append(f"- **{f['label']}:** {f['error']}")

    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\nSummary saved to {RESULTS_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
