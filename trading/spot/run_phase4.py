"""Phase 4 batch backtester - all remaining runs + resampling."""
import sys, os, json, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.spot.backtest_engine import SpotBacktestEngine

RESULTS_DIR = Path(__file__).parent / "backtest_results"
DATA_DIR = Path(__file__).parent / "data"
DATE_STR = datetime.now().strftime("%Y%m%d")

def load_csv(filename):
    df = pd.read_csv(DATA_DIR / filename)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
    # Convert epoch ms to ISO strings for backtest engine compatibility
    if df['timestamp'].dtype in ['int64', 'float64'] and df['timestamp'].iloc[0] > 1e12:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').astype(str)
    return df

def resample_ohlcv(df, factor):
    """Resample 5m OHLCV by grouping `factor` consecutive candles."""
    n = len(df) // factor * factor
    df2 = df.iloc[:n].copy()
    groups = np.arange(n) // factor
    result = df2.groupby(groups).agg({
        'timestamp': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    return result

def run_backtest(exchange, symbol, profile, timeframe, df, capital=10000):
    engine = SpotBacktestEngine(
        profile=profile, capital=capital, exchange=exchange,
        symbol=symbol, timeframe=timeframe
    )
    result = engine.run(df)
    return result

def save_result(result, name):
    path = RESULTS_DIR / f"{name}_{DATE_STR}.json"
    with open(path, 'w') as f:
        f.write(result.to_json())
    return path

def run_all():
    all_results = {}
    
    # ── 5m runs for remaining coins ──
    runs_5m = [
        ("aster", "BNB/USDT", "aster_BNB_USDT_5m.csv"),
        ("aster", "ASTER/USDT", "aster_ASTER_USDT_5m.csv"),
        ("hyperliquid", "HYPE/USDC", "hyperliquid_HYPE_USDC_5m.csv"),
        ("hyperliquid", "BTC/USDC", "hyperliquid_BTC_USDC_5m.csv"),
        ("hyperliquid", "ETH/USDC", "hyperliquid_ETH_USDC_5m.csv"),
    ]
    
    profiles = ["low", "medium", "high"]
    
    # Load and cache all data
    data_cache = {}
    for exchange, symbol, filename in runs_5m:
        print(f"Loading {filename}...")
        data_cache[filename] = load_csv(filename)
        print(f"  -> {len(data_cache[filename])} rows")
    
    # Also load aster coins for resampling
    aster_files = {
        "BTC": ("aster", "BTC/USDT", "aster_BTC_USDT_5m.csv"),
        "ETH": ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
        "BNB": ("aster", "BNB/USDT", "aster_BNB_USDT_5m.csv"),
        "ASTER": ("aster", "ASTER/USDT", "aster_ASTER_USDT_5m.csv"),
    }
    for coin, (ex, sym, fn) in aster_files.items():
        if fn not in data_cache:
            print(f"Loading {fn}...")
            data_cache[fn] = load_csv(fn)
            print(f"  -> {len(data_cache[fn])} rows")
    
    # Run 5m backtests for remaining coins
    for exchange, symbol, filename in runs_5m:
        df = data_cache[filename]
        sym_short = symbol.replace("/", "_")
        for profile in profiles:
            key = f"{exchange}_{sym_short}_{profile}_5m"
            print(f"\nRunning {key}...", flush=True)
            t0 = time.time()
            result = run_backtest(exchange, symbol, profile, "5m", df)
            elapsed = time.time() - t0
            print(f"  -> {result.total_return_pct:+.2f}% | {result.total_deals_completed} deals | {elapsed:.1f}s")
            save_result(result, f"{exchange}_{sym_short}_{profile}_5m")
            all_results[key] = {
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "deals_per_day": result.deals_per_day,
                "win_rate": result.win_rate,
                "avg_profit_per_deal_pct": result.avg_profit_per_deal_pct,
                "avg_hold_time_hours": result.avg_hold_time_hours,
                "total_fees_paid": result.total_fees_paid,
                "total_deals_completed": result.total_deals_completed,
                "final_equity": result.final_equity,
                "capital_utilization_pct": result.capital_utilization_pct,
            }
    
    # ── Resampled runs (15m, 1h) for Aster coins ──
    for tf_label, factor in [("15m", 3), ("1h", 12)]:
        for coin, (exchange, symbol, filename) in aster_files.items():
            df_5m = data_cache[filename]
            print(f"\nResampling {filename} to {tf_label} ({len(df_5m)} -> ~{len(df_5m)//factor} rows)...")
            df_resampled = resample_ohlcv(df_5m, factor)
            print(f"  -> {len(df_resampled)} rows")
            
            sym_short = symbol.replace("/", "_")
            for profile in profiles:
                key = f"{exchange}_{sym_short}_{profile}_{tf_label}"
                print(f"Running {key}...", flush=True)
                t0 = time.time()
                result = run_backtest(exchange, symbol, profile, tf_label, df_resampled)
                elapsed = time.time() - t0
                print(f"  -> {result.total_return_pct:+.2f}% | {result.total_deals_completed} deals | {elapsed:.1f}s")
                save_result(result, f"{exchange}_{sym_short}_{profile}_{tf_label}")
                all_results[key] = {
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "deals_per_day": result.deals_per_day,
                    "win_rate": result.win_rate,
                    "avg_profit_per_deal_pct": result.avg_profit_per_deal_pct,
                    "avg_hold_time_hours": result.avg_hold_time_hours,
                    "total_fees_paid": result.total_fees_paid,
                    "total_deals_completed": result.total_deals_completed,
                    "final_equity": result.final_equity,
                    "capital_utilization_pct": result.capital_utilization_pct,
                }
    
    # ── Load previously completed results ──
    prev_results = {
        "aster_BTC_USDT_low_5m": {"total_return_pct": -13.62},
        "aster_BTC_USDT_medium_5m": {"total_return_pct": -22.4},
        "aster_BTC_USDT_high_5m": {"total_return_pct": -23.43},
        "aster_ETH_USDT_low_5m": {"total_return_pct": 2.63},
        "aster_ETH_USDT_medium_5m": {"total_return_pct": 5.68},
        "aster_ETH_USDT_high_5m": {"total_return_pct": -24.27},
    }
    # Enrich previous results from saved JSON files
    prev_file_map = {
        "aster_BTC_USDT_low_5m": "aster_BTC_USDT_low_20260217.json",
        "aster_BTC_USDT_medium_5m": "aster_BTC_USDT_medium_20260217.json",
        "aster_BTC_USDT_high_5m": "aster_BTC_USDT_high_20260217.json",
        "aster_ETH_USDT_low_5m": "aster_ETH_USDT_low_20260217.json",
        "aster_ETH_USDT_medium_5m": "aster_ETH_USDT_medium_5m_20260217.json",
        "aster_ETH_USDT_high_5m": "aster_ETH_USDT_high_5m_20260217.json",
    }
    for key, fname in prev_file_map.items():
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            try:
                with open(fpath) as f:
                    data = json.load(f)
                prev_results[key] = {
                    "total_return_pct": data.get("total_return_pct", prev_results[key]["total_return_pct"]),
                    "max_drawdown_pct": data.get("max_drawdown_pct", 0),
                    "sharpe_ratio": data.get("sharpe_ratio", 0),
                    "deals_per_day": data.get("deals_per_day", 0),
                    "win_rate": data.get("win_rate", 0),
                    "avg_profit_per_deal_pct": data.get("avg_profit_per_deal_pct", 0),
                    "avg_hold_time_hours": data.get("avg_hold_time_hours", 0),
                    "total_fees_paid": data.get("total_fees_paid", 0),
                    "total_deals_completed": data.get("total_deals_completed", 0),
                    "final_equity": data.get("final_equity", 0),
                    "capital_utilization_pct": data.get("capital_utilization_pct", 0),
                }
            except:
                pass
    
    # Merge
    combined = {**prev_results, **all_results}
    
    # Save comparison report
    with open(RESULTS_DIR / "comparison_report.json", 'w') as f:
        json.dump(combined, f, indent=2)
    
    # ── Print comparison table ──
    print("\n" + "="*120)
    print("BACKTEST COMPARISON - ALL RESULTS (sorted by total_return descending)")
    print("="*120)
    header = f"{'Run':<45} {'Return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'WinRate':>8} {'Deals':>6} {'D/Day':>6} {'AvgPnl%':>8} {'Fees$':>8} {'Util%':>6}"
    print(header)
    print("-"*120)
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1].get("total_return_pct", -999), reverse=True)
    for key, data in sorted_results:
        r = data.get("total_return_pct", 0)
        dd = data.get("max_drawdown_pct", 0)
        sr = data.get("sharpe_ratio", 0)
        wr = data.get("win_rate", 0)
        deals = data.get("total_deals_completed", 0)
        dpd = data.get("deals_per_day", 0)
        apnl = data.get("avg_profit_per_deal_pct", 0)
        fees = data.get("total_fees_paid", 0)
        util = data.get("capital_utilization_pct", 0)
        print(f"{key:<45} {r:>+8.2f} {dd:>8.2f} {sr:>7.2f} {wr:>7.1f}% {deals:>6} {dpd:>6.2f} {apnl:>+8.2f} {fees:>8.2f} {util:>6.1f}")
    
    print("="*120)
    
    # ── Generate SUMMARY.md ──
    summary_lines = ["# Spot DCA Backtest Results — Phase 4 Complete\n"]
    summary_lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    summary_lines.append(f"**Initial Capital:** $10,000 per run\n")
    summary_lines.append(f"**Total Configurations Tested:** {len(combined)}\n\n")
    
    summary_lines.append("## Results Table (sorted by return)\n\n")
    summary_lines.append(f"| Run | Return% | MaxDD% | Sharpe | WinRate | Deals | Deals/Day | AvgPnl% | Fees$ | Util% |\n")
    summary_lines.append(f"|-----|---------|--------|--------|---------|-------|-----------|---------|-------|-------|\n")
    for key, data in sorted_results:
        r = data.get("total_return_pct", 0)
        dd = data.get("max_drawdown_pct", 0)
        sr = data.get("sharpe_ratio", 0)
        wr = data.get("win_rate", 0)
        deals = data.get("total_deals_completed", 0)
        dpd = data.get("deals_per_day", 0)
        apnl = data.get("avg_profit_per_deal_pct", 0)
        fees = data.get("total_fees_paid", 0)
        util = data.get("capital_utilization_pct", 0)
        summary_lines.append(f"| {key} | {r:+.2f} | {dd:.2f} | {sr:.2f} | {wr:.1f}% | {deals} | {dpd:.2f} | {apnl:+.2f} | {fees:.2f} | {util:.1f} |\n")
    
    summary_lines.append("\n## Key Findings\n\n")
    
    # Best/worst
    best_key, best_data = sorted_results[0]
    worst_key, worst_data = sorted_results[-1]
    summary_lines.append(f"- **Best performer:** {best_key} ({best_data['total_return_pct']:+.2f}%)\n")
    summary_lines.append(f"- **Worst performer:** {worst_key} ({worst_data['total_return_pct']:+.2f}%)\n")
    
    # By profile
    for profile in profiles:
        subset = [(k, v) for k, v in combined.items() if f"_{profile}_" in k]
        if subset:
            avg_ret = np.mean([v["total_return_pct"] for _, v in subset])
            summary_lines.append(f"- **{profile.upper()} profile avg return:** {avg_ret:+.2f}% ({len(subset)} runs)\n")
    
    # By timeframe
    for tf in ["5m", "15m", "1h"]:
        subset = [(k, v) for k, v in combined.items() if f"_{tf}" in k]
        if subset:
            avg_ret = np.mean([v["total_return_pct"] for _, v in subset])
            summary_lines.append(f"- **{tf} timeframe avg return:** {avg_ret:+.2f}% ({len(subset)} runs)\n")
    
    # By exchange
    for ex in ["aster", "hyperliquid"]:
        subset = [(k, v) for k, v in combined.items() if k.startswith(ex)]
        if subset:
            avg_ret = np.mean([v["total_return_pct"] for _, v in subset])
            summary_lines.append(f"- **{ex} avg return:** {avg_ret:+.2f}% ({len(subset)} runs)\n")
    
    summary_lines.append("\n## Analysis\n\n")
    summary_lines.append("### Profile Comparison\n")
    summary_lines.append("- Low risk profiles use fewer safety orders with wider deviation, preserving capital\n")
    summary_lines.append("- Medium profiles balance deal frequency with risk exposure\n")
    summary_lines.append("- High risk profiles deploy more capital via aggressive safety orders\n\n")
    
    summary_lines.append("### Timeframe Comparison\n")
    summary_lines.append("- 5m: highest signal frequency, most deals, but more noise\n")
    summary_lines.append("- 15m: reduced noise, potentially better regime detection\n")
    summary_lines.append("- 1h: fewest deals but cleaner signals\n\n")
    
    with open(RESULTS_DIR / "SUMMARY.md", 'w') as f:
        f.writelines(summary_lines)
    
    print(f"\nSaved: comparison_report.json, SUMMARY.md")
    print("Phase 4 backtests complete!")

if __name__ == "__main__":
    run_all()
