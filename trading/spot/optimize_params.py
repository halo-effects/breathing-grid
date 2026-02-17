"""Phase 5: Parameter sensitivity analysis and optimization for Spot DCA.

Key optimization: pre-compute regimes and ATR once per (dataset, timeframe),
then monkey-patch the engine to reuse cached values.
"""
import sys, json, time
from datetime import datetime
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.spot.backtest_engine import SpotBacktestEngine, PROFILES
from trading.regime_detector import classify_regime_v2
from trading.indicators import atr_pct as compute_atr_pct

RESULTS_DIR = Path(__file__).parent / "backtest_results" / "optimization"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 10000


def load_csv(filename):
    df = pd.read_csv(DATA_DIR / filename)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
    if df['timestamp'].dtype in ['int64', 'float64'] and df['timestamp'].iloc[0] > 1e12:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').astype(str)
    return df


def resample_ohlcv(df, factor):
    n = len(df) // factor * factor
    df2 = df.iloc[:n].copy()
    groups = np.arange(n) // factor
    return df2.groupby(groups).agg({
        'timestamp': 'first', 'open': 'first', 'high': 'max',
        'low': 'min', 'close': 'last', 'volume': 'sum'
    }).reset_index(drop=True)


def get_df(csv_file, timeframe):
    df = load_csv(csv_file)
    if timeframe == "15m":
        df = resample_ohlcv(df, 3)
    elif timeframe == "1h":
        df = resample_ohlcv(df, 12)
    return df


# Cache for pre-computed indicators
_indicator_cache = {}


def precompute_indicators(df, timeframe, cache_key):
    """Pre-compute regime classification and ATR for a dataset."""
    if cache_key in _indicator_cache:
        return _indicator_cache[cache_key]
    print(f"      Pre-computing indicators for {cache_key}...", flush=True)
    t0 = time.time()
    regimes = classify_regime_v2(df, timeframe)
    atr_pct = compute_atr_pct(df, 14)
    sma50 = df["close"].rolling(50).mean()
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s", flush=True)
    _indicator_cache[cache_key] = (regimes, atr_pct, sma50)
    return regimes, atr_pct, sma50


def run_with_cached_indicators(exchange, symbol, profile_name, timeframe, df,
                                regimes, atr_pct_series, sma50,
                                base_order_pct=None, max_safety_orders=None,
                                tp_min=None, deviation_baseline=None):
    """Run backtest using pre-computed indicators (monkey-patched engine)."""
    engine = SpotBacktestEngine(
        profile=profile_name, capital=CAPITAL, exchange=exchange,
        symbol=symbol, timeframe=timeframe
    )
    # Override profile params
    orig = PROFILES[profile_name.lower()]
    if base_order_pct is not None:
        engine.profile.base_order_pct = base_order_pct
    if max_safety_orders is not None:
        engine.profile.max_safety_orders = max_safety_orders
    if tp_min is not None:
        ratio = tp_min / orig.tp_min
        engine.profile.tp_min = tp_min
        engine.profile.tp_baseline = orig.tp_baseline * ratio
        engine.profile.tp_max = orig.tp_max * ratio
    if deviation_baseline is not None:
        ratio = deviation_baseline / orig.deviation_baseline
        engine.profile.deviation_baseline = deviation_baseline
        engine.profile.deviation_min = orig.deviation_min * ratio
        engine.profile.deviation_max = orig.deviation_max * ratio

    # Run the engine's simulation loop directly with cached indicators
    result = _run_engine_cached(engine, df, regimes, atr_pct_series, sma50)
    return {
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "deals_completed": result.total_deals_completed,
        "win_rate": result.win_rate,
        "avg_profit_per_deal_pct": result.avg_profit_per_deal_pct,
        "total_fees_paid": result.total_fees_paid,
        "capital_utilization_pct": result.capital_utilization_pct,
    }


def _run_engine_cached(engine, df, regimes, atr_pct_series, sma50):
    """Replicate engine.run() but using pre-computed indicators."""
    from trading.spot.backtest_engine import BLOCKED_REGIMES
    
    if len(df) < 100:
        from trading.spot.backtest_engine import BacktestResult
        return BacktestResult()

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
            if not engine.deals and regime not in BLOCKED_REGIMES:
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


# ── Sweep configurations ──────────────────────────────────────────────

SWEEP_CONFIGS = {
    "low": {
        "profile": "low",
        "coins": [
            ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
            ("aster", "BTC/USDT", "aster_BTC_USDT_5m.csv"),
        ],
        "timeframes": ["15m", "1h"],
        "baseline": {"base_order_pct": 0.03, "max_safety_orders": 5,
                      "tp_min": 1.5, "deviation_baseline": 3.5},
        "sweeps": {
            "base_order_pct": [0.02, 0.03, 0.04, 0.05],
            "max_safety_orders": [3, 4, 5, 6],
            "tp_min": [1.0, 1.25, 1.5, 2.0],
            "deviation_baseline": [2.5, 3.0, 3.5, 4.0],
        },
    },
    "medium": {
        "profile": "medium",
        "coins": [
            ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
            ("aster", "BTC/USDT", "aster_BTC_USDT_5m.csv"),
        ],
        "timeframes": ["15m", "1h"],
        "baseline": {"base_order_pct": 0.04, "max_safety_orders": 8,
                      "tp_min": 1.0, "deviation_baseline": 2.5},
        "sweeps": {
            "base_order_pct": [0.03, 0.04, 0.05, 0.06],
            "max_safety_orders": [6, 7, 8, 9],
            "tp_min": [0.8, 1.0, 1.25, 1.5],
            "deviation_baseline": [2.0, 2.5, 3.0, 3.5],
        },
    },
    "high": {
        "profile": "high",
        "coins": [
            ("aster", "ETH/USDT", "aster_ETH_USDT_5m.csv"),
            ("hyperliquid", "HYPE/USDC", "hyperliquid_HYPE_USDC_5m.csv"),
        ],
        "timeframes": ["5m", "15m"],
        "baseline": {"base_order_pct": 0.05, "max_safety_orders": 12,
                      "tp_min": 0.8, "deviation_baseline": 2.0},
        "sweeps": {
            "base_order_pct": [0.04, 0.05, 0.06, 0.08],
            "max_safety_orders": [10, 11, 12, 14],
            "tp_min": [0.6, 0.8, 1.0, 1.2],
            "deviation_baseline": [1.5, 2.0, 2.5, 3.0],
        },
    },
}


def run_one_at_a_time_sweeps():
    """Vary one parameter at a time from baseline, record results."""
    all_results = {}
    run_count = 0

    for profile_key, cfg in SWEEP_CONFIGS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"  PROFILE: {profile_key.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        profile_results = {"baseline": cfg["baseline"], "sweeps": {}}

        for exchange, symbol, csv_file in cfg["coins"]:
            for tf in cfg["timeframes"]:
                combo_key = f"{exchange}_{symbol.replace('/', '_')}_{tf}"
                print(f"\n  [{combo_key}]", flush=True)
                df = get_df(csv_file, tf)
                print(f"    Data: {len(df)} candles", flush=True)

                # Pre-compute indicators ONCE for this dataset
                cache_key = f"{csv_file}_{tf}"
                regimes, atr_pct, sma50 = precompute_indicators(df, tf, cache_key)

                # Run baseline
                bl = cfg["baseline"]
                t0 = time.time()
                baseline_result = run_with_cached_indicators(
                    exchange, symbol, cfg["profile"], tf, df,
                    regimes, atr_pct, sma50, **bl)
                elapsed = time.time() - t0
                profile_results["sweeps"].setdefault(combo_key, {})
                profile_results["sweeps"][combo_key]["baseline"] = baseline_result
                run_count += 1
                print(f"    Baseline: {baseline_result['total_return_pct']:+.2f}% return, "
                      f"{baseline_result['max_drawdown_pct']:.2f}% DD ({elapsed:.1f}s)", flush=True)

                # Sweep each parameter
                for param_name, values in cfg["sweeps"].items():
                    param_results = []
                    for val in values:
                        params = dict(bl)
                        params[param_name] = val
                        r = run_with_cached_indicators(
                            exchange, symbol, cfg["profile"], tf, df,
                            regimes, atr_pct, sma50, **params)
                        r["param_value"] = val
                        param_results.append(r)
                        run_count += 1

                    profile_results["sweeps"][combo_key][param_name] = param_results

                    best = max(param_results, key=lambda x: x["total_return_pct"])
                    print(f"    {param_name}: best={best['param_value']} -> "
                          f"{best['total_return_pct']:+.2f}% (DD {best['max_drawdown_pct']:.2f}%)", flush=True)

        all_results[profile_key] = profile_results

        with open(RESULTS_DIR / f"per_param_sweep_{profile_key}.json", "w") as f:
            json.dump(profile_results, f, indent=2, default=str)

    print(f"\n  Total one-at-a-time runs: {run_count}", flush=True)
    return all_results


def find_optimal_params(all_results):
    """Analyze sweep results to find optimal parameters per profile."""
    optimal = {}

    for profile_key, profile_data in all_results.items():
        baseline = profile_data["baseline"]
        sweeps = profile_data["sweeps"]

        param_impact = {}

        for combo_key, combo_data in sweeps.items():
            for param_name in SWEEP_CONFIGS[profile_key]["sweeps"]:
                if param_name not in combo_data:
                    continue
                if param_name not in param_impact:
                    param_impact[param_name] = {}

                for entry in combo_data[param_name]:
                    val = entry["param_value"]
                    if val not in param_impact[param_name]:
                        param_impact[param_name][val] = {"returns": [], "dds": [], "sharpes": []}
                    param_impact[param_name][val]["returns"].append(entry["total_return_pct"])
                    param_impact[param_name][val]["dds"].append(entry["max_drawdown_pct"])
                    param_impact[param_name][val]["sharpes"].append(entry["sharpe_ratio"])

        profile_optimal = {"profile": profile_key, "params": {}, "sensitivity": {}}
        for param_name, val_data in param_impact.items():
            best_val = None
            best_avg_ret = -999
            all_avg_rets = []

            for val, metrics in sorted(val_data.items()):
                avg_ret = np.mean(metrics["returns"])
                avg_dd = np.mean(metrics["dds"])
                all_avg_rets.append((val, avg_ret, avg_dd))
                if avg_ret > best_avg_ret:
                    best_avg_ret = avg_ret
                    best_val = val

            rets_only = [r for _, r, _ in all_avg_rets]
            sensitivity = max(rets_only) - min(rets_only) if rets_only else 0

            profile_optimal["params"][param_name] = best_val
            profile_optimal["sensitivity"][param_name] = {
                "range_pct": round(sensitivity, 3),
                "values": [(v, round(r, 3), round(d, 3)) for v, r, d in all_avg_rets]
            }

        ranked = sorted(profile_optimal["sensitivity"].items(),
                        key=lambda x: x[1]["range_pct"], reverse=True)
        profile_optimal["sensitivity_ranking"] = [
            (name, data["range_pct"]) for name, data in ranked]

        optimal[profile_key] = profile_optimal

    return optimal


def run_focused_grid(all_results, optimal_info):
    """Run focused grid on top 2 most impactful parameters per profile."""
    print(f"\n{'='*60}", flush=True)
    print(f"  FOCUSED GRID SEARCH (top 2 params)", flush=True)
    print(f"{'='*60}", flush=True)

    grid_results = {}
    run_count = 0

    for profile_key, opt in optimal_info.items():
        ranking = opt["sensitivity_ranking"]
        if len(ranking) < 2:
            continue

        top2 = [ranking[0][0], ranking[1][0]]
        print(f"\n  {profile_key.upper()}: grid on {top2[0]} x {top2[1]}", flush=True)

        cfg = SWEEP_CONFIGS[profile_key]
        vals0 = cfg["sweeps"][top2[0]]
        vals1 = cfg["sweeps"][top2[1]]

        grid = []
        for exchange, symbol, csv_file in cfg["coins"]:
            for tf in cfg["timeframes"]:
                combo_key = f"{exchange}_{symbol.replace('/', '_')}_{tf}"
                df = get_df(csv_file, tf)
                cache_key = f"{csv_file}_{tf}"
                regimes, atr_pct, sma50 = precompute_indicators(df, tf, cache_key)

                for v0, v1 in product(vals0, vals1):
                    params = dict(cfg["baseline"])
                    params[top2[0]] = v0
                    params[top2[1]] = v1
                    r = run_with_cached_indicators(
                        exchange, symbol, cfg["profile"], tf, df,
                        regimes, atr_pct, sma50, **params)
                    r["combo"] = combo_key
                    r[top2[0]] = v0
                    r[top2[1]] = v1
                    grid.append(r)
                    run_count += 1

        best = max(grid, key=lambda x: x["total_return_pct"])
        print(f"    Best: {top2[0]}={best[top2[0]]}, {top2[1]}={best[top2[1]]} -> "
              f"{best['total_return_pct']:+.2f}% (DD {best['max_drawdown_pct']:.2f}%)", flush=True)

        best_ra = max(grid, key=lambda x: x["total_return_pct"] / max(x["max_drawdown_pct"], 0.5))
        print(f"    Best risk-adj: {top2[0]}={best_ra[top2[0]]}, {top2[1]}={best_ra[top2[1]]} -> "
              f"{best_ra['total_return_pct']:+.2f}% (DD {best_ra['max_drawdown_pct']:.2f}%)", flush=True)

        grid_results[profile_key] = {
            "grid_params": top2,
            "grid": grid,
            "best_return": {top2[0]: best[top2[0]], top2[1]: best[top2[1]],
                            "return": best["total_return_pct"], "dd": best["max_drawdown_pct"]},
            "best_risk_adjusted": {top2[0]: best_ra[top2[0]], top2[1]: best_ra[top2[1]],
                                    "return": best_ra["total_return_pct"], "dd": best_ra["max_drawdown_pct"]},
        }

        opt["params"][top2[0]] = best_ra[top2[0]]
        opt["params"][top2[1]] = best_ra[top2[1]]
        opt["grid_results"] = grid_results[profile_key]

    print(f"\n  Grid runs: {run_count}", flush=True)
    return grid_results


def generate_report(optimal_info, all_results):
    """Generate OPTIMIZATION_REPORT.md."""
    lines = ["# Phase 5: Parameter Optimization Report\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"Capital: ${CAPITAL:,}\n")

    for profile_key in ["low", "medium", "high"]:
        opt = optimal_info[profile_key]
        cfg = SWEEP_CONFIGS[profile_key]
        bl = cfg["baseline"]

        lines.append(f"\n## {profile_key.upper()} RISK Profile\n")
        lines.append(f"Coins: {', '.join(s for _, s, _ in cfg['coins'])}")
        lines.append(f"Timeframes: {', '.join(cfg['timeframes'])}\n")

        # Sensitivity ranking
        lines.append("### Parameter Sensitivity (most to least impactful)\n")
        lines.append("| Rank | Parameter | Impact Range (%) | Most Impactful? |")
        lines.append("|------|-----------|-----------------|-----------------|")
        for i, (pname, impact) in enumerate(opt["sensitivity_ranking"], 1):
            flag = "⭐" if i <= 2 else ""
            lines.append(f"| {i} | {pname} | {impact:.3f} | {flag} |")

        # Per-parameter tables
        lines.append("\n### Parameter Sweep Details\n")
        for pname, sdata in opt["sensitivity"].items():
            lines.append(f"#### {pname}\n")
            lines.append("| Value | Avg Return (%) | Avg DD (%) |")
            lines.append("|-------|---------------|------------|")
            for val, ret, dd in sdata["values"]:
                marker = " ◀ optimal" if val == opt["params"][pname] else ""
                lines.append(f"| {val} | {ret:+.3f} | {dd:.3f} |{marker}")
            lines.append("")

        # Optimal params
        lines.append("### Recommended Parameters\n")
        lines.append("| Parameter | Default | Optimized | Change |")
        lines.append("|-----------|---------|-----------|--------|")
        for pname, opt_val in opt["params"].items():
            default_val = bl[pname]
            if isinstance(opt_val, float) and opt_val < 1:
                lines.append(f"| {pname} | {default_val*100:.1f}% | {opt_val*100:.1f}% | "
                             f"{'↑' if opt_val > default_val else '↓' if opt_val < default_val else '='} |")
            else:
                lines.append(f"| {pname} | {default_val} | {opt_val} | "
                             f"{'↑' if opt_val > default_val else '↓' if opt_val < default_val else '='} |")

        # Grid results
        if "grid_results" in opt:
            gr = opt["grid_results"]
            lines.append(f"\n### Focused Grid Search ({gr['grid_params'][0]} × {gr['grid_params'][1]})\n")
            lines.append(f"- **Best return:** {gr['best_return']['return']:+.2f}% "
                         f"(DD {gr['best_return']['dd']:.2f}%)")
            lines.append(f"- **Best risk-adjusted:** {gr['best_risk_adjusted']['return']:+.2f}% "
                         f"(DD {gr['best_risk_adjusted']['dd']:.2f}%)")

    # Final summary
    lines.append("\n## Final Recommended Configurations\n")
    for profile_key in ["low", "medium", "high"]:
        opt = optimal_info[profile_key]
        lines.append(f"\n### {profile_key.upper()} RISK\n")
        lines.append("```")
        for pname, val in opt["params"].items():
            if isinstance(val, float) and val < 1:
                lines.append(f"  {pname}: {val*100:.1f}%")
            else:
                lines.append(f"  {pname}: {val}")
        lines.append("```")
        lines.append(f"Sensitivity ranking: {' > '.join(n for n, _ in opt['sensitivity_ranking'])}")

    report = "\n".join(lines)
    with open(RESULTS_DIR / "OPTIMIZATION_REPORT.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to {RESULTS_DIR / 'OPTIMIZATION_REPORT.md'}", flush=True)
    return report


def main():
    t0 = time.time()

    print("Phase 5.1: One-at-a-time parameter sweeps...", flush=True)
    all_results = run_one_at_a_time_sweeps()

    print("\n\nPhase 5.2: Analyzing parameter sensitivity...", flush=True)
    optimal_info = find_optimal_params(all_results)

    for pk, opt in optimal_info.items():
        print(f"\n  {pk.upper()} sensitivity: {opt['sensitivity_ranking']}", flush=True)
        print(f"  {pk.upper()} optimal: {opt['params']}", flush=True)

    print("\n\nPhase 5.3: Focused grid search...", flush=True)
    run_focused_grid(all_results, optimal_info)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_DIR / "optimal_params.json", "w") as f:
        json.dump(optimal_info, f, indent=2, default=convert)

    report = generate_report(optimal_info, all_results)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Phase 5 complete in {elapsed:.0f}s", flush=True)
    print(f"{'='*60}", flush=True)

    print("\n\n=== FINAL RECOMMENDED PARAMETERS ===\n", flush=True)
    for pk in ["low", "medium", "high"]:
        opt = optimal_info[pk]
        print(f"\n{pk.upper()} RISK:", flush=True)
        for pname, val in opt["params"].items():
            if isinstance(val, float) and val < 1:
                print(f"  {pname}: {val*100:.1f}%", flush=True)
            else:
                print(f"  {pname}: {val}", flush=True)
        print(f"  Sensitivity: {' > '.join(n for n, _ in opt['sensitivity_ranking'])}", flush=True)


if __name__ == "__main__":
    main()
