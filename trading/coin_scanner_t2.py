"""Tier 2: Deep backtest scan for shortlisted coins.

Runs martingale backtest on 14-day 5m data for each candidate.
Standalone: python trading/coin_scanner_t2.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

from trading.config import MartingaleConfig
from trading.martingale_engine import MartingaleBot
from trading.regime_detector import classify_regime_v2, is_martingale_friendly_v2
from trading.data_fetcher import fetch_ohlcv

LIVE_DIR = Path(__file__).parent / "live"
LIVE_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = LIVE_DIR / "scanner_t2.json"

# Live bot params
LIVE_CONFIG = MartingaleConfig(
    base_order_size=335 * 0.04,       # 4% of $335 = $13.40
    safety_order_size=335 * 0.04,     # same as base
    safety_order_multiplier=2.0,
    price_deviation_pct=2.5,
    deviation_multiplier=1.5,
    max_safety_orders=8,
    max_active_deals=1,
    take_profit_pct=1.5,
    trailing_tp_pct=None,
    fee_pct=0.05,
    slippage_pct=0.03,
    initial_capital=335.0,
)


def fetch_5m_data(ccxt_symbol: str, days: int = 14) -> pd.DataFrame | None:
    """Fetch 5m klines for deep analysis."""
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    limit = days * 24 * 12 + 20  # 5m candles per day = 288
    try:
        df = fetch_ohlcv(ccxt_symbol, "5m", since=since_ms, limit=limit)
        return df if df is not None and len(df) > 100 else None
    except Exception as e:
        print(f"  [error] {ccxt_symbol} 5m: {e}")
        return None


def deep_scan_coin(ccxt_symbol: str, df_5m: pd.DataFrame) -> dict | None:
    """Run backtest and compute deep metrics for a single coin."""
    # Classify regimes
    regimes = classify_regime_v2(df_5m, timeframe="5m")

    # Regime distribution
    total_bars = len(regimes)
    friendly_regimes = ["RANGING", "CHOPPY", "ACCUMULATION"]
    friendly_count = regimes.isin(friendly_regimes).sum()
    friendly_pct = friendly_count / total_bars * 100

    regime_dist = regimes.value_counts(normalize=True).to_dict()
    regime_dist = {k: round(v * 100, 1) for k, v in regime_dist.items()}

    # Run backtest
    bot = MartingaleBot(LIVE_CONFIG, bidirectional=False)
    result = bot.run(df_5m, ccxt_symbol, "5m",
                     precomputed_regimes=regimes,
                     friendly_fn=is_martingale_friendly_v2)

    # Metrics
    n_deals = result.total_trades
    if n_deals == 0:
        return None

    total_profit = result.total_profit
    total_profit_pct = result.total_profit_pct
    max_dd = abs(result.max_drawdown) if result.max_drawdown != 0 else 0.01

    # Duration in days
    if len(df_5m) > 1:
        ts_first = pd.Timestamp(df_5m["timestamp"].iloc[0])
        ts_last = pd.Timestamp(df_5m["timestamp"].iloc[-1])
        duration_days = max((ts_last - ts_first).total_seconds() / 86400, 1)
    else:
        duration_days = 1

    daily_roi = total_profit_pct / duration_days
    deals_per_day = n_deals / duration_days

    # Risk-adjusted score
    risk_adj = daily_roi / max_dd if max_dd > 0 else 0

    # Composite score (weighted)
    # daily_roi weight=40%, deals_per_day weight=20%, friendly_pct weight=20%, risk_adj weight=20%
    score = (
        0.40 * min(daily_roi / 2.0, 1.0) * 100 +  # normalize: 2% daily = perfect
        0.20 * min(deals_per_day / 5.0, 1.0) * 100 +  # 5 deals/day = perfect
        0.20 * (friendly_pct / 100) * 100 +
        0.20 * min(abs(risk_adj) * 10, 1.0) * 100
    )

    return {
        "symbol": ccxt_symbol,
        "total_deals": n_deals,
        "win_rate": round(result.win_rate, 1),
        "total_profit": round(total_profit, 2),
        "total_profit_pct": round(total_profit_pct, 2),
        "max_drawdown_pct": round(result.max_drawdown, 2),
        "daily_roi_pct": round(daily_roi, 4),
        "deals_per_day": round(deals_per_day, 2),
        "friendly_regime_pct": round(friendly_pct, 1),
        "regime_distribution": regime_dist,
        "risk_adjusted_score": round(risk_adj, 4),
        "profit_factor": round(result.profit_factor, 2) if result.profit_factor != float('inf') else 9999,
        "composite_score": round(score, 1),
        "duration_days": round(duration_days, 1),
        "avg_so_per_deal": round(np.mean([d.so_count for d in result.closed_deals]), 1) if result.closed_deals else 0,
    }


def run_tier2(candidates: list[dict]) -> list[dict]:
    """Run Tier 2 deep scan on candidate list from Tier 1.
    
    candidates: list of dicts with at least 'symbol' key (ccxt format).
    """
    print("=" * 70)
    print("TIER 2: DEEP BACKTEST SCAN")
    print("=" * 70)

    results = []
    for i, cand in enumerate(candidates):
        sym = cand["symbol"]
        print(f"\n  ({i+1}/{len(candidates)}) {sym} — fetching 14d 5m data...", flush=True)

        df_5m = fetch_5m_data(sym)
        if df_5m is None:
            print(f"    skip (no 5m data)")
            continue

        print(f"    Got {len(df_5m)} candles. Running backtest...", flush=True)
        metrics = deep_scan_coin(sym, df_5m)
        if metrics is None:
            print(f"    skip (0 deals in backtest)")
            continue

        # Carry forward T1 score
        metrics["t1_score"] = cand.get("total_score", 0)
        results.append(metrics)
        print(f"    score={metrics['composite_score']} | deals={metrics['total_deals']} | ROI/day={metrics['daily_roi_pct']}% | DD={metrics['max_drawdown_pct']}%")

    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates_tested": len(candidates),
        "passed": len(results),
        "rankings": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'Rank':<5} {'Symbol':<14} {'Score':<7} {'Deals':<7} {'WinR%':<7} {'ROI/d%':<8} {'DD%':<8} {'Deals/d':<8} {'FriendR%':<9}")
    print("-" * 80)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['symbol']:<14} {r['composite_score']:<7} {r['total_deals']:<7} {r['win_rate']:<7} {r['daily_roi_pct']:<8} {r['max_drawdown_pct']:<8} {r['deals_per_day']:<8} {r['friendly_regime_pct']:<9}")

    return results


if __name__ == "__main__":
    # If run standalone, load T1 results
    t1_path = LIVE_DIR / "scanner_t1.json"
    if t1_path.exists():
        t1_data = json.loads(t1_path.read_text())
        candidates = t1_data.get("candidates", [])[:8]
        run_tier2(candidates)
    else:
        print("No Tier 1 results found. Run coin_scanner_t1.py first.")
