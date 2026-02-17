"""Phase 3: Coin screening for spot DCA suitability.

Fetches all spot pairs from Aster and Hyperliquid, applies maturity and 
DCA-suitability filters, scores and ranks coins, outputs results to JSON.
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trading.spot.exchange_client import SpotExchangeClient
from trading.indicators import atr_pct, hurst_exponent

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = Path(__file__).parent / "screening_results.json"
CONFIG_PATH = Path(__file__).parent / "spot_config.json"

# ── Filter thresholds (from strategy spec Section 5) ──────────────────
MIN_HISTORY_DAYS = 60          # 60+ days of price history
MIN_DAILY_VOL_USD = 1_000_000  # >$1M daily average
MAX_30D_GAIN_PCT = 100         # reject >100% gain in 30d (parabolic)
VOL_SPIKE_RATIO = 4.0          # flag if any day >4x average volume
ATR_IDEAL_LOW = 1.0            # ideal daily ATR% range
ATR_IDEAL_HIGH = 3.0
ATR_MAX = 5.0                  # reject above 5%
HURST_IDEAL_LOW = 0.3          # mean-reverting range (good for DCA)
HURST_IDEAL_HIGH = 0.5

CANDLES_PER_BATCH = 1000
MS_1D = 24 * 60 * 60 * 1000


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def fetch_daily_candles(client, symbol, days=90):
    """Fetch daily candles for screening."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    cursor = since_ms
    
    while cursor < now_ms:
        try:
            batch = client.exchange.fetch_ohlcv(symbol, "1d", since=cursor, limit=CANDLES_PER_BATCH)
        except Exception:
            break
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + MS_1D
        time.sleep(0.3)
    
    # Deduplicate
    seen = set()
    unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    unique.sort(key=lambda x: x[0])
    return unique


def candles_to_df(candles):
    """Convert raw candles to DataFrame."""
    if not candles:
        return None
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def screen_coin(symbol: str, candles, exchange_name: str, quote: str):
    """Apply all screening filters to a single coin. Returns dict with scores."""
    df = candles_to_df(candles)
    if df is None or len(df) < 14:
        return None
    
    result = {
        "symbol": symbol,
        "exchange": exchange_name,
        "quote": quote,
        "days_history": len(df),
        "filters": {},
        "scores": {},
        "passed": True,
        "flags": [],
        "total_score": 0,
    }
    
    # ── Filter 1: Age (history) ────────────────────────────────
    age_ok = len(df) >= MIN_HISTORY_DAYS
    result["filters"]["age"] = {"days": len(df), "required": MIN_HISTORY_DAYS, "passed": age_ok}
    if not age_ok:
        result["passed"] = False
        result["flags"].append(f"Insufficient history: {len(df)} days < {MIN_HISTORY_DAYS}")
    
    # ── Filter 2: Volume ───────────────────────────────────────
    df["vol_usd"] = df["volume"] * df["close"]
    avg_daily_vol = df["vol_usd"].mean()
    vol_ok = avg_daily_vol >= MIN_DAILY_VOL_USD
    result["filters"]["volume"] = {
        "avg_daily_usd": round(avg_daily_vol, 2),
        "required": MIN_DAILY_VOL_USD,
        "passed": vol_ok
    }
    if not vol_ok:
        result["passed"] = False
        result["flags"].append(f"Low volume: ${avg_daily_vol:,.0f} < ${MIN_DAILY_VOL_USD:,.0f}")
    
    # ── Filter 3: Parabolic move check ─────────────────────────
    if len(df) >= 30:
        last_30 = df.tail(30)
        price_min = last_30["low"].min()
        price_max = last_30["high"].max()
        gain_30d = ((price_max - price_min) / price_min) * 100 if price_min > 0 else 0
    else:
        gain_30d = 0
    
    parabolic_ok = gain_30d <= MAX_30D_GAIN_PCT
    result["filters"]["parabolic"] = {
        "gain_30d_pct": round(gain_30d, 2),
        "max_allowed": MAX_30D_GAIN_PCT,
        "passed": parabolic_ok
    }
    if not parabolic_ok:
        result["passed"] = False
        result["flags"].append(f"Parabolic: {gain_30d:.1f}% gain in 30d")
    
    # ── Filter 4: Volume spikes ────────────────────────────────
    vol_mean = df["vol_usd"].mean()
    vol_max = df["vol_usd"].max()
    spike_ratio = vol_max / vol_mean if vol_mean > 0 else 0
    spike_ok = spike_ratio <= VOL_SPIKE_RATIO
    result["filters"]["volume_spikes"] = {
        "max_ratio": round(spike_ratio, 2),
        "threshold": VOL_SPIKE_RATIO,
        "passed": spike_ok
    }
    if not spike_ok:
        result["flags"].append(f"Volume spike: {spike_ratio:.1f}x average")
    
    # ── Filter 5: ATR volatility ───────────────────────────────
    if len(df) >= 14:
        atr_series = atr_pct(df, 14)
        current_atr = atr_series.iloc[-1] if not pd.isna(atr_series.iloc[-1]) else atr_series.dropna().iloc[-1] if len(atr_series.dropna()) > 0 else 0
        avg_atr = atr_series.dropna().mean()
    else:
        current_atr = 0
        avg_atr = 0
    
    atr_ok = avg_atr <= ATR_MAX
    atr_ideal = ATR_IDEAL_LOW <= avg_atr <= ATR_IDEAL_HIGH
    result["filters"]["volatility"] = {
        "avg_atr_pct": round(avg_atr, 3),
        "current_atr_pct": round(current_atr, 3),
        "ideal_range": [ATR_IDEAL_LOW, ATR_IDEAL_HIGH],
        "passed": atr_ok,
        "ideal": atr_ideal
    }
    if not atr_ok:
        result["passed"] = False
        result["flags"].append(f"Extreme volatility: ATR {avg_atr:.2f}%")
    
    # ── Filter 6: Hurst exponent ───────────────────────────────
    if len(df) >= 60:
        hurst_series = hurst_exponent(df["close"], min_chunk=8, max_chunk=min(30, len(df) // 3))
        hurst_val = hurst_series.dropna().iloc[-1] if len(hurst_series.dropna()) > 0 else 0.5
        avg_hurst = hurst_series.dropna().mean()
    else:
        hurst_val = 0.5
        avg_hurst = 0.5
    
    hurst_ideal = HURST_IDEAL_LOW <= avg_hurst <= HURST_IDEAL_HIGH
    result["filters"]["hurst"] = {
        "current": round(hurst_val, 3),
        "average": round(avg_hurst, 3),
        "ideal_range": [HURST_IDEAL_LOW, HURST_IDEAL_HIGH],
        "ideal": hurst_ideal
    }
    
    # ── Scoring ────────────────────────────────────────────────
    # Volume score (0-25): higher volume = better liquidity
    vol_score = min(25, (avg_daily_vol / 10_000_000) * 25) if avg_daily_vol > 0 else 0
    result["scores"]["volume"] = round(vol_score, 2)
    
    # ATR score (0-25): closer to ideal range = better
    if ATR_IDEAL_LOW <= avg_atr <= ATR_IDEAL_HIGH:
        atr_score = 25
    elif avg_atr < ATR_IDEAL_LOW:
        atr_score = max(0, 25 * (avg_atr / ATR_IDEAL_LOW))
    else:
        atr_score = max(0, 25 * (1 - (avg_atr - ATR_IDEAL_HIGH) / ATR_IDEAL_HIGH))
    result["scores"]["volatility"] = round(atr_score, 2)
    
    # Hurst score (0-25): closer to 0.4 = better for DCA (mean-reverting)
    hurst_target = 0.4
    hurst_dist = abs(avg_hurst - hurst_target)
    hurst_score = max(0, 25 * (1 - hurst_dist / 0.3))
    result["scores"]["hurst"] = round(hurst_score, 2)
    
    # Volume consistency score (0-25): lower CV = more consistent
    vol_cv = df["vol_usd"].std() / df["vol_usd"].mean() if df["vol_usd"].mean() > 0 else 999
    consistency_score = max(0, 25 * (1 - min(vol_cv / 3, 1)))
    result["scores"]["consistency"] = round(consistency_score, 2)
    
    result["total_score"] = round(vol_score + atr_score + hurst_score + consistency_score, 2)
    
    return result


def cross_reference(aster_symbols, hyper_symbols):
    """Find coins available on both exchanges."""
    # Extract base currencies
    aster_bases = {s.split("/")[0] for s in aster_symbols}
    hyper_bases = {s.split("/")[0] for s in hyper_symbols}
    
    both = aster_bases & hyper_bases
    aster_only = aster_bases - hyper_bases
    hyper_only = hyper_bases - aster_bases
    
    return {
        "both_exchanges": sorted(both),
        "aster_only": sorted(aster_only),
        "hyperliquid_only": sorted(hyper_only),
        "note": "Aster uses USDT quote, Hyperliquid uses USDC quote"
    }


def main():
    config = load_config()
    all_results = []
    exchange_pairs = {}  # exchange -> list of symbols
    
    exchanges = {
        "aster": {"quote": "USDT"},
        "hyperliquid": {"quote": "USDC"},
    }
    
    for exch_name, info in exchanges.items():
        print(f"\n{'='*60}")
        print(f"Screening {exch_name.upper()} spot markets")
        print(f"{'='*60}")
        
        client = SpotExchangeClient()
        try:
            exch_config = config.get("exchanges", {}).get(exch_name, {})
            client.connect(exch_name, exch_config)
        except Exception as e:
            print(f"  Failed to connect to {exch_name}: {e}")
            continue
        
        # Get all available spot pairs
        try:
            symbols = client.list_spot_markets(info["quote"])
            print(f"  Found {len(symbols)} {info['quote']} spot pairs")
            exchange_pairs[exch_name] = symbols
        except Exception as e:
            print(f"  Failed to list markets: {e}")
            continue
        
        passed = 0
        failed = 0
        
        for i, symbol in enumerate(symbols):
            base = symbol.split("/")[0]
            print(f"  [{i+1}/{len(symbols)}] Screening {symbol}...", end=" ", flush=True)
            
            try:
                candles = fetch_daily_candles(client, symbol, days=90)
            except Exception as e:
                print(f"ERROR: {e}")
                failed += 1
                continue
            
            if not candles or len(candles) < 14:
                print(f"insufficient data ({len(candles) if candles else 0} days)")
                failed += 1
                continue
            
            result = screen_coin(symbol, candles, exch_name, info["quote"])
            if result:
                all_results.append(result)
                status = "✅ PASS" if result["passed"] else "❌ FAIL"
                print(f"{status} (score: {result['total_score']:.1f}, ATR: {result['filters']['volatility']['avg_atr_pct']:.2f}%)")
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
            else:
                print("no data")
                failed += 1
            
            time.sleep(0.2)  # rate limit
        
        print(f"\n  {exch_name}: {passed} passed, {failed} failed out of {len(symbols)}")
    
    # Sort by score
    all_results.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Cross-reference
    xref = cross_reference(
        exchange_pairs.get("aster", []),
        exchange_pairs.get("hyperliquid", [])
    )
    
    # Build output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cross_reference": xref,
        "summary": {
            "total_screened": len(all_results),
            "total_passed": sum(1 for r in all_results if r["passed"]),
            "by_exchange": {}
        },
        "top_10": [r for r in all_results if r["passed"]][:10],
        "all_results": all_results,
    }
    
    # Per-exchange summary
    for exch in exchanges:
        exch_results = [r for r in all_results if r["exchange"] == exch]
        output["summary"]["by_exchange"][exch] = {
            "screened": len(exch_results),
            "passed": sum(1 for r in exch_results if r["passed"]),
        }
    
    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to {OUTPUT_PATH}")
    
    # Print top 10
    print(f"\n{'='*60}")
    print("TOP 10 COINS FOR SPOT DCA")
    print(f"{'='*60}")
    for i, r in enumerate(output["top_10"][:10], 1):
        flags = ", ".join(r["flags"]) if r["flags"] else "clean"
        print(f"  {i}. {r['symbol']} ({r['exchange']}) — Score: {r['total_score']:.1f} | "
              f"ATR: {r['filters']['volatility']['avg_atr_pct']:.2f}% | "
              f"Hurst: {r['filters']['hurst']['average']:.3f} | "
              f"Vol: ${r['filters']['volume']['avg_daily_usd']:,.0f}/day | "
              f"Flags: {flags}")
    
    # Cross-reference output
    print(f"\n{'='*60}")
    print("CROSS-REFERENCE")
    print(f"{'='*60}")
    print(f"  Both exchanges: {', '.join(xref['both_exchanges']) or 'none'}")
    print(f"  Aster only: {', '.join(list(xref['aster_only'])[:10])}{'...' if len(xref['aster_only']) > 10 else ''}")
    print(f"  Hyperliquid only: {', '.join(list(xref['hyperliquid_only'])[:10])}{'...' if len(xref['hyperliquid_only']) > 10 else ''}")
    print(f"  Note: {xref['note']}")


if __name__ == "__main__":
    main()
