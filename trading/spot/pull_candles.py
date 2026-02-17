"""Phase 3: Pull historical candle data for spot DCA backtesting.

Fetches 5m candles (30-90 days) from Aster and Hyperliquid spot markets.
Saves CSV files to trading/spot/data/.
"""
import sys, os, time, json, csv
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trading.spot.exchange_client import SpotExchangeClient

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_PATH = Path(__file__).parent / "spot_config.json"

# Target coins per exchange
TARGETS = {
    "aster": {
        "quote": "USDT",
        "pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ASTER/USDT", "DOGE/USDT"],
    },
    "hyperliquid": {
        "quote": "USDC",
        "pairs": ["HYPE/USDC", "BTC/USDC", "ETH/USDC"],
    },
}

TIMEFRAME = "5m"
CANDLES_PER_BATCH = 1000  # max per request (exchange dependent)
TARGET_DAYS = 90  # try 90, fall back to 30
MS_5M = 5 * 60 * 1000


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def fetch_all_candles(client: SpotExchangeClient, symbol: str, days: int = 90):
    """Fetch candles by paginating backwards. Returns list of [ts, o, h, l, c, v]."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    cursor = since_ms
    
    while cursor < now_ms:
        try:
            # Use exchange's fetch_ohlcv with since parameter
            batch = client.exchange.fetch_ohlcv(symbol, TIMEFRAME, since=cursor, limit=CANDLES_PER_BATCH)
        except Exception as e:
            print(f"  Error fetching {symbol} from {cursor}: {e}")
            break
        
        if not batch:
            break
        
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        
        if last_ts <= cursor:
            break  # no progress
        cursor = last_ts + MS_5M
        
        # Rate limit
        time.sleep(0.5)
    
    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    
    unique.sort(key=lambda x: x[0])
    return unique


def save_csv(candles, exchange_name: str, symbol: str):
    """Save candles to CSV file."""
    safe_symbol = symbol.replace("/", "_")
    filename = f"{exchange_name}_{safe_symbol}_{TIMEFRAME}.csv"
    filepath = DATA_DIR / filename
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c[0], c[1], c[2], c[3], c[4], c[5]])
    
    return filepath


def check_data_quality(candles, symbol: str, exchange_name: str):
    """Check for gaps, thin volume, sufficient history."""
    if not candles:
        return {"symbol": symbol, "exchange": exchange_name, "status": "NO_DATA", "days": 0}
    
    ts = [c[0] for c in candles]
    volumes = [c[5] for c in candles]
    
    first_dt = datetime.fromtimestamp(ts[0] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(ts[-1] / 1000, tz=timezone.utc)
    days_span = (last_dt - first_dt).total_seconds() / 86400
    
    # Check gaps (>15m = 3 missing candles)
    gaps = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 3 * MS_5M:
            gaps += 1
    
    # Volume check
    avg_vol_usd = sum(v * candles[i][4] for i, v in enumerate(volumes)) / max(len(volumes), 1)
    daily_vol_usd = avg_vol_usd * 288  # 288 5m candles per day
    
    # ATR warmup check (need 14 candles minimum before backtest starts)
    enough_history = len(candles) >= (30 * 288 + 14)  # 30 days + warmup
    
    return {
        "symbol": symbol,
        "exchange": exchange_name,
        "status": "OK" if enough_history else "SHORT_HISTORY",
        "candle_count": len(candles),
        "days": round(days_span, 1),
        "gaps": gaps,
        "avg_daily_vol_usd": round(daily_vol_usd, 2),
        "thin_volume": daily_vol_usd < 100_000,
        "enough_for_backtest": enough_history,
    }


def main():
    config = load_config()
    results = {}
    
    for exch_name, target_info in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"Exchange: {exch_name.upper()}")
        print(f"{'='*60}")
        
        client = SpotExchangeClient()
        
        try:
            exch_config = config.get("exchanges", {}).get(exch_name, {})
            client.connect(exch_name, exch_config)
        except Exception as e:
            print(f"  Failed to connect to {exch_name}: {e}")
            continue
        
        # Check which pairs are available
        try:
            available = set(client.list_spot_markets(target_info["quote"]))
            print(f"  Available {target_info['quote']} spot pairs: {len(available)}")
        except Exception as e:
            print(f"  Failed to list markets: {e}")
            available = set()
        
        for symbol in target_info["pairs"]:
            print(f"\n  Fetching {symbol}...")
            
            if available and symbol not in available:
                print(f"    NOT AVAILABLE on {exch_name} spot")
                results[f"{exch_name}:{symbol}"] = {
                    "symbol": symbol, "exchange": exch_name,
                    "status": "NOT_AVAILABLE", "days": 0
                }
                continue
            
            # Try 90 days first, fall back to 30
            candles = fetch_all_candles(client, symbol, days=TARGET_DAYS)
            
            if not candles:
                print(f"    No data returned, trying 30 days...")
                candles = fetch_all_candles(client, symbol, days=30)
            
            if candles:
                filepath = save_csv(candles, exch_name, symbol)
                quality = check_data_quality(candles, symbol, exch_name)
                print(f"    Saved {len(candles)} candles ({quality['days']} days) to {filepath.name}")
                print(f"    Gaps: {quality['gaps']}, Daily Vol: ${quality['avg_daily_vol_usd']:,.0f}")
                if quality['thin_volume']:
                    print(f"    ⚠️ THIN VOLUME - may not be reliable for backtesting")
                results[f"{exch_name}:{symbol}"] = quality
            else:
                print(f"    No data available")
                results[f"{exch_name}:{symbol}"] = {
                    "symbol": symbol, "exchange": exch_name,
                    "status": "NO_DATA", "days": 0
                }
    
    # Save results summary
    summary_path = DATA_DIR / "data_quality_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nData quality report saved to {summary_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, info in results.items():
        status = info.get("status", "UNKNOWN")
        days = info.get("days", 0)
        emoji = "✅" if status == "OK" else ("⚠️" if status == "SHORT_HISTORY" else "❌")
        print(f"  {emoji} {key}: {status} ({days} days)")


if __name__ == "__main__":
    main()
