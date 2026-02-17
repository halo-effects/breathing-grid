"""Read-only test of SpotExchangeClient against Aster and Hyperliquid."""
import logging
import sys
import os

# Add workspace root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from trading.spot.exchange_client import SpotExchangeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def test_exchange(name: str, symbols: list, auth: bool = True):
    """Test an exchange with read-only operations."""
    section(f"{name.upper()} SPOT")
    client = SpotExchangeClient()

    try:
        if auth:
            client.connect(name)
        else:
            client.connect(name, config={"apiKey": "", "secret": ""})
        print(f"[OK] Connected to {name}")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return

    # List spot markets
    try:
        markets = client.list_spot_markets("USDT")
        print(f"\n— Spot USDT markets ({len(markets)}):")
        for m in markets[:30]:
            print(f"    {m}")
        if len(markets) > 30:
            print(f"    ... and {len(markets)-30} more")
    except Exception as e:
        print(f"✗ list_spot_markets failed: {e}")

    # Tickers
    for sym in symbols:
        try:
            t = client.fetch_ticker(sym)
            print(f"\n— Ticker {sym}:")
            print(f"    last={t.get('last')}  bid={t.get('bid')}  ask={t.get('ask')}  vol={t.get('baseVolume')}")
        except Exception as e:
            print(f"✗ Ticker {sym}: {e}")

    # OHLCV for first symbol
    try:
        candles = client.fetch_ohlcv(symbols[0], "5m", 5)
        print(f"\n— OHLCV {symbols[0]} (last 5 x 5m):")
        for c in candles[-5:]:
            print(f"    ts={c[0]}  O={c[1]}  H={c[2]}  L={c[3]}  C={c[4]}  V={c[5]}")
    except Exception as e:
        print(f"✗ OHLCV: {e}")

    # Min order sizes
    for sym in symbols:
        try:
            info = client.get_min_order_size(sym)
            print(f"\n— Min order {sym}: {info}")
        except Exception as e:
            print(f"✗ Min order {sym}: {e}")

    # Fees
    for sym in symbols[:2]:
        try:
            fees = client.get_trading_fees(sym)
            print(f"— Fees {sym}: maker={fees['maker']}  taker={fees['taker']}")
        except Exception as e:
            print(f"✗ Fees {sym}: {e}")

    # Balance (only if authenticated)
    if auth:
        try:
            bal = client.fetch_balance("USDT")
            print(f"\n— USDT Balance: free={bal['free']}  used={bal['used']}  total={bal['total']}")
        except Exception as e:
            print(f"✗ Balance: {e}")


if __name__ == "__main__":
    # Aster spot (authenticated)
    test_exchange("aster", ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ASTER/USDT"], auth=True)

    # Hyperliquid spot (public only)
    test_exchange("hyperliquid", ["HYPE/USDC", "PURR/USDC", "JEFF/USDC"], auth=False)

    # Try alternate Hyperliquid symbol formats
    section("HYPERLIQUID SYMBOL DISCOVERY")
    client = SpotExchangeClient()
    try:
        client.connect("hyperliquid", config={"apiKey": "", "secret": ""})
        all_markets = client.exchange.markets
        spot_markets = [s for s, m in all_markets.items() if m.get("spot")]
        print(f"Hyperliquid spot markets ({len(spot_markets)}):")
        for s in sorted(spot_markets)[:30]:
            print(f"    {s}")
        if len(spot_markets) > 30:
            print(f"    ... and {len(spot_markets)-30} more")
    except Exception as e:
        print(f"✗ Hyperliquid discovery: {e}")

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
