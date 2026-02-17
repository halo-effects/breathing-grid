import ccxt, sys, traceback

# Test Aster
print("Testing Aster spot...")
try:
    e = ccxt.aster({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    e.load_markets()
    spot = [s for s, m in e.markets.items() if m.get("spot")]
    print(f"  Aster spot markets: {len(spot)}")
    print(f"  First 10: {spot[:10]}")
except Exception as ex:
    print(f"  Aster failed: {ex}")
    traceback.print_exc()

# Test Hyperliquid
print("\nTesting Hyperliquid spot...")
try:
    e2 = ccxt.hyperliquid({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    e2.load_markets()
    spot2 = [s for s, m in e2.markets.items() if m.get("spot")]
    print(f"  Hyperliquid spot markets: {len(spot2)}")
    print(f"  First 10: {spot2[:10]}")
except Exception as ex:
    print(f"  Hyperliquid failed: {ex}")
    traceback.print_exc()
