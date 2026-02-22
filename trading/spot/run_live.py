#!/usr/bin/env python3
"""Entry point for the Spot DCA Scale-Out LIVE Trading Bot.

⚠️ This places REAL orders with REAL money on the exchange.
Use --test to verify connectivity before running live.
"""
import argparse
import sys
import os

# Add workspace root to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.spot.lifecycle_trader import LifecycleTrader, DEFAULT_SYMBOLS


def main():
    parser = argparse.ArgumentParser(description="Spot DCA Scale-Out LIVE Trader")
    parser.add_argument("--exchange", default="aster", choices=["aster", "hyperliquid"],
                        help="Exchange to connect to (default: aster)")
    parser.add_argument("--profile", default="medium", choices=["low", "medium", "high"],
                        help="Risk profile (default: medium)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Reference capital for position sizing (default: 10000)")
    parser.add_argument("--symbol", default="auto",
                        help="Symbol to trade or 'auto' for defaults (default: auto)")
    parser.add_argument("--timeframe", default="15m", choices=["1m", "5m", "15m", "1h", "4h"],
                        help="Candle timeframe (default: 15m)")
    parser.add_argument("--max-coins", type=int, default=None,
                        help="Max simultaneous coins (overrides profile default)")
    parser.add_argument("--test", action="store_true",
                        help="Test connectivity only — no orders placed")
    parser.add_argument("--confirm", action="store_true",
                        help="Required flag to confirm you want to trade with real money")
    args = parser.parse_args()

    # Resolve symbols
    if args.symbol == "auto":
        symbols = DEFAULT_SYMBOLS.get(args.exchange, ["ETH/USDT"])
    else:
        symbols = [s.strip() for s in args.symbol.split(",")]

    trader = LifecycleTrader(
        exchange=args.exchange,
        profile=args.profile,
        capital=args.capital,
        symbols=symbols,
        timeframe=args.timeframe,
        max_coins=args.max_coins,
        live=True,
    )

    if args.test:
        print("🧪 Running connectivity test...")
        success = trader.test_connectivity()
        sys.exit(0 if success else 1)

    if not args.confirm:
        print("⚠️  LIVE TRADING MODE")
        print("This will place REAL orders with REAL money.")
        print(f"Exchange: {args.exchange}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Profile: {args.profile}")
        print(f"Capital ref: ${args.capital:.0f}")
        print()
        print("Run with --confirm to proceed, or --test to verify connectivity first.")
        sys.exit(1)

    # Enable lifecycle phases (Wyckoff DCA/EXIT/MARKDOWN/SPRING/MARKUP + shorts)
    trader.enable_lifecycle(args.profile)

    print(f"🔴 STARTING LIVE TRADER on {args.exchange}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Profile: {args.profile}, Capital ref: ${args.capital:.0f}")
    print(f"   Lifecycle: ENABLED")
    print(f"   Output dir: {trader.paper_dir}")
    print()
    trader.run()


if __name__ == "__main__":
    main()
