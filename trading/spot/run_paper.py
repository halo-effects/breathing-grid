#!/usr/bin/env python3
"""Entry point for the Spot DCA Scale-Out Paper Trading Bot."""
import argparse
import sys
import os

# Add workspace root to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.spot.spot_trader import SpotPaperTrader, DEFAULT_SYMBOLS


def main():
    parser = argparse.ArgumentParser(description="Spot DCA Scale-Out Paper Trader")
    parser.add_argument("--exchange", default="aster", choices=["aster", "hyperliquid"],
                        help="Exchange to connect to (default: aster)")
    parser.add_argument("--profile", default="medium", choices=["low", "medium", "high"],
                        help="Risk profile (default: medium)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Virtual capital in USD (default: 10000)")
    parser.add_argument("--symbol", default="auto",
                        help="Symbol to trade or 'auto' for defaults (default: auto)")
    parser.add_argument("--timeframe", default="15m", choices=["1m", "5m", "15m", "1h", "4h"],
                        help="Candle timeframe (default: 15m)")
    parser.add_argument("--max-coins", type=int, default=None,
                        help="Max simultaneous coins (overrides profile default)")
    args = parser.parse_args()

    # Resolve symbols
    if args.symbol == "auto":
        symbols = DEFAULT_SYMBOLS.get(args.exchange, ["ETH/USDT"])
    else:
        symbols = [s.strip() for s in args.symbol.split(",")]

    trader = SpotPaperTrader(
        exchange=args.exchange,
        profile=args.profile,
        capital=args.capital,
        symbols=symbols,
        timeframe=args.timeframe,
        max_coins=args.max_coins,
    )
    trader.run()


if __name__ == "__main__":
    main()
