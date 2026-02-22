#!/usr/bin/env python3
"""V12f Smart Capital Allocation Paper Trading Bot.

Same as V12e lifecycle engine but with phase-weighted capital allocation:
- Cold start: coins in SPRING/MARKUP get larger capital share
- Fresh capital: freed cash routes to highest-opportunity coin
- V12e lifecycle logic untouched — only allocation changes

Usage:
    python trading/spot/run_v12f_paper.py --exchange hyperliquid --capital 10000
    python trading/spot/run_v12f_paper.py --exchange aster --profile high --capital 500
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.spot.lifecycle_trader import LifecycleTrader as SpotPaperTrader


def main():
    parser = argparse.ArgumentParser(description="V12f Smart Capital Allocation Paper Trader")
    parser.add_argument("--profile", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default="hyperliquid")
    args = parser.parse_args()

    # Hyperliquid has all 3 as spot; Aster lacks SOL spot
    if args.exchange == "hyperliquid":
        symbols = ["ETH/USDC", "SOL/USDC", "BTC/USDC"]
    else:
        symbols = ["ETH/USDT", "BTC/USDT"]  # Aster only has ETH+BTC spot

    # Dedicated output directory for V12f
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "v12f")

    trader = SpotPaperTrader(
        exchange=args.exchange,
        profile=args.profile,
        capital=args.capital,
        symbols=symbols,
        timeframe=args.timeframe,
        max_coins=3,
        smart_allocation=True,  # V12f: phase-weighted capital allocation
    )
    # Override output dir to dedicated V12f directory
    from pathlib import Path
    trader.paper_dir = Path(output_dir)
    trader.paper_dir.mkdir(parents=True, exist_ok=True)
    trader.run()


if __name__ == "__main__":
    main()
