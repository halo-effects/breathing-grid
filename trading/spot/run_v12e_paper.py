#!/usr/bin/env python3
"""V12e Lifecycle Paper Trading Bot — 3 certified coins on Aster.

Runs ETH/USDT, SOL/USDT, BTC/USDT with lifecycle engine enabled.
Medium risk profile, 1h timeframe, $10K capital split across 3 coins.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading.spot.lifecycle_trader import LifecycleTrader as SpotPaperTrader


def main():
    parser = argparse.ArgumentParser(description="V12e Lifecycle Paper Trader")
    parser.add_argument("--profile", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default="aster")
    parser.add_argument("--aggressiveness", default=None,
                        choices=["conservative", "balanced", "aggressive"],
                        help="Rebalancing aggressiveness (default: based on profile)")
    parser.add_argument("--no-auto-rotation", action="store_true",
                        help="Disable automatic coin rotation")
    parser.add_argument("--pipeline", action="store_true",
                        help="Enable scanner-driven coin pipeline")
    parser.add_argument("--pipeline-interval", type=float, default=4.0,
                        help="Pipeline check interval in hours (default: 4)")
    args = parser.parse_args()

    # Hyperliquid has all 3 as spot; Aster lacks SOL spot
    if args.exchange == "hyperliquid":
        symbols = ["ETH/USDC", "SOL/USDC", "BTC/USDC"]
    else:
        symbols = ["ETH/USDT", "BTC/USDT"]  # Aster only has ETH+BTC spot

    # Dedicated output directory for V12e lifecycle paper trading
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "v12e")

    trader = SpotPaperTrader(
        exchange=args.exchange,
        profile=args.profile,
        capital=args.capital,
        symbols=symbols,
        timeframe=args.timeframe,
        max_coins=3,
    )
    # Override output dir to dedicated V12e directory
    from pathlib import Path
    trader.paper_dir = Path(output_dir)
    trader.paper_dir.mkdir(parents=True, exist_ok=True)

    # Enable V12e lifecycle engine with shorts
    trader.enable_lifecycle(args.profile)

    # Enable scanner → trader pipeline if requested
    if args.pipeline:
        trader.enable_pipeline(args.pipeline_interval)

    trader.run()


if __name__ == "__main__":
    main()
