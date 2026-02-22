"""Spot DCA Scale-Out Paper Trading Bot.

Connects to real exchanges via CCXT for market data but tracks positions
virtually (no real orders). Implements the full DCA + scale-out strategy
from the backtest engine.
"""
import asyncio
import csv
import json
import logging
import os
import signal
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

from .exchange_client import SpotExchangeClient
from .exchange_adapter import create_adapter
from .capital_allocator import compute_phase_allocations, route_freed_capital
from .lifecycle_engine import LifecycleEngine, LifecycleConfig, LifecyclePhase, ShortPosition
from .phase_classifier import classify_phase
from ..regime_detector import classify_regime_v2
from ..indicators import atr as compute_atr, atr_pct as compute_atr_pct

logger = logging.getLogger(__name__)

PAPER_BASE = Path(__file__).parent / "paper"
PAPER_BASE.mkdir(exist_ok=True)

LIVE_BASE = Path(__file__).parent / "live"
LIVE_BASE.mkdir(exist_ok=True)

# ── Telegram ───────────────────────────────────────────────────────────────

TG_TOKEN = "8528958079:AAF90HSJ5Ck1urUydzS5CUvyf2EEeB7LUwc"
TG_CHAT_ID = "5221941584"
TG_ENABLED = True


def send_telegram(msg: str):
    if not TG_ENABLED:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ── Constants from backtest engine ─────────────────────────────────────────

EXCHANGE_FEES = {
    "aster":       {"maker": 0.0001, "taker": 0.00035},
    "hyperliquid": {"maker": 0.0004, "taker": 0.0007},
}

REGIME_TP_MULT = {
    "ACCUMULATION": 0.85, "CHOPPY": 0.90, "RANGING": 0.85,
    "DISTRIBUTION": 0.90, "MILD_TREND": 1.05, "TRENDING": 1.20,
    "EXTREME": 0.70, "BREAKOUT_WARNING": 0.80, "UNKNOWN": 1.0,
}

REGIME_DEV_MULT = {
    "ACCUMULATION": 0.85, "CHOPPY": 0.90, "RANGING": 0.80,
    "DISTRIBUTION": 0.90, "MILD_TREND": 1.10, "TRENDING": 1.30,
    "EXTREME": 1.50, "BREAKOUT_WARNING": 1.20, "UNKNOWN": 1.0,
}

BLOCKED_REGIMES = {"EXTREME"}
BEARISH_SPACING_MULT = 1.4

DEFAULT_SYMBOLS = {
    "aster": ["ETH/USDT"],
    "hyperliquid": ["HYPE/USDC"],
}

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400,
}


# ── Risk Profiles ──────────────────────────────────────────────────────────

@dataclass
class RiskProfile:
    name: str
    max_safety_orders: int
    base_order_pct: float
    tp_min: float
    tp_max: float
    tp_baseline: float
    deviation_min: float
    deviation_max: float
    deviation_baseline: float
    max_drawdown_pct: float
    so_size_multiplier: float = 2.0
    atr_baseline_pct: float = 0.8
    max_coins: int = 3


PROFILES = {
    "low": RiskProfile(
        name="low", max_safety_orders=5, base_order_pct=0.03,
        tp_min=1.5, tp_max=2.5, tp_baseline=2.0,
        deviation_min=3.0, deviation_max=4.0, deviation_baseline=3.5,
        max_drawdown_pct=15.0, max_coins=2,
    ),
    "medium": RiskProfile(
        name="medium", max_safety_orders=8, base_order_pct=0.04,
        tp_min=1.0, tp_max=2.0, tp_baseline=1.5,
        deviation_min=2.0, deviation_max=3.0, deviation_baseline=2.5,
        max_drawdown_pct=25.0, max_coins=3,
    ),
    "high": RiskProfile(
        name="high", max_safety_orders=12, base_order_pct=0.05,
        tp_min=0.8, tp_max=1.5, tp_baseline=1.0,
        deviation_min=1.5, deviation_max=2.5, deviation_baseline=2.0,
        max_drawdown_pct=35.0, max_coins=5,
    ),
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Lot:
    lot_id: int
    buy_price: float
    qty: float
    cost_usd: float
    buy_fee: float
    buy_time: str
    sell_price: Optional[float] = None
    sell_fee: float = 0.0
    sell_time: Optional[str] = None
    tp_target: float = 0.0
    pnl: float = 0.0

    @property
    def is_sold(self) -> bool:
        return self.sell_price is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Lot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Deal:
    deal_id: int
    symbol: str
    lots: List[Lot] = field(default_factory=list)
    open_time: str = ""
    close_time: Optional[str] = None
    regime_at_open: str = "UNKNOWN"

    @property
    def is_complete(self) -> bool:
        return len(self.lots) > 0 and all(lot.is_sold for lot in self.lots)

    @property
    def unsold_lots(self) -> List[Lot]:
        return [l for l in self.lots if not l.is_sold]

    @property
    def total_invested(self) -> float:
        return sum(l.cost_usd for l in self.lots)

    @property
    def total_pnl(self) -> float:
        return sum(l.pnl for l in self.lots if l.is_sold)

    @property
    def capital_deployed(self) -> float:
        return sum(l.cost_usd for l in self.unsold_lots)

    @property
    def avg_entry(self) -> float:
        unsold = self.unsold_lots
        if not unsold:
            return 0.0
        total_qty = sum(l.qty for l in unsold)
        if total_qty == 0:
            return 0.0
        return sum(l.buy_price * l.qty for l in unsold) / total_qty

    @property
    def total_qty(self) -> float:
        return sum(l.qty for l in self.unsold_lots)

    def unrealized_pnl(self, current_price: float) -> float:
        pnl = 0.0
        for l in self.unsold_lots:
            pnl += l.qty * (current_price - l.buy_price)
        return pnl

    def state(self) -> str:
        if not self.lots:
            return "IDLE"
        if self.is_complete:
            return "COMPLETE"
        if any(l.is_sold for l in self.lots):
            return "EXITING"
        return "ACCUMULATING"

    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id, "symbol": self.symbol,
            "lots": [l.to_dict() for l in self.lots],
            "open_time": self.open_time, "close_time": self.close_time,
            "regime_at_open": self.regime_at_open,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Deal":
        lots = [Lot.from_dict(l) for l in d.get("lots", [])]
        return cls(
            deal_id=d["deal_id"], symbol=d["symbol"], lots=lots,
            open_time=d.get("open_time", ""), close_time=d.get("close_time"),
            regime_at_open=d.get("regime_at_open", "UNKNOWN"),
        )


# ── Paper Trader ───────────────────────────────────────────────────────────

class LifecycleTrader:
    """Live paper trading bot using real market data with virtual positions."""

    def __init__(
        self,
        exchange: str = "aster",
        profile: str = "medium",
        capital: float = 10000.0,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        max_coins: Optional[int] = None,
        live: bool = False,
        smart_allocation: bool = False,
    ):
        self.exchange_name = exchange.lower()
        self.profile = PROFILES[profile.lower()]
        self.initial_capital = capital
        self.cash = capital
        self.live = live
        self.smart_allocation = smart_allocation  # V12f: phase-weighted capital allocation
        self.lifecycle_enabled = False  # V12e: set True to enable lifecycle phases + shorts
        self.timeframe = timeframe
        self.max_coins = max_coins or self.profile.max_coins
        self.symbols = symbols or DEFAULT_SYMBOLS.get(self.exchange_name, ["ETH/USDT"])

        fees = EXCHANGE_FEES.get(self.exchange_name, EXCHANGE_FEES["aster"])
        self.taker_fee = fees["taker"]
        self.maker_fee = fees["maker"]

        # State
        self.deals: Dict[str, Deal] = {}  # symbol -> active deal
        self.completed_deals: List[Deal] = []
        self._deal_counter = 0
        self._halted = False
        self._running = False
        self._start_time = datetime.now(timezone.utc)
        self._peak_equity = capital
        self._max_dd = 0.0
        self._last_regime: Dict[str, str] = {}

        # Manual control state
        self._paused_coins: set = set()
        self._manually_paused: bool = False
        self._conviction_overrides: Dict[str, float] = {}
        self._max_so_adjustments: Dict[str, int] = {}
        self._skip_next_so: set = set()

        # Live mode: order tracking
        self._open_orders: Dict[str, List[dict]] = {}  # symbol -> list of open order records
        self._tp_orders: Dict[str, dict] = {}  # symbol -> {order_id, symbol, amount, price}

        # Rebalancing aggressiveness (read from controls.json or default)
        self._rebalancing_mode = "balanced"
        self._auto_rotation = True
        self._rebalances_today = 0
        self._rebalance_cooldown_remaining = 0.0
        self._next_rebalance_available = "Now"

        # Capital utilization & opportunity scoring
        self._idle_capital_pct = 0.0
        self._best_opportunity = None  # {symbol, score, vs_worst} or None
        self._idle_capital_pct = 0.0
        self._best_opportunity = None

        # V12f: phase-weighted capital allocation
        self._coin_cfgi: Dict[str, float] = {}  # latest CFGI per coin
        self._fear_greed_index: Optional[float] = None  # BTC CFGI as market-wide FGI
        self._coin_phases: Dict[str, str] = {s: "DCA" for s in (symbols or [])}
        self._coin_allocations: Dict[str, float] = {}  # computed per-coin budgets

        # V12e lifecycle engines (one per symbol)
        self._lifecycle_engines: Dict[str, LifecycleEngine] = {}
        self._paper_shorts: Dict[str, dict] = {}  # symbol -> virtual short position

        # Coin pipeline: scanner → trader integration
        self._coin_start_times: Dict[str, str] = {
            s: datetime.now(timezone.utc).isoformat() for s in (symbols or [])
        }
        self._pipeline_enabled = False  # set True to enable scanner-driven coin management
        self._pipeline_check_interval = 4 * 3600  # check every 4h (matches scanner cron)
        self._last_pipeline_check = 0.0

        # Per-exchange output directory
        base_dir = LIVE_BASE if live else PAPER_BASE
        self.paper_dir = base_dir / self.exchange_name
        self.paper_dir.mkdir(parents=True, exist_ok=True)

        # Exchange client
        self.client = SpotExchangeClient()

        # Exchange adapter (routes spot + futures ops; paper mode uses PaperAdapter)
        self.adapter = create_adapter(
            exchange_id=self.exchange_name,
            ccxt_client=self.client,
            paper=not self.live,
        )

        # Logging
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.paper_dir / "bot.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

    # ── Live Order Execution ───────────────────────────────────────────

    def _get_balance(self, asset: str = 'USDT') -> float:
        """Get available (free) balance for asset."""
        try:
            bal = self.client.fetch_balance(asset)
            return float(bal.get("free", 0) or 0)
        except Exception as e:
            logger.error("Failed to fetch balance for %s: %s", asset, e)
            return 0.0

    def _get_spot_position(self, symbol: str) -> float:
        """Get current holdings of base asset (e.g. ETH from ETH/USDT)."""
        base = symbol.split('/')[0]
        return self._get_balance(base)

    def _get_min_order(self, symbol: str) -> dict:
        """Get minimum order constraints for a symbol."""
        try:
            return self.client.get_min_order_size(symbol)
        except Exception as e:
            logger.error("Failed to get min order size for %s: %s", symbol, e)
            return {"min_amount": None, "min_cost": 5.0}

    def _place_market_buy(self, symbol: str, cost_usd: float) -> Optional[dict]:
        """Place market buy for a given USD cost. Returns order info or None."""
        try:
            # Get current price to calculate amount
            price = self._get_current_price(symbol)
            if not price or price <= 0:
                logger.error("Cannot get price for market buy %s", symbol)
                return None

            amount = cost_usd / price

            # Check minimums
            mins = self._get_min_order(symbol)
            min_cost = float(mins.get("min_cost") or 5.0)
            min_amount = float(mins.get("min_amount") or 0)
            if cost_usd < min_cost:
                logger.warning("Order cost $%.2f below minimum $%.2f for %s", cost_usd, min_cost, symbol)
                return None
            if amount < min_amount:
                logger.warning("Order amount %.8f below minimum %.8f for %s", amount, min_amount, symbol)
                return None

            # Check balance
            quote = symbol.split('/')[1]
            available = self._get_balance(quote)
            if available < cost_usd:
                logger.warning("Insufficient %s balance: $%.2f < $%.2f", quote, available, cost_usd)
                self._alert(f"⚠️ Insufficient balance for {symbol} buy: ${available:.2f} < ${cost_usd:.2f}")
                return None

            logger.info("🔵 LIVE MARKET BUY %s: amount=%.8f (~$%.2f)", symbol, amount, cost_usd)
            order = self.client.create_market_buy(symbol, amount)

            # Wait for fill confirmation (up to 30s)
            order_id = order.get('id')
            if order_id:
                order = self._wait_for_fill(symbol, order_id, timeout=30)

            fill_price = float(order.get('average') or order.get('price') or price)
            fill_qty = float(order.get('filled') or order.get('amount') or amount)
            fill_cost = float(order.get('cost') or (fill_price * fill_qty))
            fee_cost = 0.0
            if order.get('fee'):
                fee_cost = float(order['fee'].get('cost', 0) or 0)

            logger.info("✅ FILLED: %s @ $%.6f, qty=%.8f, cost=$%.2f, fee=$%.4f",
                        symbol, fill_price, fill_qty, fill_cost, fee_cost)
            return {
                'order_id': order_id,
                'price': fill_price,
                'qty': fill_qty,
                'cost': fill_cost,
                'fee': fee_cost,
                'raw': order,
            }
        except Exception as e:
            logger.error("LIVE MARKET BUY FAILED %s: %s", symbol, e)
            self._alert(f"❌ Market buy failed for {symbol}: {e}")
            return None

    def _place_market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """Place market sell. Returns order info or None."""
        try:
            mins = self._get_min_order(symbol)
            min_amount = float(mins.get("min_amount") or 0)
            if amount < min_amount:
                logger.warning("Sell amount %.8f below minimum %.8f for %s", amount, min_amount, symbol)
                return None

            # Verify we have the asset
            actual = self._get_spot_position(symbol)
            if actual < amount * 0.99:  # 1% tolerance for rounding
                logger.warning("Insufficient %s holdings: %.8f < %.8f", symbol, actual, amount)
                amount = actual  # sell what we have

            if amount <= 0:
                return None

            logger.info("🔴 LIVE MARKET SELL %s: amount=%.8f", symbol, amount)
            order = self.client.create_market_sell(symbol, amount)

            order_id = order.get('id')
            if order_id:
                order = self._wait_for_fill(symbol, order_id, timeout=30)

            price = self._get_current_price(symbol) or 0
            fill_price = float(order.get('average') or order.get('price') or price)
            fill_qty = float(order.get('filled') or order.get('amount') or amount)
            fee_cost = 0.0
            if order.get('fee'):
                fee_cost = float(order['fee'].get('cost', 0) or 0)

            logger.info("✅ SOLD: %s @ $%.6f, qty=%.8f, fee=$%.4f",
                        symbol, fill_price, fill_qty, fee_cost)
            return {
                'order_id': order_id,
                'price': fill_price,
                'qty': fill_qty,
                'cost': fill_price * fill_qty,
                'fee': fee_cost,
                'raw': order,
            }
        except Exception as e:
            logger.error("LIVE MARKET SELL FAILED %s: %s", symbol, e)
            self._alert(f"❌ Market sell failed for {symbol}: {e}")
            return None

    def _place_limit_sell(self, symbol: str, amount: float, price: float) -> Optional[dict]:
        """Place limit sell (TP) order. Returns order info or None."""
        try:
            mins = self._get_min_order(symbol)
            min_amount = float(mins.get("min_amount") or 0)
            if amount < min_amount:
                logger.warning("Limit sell amount %.8f below min %.8f for %s", amount, min_amount, symbol)
                return None

            logger.info("📋 LIVE LIMIT SELL %s: amount=%.8f @ $%.6f", symbol, amount, price)
            order = self.client.create_limit_sell(symbol, amount, price)
            order_id = order.get('id')

            logger.info("📋 Limit sell placed: order_id=%s", order_id)
            return {
                'order_id': order_id,
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'status': 'open',
                'raw': order,
            }
        except Exception as e:
            logger.error("LIVE LIMIT SELL FAILED %s: %s", symbol, e)
            self._alert(f"❌ Limit sell failed for {symbol}: {e}")
            return None

    def _cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        try:
            self.client.cancel_order(order_id, symbol)
            logger.info("❌ Cancelled order %s on %s", order_id, symbol)
            return True
        except Exception as e:
            logger.error("Cancel order failed %s/%s: %s", symbol, order_id, e)
            return False

    def _check_order_status(self, symbol: str, order_id: str) -> dict:
        """Check order status. Returns {status, filled, remaining, price, cost}."""
        try:
            order = self.client.fetch_order(order_id, symbol)
            return {
                'status': order.get('status', 'unknown'),
                'filled': float(order.get('filled', 0) or 0),
                'remaining': float(order.get('remaining', 0) or 0),
                'price': float(order.get('average') or order.get('price') or 0),
                'cost': float(order.get('cost', 0) or 0),
                'fee': float((order.get('fee') or {}).get('cost', 0) or 0),
            }
        except Exception as e:
            logger.error("Check order status failed %s/%s: %s", symbol, order_id, e)
            return {'status': 'unknown', 'filled': 0, 'remaining': 0, 'price': 0, 'cost': 0, 'fee': 0}

    def _wait_for_fill(self, symbol: str, order_id: str, timeout: int = 30) -> dict:
        """Poll order until filled or timeout. Returns final order state."""
        import time as _time
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                order = self.client.fetch_order(order_id, symbol)
                status = order.get('status', '')
                if status in ('closed', 'filled'):
                    return order
                if status in ('canceled', 'cancelled', 'rejected', 'expired'):
                    logger.warning("Order %s status: %s", order_id, status)
                    return order
            except Exception as e:
                logger.warning("Error polling order %s: %s", order_id, e)
            _time.sleep(1)
        logger.warning("Order %s did not fill within %ds", order_id, timeout)
        # Return last known state
        try:
            return self.client.fetch_order(order_id, symbol)
        except Exception:
            return {'id': order_id, 'status': 'timeout'}

    def _cancel_all_symbol_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        try:
            open_orders = self.client.fetch_open_orders(symbol)
            for o in open_orders:
                oid = o.get('id')
                if oid:
                    self._cancel_order(symbol, oid)
        except Exception as e:
            logger.error("Failed to cancel all orders for %s: %s", symbol, e)

    def _reconcile_on_startup(self):
        """Verify exchange state matches saved state on startup (live mode only)."""
        if not self.live:
            return
        logger.info("🔄 Running startup reconciliation...")
        alerts = []

        for symbol, deal in list(self.deals.items()):
            # Check actual position
            actual_qty = self._get_spot_position(symbol)
            expected_qty = deal.total_qty

            if expected_qty > 0:
                diff_pct = abs(actual_qty - expected_qty) / expected_qty * 100
                if diff_pct > 5:  # >5% discrepancy
                    msg = (f"⚠️ Position mismatch for {symbol}: "
                           f"expected {expected_qty:.8f}, actual {actual_qty:.8f} "
                           f"(diff {diff_pct:.1f}%)")
                    logger.warning(msg)
                    alerts.append(msg)

            # Check TP orders still exist
            tp_info = self._tp_orders.get(symbol)
            if tp_info and tp_info.get('order_id'):
                status = self._check_order_status(symbol, tp_info['order_id'])
                if status['status'] in ('closed', 'filled'):
                    msg = f"ℹ️ TP order for {symbol} filled while offline!"
                    logger.info(msg)
                    alerts.append(msg)
                elif status['status'] in ('canceled', 'cancelled', 'expired'):
                    msg = f"⚠️ TP order for {symbol} was cancelled/expired"
                    logger.warning(msg)
                    alerts.append(msg)
                    self._tp_orders.pop(symbol, None)

        # Report balance
        quote = 'USDT'
        balance = self._get_balance(quote)
        logger.info("💰 %s balance: $%.2f", quote, balance)

        if alerts:
            full_msg = "🔄 <b>Startup Reconciliation</b>\n" + "\n".join(alerts)
            send_telegram(full_msg)
        else:
            logger.info("✅ Reconciliation OK — state matches exchange")

    def _alert(self, msg: str):
        """Send alert via telegram and log."""
        logger.warning(msg)
        send_telegram(msg)

    def test_connectivity(self) -> bool:
        """Test exchange connectivity without placing orders."""
        try:
            config_path = Path(__file__).parent / "spot_config.json"
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                exc_cfg = cfg.get("exchanges", {}).get(self.exchange_name, {})
                config = {"options": exc_cfg.get("options", {})}
            self.client.connect(self.exchange_name, config)

            # Load markets
            self.client._ensure_markets()
            print(f"✅ Connected to {self.exchange_name}")
            print(f"   Markets loaded: {len(self.client.exchange.markets)}")

            # Check balance
            quote = 'USDT'
            bal = self.client.fetch_balance(quote)
            print(f"   {quote} balance: free=${bal['free']}, used=${bal['used']}, total=${bal['total']}")

            # Check symbols
            for sym in self.symbols:
                if sym in self.client.exchange.markets:
                    mins = self._get_min_order(sym)
                    print(f"   {sym}: min_amount={mins.get('min_amount')}, min_cost={mins.get('min_cost')}")
                else:
                    print(f"   ⚠️ {sym} not found in markets!")

            # Check open orders
            for sym in self.symbols:
                try:
                    orders = self.client.fetch_open_orders(sym)
                    if orders:
                        print(f"   {sym}: {len(orders)} open order(s)")
                except Exception:
                    pass

            print("✅ All connectivity checks passed")
            return True
        except Exception as e:
            print(f"❌ Connectivity test failed: {e}")
            return False

    # ── Adaptive parameters ────────────────────────────────────────────

    def _adaptive_tp(self, regime: str, atr_pct: float) -> float:
        p = self.profile
        if atr_pct <= 0:
            return p.tp_baseline
        atr_ratio = atr_pct / p.atr_baseline_pct
        tp = p.tp_baseline * atr_ratio
        tp *= REGIME_TP_MULT.get(regime, 1.0)
        return max(p.tp_min, min(p.tp_max, round(tp, 3)))

    def _adaptive_deviation(self, regime: str, atr_pct: float, current_tp: float) -> float:
        p = self.profile
        if atr_pct <= 0:
            return p.deviation_baseline
        atr_ratio = atr_pct / p.atr_baseline_pct
        dev = p.deviation_baseline * atr_ratio
        dev *= REGIME_DEV_MULT.get(regime, 1.0)
        dev = max(p.deviation_min, min(p.deviation_max, dev))
        dev = max(dev, current_tp * 1.5)
        return min(p.deviation_max, round(dev, 3))

    def _so_trigger_price(self, base_price: float, so_index: int, deviation: float) -> float:
        total_drop = deviation * so_index / 100.0
        return base_price * (1.0 - total_drop)

    def _so_cost(self, base_cost: float, so_index: int) -> float:
        return base_cost * (self.profile.so_size_multiplier ** so_index)

    # ── Market data ────────────────────────────────────────────────────

    def _fetch_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            raw = self.client.fetch_ohlcv(symbol, self.timeframe, limit=200)
            if not raw or len(raw) < 100:
                logger.warning("Not enough candles for %s: %d", symbol, len(raw) if raw else 0)
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error("Failed to fetch candles for %s: %s", symbol, e)
            return None

    def _fetch_candles_from_db(self, symbol: str, limit: int = 2400, as_ms: bool = False) -> Optional[pd.DataFrame]:
        """Fetch historical candles from local candle DB for conductor warmup.
        Falls back to CCXT if DB unavailable. Needs 1200+ 1h candles for 50+ daily bars.
        If as_ms=True, keeps timestamp as integer ms (needed for conductor.prepare())."""
        try:
            import sqlite3
            db_path = self.paper_dir / "candles.db"
            if not db_path.exists():
                # Try shared collector DB
                db_path = Path(__file__).parent / "data" / "candles.db"
            if not db_path.exists():
                logger.debug("No candle DB found for conductor warmup, using CCXT")
                return None
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close, volume FROM candles "
                "WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, "1h", limit)
            ).fetchall()
            conn.close()
            if not rows or len(rows) < 200:
                logger.debug("Candle DB has only %d rows for %s, using CCXT", len(rows) if rows else 0, symbol)
                return None
            df = pd.DataFrame(rows[::-1], columns=["timestamp", "open", "high", "low", "close", "volume"])
            if not as_ms:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            logger.info("Loaded %d candles from DB for %s conductor warmup", len(df), symbol)
            return df
        except Exception as e:
            logger.warning("Candle DB read failed for %s: %s", symbol, e)
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.client.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close", 0))
        except Exception as e:
            logger.error("Failed to fetch price for %s: %s", symbol, e)
            return None

    def _compute_regime_and_atr(self, df: pd.DataFrame) -> Tuple[str, float, bool]:
        """Returns (regime, atr_pct, is_bullish)."""
        try:
            regimes = classify_regime_v2(df, self.timeframe)
            regime = regimes.iloc[-1] if len(regimes) > 0 else "UNKNOWN"
        except Exception:
            regime = "UNKNOWN"
        try:
            atr_pct_s = compute_atr_pct(df, 14)
            atr_pct_val = float(atr_pct_s.iloc[-1]) if not pd.isna(atr_pct_s.iloc[-1]) else 0.0
        except Exception:
            atr_pct_val = 0.0

        sma50 = df["close"].rolling(50).mean()
        is_bullish = float(df["close"].iloc[-1]) >= float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else True
        return regime, atr_pct_val, is_bullish

    # ── Deal management ────────────────────────────────────────────────

    def _open_deal(self, symbol: str, price: float, ts: str, regime: str, tp_pct: float):
        # Check manual controls
        if self._manually_paused or symbol in self._paused_coins:
            return

        if self.live:
            # LIVE: use actual exchange balance
            quote = symbol.split('/')[1]
            available_balance = self._get_balance(quote)
            reserve = available_balance * 0.1
            if self.smart_allocation and self._coin_phases:
                # V12f: phase-weighted allocation
                allocations = compute_phase_allocations(
                    self.initial_capital, self._coin_phases, self.profile.base_order_pct)
                coin_budget = allocations.get(symbol, self.initial_capital * self.profile.base_order_pct)
                per_coin = min(coin_budget, available_balance - reserve)
                self._coin_allocations = allocations
            else:
                # V12e: equal split
                per_coin = (available_balance - reserve) / max(1, self.max_coins - len(self.deals))
            base_cost = min(self.initial_capital * self.profile.base_order_pct, per_coin)
            if base_cost < 5.0 or base_cost > available_balance:
                logger.info("Skipping deal for %s: insufficient balance ($%.2f)", symbol, available_balance)
                return

            result = self._place_market_buy(symbol, base_cost)
            if not result:
                return

            fill_price = result['price']
            fill_qty = result['qty']
            fill_cost = result['cost']
            fee = result['fee']

            self._deal_counter += 1
            lot = Lot(
                lot_id=0, buy_price=fill_price, qty=fill_qty,
                cost_usd=fill_cost, buy_fee=fee, buy_time=ts,
                tp_target=fill_price * (1 + tp_pct / 100),
            )
            deal = Deal(
                deal_id=self._deal_counter, symbol=symbol,
                lots=[lot], open_time=ts, regime_at_open=regime,
            )
            self.deals[symbol] = deal
            # In live mode, cash tracks what exchange reports
            self.cash = self._get_balance(quote)

            # Place TP limit sell immediately
            tp_price = lot.tp_target
            tp_order = self._place_limit_sell(symbol, fill_qty, tp_price)
            if tp_order:
                self._tp_orders[symbol] = tp_order

            mode_label = "Spot Live"
            logger.info("📥 LIVE DEAL %d OPENED: %s @ $%.4f (fill), regime=%s, TP=%.2f%%",
                         deal.deal_id, symbol, fill_price, regime, tp_pct)
            send_telegram(
                f"📥 <b>{mode_label}: Deal Opened</b>\n"
                f"Coin: {symbol}\nProfile: {self.profile.name}\n"
                f"Entry: ${fill_price:.4f} (filled)\nBase size: ${fill_cost:.2f}\n"
                f"Regime: {regime}\nTP target: ${tp_price:.4f}\n"
                f"Order ID: {result['order_id']}"
            )
        else:
            # PAPER mode
            if self.smart_allocation and self._coin_phases:
                # V12f: phase-weighted allocation
                allocations = compute_phase_allocations(
                    self.initial_capital, self._coin_phases, self.profile.base_order_pct)
                coin_budget = allocations.get(symbol, self.initial_capital * self.profile.base_order_pct)
                base_cost = self.initial_capital * self.profile.base_order_pct
                base_cost = min(base_cost, coin_budget, self.cash * 0.9)
                self._coin_allocations = allocations
            else:
                # V12e: equal split
                available = self.cash * 0.9 / max(1, self.max_coins - len(self.deals))
                base_cost = self.initial_capital * self.profile.base_order_pct
                base_cost = min(base_cost, available)
            if base_cost < 5.0 or base_cost > self.cash:
                return

            fee = base_cost * self.taker_fee
            qty = (base_cost - fee) / price

            self._deal_counter += 1
            lot = Lot(
                lot_id=0, buy_price=price, qty=qty,
                cost_usd=base_cost, buy_fee=fee, buy_time=ts,
                tp_target=price * (1 + tp_pct / 100),
            )
            deal = Deal(
                deal_id=self._deal_counter, symbol=symbol,
                lots=[lot], open_time=ts, regime_at_open=regime,
            )
            self.deals[symbol] = deal
            self.cash -= base_cost

            logger.info("📥 DEAL %d OPENED: %s @ $%.4f (regime=%s, TP=%.2f%%)",
                         deal.deal_id, symbol, price, regime, tp_pct)
            send_telegram(
                f"📥 <b>Spot Paper: Deal Opened</b>\n"
                f"Coin: {symbol}\nProfile: {self.profile.name}\n"
                f"Entry: ${price:.4f}\nBase size: ${base_cost:.2f}\n"
                f"Regime: {regime}\nTP target: ${lot.tp_target:.4f}"
            )

    def _check_safety_orders(self, deal: Deal, current_price: float, ts: str,
                              regime: str, dev_pct: float, tp_pct: float, is_bullish: bool):
        # Skip if coin is paused
        if deal.symbol in self._paused_coins:
            return
        # Skip next SO (one-time)
        if deal.symbol in self._skip_next_so:
            self._skip_next_so.discard(deal.symbol)
            logger.info("⏭ Skipped SO for %s (one-time skip)", deal.symbol)
            return
        filled_sos = len(deal.lots) - 1
        # Apply max SO adjustments
        effective_max_sos = self.profile.max_safety_orders + self._max_so_adjustments.get(deal.symbol, 0)
        if filled_sos >= effective_max_sos:
            return
        if regime in BLOCKED_REGIMES:
            return

        next_so = filled_sos + 1
        base_price = deal.lots[0].buy_price
        spacing_mult = BEARISH_SPACING_MULT if not is_bullish else 1.0
        trigger = self._so_trigger_price(base_price, next_so, dev_pct * spacing_mult)

        if current_price <= trigger:
            base_cost = self.initial_capital * self.profile.base_order_pct
            so_cost = self._so_cost(base_cost, next_so)

            if self.live:
                # LIVE: check real balance and place real order
                quote = deal.symbol.split('/')[1]
                available = self._get_balance(quote)
                so_cost = min(so_cost, available * 0.95)
                if so_cost < 5.0:
                    return

                # Cancel existing TP order before adding to position
                tp_info = self._tp_orders.get(deal.symbol)
                if tp_info and tp_info.get('order_id'):
                    self._cancel_order(deal.symbol, tp_info['order_id'])
                    self._tp_orders.pop(deal.symbol, None)

                result = self._place_market_buy(deal.symbol, so_cost)
                if not result:
                    return

                lot = Lot(
                    lot_id=next_so, buy_price=result['price'], qty=result['qty'],
                    cost_usd=result['cost'], buy_fee=result['fee'], buy_time=ts,
                    tp_target=result['price'] * (1 + tp_pct / 100),
                )
                deal.lots.append(lot)
                self.cash = self._get_balance(quote)

                # Re-place TP for all unsold lots (weighted avg approach: sell all at once)
                total_unsold_qty = deal.total_qty
                # Use the highest unsold lot's TP (reverse order exit)
                unsold_sorted = sorted(deal.unsold_lots, key=lambda l: l.lot_id, reverse=True)
                if unsold_sorted:
                    tp_price = unsold_sorted[0].tp_target
                    tp_order = self._place_limit_sell(deal.symbol, total_unsold_qty, tp_price)
                    if tp_order:
                        self._tp_orders[deal.symbol] = tp_order

                logger.info("📦 LIVE DEAL %d SO%d FILLED: %s @ $%.4f ($%.2f)",
                             deal.deal_id, next_so, deal.symbol, result['price'], result['cost'])
                send_telegram(
                    f"📦 <b>Spot Live: SO{next_so} Filled</b>\n"
                    f"Coin: {deal.symbol}\nPrice: ${result['price']:.4f}\n"
                    f"Size: ${result['cost']:.2f}\nTotal layers: {len(deal.lots)}\n"
                    f"Order ID: {result['order_id']}"
                )
            else:
                # PAPER mode (unchanged)
                so_cost = min(so_cost, self.cash * 0.95)
                if so_cost < 5.0:
                    return

                fee = so_cost * self.taker_fee
                qty = (so_cost - fee) / trigger

                lot = Lot(
                    lot_id=next_so, buy_price=trigger, qty=qty,
                    cost_usd=so_cost, buy_fee=fee, buy_time=ts,
                    tp_target=trigger * (1 + tp_pct / 100),
                )
                deal.lots.append(lot)
                self.cash -= so_cost

                logger.info("📦 DEAL %d SO%d FILLED: %s @ $%.4f ($%.2f)",
                             deal.deal_id, next_so, deal.symbol, trigger, so_cost)
                send_telegram(
                    f"📦 <b>Spot Paper: SO{next_so} Filled</b>\n"
                    f"Coin: {deal.symbol}\nPrice: ${trigger:.4f}\n"
                    f"Size: ${so_cost:.2f}\nTotal layers: {len(deal.lots)}"
                )

    def _check_exits(self, deal: Deal, current_price: float, ts: str,
                      regime: str, atr_pct: float):
        """Sell lots in REVERSE order (largest SO first) when TP hit."""
        tp_pct = self._adaptive_tp(regime, atr_pct)

        if self.live:
            self._check_exits_live(deal, current_price, ts, regime, atr_pct, tp_pct)
        else:
            self._check_exits_paper(deal, current_price, ts, tp_pct)

    def _check_exits_live(self, deal: Deal, current_price: float, ts: str,
                           regime: str, atr_pct: float, tp_pct: float):
        """Live mode exit: check if TP limit order filled, or update TP targets."""
        symbol = deal.symbol
        unsold = sorted(deal.unsold_lots, key=lambda l: l.lot_id, reverse=True)

        # Update TP targets based on current regime
        for lot in unsold:
            lot.tp_target = lot.buy_price * (1 + tp_pct / 100)

        # Check if we have a TP order on the exchange
        tp_info = self._tp_orders.get(symbol)
        if tp_info and tp_info.get('order_id'):
            status = self._check_order_status(symbol, tp_info['order_id'])

            if status['status'] in ('closed', 'filled'):
                # TP filled! Close the lot(s) that were covered
                fill_price = status['price'] or tp_info.get('price', current_price)
                fill_qty = status['filled']
                fee = status['fee']
                self._tp_orders.pop(symbol, None)

                # Sell lots in reverse order up to fill_qty
                remaining_qty = fill_qty
                for lot in unsold:
                    if remaining_qty <= 0:
                        break
                    sell_qty = min(lot.qty, remaining_qty)
                    revenue = sell_qty * fill_price
                    lot_fee = fee * (sell_qty / fill_qty) if fill_qty > 0 else 0
                    net_revenue = revenue - lot_fee
                    pnl = net_revenue - lot.cost_usd

                    lot.sell_price = fill_price
                    lot.sell_fee = lot_fee
                    lot.sell_time = ts
                    lot.pnl = pnl
                    remaining_qty -= sell_qty

                    logger.info("📤 LIVE DEAL %d LOT %d SOLD: %s @ $%.4f, PnL=$%.2f",
                                 deal.deal_id, lot.lot_id, symbol, fill_price, pnl)
                    send_telegram(
                        f"📤 <b>Spot Live: Layer Sold</b>\n"
                        f"Coin: {symbol}\nLot: SO{lot.lot_id}\n"
                        f"Sell: ${fill_price:.4f}\nPnL: ${pnl:.2f}"
                    )

                # Update cash from exchange
                quote = symbol.split('/')[1]
                self.cash = self._get_balance(quote)

                # If there are still unsold lots, place new TP
                if not deal.is_complete:
                    remaining_unsold = deal.unsold_lots
                    if remaining_unsold:
                        top_lot = sorted(remaining_unsold, key=lambda l: l.lot_id, reverse=True)[0]
                        total_qty = sum(l.qty for l in remaining_unsold)
                        tp_order = self._place_limit_sell(symbol, total_qty, top_lot.tp_target)
                        if tp_order:
                            self._tp_orders[symbol] = tp_order

            elif status['status'] == 'open':
                # TP still open — check if target needs updating
                current_tp_price = tp_info.get('price', 0)
                desired_tp = unsold[0].tp_target if unsold else 0
                if desired_tp > 0 and abs(current_tp_price - desired_tp) / desired_tp > 0.005:
                    # TP target shifted >0.5%, cancel and replace
                    self._cancel_order(symbol, tp_info['order_id'])
                    total_qty = deal.total_qty
                    tp_order = self._place_limit_sell(symbol, total_qty, desired_tp)
                    if tp_order:
                        self._tp_orders[symbol] = tp_order
                    else:
                        self._tp_orders.pop(symbol, None)

            else:
                # Order cancelled/expired/unknown — remove tracking
                self._tp_orders.pop(symbol, None)
                # Re-place TP if we have unsold lots
                if unsold:
                    total_qty = deal.total_qty
                    tp_price = unsold[0].tp_target
                    tp_order = self._place_limit_sell(symbol, total_qty, tp_price)
                    if tp_order:
                        self._tp_orders[symbol] = tp_order

        elif unsold:
            # No TP order tracked — place one
            total_qty = deal.total_qty
            tp_price = unsold[0].tp_target
            tp_order = self._place_limit_sell(symbol, total_qty, tp_price)
            if tp_order:
                self._tp_orders[symbol] = tp_order

        # Check if deal complete
        if deal.is_complete:
            self._complete_deal(deal, ts)

    def _check_exits_paper(self, deal: Deal, current_price: float, ts: str, tp_pct: float):
        """Paper mode exit logic (original, unchanged)."""
        unsold = sorted(deal.unsold_lots, key=lambda l: l.lot_id, reverse=True)

        for lot in unsold:
            lot.tp_target = lot.buy_price * (1 + tp_pct / 100)

            if current_price >= lot.tp_target:
                sell_price = lot.tp_target
                revenue = lot.qty * sell_price
                fee = revenue * self.maker_fee
                net_revenue = revenue - fee
                pnl = net_revenue - lot.cost_usd

                lot.sell_price = sell_price
                lot.sell_fee = fee
                lot.sell_time = ts
                lot.pnl = pnl
                self.cash += net_revenue

                logger.info("📤 DEAL %d LOT %d SOLD: %s @ $%.4f, PnL=$%.2f",
                             deal.deal_id, lot.lot_id, deal.symbol, sell_price, pnl)
                send_telegram(
                    f"📤 <b>Spot Paper: Layer Sold</b>\n"
                    f"Coin: {deal.symbol}\nLot: SO{lot.lot_id}\n"
                    f"Sell: ${sell_price:.4f}\nPnL: ${pnl:.2f}"
                )

        # Check if deal complete
        if deal.is_complete:
            self._complete_deal(deal, ts)

    def _complete_deal(self, deal: Deal, ts: str):
        """Finalize a completed deal (shared by live and paper)."""
        deal.close_time = ts
        self.completed_deals.append(deal)
        sym = deal.symbol
        del self.deals[sym]
        self._tp_orders.pop(sym, None)

        duration_h = 0.0
        try:
            t0 = datetime.fromisoformat(deal.open_time)
            t1 = datetime.fromisoformat(deal.close_time)
            duration_h = (t1 - t0).total_seconds() / 3600
        except Exception:
            pass

        mode = "Spot Live" if self.live else "Spot Paper"
        logger.info("✅ DEAL %d COMPLETE: %s, PnL=$%.2f, Duration=%.1fh",
                     deal.deal_id, sym, deal.total_pnl, duration_h)
        send_telegram(
            f"✅ <b>{mode}: Deal Complete</b>\n"
            f"Coin: {sym}\nTotal PnL: ${deal.total_pnl:.2f}\n"
            f"Invested: ${deal.total_invested:.2f}\n"
            f"Return: {deal.total_pnl / deal.total_invested * 100:.2f}%\n"
            f"Duration: {duration_h:.1f}h"
        )
        self._append_trade_csv(deal)

        # V12f: route freed capital to best opportunity
        if self.smart_allocation and self._coin_phases:
            freed = deal.total_invested + deal.total_pnl  # capital returned
            additions = route_freed_capital(
                freed, self._coin_phases, self._coin_allocations,
                self.initial_capital, self.profile.base_order_pct)
            if additions:
                # Update allocations — the extra budget will be used on next deal open/SO
                for s, amt in additions.items():
                    self._coin_allocations[s] = self._coin_allocations.get(s, 0) + amt
                logger.info("📊 V12f capital routing: $%.0f freed from %s → %s",
                           freed, sym, ", ".join(f"{s}:+${a:.0f}" for s, a in additions.items()))

    # ── Equity ─────────────────────────────────────────────────────────

    def _equity(self, prices: Dict[str, float]) -> float:
        unsold_value = 0.0
        for sym, deal in self.deals.items():
            p = prices.get(sym, 0)
            unsold_value += sum(l.qty * p for l in deal.unsold_lots)
        # Include paper short unrealized PnL
        short_value = 0.0
        for sym, short in self._paper_shorts.items():
            p = prices.get(sym, 0)
            if p > 0 and short.get("avg_entry", 0) > 0:
                qty = short.get("total_qty", 0)
                margin = short.get("total_margin", 0)
                funding = short.get("funding_cost", 0)
                unrealized = (short["avg_entry"] - p) * qty - funding
                short_value += margin + unrealized
        # Include lifecycle engine positions (spring entries, markup positions)
        lifecycle_value = 0.0
        if self.lifecycle_enabled:
            for sym, engine in self._lifecycle_engines.items():
                p = prices.get(sym, 0)
                if p > 0:
                    try:
                        lifecycle_value += engine.unrealized_value(p)
                    except Exception:
                        pass
        return self.cash + unsold_value + short_value + lifecycle_value

    # ── CFGI Polling ──────────────────────────────────────────────

    def _poll_cfgi(self, symbol: str) -> float:
        """Get CFGI score for a symbol. Uses cached daily data."""
        try:
            token = symbol.split("/")[0].upper()
            cache_path = Path(__file__).parent / "data" / "cfgi_cache" / f"{token}_cfgi_daily.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for entry in reversed(data):
                    if entry.get("date", "") <= today:
                        val = float(entry.get("value", 50))
                        self._coin_cfgi[symbol] = val
                        return val
            # Fallback: try BTC market-wide
            btc_cache = Path(__file__).parent / "data" / "cfgi_cache" / "BTC_cfgi_daily.json"
            if btc_cache.exists():
                data = json.loads(btc_cache.read_text(encoding="utf-8"))
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for entry in reversed(data):
                    if entry.get("date", "") <= today:
                        val = float(entry.get("value", 50))
                        self._fear_greed_index = val
                        self._coin_cfgi[symbol] = val
                        return val
        except Exception as e:
            logger.debug("CFGI poll failed for %s: %s", symbol, e)
        return 50.0  # neutral default

    # ── V12e Lifecycle Integration ─────────────────────────────────

    def enable_lifecycle(self, profile: str = "medium"):
        """Enable V12e lifecycle engine for all symbols."""
        self.lifecycle_enabled = True
        KNOWN_ATH = {
            "ETH/USDT": 4878.0, "ETH/USDC": 4878.0,
            "BTC/USDT": 109000.0, "BTC/USDC": 109000.0,
            "SOL/USDT": 260.0, "SOL/USDC": 260.0,
            "HYPE/USDC": 35.0, "ZEC/USDT": 724.0,
            "ASTER/USDT": 1.50,
        }
        for symbol in self.symbols:
            config = LifecycleConfig(
                enabled=True,
                risk_profile=profile,
                ath=KNOWN_ATH.get(symbol, 0.0),
                short_enabled=True,
                rebalancing_mode=self._rebalancing_mode,
                auto_rotation=self._auto_rotation,
            )
            engine = LifecycleEngine(config, symbol, self.taker_fee)
            self._lifecycle_engines[symbol] = engine
            logger.info("🔄 Lifecycle engine created for %s (profile=%s, ATH=$%.0f)",
                        symbol, profile, config.ath)

            # Warm up conductor with historical candles from DB (need 1200+ for 50+ daily bars)
            try:
                hist_df = self._fetch_candles_from_db(symbol, limit=2400, as_ms=True)
                if hist_df is not None and len(hist_df) > 200:
                    engine.feed_candles_1h(hist_df)
                    logger.info("🔥 Conductor warmup for %s: %d candles → %s",
                                symbol, len(hist_df),
                                "ready" if engine._conductor._daily_ready else "needs more data")
            except Exception as e:
                logger.warning("Conductor warmup failed for %s: %s", symbol, e)

    def _init_lifecycle_for_symbol(self, symbol: str):
        """Initialize lifecycle engine + cold-start phase for a single newly-added coin."""
        KNOWN_ATH = {
            "ETH/USDT": 4878.0, "ETH/USDC": 4878.0,
            "BTC/USDT": 109000.0, "BTC/USDC": 109000.0,
            "SOL/USDT": 260.0, "SOL/USDC": 260.0,
            "HYPE/USDC": 35.0, "ZEC/USDT": 724.0,
            "ASTER/USDT": 1.50,
        }
        profile = self.profile.name
        ath = KNOWN_ATH.get(symbol, 0.0)
        config = LifecycleConfig(
            enabled=True,
            risk_profile=profile,
            ath=ath,
            short_enabled=True,
            rebalancing_mode=self._rebalancing_mode,
            auto_rotation=self._auto_rotation,
        )
        engine = LifecycleEngine(config, symbol, self.taker_fee)
        self._lifecycle_engines[symbol] = engine
        logger.info("🔄 Lifecycle engine created for %s (profile=%s, ATH=$%.0f)",
                    symbol, profile, ath)

        # Cold-start phase detection
        try:
            raw = self.client.fetch_ohlcv(symbol, self.timeframe, limit=500)
            if raw and len(raw) >= 200:
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                price = float(df["close"].iloc[-1])
                cfgi = self._poll_cfgi(symbol)
                phase, reason = classify_phase(symbol, price, ath, cfgi_value=cfgi, candles_df=df)
                engine.set_initial_phase(phase, reason)
                self._coin_phases[symbol] = phase.value
                logger.info("🎯 %s: Cold-start → %s (%d candles) — %s",
                            symbol, phase.value, len(df), reason)
                if phase == LifecyclePhase.MARKDOWN:
                    send_telegram(
                        f"⚠️ <b>{symbol} Added in MARKDOWN</b>\n"
                        f"DCA blocked until phase changes.\n{reason}")
            else:
                logger.warning("Insufficient candles for %s lifecycle init", symbol)
        except Exception as e:
            logger.error("Lifecycle init failed for %s: %s", symbol, e)

    def _init_lifecycle_phases(self):
        """Cold-start: classify initial phase for each symbol using warmup data.
        
        Fetches up to 500 candles for robust regime/trend detection,
        passes CFGI data, and sets the engine's starting phase based on
        actual market conditions — not blind DCA.
        """
        WARMUP_CANDLES = 500  # ~20 days of 1h data
        MIN_CANDLES = 200     # absolute minimum for regime detection

        for symbol, engine in self._lifecycle_engines.items():
            try:
                # Fetch extended candle history for warmup
                try:
                    raw = self.client.fetch_ohlcv(symbol, self.timeframe, limit=WARMUP_CANDLES)
                    if not raw or len(raw) < MIN_CANDLES:
                        logger.warning(
                            "Insufficient warmup candles for %s: %d/%d — defaulting to DCA",
                            symbol, len(raw) if raw else 0, MIN_CANDLES)
                        continue
                    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                except Exception as e:
                    logger.warning("Failed to fetch warmup candles for %s: %s — defaulting to DCA", symbol, e)
                    continue

                price = float(df["close"].iloc[-1])
                ath = engine.config.ath
                cfgi = self._poll_cfgi(symbol)

                phase, reason = classify_phase(
                    symbol, price, ath,
                    cfgi_value=cfgi,
                    candles_df=df,
                )
                engine.set_initial_phase(phase, reason)
                self._coin_phases[symbol] = phase.value
                logger.info("🎯 %s: Cold-start → %s (%d candles, CFGI=%.0f) — %s",
                            symbol, phase.value, len(df), cfgi, reason)

                # If MARKDOWN detected, log warning about avoiding DCA into falling knife
                if phase == LifecyclePhase.MARKDOWN:
                    logger.warning(
                        "⚠️ %s detected in MARKDOWN — DCA deals blocked until phase changes",
                        symbol)
                    send_telegram(
                        f"⚠️ <b>{symbol} Cold-Start: MARKDOWN</b>\n"
                        f"Price: ${price:.2f} ({((ath-price)/ath*100):.1f}% below ATH)\n"
                        f"DCA blocked until phase changes.\n{reason}")

            except Exception as e:
                logger.error("Phase classification failed for %s: %s", symbol, e)

    def _process_lifecycle_actions(self, symbol: str, actions: list, price: float, ts: str):
        """Process action dicts returned by lifecycle engine."""
        for action in actions:
            action_type = action.get("type", "")
            
            if action_type == "buy":
                # Spring/markup buy — use cash from pool
                cost = action.get("cost", 0)
                if cost > 0 and cost <= self.cash:
                    qty = action.get("qty", cost / price)
                    self.cash -= cost
                    logger.info("📥 [LIFECYCLE] %s buy: $%.2f @ $%.4f (%s)",
                               symbol, cost, price, action.get("reason", ""))

            elif action_type == "sell":
                # Exit engine lot sell
                revenue = action.get("revenue", 0)
                if revenue > 0:
                    self.cash += revenue
                    logger.info("📤 [LIFECYCLE] %s sell: +$%.2f @ $%.4f (%s)",
                               symbol, revenue, price, action.get("reason", ""))

            elif action_type == "open_short":
                tier = action.get("tier", 1)
                engine = self._lifecycle_engines[symbol]
                short_data = engine.state.short_position
                if short_data:
                    margin = short_data.get("total_margin", 0)
                    leverage = action.get("leverage", 1)
                    if self.live and not self.adapter.is_paper:
                        # Live: execute real short via exchange adapter
                        result = self.adapter.open_short(symbol, margin, leverage=leverage, price=price)
                        if result.get("success"):
                            self._paper_shorts[symbol] = short_data
                            if margin > 0:
                                self.cash -= margin
                            send_telegram(
                                f"📉 <b>SHORT OPENED</b>\n"
                                f"Symbol: {symbol}\nTier: {tier}\n"
                                f"Margin: ${margin:.0f}\nPrice: ${price:.4f}"
                            )
                            logger.info("📉 [LIFECYCLE] %s LIVE SHORT tier %d: margin=$%.0f @ $%.4f",
                                       symbol, tier, margin, price)
                        else:
                            logger.error("❌ [LIFECYCLE] %s SHORT FAILED: %s", symbol, result.get("error"))
                            send_telegram(f"❌ SHORT FAILED: {symbol}\n{result.get('error', 'unknown')}")
                    else:
                        # Paper: track virtual position
                        self._paper_shorts[symbol] = short_data
                        if margin > 0:
                            self.cash -= margin
                        logger.info("📉 [LIFECYCLE] %s SHORT tier %d: margin=$%.0f @ $%.4f",
                                   symbol, tier, margin, price)

            elif action_type == "close_short":
                short_data = self._paper_shorts.pop(symbol, None)
                if short_data:
                    pnl = action.get("pnl", 0)
                    margin = short_data.get("total_margin", 0)
                    if self.live and not self.adapter.is_paper:
                        # Live: close real short via exchange adapter
                        qty = short_data.get("qty", 0)
                        result = self.adapter.close_short(symbol, qty, price=price)
                        if result.get("success"):
                            self.cash += margin + pnl
                            send_telegram(
                                f"📈 <b>SHORT CLOSED</b>\n"
                                f"Symbol: {symbol}\nPnL: ${pnl:.2f}\n"
                                f"Reason: {action.get('reason', 'N/A')}"
                            )
                            logger.info("📈 [LIFECYCLE] %s LIVE SHORT CLOSED: pnl=$%.2f (%s)",
                                       symbol, pnl, action.get("reason", ""))
                        else:
                            # Failed to close - put short back
                            self._paper_shorts[symbol] = short_data
                            logger.error("❌ [LIFECYCLE] %s SHORT CLOSE FAILED: %s", symbol, result.get("error"))
                            send_telegram(f"❌ SHORT CLOSE FAILED: {symbol}\n{result.get('error', 'unknown')}")
                    else:
                        # Paper: virtual close
                        self.cash += margin + pnl
                        logger.info("📈 [LIFECYCLE] %s SHORT CLOSED: pnl=$%.2f (%s)",
                                   symbol, pnl, action.get("reason", ""))

            elif action_type == "phase_change":
                new_phase = action.get("phase", "DCA")
                self._coin_phases[symbol] = new_phase
                logger.info("🔄 [LIFECYCLE] %s phase → %s", symbol, new_phase)

    def _load_lifecycle_state(self):
        """Restore lifecycle engine state from persisted state.json."""
        state_path = self.paper_dir / "state.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            lc_states = state.get("lifecycle_states", {})
            for symbol, lc_data in lc_states.items():
                if symbol in self._lifecycle_engines:
                    from .lifecycle_engine import LifecycleState
                    self._lifecycle_engines[symbol].state = LifecycleState.from_dict(lc_data)
                    phase = self._lifecycle_engines[symbol].state.phase.value
                    self._coin_phases[symbol] = phase
                    logger.info("[LIFECYCLE] Restored state for %s: phase=%s", symbol, phase)
        except Exception as e:
            logger.error("Failed to restore lifecycle state: %s", e)

    def _lifecycle_allows_dca(self, symbol: str) -> bool:
        """Check if lifecycle phase allows normal DCA operations."""
        if not self.lifecycle_enabled or symbol not in self._lifecycle_engines:
            return True  # No lifecycle = always allow DCA
        phase = self._lifecycle_engines[symbol].state.phase
        if phase != LifecyclePhase.DCA:
            logger.debug("[LIFECYCLE] %s phase=%s, skipping DCA", symbol, phase.value)
            return False
        return True

    # ── V12f Phase Tracking ─────────────────────────────────────────

    def _update_coin_phase(self, symbol: str, price: float, df: pd.DataFrame):
        """Update phase tracking for smart capital allocation.
        
        Uses lifecycle engine state if available (set externally),
        otherwise falls back to phase_classifier for cold-start detection.
        """
        # If lifecycle engine has set the phase externally, use that
        # (lifecycle integration sets _coin_phases directly)
        if hasattr(self, '_lifecycle_phase_source') and symbol in self._lifecycle_phase_source:
            self._coin_phases[symbol] = self._lifecycle_phase_source[symbol]
            return
        
        # Fallback: use phase_classifier for autonomous detection
        try:
            from .phase_classifier import classify_phase
            from .lifecycle_engine import LifecyclePhase
            
            # Known ATHs
            ATH_MAP = {
                "ETH/USDT": 4878.0, "ETH/USDC": 4878.0,
                "BTC/USDT": 109000.0, "BTC/USDC": 109000.0,
                "SOL/USDT": 260.0, "SOL/USDC": 260.0,
                "HYPE/USDC": 35.0, "ZEC/USDT": 724.0,
            }
            ath = ATH_MAP.get(symbol, price * 2)  # fallback: assume 50% below ATH
            
            cfgi = self._poll_cfgi(symbol) if hasattr(self, '_poll_cfgi') else None
            phase, reason = classify_phase(symbol, price, ath, cfgi_value=cfgi, candles_df=df)
            new_phase = phase.value
            
            old_phase = self._coin_phases.get(symbol)
            if old_phase != new_phase:
                logger.info("📊 [V12f] %s phase: %s → %s (%s)", symbol, old_phase, new_phase, reason)
            self._coin_phases[symbol] = new_phase
        except Exception as e:
            logger.debug("Phase classification failed for %s: %s", symbol, e)

    def set_coin_phase(self, symbol: str, phase: str):
        """External setter for lifecycle engine integration."""
        self._coin_phases[symbol] = phase

    # ── Rebalancing Controls ──────────────────────────────────────────

    def _read_rebalancing_controls(self) -> tuple:
        """Read rebalancing_mode and auto_rotation from controls.json.
        Returns (mode: str, auto_rotation: bool)."""
        controls_path = self.paper_dir / "controls.json"
        default_map = {"low": "conservative", "medium": "balanced", "high": "aggressive"}
        default_mode = default_map.get(self.profile.name, "balanced")
        default_auto = True
        try:
            if controls_path.exists():
                controls = json.loads(controls_path.read_text(encoding="utf-8"))
                mode = controls.get("rebalancing_mode", default_mode)
                if mode not in ("conservative", "balanced", "aggressive"):
                    mode = default_mode
                auto = controls.get("auto_rotation", default_auto)
                return mode, bool(auto)
        except Exception as e:
            logger.warning("Failed to read controls.json: %s", e)
        return default_mode, default_auto

    def _check_deployment_opportunities(self, prices: Dict[str, float]) -> dict:
        """Check scanner results for rotation candidates. Returns best opportunity or empty dict.
        
        Reads scanner_recommendation.json and compares top candidates against
        worst currently-held coin using the 20% rotation threshold from portfolio_rebalancer.
        """
        ROTATION_THRESHOLD = 0.20
        IDLE_THRESHOLDS = {"conservative": 0.70, "balanced": 0.50, "aggressive": 0.30}

        result = {"idle_capital_pct": 0.0, "best_opportunity": None}

        if not self._auto_rotation:
            return result

        # Compute idle capital
        equity = self._equity(prices)
        deployed = sum(d.capital_deployed for d in self.deals.values())
        idle_pct = (equity - deployed) / equity * 100 if equity > 0 else 0
        result["idle_capital_pct"] = round(idle_pct, 1)

        threshold = IDLE_THRESHOLDS.get(self._rebalancing_mode, 0.50) * 100
        if idle_pct < threshold:
            return result  # enough capital deployed, no need to seek

        # Read scanner results
        scanner_paths = [
            Path(__file__).parent.parent / "live" / "scanner_recommendation.json",
            Path(__file__).parent / "live" / "scanner_recommendation.json",
        ]
        scanner_data = None
        for sp in scanner_paths:
            if sp.exists():
                try:
                    scanner_data = json.loads(sp.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass

        if not scanner_data or "top_5" not in scanner_data:
            return result

        # Find worst currently-held coin score
        held_scores = {}
        for entry in scanner_data.get("top_5", []):
            sym = entry.get("symbol", "")
            if sym in self.deals or sym in self.symbols:
                held_scores[sym] = entry.get("score", 0)

        worst_held_score = min(held_scores.values()) if held_scores else 0
        worst_held_sym = min(held_scores, key=held_scores.get) if held_scores else None

        # Find best non-held candidate
        for entry in scanner_data.get("top_5", []):
            sym = entry.get("symbol", "")
            score = entry.get("score", 0)
            if sym in self.deals or sym in self.symbols:
                continue
            # Check rotation threshold
            if worst_held_score > 0 and score < worst_held_score * (1 + ROTATION_THRESHOLD):
                continue
            result["best_opportunity"] = {
                "symbol": sym,
                "score": score,
                "vs_worst": worst_held_sym,
                "vs_worst_score": worst_held_score,
                "advantage_pct": round((score - worst_held_score) / worst_held_score * 100, 1) if worst_held_score > 0 else None,
            }
            break  # top_5 is already ranked

        return result

    def _update_rebalancing_status(self, lifecycle_engines: dict = None, prices: dict = None):
        """Update rebalancing status fields from lifecycle engines."""
        self._rebalancing_mode, self._auto_rotation = self._read_rebalancing_controls()
        if lifecycle_engines:
            # Aggregate from all engines
            total_today = 0
            max_cooldown = 0.0
            for eng in lifecycle_engines.values():
                total_today += eng.state.rebalances_today
                cd = eng.rebalance_cooldown_remaining()
                if cd > max_cooldown:
                    max_cooldown = cd
            self._rebalances_today = total_today
            self._rebalance_cooldown_remaining = max_cooldown
            if max_cooldown <= 0:
                self._next_rebalance_available = "Now"
            else:
                from datetime import timedelta
                avail = datetime.now(timezone.utc) + timedelta(hours=max_cooldown)
                self._next_rebalance_available = avail.strftime("%H:%M UTC")

        # Check deployment opportunities
        if prices:
            opp = self._check_deployment_opportunities(prices)
            self._idle_capital_pct = opp.get("idle_capital_pct", 0.0)
            self._best_opportunity = opp.get("best_opportunity")

    # ── Capital Utilization & Opportunity Scoring ─────────────────

    def _update_capital_utilization(self, prices: Dict[str, float]):
        """Calculate idle capital percentage and check for deployment opportunities."""
        equity = self._equity(prices) if prices else self.initial_capital
        if equity <= 0:
            self._idle_capital_pct = 0.0
            return

        deployed = sum(
            deal.capital_deployed for deal in self.deals.values()
        )
        self._idle_capital_pct = round(max(0, (equity - deployed) / equity * 100), 1)

        # Check for opportunities from scanner results (only if auto_rotation ON)
        if not self._auto_rotation:
            self._best_opportunity = None
            return

        self._best_opportunity = self._check_scanner_opportunities()

    def _check_scanner_opportunities(self) -> Optional[dict]:
        """Read scanner results and find rotation candidates.
        Returns {symbol, score, vs_worst} if a candidate beats worst held by 20%."""
        import json as _json
        from pathlib import Path as _Path

        # Try multiple scanner result locations
        scanner_paths = [
            _Path("trading/live/scanner_recommendation.json"),
            _Path("trading/live/scanner_t1.json"),
        ]
        scanner_data = None
        for sp in scanner_paths:
            if sp.exists():
                try:
                    scanner_data = _json.loads(sp.read_text(encoding="utf-8"))
                    break
                except Exception:
                    continue

        if not scanner_data:
            return None

        # Extract scored coins from scanner
        scored = []
        if isinstance(scanner_data, dict):
            # scanner_recommendation.json format
            if "recommendation" in scanner_data:
                rec = scanner_data["recommendation"]
                if isinstance(rec, dict) and "symbol" in rec:
                    scored = [{"symbol": rec["symbol"], "score": rec.get("score", 0)}]
            # scanner_t1.json format
            elif "results" in scanner_data:
                scored = [
                    {"symbol": r.get("symbol", ""), "score": r.get("score", 0)}
                    for r in scanner_data["results"]
                    if r.get("symbol")
                ]

        if not scored:
            return None

        # Find worst currently-held coin score
        held_symbols = set(self.deals.keys())
        if not held_symbols:
            # No holdings — best scanner coin is the opportunity
            best = max(scored, key=lambda x: x["score"])
            return {"symbol": best["symbol"], "score": round(best["score"], 1), "vs_worst": "no_holdings"}

        worst_score = float("inf")
        worst_sym = None
        for s in scored:
            if s["symbol"] in held_symbols and s["score"] < worst_score:
                worst_score = s["score"]
                worst_sym = s["symbol"]

        if worst_sym is None or worst_score == float("inf"):
            return None

        # Check if any non-held coin beats worst by rotation threshold (20%)
        rotation_threshold = 0.20
        best_candidate = None
        for s in scored:
            if s["symbol"] not in held_symbols and s["score"] > worst_score * (1 + rotation_threshold):
                if best_candidate is None or s["score"] > best_candidate["score"]:
                    best_candidate = s

        if best_candidate:
            return {
                "symbol": best_candidate["symbol"],
                "score": round(best_candidate["score"], 1),
                "vs_worst": f"beats {worst_sym} ({worst_score:.1f}) by {((best_candidate['score'] / worst_score) - 1) * 100:.0f}%",
            }

        return None

    # ── Coin Pipeline (Scanner → Trader) ────────────────────────────

    def enable_pipeline(self, check_interval_hours: float = 4.0):
        """Enable scanner-driven coin management."""
        self._pipeline_enabled = True
        self._pipeline_check_interval = check_interval_hours * 3600
        logger.info("🔗 Coin pipeline enabled (check every %.1fh)", check_interval_hours)

    def _check_pipeline(self):
        """Periodically evaluate scanner results and generate add/remove commands."""
        if not self._pipeline_enabled:
            return
        now = time.time()
        if now - self._last_pipeline_check < self._pipeline_check_interval:
            return
        self._last_pipeline_check = now
        try:
            from .coin_pipeline import evaluate_coins
            commands = evaluate_coins(
                output_dir=self.paper_dir,
                max_coins=self.max_coins,
                dry_run=False,
            )
            if commands:
                logger.info("🔗 Pipeline generated %d command(s)", len(commands))
                send_telegram(
                    f"🔗 <b>Coin Pipeline Update</b>\n"
                    f"{len(commands)} change(s) queued from scanner results"
                )
        except Exception as e:
            logger.error("Pipeline check failed: %s", e)

    # ── Command Processing ────────────────────────────────────────────

    def _check_commands(self):
        """Check for and process commands from commands.json."""
        cmd_path = self.paper_dir / "commands.json"
        if not cmd_path.exists():
            return
        try:
            commands = json.loads(cmd_path.read_text(encoding="utf-8"))
            if not isinstance(commands, list) or not commands:
                cmd_path.unlink(missing_ok=True)
                return
            logger.info("📋 Processing %d command(s)", len(commands))
            for cmd in commands:
                action = cmd.get("action", "")
                try:
                    if action == "add_coin":
                        self._cmd_add_coin(cmd["symbol"])
                    elif action == "remove_coin":
                        self._cmd_remove_coin(cmd["symbol"])
                    elif action == "switch_coin":
                        self._cmd_remove_coin(cmd["from"])
                        self._cmd_add_coin(cmd["to"])
                    elif action == "pause_coin":
                        self._cmd_pause_coin(cmd["symbol"])
                    elif action == "resume_coin":
                        self._cmd_resume_coin(cmd["symbol"])
                    elif action == "pause_account":
                        self._cmd_pause_account()
                    elif action == "resume_account":
                        self._cmd_resume_account()
                    elif action == "force_close":
                        self._cmd_force_close(cmd["symbol"])
                    elif action == "emergency_exit":
                        self._cmd_emergency_exit()
                    elif action == "override_conviction":
                        self._cmd_override_conviction(cmd["symbol"], cmd["score"])
                    elif action == "clear_conviction_override":
                        self._cmd_clear_conviction_override(cmd["symbol"])
                    elif action == "adjust_max_sos":
                        self._cmd_adjust_max_sos(cmd["symbol"], cmd["delta"])
                    elif action == "reset_max_sos":
                        self._cmd_reset_max_sos(cmd["symbol"])
                    elif action == "skip_next_so":
                        self._cmd_skip_next_so(cmd["symbol"])
                    else:
                        logger.warning("Unknown command action: %s", action)
                except Exception as e:
                    logger.error("Error processing command %s: %s", cmd, e)
            cmd_path.unlink(missing_ok=True)
            self._save_state()
            self._write_status({}, {})
        except Exception as e:
            logger.error("Failed to read commands.json: %s", e)
            try:
                cmd_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _cmd_add_coin(self, symbol: str):
        """Add a coin to the active symbols list."""
        if symbol in self.symbols:
            logger.info("⚠️ %s already in symbols, skipping add", symbol)
            return
        if len(self.symbols) >= self.max_coins:
            logger.warning("⚠️ Cannot add %s — at max_coins (%d)", symbol, self.max_coins)
            send_telegram(f"⚠️ Cannot add {symbol} — already at max coins ({self.max_coins})")
            return
        self.symbols.append(symbol)
        self._coin_phases[symbol] = "DCA"
        self._coin_start_times[symbol] = datetime.now(timezone.utc).isoformat()

        # Initialize lifecycle engine for the new coin if lifecycle is enabled
        if self.lifecycle_enabled:
            try:
                self._init_lifecycle_for_symbol(symbol)
            except Exception as e:
                logger.error("Failed to init lifecycle for %s: %s", symbol, e)

        logger.info("➕ Added %s to symbols: %s", symbol, self.symbols)
        send_telegram(
            f"➕ <b>Coin Added</b>\n"
            f"Symbol: {symbol}\n"
            f"Active coins: {', '.join(self.symbols)}"
        )

    def _cmd_remove_coin(self, symbol: str):
        """Remove a coin, force-closing any active deal."""
        if symbol not in self.symbols:
            logger.info("⚠️ %s not in symbols, skipping remove", symbol)
            return
        # Force close active deal if exists
        if symbol in self.deals:
            self._force_close_deal(symbol)
        self.symbols.remove(symbol)
        logger.info("➖ Removed %s from symbols: %s", symbol, self.symbols)
        send_telegram(
            f"➖ <b>Coin Removed</b>\n"
            f"Symbol: {symbol}\n"
            f"Active coins: {', '.join(self.symbols) if self.symbols else 'none'}"
        )

    # ── Manual Control Command Handlers ──────────────────────────────

    def _cmd_pause_coin(self, symbol: str):
        self._paused_coins.add(symbol)
        logger.info("⏸ Paused coin: %s", symbol)
        send_telegram(f"⏸ <b>Coin Paused</b>\nSymbol: {symbol}\nNo new deals or SOs. Existing TPs still active.")

    def _cmd_resume_coin(self, symbol: str):
        self._paused_coins.discard(symbol)
        logger.info("▶️ Resumed coin: %s", symbol)
        send_telegram(f"▶️ <b>Coin Resumed</b>\nSymbol: {symbol}\nNormal trading restored.")

    def _cmd_pause_account(self):
        self._manually_paused = True
        logger.info("⏸ Account manually paused")
        send_telegram("⏸ <b>Account Paused</b>\nNo new deals or SOs. Existing TPs still execute.")

    def _cmd_resume_account(self):
        self._manually_paused = False
        logger.info("▶️ Account manually resumed")
        send_telegram("▶️ <b>Account Resumed</b>\nNormal trading restored.")

    def _cmd_force_close(self, symbol: str):
        """Force close active deal but keep the symbol (can reopen next cycle)."""
        if symbol not in self.deals:
            logger.info("⚠️ No active deal for %s to force close", symbol)
            return
        self._force_close_deal(symbol)
        # Note: unlike remove_coin, we do NOT remove from self.symbols

    def _cmd_emergency_exit(self):
        """Force close ALL active deals and pause account."""
        symbols = list(self.deals.keys())
        logger.warning("🛑 EMERGENCY EXIT: closing %d active deals", len(symbols))
        send_telegram(f"🛑 <b>EMERGENCY EXIT</b>\nClosing {len(symbols)} active deal(s) and pausing account.")
        for sym in symbols:
            self._force_close_deal(sym)
        self._manually_paused = True
        logger.info("🛑 Emergency exit complete. Account paused.")
        send_telegram("🛑 <b>Emergency Exit Complete</b>\nAll deals closed. Account paused.")

    def _cmd_override_conviction(self, symbol: str, score: float):
        self._conviction_overrides[symbol] = float(score)
        logger.info("🎯 Conviction override set for %s: %.1f", symbol, score)
        send_telegram(f"🎯 <b>Conviction Override</b>\nSymbol: {symbol}\nOverride score: {score}")

    def _cmd_clear_conviction_override(self, symbol: str):
        self._conviction_overrides.pop(symbol, None)
        logger.info("🎯 Conviction override cleared for %s", symbol)
        send_telegram(f"🎯 <b>Conviction Override Cleared</b>\nSymbol: {symbol}\nUsing auto-calculated score.")

    def _cmd_adjust_max_sos(self, symbol: str, delta: int):
        current = self._max_so_adjustments.get(symbol, 0)
        self._max_so_adjustments[symbol] = current + int(delta)
        logger.info("📊 Max SO adjustment for %s: %+d (total: %+d)", symbol, delta, self._max_so_adjustments[symbol])
        send_telegram(f"📊 <b>Max SO Adjusted</b>\nSymbol: {symbol}\nDelta: {delta:+d}\nTotal adjustment: {self._max_so_adjustments[symbol]:+d}")

    def _cmd_reset_max_sos(self, symbol: str):
        self._max_so_adjustments.pop(symbol, None)
        logger.info("📊 Max SO adjustment reset for %s", symbol)
        send_telegram(f"📊 <b>Max SO Reset</b>\nSymbol: {symbol}\nUsing default max SOs.")

    def _cmd_skip_next_so(self, symbol: str):
        self._skip_next_so.add(symbol)
        logger.info("⏭ Skip next SO for %s", symbol)
        send_telegram(f"⏭ <b>Skip Next SO</b>\nSymbol: {symbol}\nNext safety order will be skipped (one-time).")

    def _force_close_deal(self, symbol: str):
        """Force close an active deal at current market price."""
        deal = self.deals.get(symbol)
        if not deal or not deal.unsold_lots:
            return

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if self.live:
            # Cancel all open orders for this symbol
            self._cancel_all_symbol_orders(symbol)
            self._tp_orders.pop(symbol, None)

            # Market sell all holdings
            total_qty = deal.total_qty
            result = self._place_market_sell(symbol, total_qty)
            if not result:
                # Try fetching actual position and selling that
                actual = self._get_spot_position(symbol)
                if actual > 0:
                    result = self._place_market_sell(symbol, actual)
                if not result:
                    logger.error("Force close FAILED for %s — could not sell", symbol)
                    self._alert(f"❌ Force close FAILED for {symbol}")
                    return

            fill_price = result['price']
            fill_qty = result['qty']
            total_revenue = fill_price * fill_qty
            fee = result['fee']

            # Distribute PnL across lots
            for lot in deal.unsold_lots:
                lot_share = lot.qty / total_qty if total_qty > 0 else 1.0 / len(deal.unsold_lots)
                lot_revenue = total_revenue * lot_share
                lot_fee = fee * lot_share
                lot.sell_price = fill_price
                lot.sell_fee = lot_fee
                lot.sell_time = ts
                lot.pnl = (lot_revenue - lot_fee) - lot.cost_usd

            quote = symbol.split('/')[1]
            self.cash = self._get_balance(quote)
        else:
            # PAPER mode (unchanged)
            try:
                df = self._fetch_candles(symbol)
                if df is None or df.empty:
                    logger.error("Cannot fetch price for force close of %s", symbol)
                    return
                price = float(df["close"].iloc[-1])
            except Exception as e:
                logger.error("Error fetching price for force close of %s: %s", symbol, e)
                return

            total_revenue = 0.0
            for lot in deal.unsold_lots:
                revenue = lot.qty * price
                fee = revenue * self.taker_fee
                net = revenue - fee
                lot.sell_price = price
                lot.sell_fee = fee
                lot.sell_time = ts
                lot.pnl = net - lot.cost_usd
                total_revenue += net
            self.cash += total_revenue
            fill_price = price

        deal.close_time = ts
        self.completed_deals.append(deal)
        del self.deals[symbol]
        self._tp_orders.pop(symbol, None)

        pnl = deal.total_pnl
        mode = "Live" if self.live else "Paper"
        logger.info("🔴 FORCE CLOSED %s (%s): PnL=$%.2f", symbol, mode, pnl)
        send_telegram(
            f"🔴 <b>Force Closed Deal ({mode})</b>\n"
            f"Coin: {symbol}\nClose price: ${fill_price:.4f}\n"
            f"PnL: ${pnl:.2f}\nInvested: ${deal.total_invested:.2f}\n"
            f"Return: {pnl / deal.total_invested * 100:.2f}%"
        )
        self._append_trade_csv(deal)

    # ── Persistence ────────────────────────────────────────────────────

    def _save_state(self):
        state = {
            "cash": self.cash,
            "deal_counter": self._deal_counter,
            "halted": self._halted,
            "peak_equity": self._peak_equity,
            "max_dd": self._max_dd,
            "start_time": self._start_time.isoformat(),
            "symbols": self.symbols,
            "deals": {s: d.to_dict() for s, d in self.deals.items()},
            "completed_deals": [d.to_dict() for d in self.completed_deals[-50:]],
            "paused_coins": list(self._paused_coins),
            "manually_paused": self._manually_paused,
            "conviction_overrides": self._conviction_overrides,
            "max_so_adjustments": self._max_so_adjustments,
            "skip_next_so": list(self._skip_next_so),
            "live": self.live,
            "tp_orders": {s: {k: v for k, v in o.items() if k != 'raw'} for s, o in self._tp_orders.items()},
            "coin_phases": self._coin_phases,
            "coin_allocations": self._coin_allocations,
            "coin_start_times": self._coin_start_times,
            "smart_allocation": self.smart_allocation,
            "pipeline_enabled": self._pipeline_enabled,
            "lifecycle_enabled": self.lifecycle_enabled,
            "lifecycle_states": {s: e.state.to_dict() for s, e in self._lifecycle_engines.items()} if self.lifecycle_enabled else {},
            "paper_shorts": self._paper_shorts,
        }
        state_path = self.paper_dir / "state.json"
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _load_state(self):
        state_path = self.paper_dir / "state.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.cash = state.get("cash", self.initial_capital)
            self._deal_counter = state.get("deal_counter", 0)
            self._halted = state.get("halted", False)
            self._peak_equity = state.get("peak_equity", self.initial_capital)
            self._max_dd = state.get("max_dd", 0.0)
            if state.get("start_time"):
                try:
                    self._start_time = datetime.fromisoformat(state["start_time"])
                except Exception:
                    pass
            if state.get("symbols"):
                self.symbols = state["symbols"]
            for sym, d in state.get("deals", {}).items():
                self.deals[sym] = Deal.from_dict(d)
            for d in state.get("completed_deals", []):
                self.completed_deals.append(Deal.from_dict(d))
            # Manual control state
            self._paused_coins = set(state.get("paused_coins", []))
            self._manually_paused = state.get("manually_paused", False)
            self._conviction_overrides = state.get("conviction_overrides", {})
            self._max_so_adjustments = state.get("max_so_adjustments", {})
            self._skip_next_so = set(state.get("skip_next_so", []))
            # Live order tracking
            self._tp_orders = state.get("tp_orders", {})
            # V12f state
            if state.get("coin_phases"):
                self._coin_phases = state["coin_phases"]
            if state.get("coin_allocations"):
                self._coin_allocations = state["coin_allocations"]
            if state.get("coin_start_times"):
                self._coin_start_times = state["coin_start_times"]
            if state.get("pipeline_enabled"):
                self._pipeline_enabled = state["pipeline_enabled"]
            # Lifecycle state
            if state.get("paper_shorts"):
                self._paper_shorts = state["paper_shorts"]
            # Prune stale entries from dicts that reference removed symbols
            active_syms = set(self.symbols)
            for d in [self._coin_phases, self._coin_start_times, self._coin_allocations]:
                stale = [k for k in d if k not in active_syms]
                for k in stale:
                    del d[k]
            # Ensure all active symbols have entries
            for sym in self.symbols:
                if sym not in self._coin_phases:
                    self._coin_phases[sym] = "DCA"
                if sym not in self._coin_start_times:
                    self._coin_start_times[sym] = datetime.now(timezone.utc).isoformat()

            logger.info("State loaded: cash=$%.2f, %d active deals, %d completed, symbols=%s, live=%s",
                        self.cash, len(self.deals), len(self.completed_deals), self.symbols, self.live)
        except Exception as e:
            logger.error("Failed to load state: %s", e)

    def _build_lifecycle_status(self, prices: Dict[str, float]) -> Dict[str, dict]:
        """Build lifecycle status dict for dashboard consumption.
        
        Returns {symbol: {phase, score, short_active, spring_deployed, metrics, ...}}
        """
        lifecycle = {}
        for sym in self.symbols:
            phase = self._coin_phases.get(sym, "DCA")
            lc_data = {"phase": phase}

            # If lifecycle engine exists, extract rich state
            if sym in self._lifecycle_engines:
                engine = self._lifecycle_engines[sym]
                state = engine.state
                lc_data["score"] = round(state.conductor_cached_score, 1)
                lc_data["daily_score"] = round(state.conductor_cached_score, 1)
                lc_data["short_active"] = bool(self._paper_shorts.get(sym))

                # Short position details
                short = self._paper_shorts.get(sym)
                if short:
                    lc_data["short_entry"] = short.get("entry_price", 0)
                    lc_data["short_margin"] = short.get("total_margin", 0)
                    lc_data["short_tier"] = short.get("tier", 0)

                # Spring/markup deployed capital
                if phase == "SPRING":
                    lc_data["spring_deployed"] = getattr(state, 'spring_deployed', 0)
                elif phase == "MARKUP":
                    mp = getattr(state, 'markup_position', None)
                    lc_data["markup_deployed"] = mp.get("cost", 0) if mp else 0

                # Phase transition metrics
                lc_data["metrics"] = {
                    "exit_phases": getattr(state, 'exit_phases', 0),
                    "markdown_phases": getattr(state, 'markdown_phases', 0),
                    "spring_phases": getattr(state, 'spring_phases', 0),
                    "markup_phases": getattr(state, 'markup_phases', 0),
                }

                # PnL by phase
                lc_data["short_pnl"] = round(getattr(state, 'short_pnl', 0), 2)
                lc_data["spring_pnl"] = round(getattr(state, 'spring_pnl', 0), 2)
                lc_data["markup_pnl"] = round(getattr(state, 'markup_pnl', 0), 2)
            else:
                # No lifecycle engine — use phase_classifier result only
                lc_data["score"] = 0
                lc_data["short_active"] = False
                lc_data["metrics"] = {
                    "exit_phases": 0, "markdown_phases": 0,
                    "spring_phases": 0, "markup_phases": 0,
                }

            lifecycle[sym] = lc_data
        return lifecycle

    def _write_status(self, prices: Dict[str, float], regime_info: Dict[str, Tuple[str, bool]]):
        equity = self._equity(prices)
        pnl_pct = (equity - self.initial_capital) / self.initial_capital * 100
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds() / 3600

        coins = {}
        for sym, deal in self.deals.items():
            p = prices.get(sym, 0)
            unsold = deal.unsold_lots
            regime, is_bull = regime_info.get(sym, ("UNKNOWN", True))

            tp_pct = self._adaptive_tp(regime, 0.8)  # approximate
            dev_pct = self._adaptive_deviation(regime, 0.8, tp_pct)
            next_so_idx = len(deal.lots)
            base_price = deal.lots[0].buy_price if deal.lots else p
            spacing_mult = BEARISH_SPACING_MULT if not is_bull else 1.0
            next_so_price = self._so_trigger_price(base_price, next_so_idx, dev_pct * spacing_mult)

            # Next TP: the highest unsold lot's TP (sells first in reverse order)
            unsold_sorted = sorted(unsold, key=lambda l: l.lot_id, reverse=True)
            next_tp = unsold_sorted[0].tp_target if unsold_sorted else 0

            coin_info = {
                "state": deal.state(),
                "side": "long",
                "layers": len(deal.lots),
                "avg_entry": round(deal.avg_entry, 4),
                "current_price": round(p, 4),
                "unrealized_pnl": round(deal.unrealized_pnl(p), 2),
                "next_so_price": round(next_so_price, 4),
                "next_tp_price": round(next_tp, 4),
                "invested": round(deal.total_invested, 2),
                "realized_pnl": round(deal.total_pnl, 2),
            }
            if self.lifecycle_enabled or self.smart_allocation:
                coin_info["lifecycle_phase"] = self._coin_phases.get(sym, "DCA")
            if self.smart_allocation:
                coin_info["allocation"] = round(self._coin_allocations.get(sym, 0), 2)
            # CFGI per coin (dashboard reads c.cfgi)
            cfgi_val = self._coin_cfgi.get(sym)
            if cfgi_val is not None:
                coin_info["cfgi"] = round(cfgi_val, 1)
            coins[sym] = coin_info

        n_completed = len(self.completed_deals)
        wins = sum(1 for d in self.completed_deals if d.total_pnl > 0)
        total_realized = sum(d.total_pnl for d in self.completed_deals)

        # Global regime (from first symbol)
        first_regime = "UNKNOWN"
        first_trend = "neutral"
        if regime_info:
            r, b = next(iter(regime_info.values()))
            first_regime = r
            first_trend = "bullish" if b else "bearish"

        status = {
            "running": self._running,
            "mode": "live" if self.live else "paper",
            "profile": self.profile.name,
            "exchange": self.exchange_name,
            "capital": self.initial_capital,
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "pnl_pct": round(pnl_pct, 2),
            "coins": coins,
            "deals_completed": n_completed,
            "win_rate": round(wins / n_completed * 100, 1) if n_completed > 0 else 0.0,
            "total_realized_pnl": round(total_realized, 2),
            "max_drawdown_pct": round(self._max_dd, 2),
            "regime": first_regime,
            "trend_direction": first_trend,
            "last_update": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "uptime_hours": round(uptime, 1),
            "halted": self._halted,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "paused_coins": list(self._paused_coins),
            "account_paused": self._manually_paused,
            "conviction_overrides": self._conviction_overrides,
            "max_so_adjustments": self._max_so_adjustments,
            "controls": {
                "paused_coins": list(self._paused_coins),
                "account_paused": self._manually_paused,
                "conviction_overrides": self._conviction_overrides,
                "max_so_adjustments": self._max_so_adjustments,
                "skip_next_so": list(self._skip_next_so),
            },
            "rebalancing_mode": self._rebalancing_mode,
            "auto_rotation": self._auto_rotation,
            "rebalances_today": self._rebalances_today,
            "rebalance_cooldown_remaining": round(self._rebalance_cooldown_remaining, 2),
            "next_rebalance_available": self._next_rebalance_available,
            "idle_capital_pct": self._idle_capital_pct,
            "best_opportunity": self._best_opportunity,
            "idle_capital_pct": self._idle_capital_pct,
            "best_opportunity": self._best_opportunity,
            "fear_greed_index": self._fear_greed_index,
            "lifecycle_enabled": self.lifecycle_enabled,
            "smart_allocation": self.smart_allocation,
            "coin_phases": self._coin_phases if (self.lifecycle_enabled or self.smart_allocation) else {},
            "coin_allocations": self._coin_allocations if self.smart_allocation else {},
            "pipeline_enabled": self._pipeline_enabled,
            "lifecycle": self._build_lifecycle_status(prices),
        }
        (self.paper_dir / "status.json").write_text(
            json.dumps(status, indent=2, default=str), encoding="utf-8"
        )

    def _append_trade_csv(self, deal: Deal):
        csv_path = self.paper_dir / "trades.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "deal_id", "symbol", "open_time", "close_time", "regime",
                    "layers", "invested", "pnl", "return_pct", "duration_h",
                ])
            duration_h = 0.0
            try:
                t0 = datetime.fromisoformat(deal.open_time)
                t1 = datetime.fromisoformat(deal.close_time)
                duration_h = (t1 - t0).total_seconds() / 3600
            except Exception:
                pass
            ret_pct = deal.total_pnl / deal.total_invested * 100 if deal.total_invested > 0 else 0
            w.writerow([
                deal.deal_id, deal.symbol, deal.open_time, deal.close_time,
                deal.regime_at_open, len(deal.lots), round(deal.total_invested, 2),
                round(deal.total_pnl, 2), round(ret_pct, 2), round(duration_h, 2),
            ])

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self):
        """Main blocking run loop."""
        mode = "LIVE" if self.live else "PAPER"
        logger.info("=" * 60)
        logger.info("Spot %s Trader starting", mode)
        logger.info("Exchange: %s | Profile: %s | Capital: $%.0f",
                     self.exchange_name, self.profile.name, self.initial_capital)
        logger.info("Symbols: %s | Timeframe: %s | Max coins: %d",
                     self.symbols, self.timeframe, self.max_coins)
        logger.info("=" * 60)

        # Connect to exchange
        try:
            config_path = Path(__file__).parent / "spot_config.json"
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                exc_cfg = cfg.get("exchanges", {}).get(self.exchange_name, {})
                config = {"options": exc_cfg.get("options", {})}
            self.client.connect(self.exchange_name, config)
            logger.info("Connected to %s", self.exchange_name)
        except Exception as e:
            logger.error("Failed to connect: %s", e)
            send_telegram(f"❌ Spot Paper: Failed to connect to {self.exchange_name}: {e}")
            return

        # Load persisted state
        self._load_state()

        # Live mode: sync cash from exchange and reconcile
        if self.live:
            quote = 'USDT'  # default, could be parameterized
            self.cash = self._get_balance(quote)
            self.initial_capital = self.initial_capital or self.cash
            self._reconcile_on_startup()

        # Initialize lifecycle engines if enabled
        if self.lifecycle_enabled and self._lifecycle_engines:
            self._init_lifecycle_phases()
            self._load_lifecycle_state()

        self._running = True
        self._start_time = datetime.now(timezone.utc)
        poll_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 900)

        send_telegram(
            f"🚀 <b>Spot {mode} Trader Started</b>\n"
            f"Exchange: {self.exchange_name}\nProfile: {self.profile.name}\n"
            f"Capital: ${self.initial_capital:.0f}\nCash: ${self.cash:.2f}\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Timeframe: {self.timeframe}\nPoll: {poll_seconds}s"
        )

        # Signal handling for graceful shutdown
        def _shutdown(sig, frame):
            logger.info("Shutdown signal received")
            self._running = False
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        cycle = 0
        while self._running:
            cycle += 1
            try:
                self._run_cycle(cycle)
            except Exception as e:
                logger.error("Cycle %d error: %s\n%s", cycle, e, traceback.format_exc())

            # Save state every cycle
            self._save_state()

            if self._running:
                logger.debug("Sleeping %ds until next cycle...", poll_seconds)
                # Sleep in small increments for responsive shutdown
                for _ in range(poll_seconds):
                    if not self._running:
                        break
                    time.sleep(1)

        # Final save
        self._running = False
        self._save_state()
        self._write_status({}, {})
        logger.info("%s trader stopped.", mode)
        send_telegram(f"🛑 <b>Spot {mode} Trader Stopped</b>")

    def _run_cycle(self, cycle: int):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("── Cycle %d at %s ──", cycle, ts)

        # Check scanner pipeline for coin changes
        self._check_pipeline()

        # Check for coin management commands (including pipeline-generated ones)
        self._check_commands()

        prices: Dict[str, float] = {}
        regime_info: Dict[str, Tuple[str, bool]] = {}

        # Poll BTC CFGI as market-wide fear & greed index
        try:
            btc_cache = Path(__file__).parent / "data" / "cfgi_cache" / "BTC_cfgi_daily.json"
            if btc_cache.exists():
                data = json.loads(btc_cache.read_text(encoding="utf-8"))
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for entry in reversed(data):
                    if entry.get("date", "") <= today:
                        self._fear_greed_index = float(entry.get("value", 50))
                        break
        except Exception:
            pass

        for symbol in self.symbols:
            # Fetch candles and compute regime
            df = self._fetch_candles(symbol)
            if df is None:
                continue

            price = float(df["close"].iloc[-1])
            prices[symbol] = price

            regime, atr_pct_val, is_bullish = self._compute_regime_and_atr(df)
            regime_info[symbol] = (regime, is_bullish)

            # Detect regime change
            prev_regime = self._last_regime.get(symbol)
            if prev_regime and prev_regime != regime:
                logger.info("🔄 Regime change for %s: %s → %s", symbol, prev_regime, regime)
                send_telegram(
                    f"🔄 <b>Regime Change</b>\n"
                    f"Coin: {symbol}\n{prev_regime} → {regime}\n"
                    f"Trend: {'Bullish' if is_bullish else 'Bearish'}"
                )
            self._last_regime[symbol] = regime

            tp_pct = self._adaptive_tp(regime, atr_pct_val)
            dev_pct = self._adaptive_deviation(regime, atr_pct_val, tp_pct)

            # Poll CFGI for all symbols (dashboard needs it)
            self._poll_cfgi(symbol)

            logger.debug("%s: $%.4f | regime=%s | ATR%%=%.2f | TP=%.2f%% | Dev=%.2f%%",
                         symbol, price, regime, atr_pct_val, tp_pct, dev_pct)

            # V12f: update phase tracking for smart allocation
            if self.smart_allocation:
                self._update_coin_phase(symbol, price, df)

            # V12e lifecycle engine processing
            if self.lifecycle_enabled and symbol in self._lifecycle_engines:
                try:
                    engine = self._lifecycle_engines[symbol]
                    high = float(df["high"].iloc[-1])
                    low = float(df["low"].iloc[-1])
                    
                    # Feed candles for conductor scoring (needs timestamp as ms int)
                    conductor_df = df.copy()
                    if pd.api.types.is_datetime64_any_dtype(conductor_df["timestamp"]):
                        conductor_df["timestamp"] = conductor_df["timestamp"].astype(np.int64) // 10**6
                    engine.feed_candles_1h(conductor_df)
                    
                    # Get CFGI score
                    cfgi = self._poll_cfgi(symbol)
                    
                    # Process lifecycle cycle
                    actions = engine.process_cycle(
                        price=price, high=high, low=low, ts=ts,
                        regime=regime, cfgi_score=cfgi,
                        cash=self.cash, deals=self.deals,
                        send_telegram_fn=send_telegram,
                    )
                    
                    # Execute returned actions
                    if actions:
                        self._process_lifecycle_actions(symbol, actions, price, ts)
                    
                    # Update phase tracking
                    self._coin_phases[symbol] = engine.state.phase.value
                except Exception as e:
                    logger.error("[LIFECYCLE] process_cycle error for %s: %s", symbol, e)

            # Process active deal (gated by lifecycle phase)
            if symbol in self.deals:
                deal = self.deals[symbol]
                if not self._halted and not self._manually_paused and self._lifecycle_allows_dca(symbol):
                    self._check_safety_orders(deal, price, ts, regime, dev_pct, tp_pct, is_bullish)
                # TPs always execute even when paused or in non-DCA phase
                self._check_exits(deal, price, ts, regime, atr_pct_val)
            elif not self._halted and not self._manually_paused and regime not in BLOCKED_REGIMES and len(self.deals) < self.max_coins and self._lifecycle_allows_dca(symbol):
                # Open new deal (only in DCA phase)
                self._open_deal(symbol, price, ts, regime, tp_pct)

        # Update equity and drawdown
        equity = self._equity(prices)
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = (self._peak_equity - equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0
        if dd > self._max_dd:
            self._max_dd = dd
        if dd >= self.profile.max_drawdown_pct and not self._halted:
            self._halted = True
            logger.warning("⚠️ DRAWDOWN HALT: %.1f%% >= %.1f%%", dd, self.profile.max_drawdown_pct)
            send_telegram(f"⚠️ <b>Drawdown Halt!</b>\nDD: {dd:.1f}%\nEquity: ${equity:.2f}")

        # Update capital utilization and opportunity scoring
        self._update_capital_utilization(prices)

        # Write status
        self._write_status(prices, regime_info)

        logger.info("Equity: $%.2f (%.2f%%) | Cash: $%.2f | Deals: %d active, %d completed | DD: %.1f%%",
                     equity, (equity - self.initial_capital) / self.initial_capital * 100,
                     self.cash, len(self.deals), len(self.completed_deals), dd)


# Backwards compatibility alias
SpotPaperTrader = LifecycleTrader
