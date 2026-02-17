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
from ..regime_detector import classify_regime_v2
from ..indicators import atr as compute_atr, atr_pct as compute_atr_pct

logger = logging.getLogger(__name__)

PAPER_BASE = Path(__file__).parent / "paper"
PAPER_BASE.mkdir(exist_ok=True)

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

class SpotPaperTrader:
    """Live paper trading bot using real market data with virtual positions."""

    def __init__(
        self,
        exchange: str = "aster",
        profile: str = "medium",
        capital: float = 10000.0,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        max_coins: Optional[int] = None,
    ):
        self.exchange_name = exchange.lower()
        self.profile = PROFILES[profile.lower()]
        self.initial_capital = capital
        self.cash = capital
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

        # Per-exchange output directory
        self.paper_dir = PAPER_BASE / self.exchange_name
        self.paper_dir.mkdir(parents=True, exist_ok=True)

        # Exchange client
        self.client = SpotExchangeClient()

        # Logging
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.paper_dir / "bot.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

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
        # Capital per coin with 10% reserve
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
        filled_sos = len(deal.lots) - 1
        if filled_sos >= self.profile.max_safety_orders:
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
            so_cost = min(so_cost, self.cash * 0.95)  # leave small buffer
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
            deal.close_time = ts
            self.completed_deals.append(deal)
            del self.deals[deal.symbol]

            duration_h = 0.0
            try:
                t0 = datetime.fromisoformat(deal.open_time)
                t1 = datetime.fromisoformat(deal.close_time)
                duration_h = (t1 - t0).total_seconds() / 3600
            except Exception:
                pass

            logger.info("✅ DEAL %d COMPLETE: %s, PnL=$%.2f, Duration=%.1fh",
                         deal.deal_id, deal.symbol, deal.total_pnl, duration_h)
            send_telegram(
                f"✅ <b>Spot Paper: Deal Complete</b>\n"
                f"Coin: {deal.symbol}\nTotal PnL: ${deal.total_pnl:.2f}\n"
                f"Invested: ${deal.total_invested:.2f}\n"
                f"Return: {deal.total_pnl / deal.total_invested * 100:.2f}%\n"
                f"Duration: {duration_h:.1f}h"
            )
            self._append_trade_csv(deal)

    # ── Equity ─────────────────────────────────────────────────────────

    def _equity(self, prices: Dict[str, float]) -> float:
        unsold_value = 0.0
        for sym, deal in self.deals.items():
            p = prices.get(sym, 0)
            unsold_value += sum(l.qty * p for l in deal.unsold_lots)
        return self.cash + unsold_value

    # ── Persistence ────────────────────────────────────────────────────

    def _save_state(self):
        state = {
            "cash": self.cash,
            "deal_counter": self._deal_counter,
            "halted": self._halted,
            "peak_equity": self._peak_equity,
            "max_dd": self._max_dd,
            "start_time": self._start_time.isoformat(),
            "deals": {s: d.to_dict() for s, d in self.deals.items()},
            "completed_deals": [d.to_dict() for d in self.completed_deals[-50:]],  # keep last 50
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
            for sym, d in state.get("deals", {}).items():
                self.deals[sym] = Deal.from_dict(d)
            for d in state.get("completed_deals", []):
                self.completed_deals.append(Deal.from_dict(d))
            logger.info("State loaded: cash=$%.2f, %d active deals, %d completed",
                        self.cash, len(self.deals), len(self.completed_deals))
        except Exception as e:
            logger.error("Failed to load state: %s", e)

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

            coins[sym] = {
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
        logger.info("=" * 60)
        logger.info("Spot Paper Trader starting")
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

        self._running = True
        self._start_time = datetime.now(timezone.utc)
        poll_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 900)

        send_telegram(
            f"🚀 <b>Spot Paper Trader Started</b>\n"
            f"Exchange: {self.exchange_name}\nProfile: {self.profile.name}\n"
            f"Capital: ${self.initial_capital:.0f}\nSymbols: {', '.join(self.symbols)}\n"
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
        logger.info("Paper trader stopped.")
        send_telegram("🛑 <b>Spot Paper Trader Stopped</b>")

    def _run_cycle(self, cycle: int):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("── Cycle %d at %s ──", cycle, ts)

        prices: Dict[str, float] = {}
        regime_info: Dict[str, Tuple[str, bool]] = {}

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

            logger.debug("%s: $%.4f | regime=%s | ATR%%=%.2f | TP=%.2f%% | Dev=%.2f%%",
                         symbol, price, regime, atr_pct_val, tp_pct, dev_pct)

            # Process active deal
            if symbol in self.deals:
                deal = self.deals[symbol]
                if not self._halted:
                    self._check_safety_orders(deal, price, ts, regime, dev_pct, tp_pct, is_bullish)
                self._check_exits(deal, price, ts, regime, atr_pct_val)
            elif not self._halted and regime not in BLOCKED_REGIMES and len(self.deals) < self.max_coins:
                # Open new deal
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

        # Write status
        self._write_status(prices, regime_info)

        logger.info("Equity: $%.2f (%.2f%%) | Cash: $%.2f | Deals: %d active, %d completed | DD: %.1f%%",
                     equity, (equity - self.initial_capital) / self.initial_capital * 100,
                     self.cash, len(self.deals), len(self.completed_deals), dd)
