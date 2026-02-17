"""Live Martingale DCA trader for Aster DEX (Binance-compatible futures API).
Supports simultaneous LONG + SHORT deals (dual-tracking) with regime-based allocation."""
import hashlib
import hmac
import json
import csv
import time
import os
import traceback
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
import numpy as np

from .regime_detector import classify_regime_v2, is_martingale_friendly_v2

LIVE_DIR = Path(__file__).parent / "live"
LIVE_DIR.mkdir(exist_ok=True)

# Telegram config
TG_TOKEN = "8528958079:AAF90HSJ5Ck1urUydzS5CUvyf2EEeB7LUwc"
TG_CHAT_ID = "5221941584"
TG_ENABLED = True

# HYPEUSDT market rules
TICK_SIZE = 0.001
STEP_SIZE = 0.01
MIN_QTY = 0.01
MIN_NOTIONAL = 5.0
PRICE_PRECISION = 3  # decimals for tick 0.001
QTY_PRECISION = 2

# Regime-based capital allocation: (long_fraction, short_fraction)
# For directional regimes (TRENDING, MILD_TREND, DISTRIBUTION), these are
# the BULLISH allocations. They get flipped when trend direction is bearish.
REGIME_ALLOC = {
    "ACCUMULATION": (0.70, 0.30),
    "CHOPPY": (0.50, 0.50),
    "RANGING": (0.50, 0.50),
    "DISTRIBUTION": (0.30, 0.70),
    "MILD_TREND": (0.60, 0.40),
    "TRENDING": (0.75, 0.25),
    "EXTREME": (0.0, 0.0),       # no new deals
    "BREAKOUT_WARNING": (0.0, 0.0),
    "UNKNOWN": (0.50, 0.50),
}
# Regimes where allocation should flip based on trend direction
DIRECTIONAL_REGIMES = {"TRENDING", "MILD_TREND", "DISTRIBUTION"}


def round_price(p: float) -> float:
    return round(round(p / TICK_SIZE) * TICK_SIZE, PRICE_PRECISION)


def round_qty(q: float) -> float:
    return round(round(q / STEP_SIZE) * STEP_SIZE, QTY_PRECISION)


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


class AsterAPI:
    """Thin wrapper around Aster DEX Binance-compatible futures API."""

    def __init__(self, base_url: str = "https://fapi.asterdex.com"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("ASTER_API_KEY", "")
        self.api_secret = os.environ.get("ASTER_API_SECRET", "")
        # Fallback: read from Windows registry (setx stores there)
        if not self.api_key:
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment') as k:
                    self.api_key = winreg.QueryValueEx(k, 'ASTER_API_KEY')[0]
                    self.api_secret = winreg.QueryValueEx(k, 'ASTER_API_SECRET')[0]
            except Exception:
                pass
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _get(self, path: str, params: dict = None, signed: bool = False) -> Any:
        params = params or {}
        if signed:
            params = self._sign(params)
        r = self.session.get(f"{self.base_url}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict = None, signed: bool = True) -> Any:
        params = params or {}
        if signed:
            params = self._sign(params)
        r = self.session.post(f"{self.base_url}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict = None, signed: bool = True) -> Any:
        params = params or {}
        if signed:
            params = self._sign(params)
        r = self.session.delete(f"{self.base_url}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def ping(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/fapi/v1/ping", timeout=5)
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"    [PING] {e}")
            return False

    def klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        data = self._get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"
        ])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def set_leverage(self, symbol: str, leverage: int):
        return self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def set_hedge_mode(self, enabled: bool = True) -> bool:
        """Try to enable dual-side (hedge) position mode. Returns True if successful."""
        try:
            self._post("/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if enabled else "false"})
            return True
        except Exception as e:
            # May already be set, or not supported
            err_str = str(e)
            if "No need to change" in err_str or "4059" in err_str:
                return True  # already in desired mode
            return False

    def get_position_mode(self) -> bool:
        """Check if hedge mode is enabled. Returns True if dual-side."""
        try:
            result = self._get("/fapi/v1/positionSide/dual", {}, signed=True)
            return result.get("dualSidePosition", False)
        except Exception:
            return False

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                    price: float = None, reduce_only: bool = False, time_in_force: str = None,
                    position_side: str = None) -> dict:
        params = {"symbol": symbol, "side": side, "type": order_type, "quantity": round_qty(quantity)}
        if price is not None:
            params["price"] = round_price(price)
        if reduce_only:
            params["reduceOnly"] = "true"
        if position_side:
            params["positionSide"] = position_side
        if time_in_force:
            params["timeInForce"] = time_in_force
        elif order_type == "LIMIT":
            params["timeInForce"] = "GTC"
        return self._post("/fapi/v1/order", params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    def open_orders(self, symbol: str) -> list:
        return self._get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def query_order(self, symbol: str, order_id: int) -> dict:
        return self._get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def position_risk(self, symbol: str = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/fapi/v2/positionRisk", params, signed=True)

    def balance(self) -> list:
        return self._get("/fapi/v2/balance", {}, signed=True)

    def usdt_balance(self) -> float:
        for b in self.balance():
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    def usdt_equity(self) -> float:
        for b in self.balance():
            if b["asset"] == "USDT":
                return float(b["balance"]) + float(b.get("crossUnPnl", 0))
        return 0.0

    def usdt_available(self) -> float:
        for b in self.balance():
            if b["asset"] == "USDT":
                return float(b.get("availableBalance", b.get("balance", 0)))
        return 0.0

    def funding_rate_history(self, symbol: str, limit: int = 50) -> list:
        """Get funding rate history (public endpoint)."""
        return self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}, signed=False)

    def premium_index(self, symbol: str) -> dict:
        """Get current funding rate info (public endpoint)."""
        return self._get("/fapi/v1/premiumIndex", {"symbol": symbol}, signed=False)


@dataclass
class LiveDeal:
    deal_id: int
    symbol: str
    entry_price: float
    entry_qty: float
    entry_cost: float  # notional
    entry_time: str
    direction: str = "LONG"  # "LONG" or "SHORT"
    safety_orders_filled: int = 0
    safety_order_ids: List[int] = field(default_factory=list)  # exchange order IDs for pending SOs
    tp_order_id: Optional[int] = None
    total_qty: float = 0.0
    total_cost: float = 0.0
    avg_entry: float = 0.0
    closed: bool = False
    close_price: Optional[float] = None
    close_time: Optional[str] = None
    realized_pnl: float = 0.0

    def __post_init__(self):
        if self.total_qty == 0:
            self.total_qty = self.entry_qty
            self.total_cost = self.entry_cost
            self.avg_entry = self.entry_price

    def add_fill(self, price: float, qty: float, cost: float):
        self.total_qty += qty
        self.total_cost += cost
        self.avg_entry = self.total_cost / self.total_qty if self.total_qty > 0 else 0
        self.safety_orders_filled += 1

    def tp_price(self, tp_pct: float) -> float:
        if self.direction == "SHORT":
            return round_price(self.avg_entry * (1 - tp_pct / 100))
        return round_price(self.avg_entry * (1 + tp_pct / 100))

    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id, "symbol": self.symbol,
            "entry_price": self.entry_price, "entry_qty": self.entry_qty,
            "entry_cost": self.entry_cost, "entry_time": self.entry_time,
            "direction": self.direction,
            "safety_orders_filled": self.safety_orders_filled,
            "safety_order_ids": self.safety_order_ids,
            "tp_order_id": self.tp_order_id,
            "total_qty": self.total_qty, "total_cost": self.total_cost,
            "avg_entry": self.avg_entry, "closed": self.closed,
            "close_price": self.close_price, "close_time": self.close_time,
            "realized_pnl": self.realized_pnl,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiveDeal":
        # Backward compat: default direction to LONG if missing
        if "direction" not in d:
            d["direction"] = "LONG"
        return cls(**d)


class AsterTrader:
    """Martingale DCA bot for Aster DEX perpetual futures (dual-tracking: simultaneous LONG + SHORT)."""

    # Strategy defaults
    TP_PCT = 1.5  # baseline / fallback
    TP_MIN = 0.6
    TP_MAX = 2.5
    TP_UPDATE_THRESHOLD = 0.1  # min difference to trigger TP order update
    ATR_PERIOD = 14
    ATR_BASELINE_PCT = 0.8  # baseline ATR% (calibrated to HYPE 5m)
    MAX_SOS = 8
    DEVIATION_PCT = 2.5  # baseline / fallback
    DEV_MIN = 1.2
    DEV_MAX = 4.0
    DEV_UPDATE_THRESHOLD = 0.15  # min difference to trigger SO re-placement
    DEV_TP_FLOOR_MULT = 1.5  # deviation must be >= TP * this
    DEVIATION_MULT = 1.0
    SO_VOL_MULT = 2.0
    BASE_ORDER_PCT = 0.04  # 4% of capital
    FEE_PCT = 0.05
    MARGIN_RESERVE_PCT = 0.10  # keep 10% of capital as margin safety buffer

    def __init__(self, symbol: str = "HYPEUSDT", timeframe: str = "5m",
                 capital: float = None, max_drawdown_pct: float = 25.0,
                 leverage: int = 1, dry_run: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_drawdown_pct = max_drawdown_pct
        self.leverage = leverage
        self.dry_run = dry_run

        self.api = AsterAPI()
        self.capital = capital  # set later from balance if None

        # Dual-tracking: independent long and short deal slots
        self.long_deal: Optional[LiveDeal] = None
        self.short_deal: Optional[LiveDeal] = None
        self.hedge_mode: bool = False  # set during init

        self.deal_counter = 0
        self.current_regime = "UNKNOWN"
        self.current_price = 0.0
        self.peak_equity = 0.0
        self.start_equity = 0.0
        self.total_deposits = 0.0  # Track deposits to separate from trading PnL
        self.daily_start_equity = 0.0
        self.daily_start_date = ""
        self.consecutive_errors = 0
        self._trend_bullish = True  # default bullish until first regime detection
        self.halted = False
        self.halt_reason = ""
        self.paused_until: Optional[float] = None
        self._running = False
        self.start_time = ""
        self.cycle_count = 0
        self.last_daily_summary = ""
        self._last_klines_df: Optional[pd.DataFrame] = None

        # Funding fee tracking
        self.total_funding_fees = 0.0  # Negative = paid, positive = received
        self.last_funding_time: Optional[str] = None
        self.last_funding_rate: float = 0.0
        self.last_funding_cost: float = 0.0
        self.current_funding_rate: float = 0.0

        # Adaptive TP/deviation state
        self.current_tp_pct: float = self.TP_PCT
        self.current_dev_pct: float = self.DEVIATION_PCT
        self.current_atr_pct: float = 0.0

        # Margin cache (refreshed once per cycle)
        self._cached_available_margin: float = 0.0
        self._margin_cache_time: float = 0.0

        self.load_state()

    # ── Helper: get deal by direction ──────────────────────────────────

    def _get_deal(self, direction: str) -> Optional[LiveDeal]:
        return self.long_deal if direction == "LONG" else self.short_deal

    def _set_deal(self, direction: str, deal: Optional[LiveDeal]):
        if direction == "LONG":
            self.long_deal = deal
        else:
            self.short_deal = deal

    def _active_deals(self) -> List[LiveDeal]:
        """Return list of active deals."""
        deals = []
        if self.long_deal:
            deals.append(self.long_deal)
        if self.short_deal:
            deals.append(self.short_deal)
        return deals

    # ── Safety order math ──────────────────────────────────────────────

    def so_deviation(self, n: int) -> float:
        """Cumulative deviation % for SO #n (1-indexed)."""
        return self.current_dev_pct * n  # mult=1.0 so linear

    def so_price(self, entry: float, n: int, direction: str = "LONG") -> float:
        if direction == "SHORT":
            return round_price(entry * (1 + self.so_deviation(n) / 100))
        return round_price(entry * (1 - self.so_deviation(n) / 100))

    def so_qty(self, base_qty_usd: float, n: int) -> float:
        """USD size for SO #n."""
        size = base_qty_usd * (self.SO_VOL_MULT ** n)
        return size

    def base_order_usd(self, alloc_fraction: float = 1.0) -> float:
        """Base order in USD. Not scaled by allocation — capital too small.
        Allocation gates whether we open, not the order size."""
        return self.capital * self.BASE_ORDER_PCT

    # ── Margin management ──────────────────────────────────────────────

    def _get_available_margin(self, force: bool = False) -> float:
        """Get available margin, cached for 25s per cycle to minimize API calls."""
        if self.dry_run:
            return float('inf')
        now = time.time()
        if not force and (now - self._margin_cache_time) < 25:
            return self._cached_available_margin
        try:
            self._cached_available_margin = self.api.usdt_available()
            self._margin_cache_time = now
        except Exception as e:
            print(f"  [WARN] Margin check failed: {e}")
        return self._cached_available_margin

    def _margin_reserve(self) -> float:
        """Minimum margin that must remain available (safety buffer)."""
        return (self.capital or 0) * self.MARGIN_RESERVE_PCT

    def _estimate_order_margin(self, qty: float, price: float) -> float:
        """Estimate margin required for an order. In cross margin with leverage,
        margin ≈ notional / leverage. Add a 20% buffer for fees/funding."""
        notional = qty * price
        margin = notional / self.leverage if self.leverage > 0 else notional
        return margin * 1.2  # 20% safety buffer

    def _ensure_tp_margin(self, deal: LiveDeal, tp_qty: float, tp_price: float) -> bool:
        """Ensure enough margin for TP order. If not, cancel deepest SOs to free margin.
        Returns True if margin is now sufficient."""
        if self.dry_run:
            return True
        needed = self._estimate_order_margin(tp_qty, tp_price)
        available = self._get_available_margin(force=True)
        reserve = self._margin_reserve()

        if available - needed >= reserve:
            return True

        # Cancel deepest SOs to free margin
        print(f"  ⚠️  Insufficient margin for {deal.direction} TP (need ${needed:.2f}, avail ${available:.2f}) — freeing margin from SOs")
        # SOs are ordered shallow→deep, cancel from the end
        cancelled = 0
        while deal.safety_order_ids and (available - needed < reserve):
            deepest_oid = deal.safety_order_ids.pop()  # remove deepest
            try:
                self.api.cancel_order(self.symbol, deepest_oid)
                cancelled += 1
                # Refresh margin after cancellation
                available = self._get_available_margin(force=True)
            except Exception as e:
                print(f"    [WARN] Failed to cancel SO {deepest_oid}: {e}")

        if cancelled > 0:
            print(f"    🔓 Cancelled {cancelled} deepest SO(s) to free margin for TP (avail now: ${available:.2f})")
            send_telegram(f"⚠️ Cancelled {cancelled} SO(s) for {deal.direction} deal to ensure TP placement")

        return (available - needed) >= reserve

    # ── State persistence ──────────────────────────────────────────────

    def save_state(self):
        state = {
            "long_deal": self.long_deal.to_dict() if self.long_deal else None,
            "short_deal": self.short_deal.to_dict() if self.short_deal else None,
            "deal_counter": self.deal_counter,
            "current_regime": self.current_regime,
            "current_price": self.current_price,
            "peak_equity": self.peak_equity,
            "start_equity": self.start_equity,
            "total_deposits": self.total_deposits,
            "daily_start_equity": self.daily_start_equity,
            "daily_start_date": self.daily_start_date,
            "capital": self.capital,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "paused_until": self.paused_until,
            "start_time": self.start_time,
            "cycle_count": self.cycle_count,
            "last_daily_summary": self.last_daily_summary,
            "hedge_mode": self.hedge_mode,
            "current_tp_pct": self.current_tp_pct,
            "current_dev_pct": self.current_dev_pct,
            "current_atr_pct": self.current_atr_pct,
            # Funding fee tracking
            "total_funding_fees": self.total_funding_fees,
            "last_funding_time": self.last_funding_time,
            "last_funding_rate": self.last_funding_rate,
            "last_funding_cost": self.last_funding_cost,
            "current_funding_rate": self.current_funding_rate,
        }
        with open(LIVE_DIR / "state.json", "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        path = LIVE_DIR / "state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                s = json.load(f)

            # ── Backward compat: migrate old single-deal state ──
            if "deal" in s and "long_deal" not in s:
                old_deal = s.get("deal")
                if old_deal:
                    deal_obj = LiveDeal.from_dict(old_deal)
                    if deal_obj.direction == "SHORT":
                        self.short_deal = deal_obj
                    else:
                        self.long_deal = deal_obj
                    print(f"  ♻️  Migrated old single deal → {deal_obj.direction.lower()}_deal")
                else:
                    self.long_deal = None
                    self.short_deal = None
            else:
                self.long_deal = LiveDeal.from_dict(s["long_deal"]) if s.get("long_deal") else None
                self.short_deal = LiveDeal.from_dict(s["short_deal"]) if s.get("short_deal") else None

            self.deal_counter = s.get("deal_counter", 0)
            self.current_regime = s.get("current_regime", "UNKNOWN")
            self.current_price = s.get("current_price", 0)
            self.peak_equity = s.get("peak_equity", 0)
            self.start_equity = s.get("start_equity", 0)
            self.total_deposits = s.get("total_deposits", 0)
            self.daily_start_equity = s.get("daily_start_equity", 0)
            self.daily_start_date = s.get("daily_start_date", "")
            self.capital = s.get("capital", self.capital)
            self.halted = s.get("halted", False)
            self.halt_reason = s.get("halt_reason", "")
            self.paused_until = s.get("paused_until")
            self.start_time = s.get("start_time", "")
            self.cycle_count = s.get("cycle_count", 0)
            self.last_daily_summary = s.get("last_daily_summary", "")
            self.hedge_mode = s.get("hedge_mode", False)
            self.current_tp_pct = s.get("current_tp_pct", self.TP_PCT)
            self.current_dev_pct = s.get("current_dev_pct", self.DEVIATION_PCT)
            self.current_atr_pct = s.get("current_atr_pct", 0.0)
            
            # Load funding fee tracking
            self.total_funding_fees = s.get("total_funding_fees", 0.0)
            self.last_funding_time = s.get("last_funding_time")
            self.last_funding_rate = s.get("last_funding_rate", 0.0)
            self.last_funding_cost = s.get("last_funding_cost", 0.0)
            self.current_funding_rate = s.get("current_funding_rate", 0.0)

            long_str = f"LONG(#{self.long_deal.deal_id})" if self.long_deal else "none"
            short_str = f"SHORT(#{self.short_deal.deal_id})" if self.short_deal else "none"
            print(f"  ♻️  Resumed state: long={long_str}, short={short_str}, cycles={self.cycle_count}")

            # Auto-clear transient halts (API errors) on restart — they're connectivity issues, not fatal
            if self.halted and "API error" in self.halt_reason:
                print(f"  🔄 Clearing transient halt on restart (was: {self.halt_reason})")
                self.halted = False
                self.halt_reason = ""
                self.consecutive_errors = 0
        except Exception as e:
            print(f"  [WARN] State load failed: {e}")

    def write_status(self):
        equity = self._get_equity()
        gross_pnl = equity - self.start_equity if self.start_equity else 0
        trading_pnl = gross_pnl - self.total_deposits  # Exclude deposits from trading returns
        pnl = trading_pnl  # Dashboard shows true trading PnL
        dd = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0

        long_info = self.long_deal.to_dict() if self.long_deal else None
        short_info = self.short_deal.to_dict() if self.short_deal else None
        long_alloc, short_alloc = REGIME_ALLOC.get(self.current_regime, (0.5, 0.5))
        if self.current_regime in DIRECTIONAL_REGIMES and not self._trend_bullish:
            long_alloc, short_alloc = short_alloc, long_alloc

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "running": self._running,
            "dry_run": self.dry_run,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "regime": self.current_regime,
            "price": self.current_price,
            "equity": round(equity, 2),
            "start_equity": round(self.start_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "total_deposits": round(self.total_deposits, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (self.start_equity + self.total_deposits) * 100, 2) if (self.start_equity + self.total_deposits) else 0,
            "drawdown_pct": round(dd, 2),
            "max_drawdown_threshold": self.max_drawdown_pct,
            "leverage": self.leverage,
            "capital": round(self.capital, 2) if self.capital else 0,
            "available_balance": round(self.api.usdt_available(), 2) if not self.dry_run else 0,
            "cycle_count": self.cycle_count,
            "start_time": self.start_time,
            "hedge_mode": self.hedge_mode,
            "long_deal": long_info,
            "short_deal": short_info,
            "trend_direction": "bullish" if self._trend_bullish else "bearish",
            "regime_alloc": {"long": long_alloc, "short": short_alloc},
            "deal_counter": self.deal_counter,
            "adaptive_tp": {
                "current_tp_pct": round(self.current_tp_pct, 3),
                "baseline_tp_pct": self.TP_PCT,
                "current_dev_pct": round(self.current_dev_pct, 3),
                "baseline_dev_pct": self.DEVIATION_PCT,
                "atr_pct": round(self.current_atr_pct, 4),
                "atr_baseline_pct": self.ATR_BASELINE_PCT,
                "tp_min": self.TP_MIN,
                "tp_max": self.TP_MAX,
                "dev_min": self.DEV_MIN,
                "dev_max": self.DEV_MAX,
                "regime_tp_mult": self.REGIME_TP_MULT.get(self.current_regime, 1.0),
                "regime_dev_mult": self.REGIME_DEV_MULT.get(self.current_regime, 1.0),
            },
            "config": {
                "tp_pct": round(self.current_tp_pct, 3),
                "max_sos": self.MAX_SOS,
                "deviation_pct": round(self.current_dev_pct, 3),
                "so_vol_mult": self.SO_VOL_MULT,
                "base_order_pct": self.BASE_ORDER_PCT,
            },
            # Funding fee information
            "total_funding_fees": round(self.total_funding_fees, 2),
            "last_funding_time": self.last_funding_time,
            "last_funding_rate": self.last_funding_rate,
            "last_funding_cost": round(self.last_funding_cost, 4),
            "current_funding_rate": self.current_funding_rate,
        }
        with open(LIVE_DIR / "status.json", "w") as f:
            json.dump(status, f, indent=2)

        self._write_history(status)

    def _write_history(self, status: dict):
        """Append a data point to history.json for dashboard charts."""
        history_path = LIVE_DIR / "history.json"
        try:
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
            else:
                history = []

            dirs = []
            if self.long_deal:
                dirs.append("L")
            if self.short_deal:
                dirs.append("S")

            history.append({
                "t": status["timestamp"],
                "p": status["price"],
                "eq": status["equity"],
                "pnl": status["pnl"],
                "dd": status["drawdown_pct"],
                "r": status["regime"][:3],
                "dir": "+".join(dirs) if dirs else "",
            })
            if len(history) > 2000:
                history = history[-2000:]
            with open(history_path, "w") as f:
                json.dump(history, f)
        except Exception:
            pass

    def log_trade(self, action: str, price: float, qty: float, notional: float,
                  so_count: int = 0, pnl: float = 0, direction: str = "LONG", deal_id: int = 0,
                  tp_analysis: dict = None):
        path = LIVE_DIR / "trades.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "action", "symbol", "deal_id", "direction", "price", "qty",
                            "notional", "so_count", "pnl", "regime",
                            "duration", "tp_pct", "dev_pct", "atr_pct", "analysis_note"])
            row = [
                datetime.now(timezone.utc).isoformat(), action, self.symbol,
                deal_id, direction,
                f"{price:.3f}", f"{qty:.2f}", f"{notional:.2f}", so_count, f"{pnl:.2f}", self.current_regime,
            ]
            if tp_analysis:
                row.extend([
                    tp_analysis.get("duration", ""),
                    tp_analysis.get("tp_pct", ""),
                    tp_analysis.get("dev_pct", ""),
                    tp_analysis.get("atr_pct", ""),
                    tp_analysis.get("note", ""),
                ])
            else:
                row.extend(["", "", "", "", ""])
            w.writerow(row)

    # ── Equity / balance ───────────────────────────────────────────────

    def _get_equity(self) -> float:
        if self.dry_run:
            return self.capital or 1000
        try:
            return self.api.usdt_equity()
        except Exception:
            return self.capital or 0

    # ── Regime detection ───────────────────────────────────────────────

    def detect_regime(self) -> str:
        try:
            df = self.api.klines(self.symbol, self.timeframe, limit=300)
            if len(df) < 100:
                return "UNKNOWN"
            self.current_price = float(df["close"].iloc[-1])
            self._last_klines_df = df
            regimes = classify_regime_v2(df, self.timeframe)
            # Detect trend direction via SMA50
            sma50 = df["close"].rolling(50).mean().iloc[-1]
            self._trend_bullish = self.current_price >= sma50
            return regimes.iloc[-1]
        except Exception as e:
            print(f"  [WARN] Regime detection error: {e}")
            return self.current_regime

    # ── Adaptive Take Profit ───────────────────────────────────────────

    # Regime TP multipliers: tighter in ranging/accumulation, wider in trending
    REGIME_TP_MULT = {
        "ACCUMULATION": 0.85,
        "CHOPPY": 0.90,
        "RANGING": 0.85,
        "DISTRIBUTION": 0.90,
        "MILD_TREND": 1.05,
        "TRENDING": 1.20,
        "EXTREME": 0.70,  # take quick profits in extreme vol
        "BREAKOUT_WARNING": 0.80,
        "UNKNOWN": 1.0,
    }

    # Regime deviation multipliers: tighter grid in ranging, wider in trending
    REGIME_DEV_MULT = {
        "ACCUMULATION": 0.85,
        "CHOPPY": 0.90,
        "RANGING": 0.80,
        "DISTRIBUTION": 0.90,
        "MILD_TREND": 1.10,
        "TRENDING": 1.30,
        "EXTREME": 1.50,  # wide grid in extreme vol to avoid rapid SO fills
        "BREAKOUT_WARNING": 1.20,
        "UNKNOWN": 1.0,
    }

    def _calculate_atr_pct(self) -> float:
        """Calculate 14-period ATR as a percentage of current price from cached klines."""
        df = self._last_klines_df
        if df is None or len(df) < self.ATR_PERIOD + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        # True Range
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        # Simple moving average of last ATR_PERIOD TRs
        atr = float(np.mean(tr[-self.ATR_PERIOD:]))
        price = float(close[-1])
        return (atr / price * 100) if price > 0 else 0.0

    def _calculate_adaptive_tp(self) -> float:
        """Calculate adaptive TP% based on ATR volatility and regime. Returns TP%."""
        atr_pct = self._calculate_atr_pct()
        self.current_atr_pct = atr_pct

        if atr_pct <= 0:
            return self.TP_PCT  # fallback

        # Scale TP based on ATR relative to baseline
        atr_ratio = atr_pct / self.ATR_BASELINE_PCT
        tp = self.TP_PCT * atr_ratio

        # Apply regime multiplier
        regime_mult = self.REGIME_TP_MULT.get(self.current_regime, 1.0)
        tp *= regime_mult

        # Clamp
        tp = max(self.TP_MIN, min(self.TP_MAX, tp))
        return round(tp, 3)

    def _calculate_adaptive_deviation(self) -> float:
        """Calculate adaptive SO deviation% using same ATR/regime scaling as TP.
        Enforces deviation > TP so SOs always sit beyond the TP target."""
        if self.current_atr_pct <= 0:
            return self.DEVIATION_PCT  # fallback

        atr_ratio = self.current_atr_pct / self.ATR_BASELINE_PCT
        dev = self.DEVIATION_PCT * atr_ratio

        # Apply regime multiplier (deviation has its own multipliers)
        regime_mult = self.REGIME_DEV_MULT.get(self.current_regime, 1.0)
        dev *= regime_mult

        # Clamp
        dev = max(self.DEV_MIN, min(self.DEV_MAX, dev))

        # Critical constraint: deviation must always be wider than TP * 1.5
        tp_floor = self.current_tp_pct * self.DEV_TP_FLOOR_MULT
        dev = max(dev, tp_floor)
        dev = min(dev, self.DEV_MAX)

        return round(dev, 3)

    def _update_adaptive_tp(self):
        """Recalculate adaptive TP + deviation and update orders if changed significantly."""
        new_tp = self._calculate_adaptive_tp()
        old_tp = self.current_tp_pct
        tp_diff = abs(new_tp - old_tp)

        new_dev = self._calculate_adaptive_deviation()
        old_dev = self.current_dev_pct
        dev_diff = abs(new_dev - old_dev)

        tp_changed = tp_diff >= self.TP_UPDATE_THRESHOLD
        dev_changed = dev_diff >= self.TP_UPDATE_THRESHOLD  # same threshold

        if tp_changed or dev_changed:
            parts = []
            if tp_changed:
                parts.append(f"TP: {old_tp:.2f}%→{new_tp:.2f}%")
            if dev_changed:
                parts.append(f"Dev: {old_dev:.2f}%→{new_dev:.2f}%")
            print(f"  📐 Adaptive {' | '.join(parts)} (ATR: {self.current_atr_pct:.3f}%, regime: {self.current_regime})")

        self.current_tp_pct = new_tp
        self.current_dev_pct = new_dev

        if tp_changed or dev_changed:
            for deal in self._active_deals():
                if tp_changed:
                    old_tp_price = deal.tp_price(old_tp)
                    new_tp_price = deal.tp_price(new_tp)
                    price_diff_pct = abs(new_tp_price - old_tp_price) / deal.avg_entry * 100 if deal.avg_entry > 0 else 0
                    if price_diff_pct > 0.05:
                        print(f"    ↻ {deal.direction} TP order: ${old_tp_price:.3f} → ${new_tp_price:.3f}")
                        self._update_tp_order_adaptive(deal)
                if dev_changed:
                    print(f"    ↻ {deal.direction} SO grid: {old_dev:.2f}% → {new_dev:.2f}% step — re-placing SOs")
                    self._update_so_orders_adaptive(deal)

    def _update_so_orders_adaptive(self, deal: LiveDeal):
        """Cancel remaining SO orders and re-place at new deviation levels."""
        if not deal:
            return
        # Cancel existing SOs
        if not self.dry_run:
            for oid in list(deal.safety_order_ids):
                try:
                    self.api.cancel_order(self.symbol, oid)
                except Exception:
                    pass
        deal.safety_order_ids = []

        # Re-place SOs from next unfilled level
        remaining_sos = self.MAX_SOS - deal.safety_orders_filled
        if remaining_sos <= 0:
            return

        if self.dry_run:
            self._place_remaining_sos_dry(deal)
        else:
            self._place_remaining_sos(deal)

    def _place_remaining_sos(self, deal: LiveDeal):
        """Place SO orders from current fill level onward (for adaptive re-placement)."""
        if not deal:
            return
        bo_usd = self.base_order_usd()
        direction = deal.direction
        so_side = self._entry_side(direction)
        pos_side = self._position_side(direction)
        deal.safety_order_ids = []
        available = self._get_available_margin()
        reserve = self._margin_reserve()

        for n in range(deal.safety_orders_filled + 1, self.MAX_SOS + 1):
            price = self.so_price(deal.entry_price, n, direction)
            size_usd = self.so_qty(bo_usd, n)
            qty = round_qty(size_usd / price)
            if qty < MIN_QTY or qty * price < MIN_NOTIONAL:
                break
            margin_needed = self._estimate_order_margin(qty, price)
            if not self.dry_run and (available - margin_needed) < reserve:
                print(f"      ⚠️  {direction} SO#{n}+ skipped — insufficient margin (avail ${available:.2f})")
                break
            try:
                result = self.api.place_order(self.symbol, so_side, "LIMIT", qty, price=price,
                                              position_side=pos_side)
                deal.safety_order_ids.append(int(result["orderId"]))
                available -= margin_needed
                print(f"      {direction} SO#{n}: LIMIT {so_side} {qty} @ ${price:.3f} (id={result['orderId']})")
            except Exception as e:
                print(f"      ❌ {direction} SO#{n} failed: {e}")

    def _place_remaining_sos_dry(self, deal: LiveDeal):
        """Dry-run: log new SO levels after adaptive re-placement."""
        if not deal:
            return
        bo_usd = self.base_order_usd()
        direction = deal.direction
        so_side = self._entry_side(direction)
        for n in range(deal.safety_orders_filled + 1, self.MAX_SOS + 1):
            price = self.so_price(deal.entry_price, n, direction)
            size_usd = self.so_qty(bo_usd, n)
            qty = round_qty(size_usd / price)
            print(f"      {direction} SO#{n}: [DRY] LIMIT {so_side} {qty} @ ${price:.3f} (${size_usd:.0f})")

    def _update_tp_order_adaptive(self, deal: LiveDeal):
        """Cancel and replace TP order with current adaptive TP%."""
        if not deal:
            return
        if deal.tp_order_id and not self.dry_run:
            try:
                self.api.cancel_order(self.symbol, deal.tp_order_id)
            except Exception:
                pass
        deal.tp_order_id = None

        if self.dry_run:
            tp = deal.tp_price(self.current_tp_pct)
            exit_side = self._exit_side(deal.direction)
            print(f"    {deal.direction} TP: [DRY] LIMIT {exit_side} {deal.total_qty} @ ${tp:.3f}")
        else:
            self._place_tp_order(deal)

    # ── Order management ───────────────────────────────────────────────

    def _entry_side(self, direction: str) -> str:
        return "BUY" if direction == "LONG" else "SELL"

    def _exit_side(self, direction: str) -> str:
        return "SELL" if direction == "LONG" else "BUY"

    def _position_side(self, direction: str) -> Optional[str]:
        """Return positionSide param if in hedge mode, else None."""
        if self.hedge_mode:
            return "LONG" if direction == "LONG" else "SHORT"
        return None

    def _open_deal(self, direction: str, alloc_fraction: float = 1.0):
        """Open a new Martingale deal with market entry + SO limits + TP."""
        bo_usd = self.base_order_usd(alloc_fraction)
        if bo_usd < MIN_NOTIONAL:
            print(f"  [SKIP] {direction} base order ${bo_usd:.2f} < min notional ${MIN_NOTIONAL}")
            return

        qty = round_qty(bo_usd / self.current_price)
        if qty < MIN_QTY:
            print(f"  [SKIP] {direction} qty {qty} < min {MIN_QTY}")
            return

        entry_side = self._entry_side(direction)
        pos_side = self._position_side(direction)
        dir_emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
        now_str = datetime.now(timezone.utc).isoformat()
        self.deal_counter += 1

        if self.dry_run:
            print(f"  {dir_emoji} [DRY] Would open {direction} deal#{self.deal_counter}: MARKET {entry_side} {qty} @ ~${self.current_price:.3f}")
            new_deal = LiveDeal(
                deal_id=self.deal_counter, symbol=self.symbol,
                entry_price=self.current_price, entry_qty=qty,
                entry_cost=qty * self.current_price, entry_time=now_str,
                direction=direction,
            )
            self._set_deal(direction, new_deal)
            self._place_so_orders_dry(new_deal)
            self._place_tp_order_dry(new_deal)
            send_telegram(f"{dir_emoji} <b>[DRY] Trade #{self.deal_counter} {direction}</b>\n{self.symbol}\nEntry: ${self.current_price:.3f}\nQty: {qty}\nLeverage: {self.leverage}x\nRegime: {self.current_regime}")
            return

        # Place market entry
        try:
            result = self.api.place_order(self.symbol, entry_side, "MARKET", qty, position_side=pos_side)
            order_id = result.get("orderId")
            fill_price = float(result.get("avgPrice", 0))
            fill_qty = float(result.get("executedQty", 0))

            # Aster returns NEW status initially -- poll for fill
            if fill_qty == 0 or fill_price == 0:
                for _ in range(10):
                    time.sleep(1)
                    status = self.api.query_order(self.symbol, order_id)
                    fill_price = float(status.get("avgPrice", 0))
                    fill_qty = float(status.get("executedQty", 0))
                    if fill_qty > 0 and fill_price > 0:
                        break
                if fill_qty == 0 or fill_price == 0:
                    print(f"  ❌ {direction} market order {order_id} not filled after 10s -- cancelling")
                    try:
                        self.api.cancel_order(self.symbol, order_id)
                    except Exception:
                        pass
                    self.deal_counter -= 1
                    return

            cost = fill_price * fill_qty

            new_deal = LiveDeal(
                deal_id=self.deal_counter, symbol=self.symbol,
                entry_price=fill_price, entry_qty=fill_qty,
                entry_cost=cost, entry_time=now_str,
                direction=direction,
            )
            self._set_deal(direction, new_deal)
            print(f"  {dir_emoji} Trade #{self.deal_counter} {direction}: {entry_side} {fill_qty} @ ${fill_price:.3f}")
            self.log_trade("OPEN", fill_price, fill_qty, cost, direction=direction, deal_id=self.deal_counter)

            # Place safety orders
            self._place_so_orders(new_deal)
            # Place TP
            self._place_tp_order(new_deal)

            send_telegram(
                f"{dir_emoji} <b>Trade #{self.deal_counter} {direction}</b>\n{self.symbol}\n"
                f"Entry: ${fill_price:.3f}\nQty: {fill_qty}\n"
                f"TP: ${new_deal.tp_price(self.current_tp_pct):.3f} ({'+' if direction == 'LONG' else '-'}{self.current_tp_pct:.2f}%)\n"
                f"SOs: {self.MAX_SOS} pending\nLeverage: {self.leverage}x\nRegime: {self.current_regime}"
            )
        except Exception as e:
            print(f"  ❌ Failed to open {direction} deal: {e}")
            self.deal_counter -= 1

    def _place_so_orders(self, deal: LiveDeal):
        """Place all safety order limits for a deal, respecting margin constraints."""
        if not deal:
            return
        bo_usd = self.base_order_usd()  # SOs use full base (not alloc-scaled)
        direction = deal.direction
        so_side = self._entry_side(direction)
        pos_side = self._position_side(direction)
        deal.safety_order_ids = []
        available = self._get_available_margin()
        reserve = self._margin_reserve()

        for n in range(1, self.MAX_SOS + 1):
            price = self.so_price(deal.entry_price, n, direction)
            size_usd = self.so_qty(bo_usd, n)
            qty = round_qty(size_usd / price)
            if qty < MIN_QTY or qty * price < MIN_NOTIONAL:
                break
            # Margin check: skip if insufficient
            margin_needed = self._estimate_order_margin(qty, price)
            if not self.dry_run and (available - margin_needed) < reserve:
                print(f"    ⚠️  {direction} SO#{n}+ skipped — insufficient margin (avail ${available:.2f}, need ${margin_needed:.2f}, reserve ${reserve:.2f})")
                send_telegram(f"⚠️ {direction} SO#{n}-{self.MAX_SOS} skipped: margin insufficient (${available:.0f} avail)")
                break
            try:
                result = self.api.place_order(self.symbol, so_side, "LIMIT", qty, price=price,
                                              position_side=pos_side)
                deal.safety_order_ids.append(int(result["orderId"]))
                available -= margin_needed  # track locally without re-fetching
                print(f"    {direction} SO#{n}: LIMIT {so_side} {qty} @ ${price:.3f} (id={result['orderId']})")
            except Exception as e:
                print(f"    ❌ {direction} SO#{n} failed: {e}")

    def _place_so_orders_dry(self, deal: LiveDeal):
        if not deal:
            return
        bo_usd = self.base_order_usd()
        direction = deal.direction
        so_side = self._entry_side(direction)
        for n in range(1, self.MAX_SOS + 1):
            price = self.so_price(deal.entry_price, n, direction)
            size_usd = self.so_qty(bo_usd, n)
            qty = round_qty(size_usd / price)
            print(f"    {direction} SO#{n}: [DRY] LIMIT {so_side} {qty} @ ${price:.3f} (${size_usd:.0f})")

    def _place_tp_order(self, deal: LiveDeal):
        if not deal:
            return
        tp = deal.tp_price(self.current_tp_pct)
        exit_side = self._exit_side(deal.direction)
        pos_side = self._position_side(deal.direction)

        # Ensure margin for TP — cancel SOs if needed (TP > SO priority)
        self._ensure_tp_margin(deal, deal.total_qty, tp)

        try:
            params = dict(position_side=pos_side)
            # In net mode, don't use reduceOnly — it fails when the other side has a larger position
            if self.hedge_mode:
                pass  # hedge mode handles it via positionSide
            result = self.api.place_order(
                self.symbol, exit_side, "LIMIT", deal.total_qty,
                price=tp, **params
            )
            deal.tp_order_id = int(result["orderId"])
            print(f"    {deal.direction} TP: LIMIT {exit_side} {deal.total_qty} @ ${tp:.3f} (id={result['orderId']})")
        except Exception as e:
            print(f"    ❌ {deal.direction} TP order failed: {e}")
            # Last resort: cancel ALL SOs for this deal and retry once
            if deal.safety_order_ids:
                print(f"    🔓 Emergency: cancelling ALL {len(deal.safety_order_ids)} SOs for TP retry")
                self._cancel_remaining_sos(deal)
                try:
                    result = self.api.place_order(
                        self.symbol, exit_side, "LIMIT", deal.total_qty,
                        price=tp, **params
                    )
                    deal.tp_order_id = int(result["orderId"])
                    print(f"    ✅ {deal.direction} TP retry succeeded (id={result['orderId']})")
                    send_telegram(f"⚠️ {deal.direction} TP placed after emergency SO cancellation")
                except Exception as e2:
                    print(f"    ❌❌ {deal.direction} TP retry ALSO failed: {e2}")
                    send_telegram(f"🚨 CRITICAL: {deal.direction} TP placement failed even after cancelling all SOs: {e2}")

    def _place_tp_order_dry(self, deal: LiveDeal):
        if not deal:
            return
        tp = deal.tp_price(self.current_tp_pct)
        exit_side = self._exit_side(deal.direction)
        print(f"    {deal.direction} TP: [DRY] LIMIT {exit_side} {deal.total_qty} @ ${tp:.3f}")

    def _update_tp_order(self, deal: LiveDeal):
        """Cancel old TP and place new one after SO fill."""
        if not deal:
            return
        if deal.tp_order_id and not self.dry_run:
            try:
                self.api.cancel_order(self.symbol, deal.tp_order_id)
            except Exception:
                pass
        deal.tp_order_id = None

        if self.dry_run:
            self._place_tp_order_dry(deal)
        else:
            self._place_tp_order(deal)

    # ── PnL calculation ────────────────────────────────────────────────

    def _calc_pnl(self, close_price: float, total_qty: float, total_cost: float, direction: str) -> float:
        if direction == "SHORT":
            return total_cost - (total_qty * close_price)
        else:
            return (total_qty * close_price) - total_cost

    # ── TP-hit analysis ───────────────────────────────────────────────

    def _tp_hit_analysis(self, deal: LiveDeal, close_price: float, pnl: float, pnl_pct: float) -> dict:
        """Generate analysis of a completed deal for logging."""
        now = datetime.now(timezone.utc)
        try:
            open_time = datetime.fromisoformat(deal.entry_time.replace('Z', '+00:00'))
            duration = now - open_time
            dur_str = str(duration).split('.')[0]  # H:MM:SS
        except Exception:
            duration = timedelta(0)
            dur_str = "unknown"

        # Determine how adaptive params influenced outcome
        tp_label = "Tight" if self.current_tp_pct < 1.0 else ("Wide" if self.current_tp_pct > 1.8 else "Normal")
        dev_label = "Tight" if self.current_dev_pct < 2.0 else ("Wide" if self.current_dev_pct > 3.0 else "Normal")
        regime_lower = self.current_regime.lower().replace('_', ' ')

        if deal.safety_orders_filled == 0:
            outcome_note = f"{tp_label} TP ({self.current_tp_pct:.1f}%) hit with no SOs — clean entry in {regime_lower} market"
        elif deal.safety_orders_filled <= 2:
            outcome_note = f"{tp_label} TP ({self.current_tp_pct:.1f}%) + {deal.safety_orders_filled} SO(s) with {dev_label} dev ({self.current_dev_pct:.1f}%) in {regime_lower} market"
        else:
            outcome_note = f"Deep DCA ({deal.safety_orders_filled} SOs) with {dev_label} dev ({self.current_dev_pct:.1f}%) recovered via {tp_label} TP ({self.current_tp_pct:.1f}%) in {regime_lower} market"

        analysis = {
            "duration": dur_str,
            "duration_sec": int(duration.total_seconds()),
            "sos_filled": deal.safety_orders_filled,
            "tp_pct": round(self.current_tp_pct, 3),
            "dev_pct": round(self.current_dev_pct, 3),
            "atr_pct": round(self.current_atr_pct, 4),
            "regime": self.current_regime,
            "note": outcome_note,
        }

        # Print formatted analysis block
        print(f"\n  {'═' * 60}")
        print(f"  📊 TP-HIT ANALYSIS — Deal #{deal.deal_id} {deal.direction}")
        print(f"  {'─' * 60}")
        print(f"  Duration:    {dur_str}")
        print(f"  SOs filled:  {deal.safety_orders_filled}/{self.MAX_SOS}")
        print(f"  Adaptive TP: {self.current_tp_pct:.2f}%  (range {self.TP_MIN}-{self.TP_MAX})")
        print(f"  Adaptive Dev: {self.current_dev_pct:.2f}%  (range {self.DEV_MIN}-{self.DEV_MAX})")
        print(f"  ATR%:        {self.current_atr_pct:.3f}%  (baseline {self.ATR_BASELINE_PCT}%)")
        print(f"  Regime:      {self.current_regime}")
        print(f"  Insight:     {outcome_note}")
        print(f"  {'═' * 60}\n")

        return analysis

    # ── Monitoring loop ────────────────────────────────────────────────

    def _check_orders_for_deal(self, deal: LiveDeal) -> bool:
        """Check if any SOs or TP filled for a specific deal. Returns True if deal closed."""
        if not deal or self.dry_run:
            return False

        direction = deal.direction

        # Retry placing TP if missing
        if not deal.tp_order_id and not self.dry_run:
            print(f"  [RETRY] Placing missing {direction} TP order...")
            self._place_tp_order(deal)

        # Check TP first
        if deal.tp_order_id:
            try:
                tp_info = self.api.query_order(self.symbol, deal.tp_order_id)
                if tp_info["status"] == "FILLED":
                    fill_price = float(tp_info.get("avgPrice", tp_info.get("price", 0)))
                    pnl = self._calc_pnl(fill_price, deal.total_qty, deal.total_cost, direction)
                    pnl_pct = pnl / deal.total_cost * 100
                    print(f"  ✅ TP HIT! Trade #{deal.deal_id} {direction} @ ${fill_price:.3f} | PnL: ${pnl:.2f} ({pnl_pct:+.1f}%)")

                    # TP-hit analysis
                    analysis = self._tp_hit_analysis(deal, fill_price, pnl, pnl_pct)

                    self.log_trade("TP_HIT", fill_price, deal.total_qty, deal.total_qty * fill_price,
                                   pnl=pnl, direction=direction, deal_id=deal.deal_id,
                                   tp_analysis=analysis)

                    self._cancel_remaining_sos(deal)

                    send_telegram(
                        f"✅ <b>TP HIT — Trade #{deal.deal_id} {direction}</b>\n"
                        f"{self.symbol} closed @ ${fill_price:.3f}\n"
                        f"PnL: <b>${pnl:.2f} ({pnl_pct:+.1f}%)</b>\n"
                        f"SOs used: {deal.safety_orders_filled}/{self.MAX_SOS}\n"
                        f"Leverage: {self.leverage}x\n"
                        f"Duration: {analysis.get('duration', '?')} | TP%: {analysis.get('tp_pct', '?')}%"
                    )
                    deal.closed = True
                    deal.close_price = fill_price
                    deal.close_time = datetime.now(timezone.utc).isoformat()
                    deal.realized_pnl = pnl
                    self._set_deal(direction, None)
                    return True
            except Exception as e:
                print(f"  [WARN] {direction} TP check error: {e}")

        # Check SOs
        filled_ids = []
        for oid in list(deal.safety_order_ids):
            try:
                info = self.api.query_order(self.symbol, oid)
                if info["status"] == "FILLED":
                    fill_price = float(info.get("avgPrice", info.get("price", 0)))
                    fill_qty = float(info.get("executedQty", 0))
                    cost = fill_price * fill_qty
                    deal.add_fill(fill_price, fill_qty, cost)
                    filled_ids.append(oid)
                    so_num = deal.safety_orders_filled
                    print(f"  📉 {direction} SO#{so_num} filled @ ${fill_price:.3f} | Avg: ${deal.avg_entry:.3f}")
                    self.log_trade("SO_FILL", fill_price, fill_qty, cost, so_count=so_num,
                                   direction=direction, deal_id=deal.deal_id)
                    send_telegram(
                        f"📉 <b>SO #{so_num} Filled ({direction})</b>\n"
                        f"{self.symbol} @ ${fill_price:.3f}\n"
                        f"New avg: ${deal.avg_entry:.3f}\n"
                        f"New TP: ${deal.tp_price(self.current_tp_pct):.3f}"
                    )
            except Exception as e:
                print(f"  [WARN] {direction} SO check error: {e}")

        for oid in filled_ids:
            deal.safety_order_ids.remove(oid)

        if filled_ids:
            self._update_tp_order(deal)

        return False

    def _check_orders(self):
        """Check orders for both active deals."""
        if self.long_deal:
            self._check_orders_for_deal(self.long_deal)
        if self.short_deal:
            self._check_orders_for_deal(self.short_deal)

    def _check_position_sync(self):
        """Reconcile deal state with actual exchange position(s).
        
        In net position mode, the exchange holds a single net position.
        We track virtual long_deal and short_deal separately.
        This function detects drift between exchange reality and our tracking,
        and logs warnings when quantities don't match.
        
        Critical: also detects orphaned position excess that the bot isn't tracking.
        """
        if self.dry_run:
            return
        try:
            positions = self.api.position_risk(self.symbol)
            for p in positions:
                if p["symbol"] != self.symbol:
                    continue
                pos_side = p.get("positionSide", "BOTH")
                pos_amt = float(p.get("positionAmt", 0))

                if self.hedge_mode:
                    # In hedge mode, we get separate LONG and SHORT position entries
                    if pos_side == "LONG" and self.long_deal and pos_amt <= 0:
                        print("  ⚠️  LONG position closed on exchange — clearing long_deal")
                        self._cancel_remaining_sos(self.long_deal)
                        self.long_deal = None
                    elif pos_side == "SHORT" and self.short_deal and pos_amt >= 0:
                        print("  ⚠️  SHORT position closed on exchange — clearing short_deal")
                        self._cancel_remaining_sos(self.short_deal)
                        self.short_deal = None
                else:
                    # Net position mode: single entry with positionSide=BOTH
                    if pos_side != "BOTH":
                        continue

                    # Calculate expected net from our tracked deals
                    tracked_long_qty = self.long_deal.total_qty if self.long_deal else 0
                    tracked_short_qty = self.short_deal.total_qty if self.short_deal else 0
                    expected_net = round(tracked_long_qty - tracked_short_qty, 4)
                    actual_net = round(pos_amt, 4)
                    drift = round(actual_net - expected_net, 4)

                    # --- Case 1: No deals tracked ---
                    if not self.long_deal and not self.short_deal:
                        if abs(actual_net) > 0.01:
                            print(f"  🚨 POSITION DRIFT: No deals tracked but exchange has {actual_net:+.4f} net position!")
                            send_telegram(
                                f"🚨 POSITION DRIFT DETECTED\n"
                                f"Exchange: {actual_net:+.4f} net\n"
                                f"Tracked: no deals\n"
                                f"Action needed: orphaned position"
                            )
                        return

                    # --- Case 2: Deals tracked, check for zero position ---
                    if pos_amt == 0 and (self.long_deal or self.short_deal):
                        if self.long_deal and self.short_deal:
                            # Both tracked, net zero could be normal if quantities match
                            if abs(tracked_long_qty - tracked_short_qty) > 0.01:
                                print(f"  ⚠️  Net position is zero but tracked deals are unequal (L:{tracked_long_qty} S:{tracked_short_qty})")
                        elif self.long_deal and not self.short_deal:
                            print("  ⚠️  Net position is zero but LONG deal tracked (no short) — clearing long_deal")
                            self._cancel_remaining_sos(self.long_deal)
                            self.long_deal = None
                        elif self.short_deal and not self.long_deal:
                            print("  ⚠️  Net position is zero but SHORT deal tracked (no long) — clearing short_deal")
                            self._cancel_remaining_sos(self.short_deal)
                            self.short_deal = None
                        return

                    # --- Case 3: Quantity drift detection ---
                    # Allow small tolerance for rounding (0.5% of tracked qty or 0.02, whichever is larger)
                    tolerance = max(0.02, (tracked_long_qty + tracked_short_qty) * 0.005)
                    if abs(drift) > tolerance:
                        drift_value = abs(drift) * self.current_price
                        print(f"  🚨 POSITION DRIFT: exchange={actual_net:+.4f} expected={expected_net:+.4f} drift={drift:+.4f} (~${drift_value:.2f})")
                        print(f"     Tracked: L={tracked_long_qty:.4f} S={tracked_short_qty:.4f}")
                        send_telegram(
                            f"🚨 POSITION DRIFT DETECTED\n"
                            f"Exchange net: {actual_net:+.4f}\n"
                            f"Expected net: {expected_net:+.4f}\n"
                            f"Drift: {drift:+.4f} (~${drift_value:.2f})\n"
                            f"Tracked: L={tracked_long_qty:.4f} S={tracked_short_qty:.4f}"
                        )

                    # --- Case 4: Sign mismatch (deal tracking thinks opposite) ---
                    if pos_amt > 0 and self.long_deal and self.short_deal:
                        if abs(pos_amt - tracked_long_qty) < tolerance:
                            # Net matches long only — short was likely closed on exchange
                            print("  ⚠️  Net position matches LONG only — clearing short_deal")
                            self._cancel_remaining_sos(self.short_deal)
                            self.short_deal = None
                    elif pos_amt < 0 and self.long_deal and self.short_deal:
                        if abs(abs(pos_amt) - tracked_short_qty) < tolerance:
                            print("  ⚠️  Net position matches SHORT only — clearing long_deal")
                            self._cancel_remaining_sos(self.long_deal)
                            self.long_deal = None
        except Exception as e:
            print(f"  [WARN] Position sync error: {e}")

    def _cancel_remaining_sos(self, deal: LiveDeal):
        if not deal or self.dry_run:
            return
        for oid in deal.safety_order_ids:
            try:
                self.api.cancel_order(self.symbol, oid)
            except Exception:
                pass
        deal.safety_order_ids = []

    def _cancel_all(self):
        if self.dry_run:
            return
        try:
            self.api.cancel_all_orders(self.symbol)
            print("  ⛔ All orders cancelled")
        except Exception as e:
            print(f"  [WARN] Cancel all failed: {e}")

    def _close_position(self):
        """Market close any open position."""
        if self.dry_run:
            return
        try:
            positions = self.api.position_risk(self.symbol)
            for p in positions:
                if p["symbol"] == self.symbol:
                    amt = float(p.get("positionAmt", 0))
                    pos_side = p.get("positionSide", "BOTH")
                    if amt > 0:
                        self.api.place_order(self.symbol, "SELL", "MARKET", amt, reduce_only=True,
                                             position_side=pos_side if self.hedge_mode else None)
                        print(f"  ⛔ Closed {pos_side} position: SELL {amt}")
                    elif amt < 0:
                        self.api.place_order(self.symbol, "BUY", "MARKET", abs(amt), reduce_only=True,
                                             position_side=pos_side if self.hedge_mode else None)
                        print(f"  ⛔ Closed {pos_side} position: BUY {abs(amt)}")
        except Exception as e:
            print(f"  [WARN] Close position failed: {e}")

    # ── Kill switches ──────────────────────────────────────────────────

    def _check_kill_switches(self) -> bool:
        equity = self._get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity * 100
            if dd >= self.max_drawdown_pct:
                self._trigger_halt(f"Max drawdown {dd:.1f}% >= {self.max_drawdown_pct}%")
                return True

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_equity = equity
        elif self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity * 100
            if daily_loss >= 15:
                self.paused_until = time.time() + 86400
                self._trigger_halt(f"Daily loss {daily_loss:.1f}% >= 15% — paused 24h")
                return True

        if self.paused_until and time.time() < self.paused_until:
            return True
        elif self.paused_until and time.time() >= self.paused_until:
            self.paused_until = None
            print("  ▶️  Pause expired, resuming")

        return False

    def _trigger_halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        print(f"\n  🚨 HALT: {reason}")
        self._cancel_all()
        self._close_position()
        send_telegram(f"🚨 <b>KILL SWITCH</b>\n{reason}\nAll orders cancelled, position closed.")

    # ── Daily summary ──────────────────────────────────────────────────

    def _check_daily_summary(self):
        now = datetime.now(timezone.utc)
        today_key = now.strftime("%Y-%m-%d")
        if now.hour == 0 and now.minute < 2 and self.last_daily_summary != today_key:
            self.last_daily_summary = today_key
            equity = self._get_equity()
            pnl = equity - self.start_equity if self.start_equity else 0

            long_str = f"LONG #{self.long_deal.deal_id} (avg ${self.long_deal.avg_entry:.3f}, {self.long_deal.safety_orders_filled} SOs)" if self.long_deal else "none"
            short_str = f"SHORT #{self.short_deal.deal_id} (avg ${self.short_deal.avg_entry:.3f}, {self.short_deal.safety_orders_filled} SOs)" if self.short_deal else "none"
            dd_str = f"Drawdown: {(self.peak_equity - equity) / self.peak_equity * 100:.1f}%" if self.peak_equity > 0 else ""

            send_telegram(
                f"📊 <b>Daily Summary</b> ({today_key})\n"
                f"Equity: ${equity:.2f}\n"
                f"Total PnL: ${pnl:+.2f}\n"
                f"Deals completed: {self.deal_counter}\n"
                f"Long: {long_str}\n"
                f"Short: {short_str}\n"
                f"Regime: {self.current_regime}\n"
                f"{dd_str}"
            )

    # ── Reconcile on startup ───────────────────────────────────────────

    def reconcile(self):
        """On startup, check exchange state and reconcile with saved state."""
        if self.dry_run:
            return
        print("  🔄 Reconciling with exchange...")
        try:
            positions = self.api.position_risk(self.symbol)
            found_positions = {}  # pos_side -> (amt, entry)

            for p in positions:
                if p["symbol"] == self.symbol:
                    pos_amt = float(p.get("positionAmt", 0))
                    pos_side = p.get("positionSide", "BOTH")
                    entry = float(p.get("entryPrice", 0))
                    if pos_amt != 0:
                        found_positions[pos_side] = (pos_amt, entry)
                        print(f"    Exchange position ({pos_side}): {pos_amt} @ ${entry:.3f}")

            if self.hedge_mode:
                # In hedge mode: separate LONG and SHORT positions
                if "LONG" in found_positions and not self.long_deal:
                    amt, entry = found_positions["LONG"]
                    if amt > 0:
                        print(f"    ⚠️  LONG position exists but no saved deal — creating stub")
                        self.deal_counter += 1
                        self.long_deal = LiveDeal(
                            deal_id=self.deal_counter, symbol=self.symbol,
                            entry_price=entry, entry_qty=abs(amt),
                            entry_cost=abs(amt) * entry,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            direction="LONG",
                        )
                if "SHORT" in found_positions and not self.short_deal:
                    amt, entry = found_positions["SHORT"]
                    if amt < 0:
                        print(f"    ⚠️  SHORT position exists but no saved deal — creating stub")
                        self.deal_counter += 1
                        self.short_deal = LiveDeal(
                            deal_id=self.deal_counter, symbol=self.symbol,
                            entry_price=entry, entry_qty=abs(amt),
                            entry_cost=abs(amt) * entry,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            direction="SHORT",
                        )
                # Clear deals with no matching position
                if "LONG" not in found_positions and self.long_deal:
                    print("    ⚠️  No LONG position on exchange — clearing long_deal")
                    self.long_deal = None
                if "SHORT" not in found_positions and self.short_deal:
                    print("    ⚠️  No SHORT position on exchange — clearing short_deal")
                    self.short_deal = None
            else:
                # Net position mode (BOTH)
                if "BOTH" in found_positions:
                    amt, entry = found_positions["BOTH"]
                    if amt > 0 and not self.long_deal:
                        print(f"    ⚠️  Net LONG position exists but no long_deal — creating stub")
                        self.deal_counter += 1
                        self.long_deal = LiveDeal(
                            deal_id=self.deal_counter, symbol=self.symbol,
                            entry_price=entry, entry_qty=abs(amt),
                            entry_cost=abs(amt) * entry,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            direction="LONG",
                        )
                    elif amt < 0 and not self.short_deal:
                        print(f"    ⚠️  Net SHORT position exists but no short_deal — creating stub")
                        self.deal_counter += 1
                        self.short_deal = LiveDeal(
                            deal_id=self.deal_counter, symbol=self.symbol,
                            entry_price=entry, entry_qty=abs(amt),
                            entry_cost=abs(amt) * entry,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            direction="SHORT",
                        )
                elif not found_positions:
                    if self.long_deal:
                        print("    ⚠️  No position on exchange — clearing long_deal")
                        self.long_deal = None
                    if self.short_deal:
                        print("    ⚠️  No position on exchange — clearing short_deal")
                        self.short_deal = None

            open_ords = self.api.open_orders(self.symbol)
            print(f"    Open orders on exchange: {len(open_ords)}")
        except Exception as e:
            print(f"    [WARN] Reconcile failed: {e}")

    # ── Hedge mode setup ───────────────────────────────────────────────

    def _setup_hedge_mode(self):
        """Try to enable hedge mode for independent long/short positions."""
        if self.dry_run:
            self.hedge_mode = True  # assume hedge in dry run
            print("  ⚙️  Hedge mode assumed (dry run)")
            return

        # First check current mode
        is_hedge = self.api.get_position_mode()
        if is_hedge:
            self.hedge_mode = True
            print("  ⚙️  Hedge mode already enabled ✓")
            return

        # Try to enable it
        success = self.api.set_hedge_mode(True)
        if success:
            self.hedge_mode = True
            print("  ⚙️  Hedge mode enabled ✓")
        else:
            self.hedge_mode = False
            print("  ⚠️  Hedge mode NOT available — running in net position mode")
            print("       Dual-tracking will use virtual position tracking.")
            print("       ⚠️  WARNING: In net mode, opposing positions partially offset on exchange!")
            send_telegram(
                "⚠️ <b>Hedge Mode Unavailable</b>\n"
                "Running dual-tracking in net position mode.\n"
                "Opposing positions will partially offset on exchange."
            )

    # ── Dry-run simulation ─────────────────────────────────────────────

    def _dry_run_check_deal(self, deal: LiveDeal) -> bool:
        """Simulate order fills for a deal. Returns True if deal closed."""
        if not deal:
            return False

        direction = deal.direction

        # Check SO fills
        filled_any = False
        for n in range(deal.safety_orders_filled + 1, self.MAX_SOS + 1):
            so_p = self.so_price(deal.entry_price, n, direction)
            so_triggered = (direction == "LONG" and self.current_price <= so_p) or \
                           (direction == "SHORT" and self.current_price >= so_p)
            if so_triggered:
                bo_usd = self.base_order_usd()
                size_usd = self.so_qty(bo_usd, n)
                qty = round_qty(size_usd / so_p)
                deal.add_fill(so_p, qty, size_usd)
                print(f"  📉 [DRY] {direction} SO#{deal.safety_orders_filled} filled @ ${so_p:.3f}")
                filled_any = True
            else:
                break

        if filled_any:
            tp = deal.tp_price(self.current_tp_pct)
            print(f"    [DRY] {direction} New avg: ${deal.avg_entry:.3f} | New TP: ${tp:.3f}")

        # Check TP
        tp = deal.tp_price(self.current_tp_pct)
        tp_hit = (direction == "LONG" and self.current_price >= tp) or \
                 (direction == "SHORT" and self.current_price <= tp)
        if tp_hit:
            pnl = self._calc_pnl(tp, deal.total_qty, deal.total_cost, direction)
            print(f"  ✅ [DRY] TP HIT {direction} @ ${tp:.3f} | PnL: ${pnl:.2f}")
            self.log_trade("TP_HIT", tp, deal.total_qty, deal.total_qty * tp,
                           pnl=pnl, direction=direction, deal_id=deal.deal_id)
            send_telegram(f"✅ <b>[DRY] TP HIT {direction}</b>\nTrade #{deal.deal_id} | PnL: ${pnl:.2f}")
            self._set_deal(direction, None)
            return True

        return False

    # ── Funding fee tracking ───────────────────────────────────────────

    def _get_funding_times_utc(self) -> List[datetime]:
        """Get funding settlement times (every 4h at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)."""
        now = datetime.now(timezone.utc)
        funding_hours = [0, 4, 8, 12, 16, 20]
        times = []
        for hour in funding_hours:
            dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            times.append(dt)
            # Also include previous day funding times
            times.append(dt - timedelta(days=1))
        return sorted(times)

    def _get_last_funding_settlement_time(self) -> datetime:
        """Find the most recent funding settlement time (past occurrence)."""
        now = datetime.now(timezone.utc)
        funding_hours = [0, 4, 8, 12, 16, 20]
        
        # Find the most recent funding hour today
        current_hour = now.hour
        last_funding_hour = None
        for hour in reversed(funding_hours):
            if current_hour >= hour:
                last_funding_hour = hour
                break
        
        if last_funding_hour is not None:
            # Today's funding time
            return now.replace(hour=last_funding_hour, minute=0, second=0, microsecond=0)
        else:
            # Use yesterday's last funding (20:00)
            yesterday = now - timedelta(days=1)
            return yesterday.replace(hour=20, minute=0, second=0, microsecond=0)

    def _track_funding_fees(self):
        """Check for new funding settlements and calculate costs."""
        if self.dry_run:
            return  # Skip funding tracking in dry run
        
        try:
            # Get current funding rate info
            premium_info = self.api.premium_index(self.symbol)
            self.current_funding_rate = float(premium_info.get("lastFundingRate", 0))
            
            # Find the most recent funding settlement
            last_settlement = self._get_last_funding_settlement_time()
            last_settlement_str = last_settlement.isoformat()
            
            # Check if we've already processed this funding settlement
            if self.last_funding_time and self.last_funding_time >= last_settlement_str:
                return  # Already processed this funding period
            
            # Get funding rate history to find the rate for this settlement
            try:
                funding_history = self.api.funding_rate_history(self.symbol, limit=10)
                if not funding_history:
                    return
                
                # Find the funding rate for the settlement time
                settlement_rate = None
                for entry in funding_history:
                    funding_time = datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc)
                    if abs((funding_time - last_settlement).total_seconds()) < 300:  # Within 5 minutes
                        settlement_rate = float(entry["fundingRate"])
                        break
                
                if settlement_rate is None:
                    # Use the most recent rate as fallback
                    settlement_rate = float(funding_history[0]["fundingRate"])
                
                # Calculate net position at settlement time
                net_position_qty = 0.0
                active_deals = self._active_deals()
                
                for deal in active_deals:
                    # Only count deals that were opened before the settlement
                    deal_open_time = datetime.fromisoformat(deal.entry_time.replace('Z', '+00:00'))
                    if deal_open_time <= last_settlement:
                        if deal.direction == "LONG":
                            net_position_qty += deal.total_qty
                        else:  # SHORT
                            net_position_qty -= deal.total_qty
                
                if abs(net_position_qty) < 0.01:
                    # No significant position, no funding cost
                    self.last_funding_time = last_settlement_str
                    return
                
                # Calculate funding cost: net_position_qty × mark_price × funding_rate
                mark_price = self.current_price  # Use current price as approximation
                funding_cost = net_position_qty * mark_price * settlement_rate
                
                # Update tracking
                self.total_funding_fees += funding_cost
                self.last_funding_time = last_settlement_str
                self.last_funding_rate = settlement_rate
                self.last_funding_cost = funding_cost
                
                # Log the funding event
                direction_str = "LONG" if net_position_qty > 0 else "SHORT"
                rate_pct = settlement_rate * 100
                print(f"  💰 Funding: rate={rate_pct:+.4f}%, net={net_position_qty:+.2f} {direction_str}, cost=${funding_cost:+.4f}")
                
                # Detailed log for tracking
                if abs(funding_cost) > 0.001:  # Only log significant costs
                    send_telegram(
                        f"💰 <b>Funding Settlement</b>\n"
                        f"Rate: {rate_pct:+.4f}% {'(longs pay)' if settlement_rate > 0 else '(shorts pay)'}\n"
                        f"Net position: {net_position_qty:+.2f} {direction_str}\n"
                        f"Cost: ${funding_cost:+.4f}\n"
                        f"Total funding: ${self.total_funding_fees:+.2f}"
                    )
                
            except Exception as e:
                print(f"  [WARN] Funding rate fetch failed: {e}")
                
        except Exception as e:
            print(f"  [WARN] Funding tracking error: {e}")

    # ── Main loop ──────────────────────────────────────────────────────

    def start(self):
        self._running = True
        if not self.start_time:
            self.start_time = datetime.now(timezone.utc).isoformat()

        # Init capital
        if self.capital is None:
            if self.dry_run:
                self.capital = 1000
            else:
                self.capital = self.api.usdt_balance()
        if not self.start_equity:
            self.start_equity = self._get_equity()
        if not self.peak_equity:
            self.peak_equity = self.start_equity

        # Set leverage
        if not self.dry_run:
            try:
                self.api.set_leverage(self.symbol, self.leverage)
                print(f"  ⚙️  Leverage set to {self.leverage}x")
            except Exception as e:
                print(f"  [WARN] Set leverage failed: {e}")

        # Setup hedge mode for dual-tracking
        self._setup_hedge_mode()

        self.reconcile()

        mode = "DRY RUN" if self.dry_run else "LIVE"
        print(f"\n  🚀 Aster Trader [{mode}] started! (dual-tracking)")
        print(f"  Symbol: {self.symbol} | TF: {self.timeframe} | Capital: ${self.capital:.2f}")
        print(f"  TP: {self.TP_PCT}% | SOs: {self.MAX_SOS} | Dev: {self.DEVIATION_PCT}% | Leverage: {self.leverage}x")
        print(f"  Hedge mode: {'✓' if self.hedge_mode else '✗ (net position)'}")

        send_telegram(
            f"🚀 <b>Bot Started [{mode}]</b>\n"
            f"{self.symbol} | {self.timeframe} | Dual-tracking\n"
            f"Capital: ${self.capital:.2f} | Leverage: {self.leverage}x\n"
            f"TP: {self.TP_PCT}% | Dev: {self.DEVIATION_PCT}% | SOs: {self.MAX_SOS}\n"
            f"Hedge mode: {'✓' if self.hedge_mode else '✗'}"
        )

        while self._running:
            try:
                self.cycle_count += 1

                # Connectivity check
                if not self.dry_run and not self.api.ping():
                    self.consecutive_errors += 1
                    print(f"  ⚠️  API ping failed ({self.consecutive_errors}/3)")
                    if self.consecutive_errors >= 3:
                        self._trigger_halt("3+ consecutive API errors (ping unreachable)")
                        break
                    time.sleep(10)
                    continue
                self.consecutive_errors = 0

                # Kill switches
                if self.halted:
                    print(f"  🛑 Halted: {self.halt_reason}")
                    break
                if self._check_kill_switches():
                    if self.halted:
                        break
                    time.sleep(30)
                    continue

                # Update capital to current equity (compounds with profits + deposits)
                _eq = self._get_equity()
                if _eq > 0:
                    self.capital = _eq

                # Detect regime
                self.current_regime = self.detect_regime()
                long_alloc, short_alloc = REGIME_ALLOC.get(self.current_regime, (0.5, 0.5))

                # Directional awareness: flip allocation in bearish trends
                if self.current_regime in DIRECTIONAL_REGIMES and not self._trend_bullish:
                    long_alloc, short_alloc = short_alloc, long_alloc

                # Adaptive TP + deviation adjustment
                self._update_adaptive_tp()

                # Track funding fees (once per cycle)
                self._track_funding_fees()

                trend_dir = "▲" if self._trend_bullish else "▼"
                long_str = f"L#{self.long_deal.deal_id}({self.long_deal.safety_orders_filled}SO)" if self.long_deal else "—"
                short_str = f"S#{self.short_deal.deal_id}({self.short_deal.safety_orders_filled}SO)" if self.short_deal else "—"
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] ${self.current_price:.3f} | {self.current_regime}{trend_dir} | TP:{self.current_tp_pct:.2f}% Dev:{self.current_dev_pct:.2f}% | {long_str} {short_str} | alloc L:{long_alloc:.0%}/S:{short_alloc:.0%}")

                # Check existing orders for both deals
                if self.dry_run:
                    if self.long_deal:
                        self._dry_run_check_deal(self.long_deal)
                    if self.short_deal:
                        self._dry_run_check_deal(self.short_deal)
                else:
                    self._check_orders()
                    self._check_position_sync()

                # Open new deals based on regime allocation
                if not self.long_deal and long_alloc > 0:
                    print(f"    Opening LONG deal (alloc: {long_alloc:.0%})")
                    self._open_deal("LONG", long_alloc)
                elif not self.long_deal and long_alloc == 0:
                    print(f"    LONG blocked by regime ({self.current_regime})")

                if not self.short_deal and short_alloc > 0:
                    print(f"    Opening SHORT deal (alloc: {short_alloc:.0%})")
                    self._open_deal("SHORT", short_alloc)
                elif not self.short_deal and short_alloc == 0:
                    print(f"    SHORT blocked by regime ({self.current_regime})")

                # Daily summary
                self._check_daily_summary()

                # Persist
                self.save_state()
                self.write_status()

                time.sleep(30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.consecutive_errors += 1
                print(f"\n  ❌ Error (#{self.consecutive_errors}): {e}")
                traceback.print_exc()
                if self.consecutive_errors >= 3:
                    self._trigger_halt(f"3+ errors: {e}")
                    break
                time.sleep(15)

        self._shutdown()

    def _shutdown(self):
        self._running = False
        print("\n  🛑 Shutting down...")
        if not self.dry_run:
            if self.long_deal or self.short_deal:
                print("  Cancelling open orders...")
                self._cancel_all()
                if self.long_deal:
                    print(f"  ⚠️  LONG position left open (deal #{self.long_deal.deal_id}). Close manually if desired.")
                if self.short_deal:
                    print(f"  ⚠️  SHORT position left open (deal #{self.short_deal.deal_id}). Close manually if desired.")
        self.save_state()
        self.write_status()
        send_telegram("🛑 <b>Bot Stopped</b>")

    def stop(self):
        self._running = False
