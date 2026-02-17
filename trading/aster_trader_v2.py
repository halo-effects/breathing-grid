"""Paper Trading Bot for AIT with Risk Profiles
Simulates trading using real market data but no actual orders placed.
"""
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

# Import regime detection from the live bot
try:
    from .regime_detector import classify_regime_v2, is_martingale_friendly_v2
except ImportError:
    try:
        from regime_detector import classify_regime_v2, is_martingale_friendly_v2
    except ImportError:
        # Fallback for standalone testing
        def classify_regime_v2(df, timeframe):
            return pd.Series(["UNKNOWN"] * len(df))
        def is_martingale_friendly_v2(df):
            return True

PAPER_DIR = Path(__file__).parent / "paper"
PAPER_DIR.mkdir(exist_ok=True)

# Telegram config (same as live)
TG_TOKEN = "8528958079:AAF90HSJ5Ck1urUydzS5CUvyf2EEeB7LUwc"
TG_CHAT_ID = "5221941584"
TG_ENABLED = True

# HYPEUSDT market rules
TICK_SIZE = 0.001
STEP_SIZE = 0.01
MIN_QTY = 0.01
MIN_NOTIONAL = 5.0
PRICE_PRECISION = 3
QTY_PRECISION = 2

# Regime-based capital allocation
REGIME_ALLOC = {
    "ACCUMULATION": (0.70, 0.30),
    "CHOPPY": (0.50, 0.50),
    "RANGING": (0.50, 0.50),
    "DISTRIBUTION": (0.30, 0.70),
    "MILD_TREND": (0.60, 0.40),
    "TRENDING": (0.75, 0.25),
    "EXTREME": (0.0, 0.0),
    "BREAKOUT_WARNING": (0.0, 0.0),
    "UNKNOWN": (0.50, 0.50),
}
DIRECTIONAL_REGIMES = {"TRENDING", "MILD_TREND", "DISTRIBUTION"}

# Risk Profile Definitions
PROFILES = {
    "low": {
        "name": "Low Risk",
        "leverage": 1,
        "max_safety_orders": 8,
        "so_volume_mult": 2.0,
        "base_order_pct": 4.0,
        "capital_reserve": 10.0,
        "tp_range": (0.6, 2.5),
        "deviation_range": (1.2, 4.0),
        "extreme_allocation": (0, 0),  # Halt in extreme
        "max_directional_bias": (75, 25),
    },
    "medium": {
        "name": "Medium Risk", 
        "leverage": 2,
        "max_safety_orders": 12,
        "so_volume_mult": 2.5,
        "base_order_pct": 6.0,
        "capital_reserve": 5.0,
        "tp_range": (0.4, 2.0),
        "deviation_range": (0.8, 3.0),
        "extreme_allocation": (20, 20),  # Reduce in extreme
        "max_directional_bias": (85, 15),
    },
    "high": {
        "name": "High Risk",
        "leverage": 5,
        "max_safety_orders": 16,
        "so_volume_mult": 3.0,
        "base_order_pct": 8.0,
        "capital_reserve": 2.0,
        "tp_range": (0.2, 1.5),
        "deviation_range": (0.5, 2.0),
        "extreme_allocation": (40, 40),  # Continue in extreme
        "max_directional_bias": (95, 5),
    }
}


def round_price(p: float) -> float:
    return round(round(p / TICK_SIZE) * TICK_SIZE, PRICE_PRECISION)


def round_qty(q: float) -> float:
    return round(round(q / STEP_SIZE) * STEP_SIZE, QTY_PRECISION)


def send_telegram(msg: str):
    if not TG_ENABLED:
        return
    try:
        # Prefix with [PAPER] to distinguish from live
        paper_msg = f"[PAPER] {msg}"
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": paper_msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


class AsterAPI:
    """Same as live bot but only used for reading data."""
    
    def __init__(self, base_url: str = "https://fapi.asterdex.com"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("ASTER_API_KEY", "")
        self.api_secret = os.environ.get("ASTER_API_SECRET", "")
        # Fallback: read from Windows registry
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

    def ping(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/fapi/v1/ping", timeout=5)
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"    [PING] {e}")
            return False

    def klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        data = self.session.get(
            f"{self.base_url}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15
        ).json()
        
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"
        ])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


@dataclass
class VirtualDeal:
    """Virtual deal for paper trading."""
    deal_id: int
    symbol: str
    entry_price: float
    entry_qty: float
    entry_cost: float
    entry_time: str
    direction: str = "LONG"
    safety_orders_filled: int = 0
    virtual_so_levels: List[Dict] = field(default_factory=list)  # [{level: n, price: float, qty: float, size_usd: float}]
    tp_price: float = 0.0
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
        """Simulate SO fill."""
        self.total_qty += qty
        self.total_cost += cost
        self.avg_entry = self.total_cost / self.total_qty if self.total_qty > 0 else 0
        self.safety_orders_filled += 1

    def calc_tp_price(self, tp_pct: float) -> float:
        """Calculate TP price for current avg entry."""
        if self.direction == "SHORT":
            return round_price(self.avg_entry * (1 - tp_pct / 100))
        return round_price(self.avg_entry * (1 + tp_pct / 100))

    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "entry_qty": self.entry_qty,
            "entry_cost": self.entry_cost,
            "entry_time": self.entry_time,
            "direction": self.direction,
            "safety_orders_filled": self.safety_orders_filled,
            "virtual_so_levels": self.virtual_so_levels,
            "tp_price": self.tp_price,
            "total_qty": self.total_qty,
            "total_cost": self.total_cost,
            "avg_entry": self.avg_entry,
            "closed": self.closed,
            "close_price": self.close_price,
            "close_time": self.close_time,
            "realized_pnl": self.realized_pnl,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VirtualDeal":
        return cls(**d)


class VirtualAccount:
    """Simulates exchange account for paper trading."""
    
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.start_balance = initial_balance
        self.equity = initial_balance
        self.virtual_positions = {}  # symbol -> {direction: {qty, cost}}
        
    def get_equity(self) -> float:
        """Return current equity (balance + unrealized PnL)."""
        return self.equity
        
    def get_balance(self) -> float:
        """Return available balance."""
        return self.balance
        
    def apply_fill(self, price: float, qty: float, direction: str, symbol: str = "HYPEUSDT"):
        """Apply a virtual fill to the account."""
        cost = price * qty
        
        if symbol not in self.virtual_positions:
            self.virtual_positions[symbol] = {"LONG": {"qty": 0, "cost": 0}, "SHORT": {"qty": 0, "cost": 0}}
        
        pos = self.virtual_positions[symbol][direction]
        pos["qty"] += qty
        pos["cost"] += cost
        
    def close_position(self, close_price: float, close_qty: float, direction: str, symbol: str = "HYPEUSDT") -> float:
        """Close position and return realized PnL."""
        if symbol not in self.virtual_positions:
            return 0.0
            
        pos = self.virtual_positions[symbol][direction]
        if pos["qty"] <= 0:
            return 0.0
            
        # Calculate PnL
        avg_cost = pos["cost"] / pos["qty"]
        if direction == "LONG":
            pnl = (close_price - avg_cost) * close_qty
        else:  # SHORT
            pnl = (avg_cost - close_price) * close_qty
            
        # Subtract fill from position
        ratio = close_qty / pos["qty"]
        pos["qty"] -= close_qty
        pos["cost"] -= pos["cost"] * ratio
        
        # Add PnL to balance
        self.balance += pnl
        self.equity = self.balance
        
        return pnl


class ProfileManager:
    """Manages risk profile allocation and parameters."""
    
    def __init__(self, allocation_path: Path):
        self.allocation_path = allocation_path
        self.allocation = self._load_allocation()
        
    def _load_allocation(self) -> dict:
        """Load allocation from JSON file."""
        try:
            with open(self.allocation_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load allocation: {e}")
            return {"low": 0, "medium": 100, "high": 0, "total_capital": 10000}
    
    def get_active_profile(self) -> str:
        """Get the currently active profile (single profile for Phase 1)."""
        for profile, pct in self.allocation.items():
            if profile in PROFILES and pct > 0:
                return profile
        return "medium"  # Default fallback
        
    def get_profile_capital(self, profile: str) -> float:
        """Get allocated capital for a profile."""
        pct = self.allocation.get(profile, 0)
        total = self.allocation.get("total_capital", 10000)
        return total * (pct / 100)
        
    def get_profile_params(self, profile: str) -> dict:
        """Get parameters for a profile."""
        return PROFILES.get(profile, PROFILES["medium"])
        
    def refresh_allocation(self):
        """Re-read allocation from disk."""
        old_allocation = self.allocation.copy()
        self.allocation = self._load_allocation()
        
        # Check if profile changed
        old_active = None
        new_active = None
        
        for profile, pct in old_allocation.items():
            if profile in PROFILES and pct > 0:
                old_active = profile
                break
                
        for profile, pct in self.allocation.items():
            if profile in PROFILES and pct > 0:
                new_active = profile
                break
                
        if old_active != new_active and new_active:
            print(f"Profile changed: {old_active or 'none'} -> {new_active}")
            send_telegram(f"Risk profile changed to {PROFILES[new_active]['name']}")
            return True
        return False


class PaperDealManager:
    """Manages virtual deals like the live DealManager."""
    
    def __init__(self, profile_manager: ProfileManager, virtual_account: VirtualAccount):
        self.profile_manager = profile_manager
        self.virtual_account = virtual_account
        self.long_deal: Optional[VirtualDeal] = None
        self.short_deal: Optional[VirtualDeal] = None
        self.deal_counter = 0
        self.current_price = 0.0
        self.current_tp_pct = 1.5
        self.current_dev_pct = 2.5
        
    def _get_deal(self, direction: str) -> Optional[VirtualDeal]:
        return self.long_deal if direction == "LONG" else self.short_deal
        
    def _set_deal(self, direction: str, deal: Optional[VirtualDeal]):
        if direction == "LONG":
            self.long_deal = deal
        else:
            self.short_deal = deal
            
    def _calc_pnl(self, close_price: float, total_qty: float, total_cost: float, direction: str) -> float:
        if direction == "SHORT":
            return total_cost - (total_qty * close_price)
        else:
            return (total_qty * close_price) - total_cost
            
    def create_so_levels(self, deal: VirtualDeal, params: dict):
        """Create virtual SO levels for a deal."""
        deal.virtual_so_levels = []
        
        base_order_usd = params["base_order_pct"] / 100 * self.profile_manager.get_profile_capital(
            self.profile_manager.get_active_profile()
        )
        
        for n in range(1, params["max_safety_orders"] + 1):
            deviation_pct = params["deviation_range"][0] + (
                params["deviation_range"][1] - params["deviation_range"][0]
            ) * min(n / params["max_safety_orders"], 1.0)
            
            if deal.direction == "LONG":
                so_price = round_price(deal.entry_price * (1 - (deviation_pct * n) / 100))
            else:
                so_price = round_price(deal.entry_price * (1 + (deviation_pct * n) / 100))
                
            size_usd = base_order_usd * (params["so_volume_mult"] ** n)
            qty = round_qty(size_usd / so_price)
            
            if qty < MIN_QTY or qty * so_price < MIN_NOTIONAL:
                break
                
            deal.virtual_so_levels.append({
                "level": n,
                "price": so_price,
                "qty": qty,
                "size_usd": size_usd,
                "filled": False
            })
            
    def open_deal(self, direction: str, profile: str, alloc_fraction: float = 1.0):
        """Open a new virtual deal."""
        if self.current_price <= 0:
            print(f"  [SKIP] {direction} deal - no valid price data")
            return
            
        params = self.profile_manager.get_profile_params(profile)
        capital = self.profile_manager.get_profile_capital(profile)
        
        base_order_usd = params["base_order_pct"] / 100 * capital
        if base_order_usd < MIN_NOTIONAL:
            print(f"  [SKIP] {direction} base order ${base_order_usd:.2f} < min notional ${MIN_NOTIONAL}")
            return
            
        # Apply small slippage (0.01%) to simulate realistic entry
        slippage = 0.0001 if direction == "LONG" else -0.0001
        entry_price = round_price(self.current_price * (1 + slippage))
        
        qty = round_qty(base_order_usd / entry_price)
        if qty < MIN_QTY:
            print(f"  [SKIP] {direction} qty {qty} < min {MIN_QTY}")
            return
            
        cost = entry_price * qty
        self.deal_counter += 1
        now_str = datetime.now(timezone.utc).isoformat()
        
        deal = VirtualDeal(
            deal_id=self.deal_counter,
            symbol="HYPEUSDT",  # Hardcoded for now
            entry_price=entry_price,
            entry_qty=qty,
            entry_cost=cost,
            entry_time=now_str,
            direction=direction
        )
        
        # Create SO levels
        self.create_so_levels(deal, params)
        
        # Set initial TP
        tp_pct = params["tp_range"][0] + (params["tp_range"][1] - params["tp_range"][0]) * 0.5  # Mid-range
        deal.tp_price = deal.calc_tp_price(tp_pct)
        self.current_tp_pct = tp_pct
        
        # Apply fill to virtual account
        self.virtual_account.apply_fill(entry_price, qty, direction)
        
        self._set_deal(direction, deal)
        
        dir_emoji = "[LONG]" if direction == "LONG" else "[SHORT]"
        print(f"  {dir_emoji} [PAPER] Deal #{self.deal_counter} {direction}: {qty} @ ${entry_price:.3f}")
        print(f"    TP: ${deal.tp_price:.3f} | SOs: {len(deal.virtual_so_levels)}")
        
        send_telegram(
            f"{dir_emoji} Deal #{self.deal_counter} {direction}\n"
            f"HYPEUSDT\nEntry: ${entry_price:.3f}\nQty: {qty}\n"
            f"TP: ${deal.tp_price:.3f}\nLeverage: {params['leverage']}x\nProfile: {PROFILES[profile]['name']}"
        )
        
    def check_deals_for_fills(self) -> List[str]:
        """Check both deals for SO fills and TP hits."""
        closed_deals = []
        
        for direction in ["LONG", "SHORT"]:
            deal = self._get_deal(direction)
            if not deal or deal.closed:
                continue
                
            # Check SO fills (in order from shallow to deep)
            for so_level in deal.virtual_so_levels:
                if so_level["filled"]:
                    continue
                    
                # Check if price crossed SO level
                so_triggered = False
                if direction == "LONG" and self.current_price <= so_level["price"]:
                    so_triggered = True
                elif direction == "SHORT" and self.current_price >= so_level["price"]:
                    so_triggered = True
                    
                if so_triggered:
                    # Apply slippage
                    fill_price = round_price(so_level["price"] * (1 + 0.0001))
                    fill_qty = so_level["qty"]
                    fill_cost = fill_price * fill_qty
                    
                    deal.add_fill(fill_price, fill_qty, fill_cost)
                    so_level["filled"] = True
                    
                    # Update TP price to new average
                    deal.tp_price = deal.calc_tp_price(self.current_tp_pct)
                    
                    # Apply to virtual account
                    self.virtual_account.apply_fill(fill_price, fill_qty, direction)
                    
                    print(f"[PAPER] {direction} SO#{deal.safety_orders_filled} filled @ ${fill_price:.3f}")
                    print(f"    New avg: ${deal.avg_entry:.3f} | New TP: ${deal.tp_price:.3f}")
                    
                    send_telegram(
                        f"SO #{deal.safety_orders_filled} Filled ({direction})\n"
                        f"HYPEUSDT @ ${fill_price:.3f}\n"
                        f"New avg: ${deal.avg_entry:.3f}\n"
                        f"New TP: ${deal.tp_price:.3f}"
                    )
                else:
                    # SOs must fill in order, so if this one didn't trigger, stop checking deeper ones
                    break
                    
            # Check TP hit
            tp_hit = False
            if direction == "LONG" and self.current_price >= deal.tp_price:
                tp_hit = True
            elif direction == "SHORT" and self.current_price <= deal.tp_price:
                tp_hit = True
                
            if tp_hit:
                # Apply slippage
                fill_price = round_price(deal.tp_price * (1 - 0.0001 if direction == "LONG" else 1 + 0.0001))
                pnl = self.virtual_account.close_position(fill_price, deal.total_qty, direction)
                
                deal.closed = True
                deal.close_price = fill_price
                deal.close_time = datetime.now(timezone.utc).isoformat()
                deal.realized_pnl = pnl
                
                pnl_pct = pnl / deal.total_cost * 100
                
                print(f"[PAPER] TP HIT! Deal #{deal.deal_id} {direction} @ ${fill_price:.3f}")
                print(f"    PnL: ${pnl:.2f} ({pnl_pct:+.1f}%) | SOs used: {deal.safety_orders_filled}")
                
                send_telegram(
                    f"TP HIT - Deal #{deal.deal_id} {direction}\n"
                    f"HYPEUSDT closed @ ${fill_price:.3f}\n"
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.1f}%)\n"
                    f"SOs used: {deal.safety_orders_filled}"
                )
                
                self._set_deal(direction, None)
                closed_deals.append(f"{direction}_{deal.deal_id}")
                
        return closed_deals


class AsterTraderV2:
    """Paper Trading Bot with Risk Profiles."""
    
    def __init__(self, symbol: str = "HYPEUSDT", timeframe: str = "5m",
                 capital: float = 10000, profile: str = "medium"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        
        self.api = AsterAPI()
        self.virtual_account = VirtualAccount(capital)
        
        # Profile management
        self.profile_manager = ProfileManager(PAPER_DIR / "allocation.json")
        
        # Deal management
        self.deal_manager = PaperDealManager(self.profile_manager, self.virtual_account)
        
        # State
        self.current_regime = "UNKNOWN"
        self.current_price = 0.0
        self.start_equity = capital
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.cycle_count = 0
        self._running = False
        self._trend_bullish = True
        self._last_klines_df: Optional[pd.DataFrame] = None
        self.guardrail_events = []
        
        print(f"Paper trading bot initialized: {self.symbol} | Capital: ${self.capital}")
        
    def detect_regime(self) -> str:
        """Detect market regime using same logic as live bot."""
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
            
    def write_status(self):
        """Write comprehensive status JSON."""
        equity = self.virtual_account.get_equity()
        pnl = equity - self.start_equity
        pnl_pct = pnl / self.start_equity * 100 if self.start_equity > 0 else 0
        drawdown_pct = max(0, (self.start_equity - equity) / self.start_equity * 100) if self.start_equity > 0 else 0
        
        active_profile = self.profile_manager.get_active_profile()
        profile_params = self.profile_manager.get_profile_params(active_profile)
        
        long_alloc, short_alloc = REGIME_ALLOC.get(self.current_regime, (0.5, 0.5))
        if self.current_regime in DIRECTIONAL_REGIMES and not self._trend_bullish:
            long_alloc, short_alloc = short_alloc, long_alloc
            
        status = {
            "mode": "paper",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "running": self._running,
            "price": self.current_price,
            "active_profile": active_profile,
            "allocation": self.profile_manager.allocation,
            "total_capital": self.capital,
            "equity": round(equity, 2),
            "start_equity": round(self.start_equity, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "drawdown_pct": round(drawdown_pct, 2),
            "regime": self.current_regime,
            "trend_direction": "bullish" if self._trend_bullish else "bearish",
            "regime_alloc": {"long": long_alloc, "short": short_alloc},
            "adaptive_tp": {
                "current_tp_pct": round(self.deal_manager.current_tp_pct, 3),
                "current_dev_pct": round(self.deal_manager.current_dev_pct, 3),
                "tp_range": profile_params["tp_range"],
                "deviation_range": profile_params["deviation_range"],
            },
            "long_deal": self.deal_manager.long_deal.to_dict() if self.deal_manager.long_deal else None,
            "short_deal": self.deal_manager.short_deal.to_dict() if self.deal_manager.short_deal else None,
            "config": {
                "profile": active_profile,
                "leverage": profile_params["leverage"],
                "max_safety_orders": profile_params["max_safety_orders"],
                "so_volume_mult": profile_params["so_volume_mult"],
                "base_order_pct": profile_params["base_order_pct"],
                "capital_reserve": profile_params["capital_reserve"],
            },
            "deal_counter": self.deal_manager.deal_counter,
            "cycle_count": self.cycle_count,
            "start_time": self.start_time,
            "guardrail_events": self.guardrail_events,
        }
        
        with open(PAPER_DIR / "status.json", "w") as f:
            json.dump(status, f, indent=2)
            
    def log_trade(self, action: str, price: float, qty: float, notional: float,
                  so_count: int = 0, pnl: float = 0, direction: str = "LONG", deal_id: int = 0):
        """Log trade to CSV file."""
        path = PAPER_DIR / "trades.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "action", "symbol", "deal_id", "direction", "price", "qty",
                            "notional", "so_count", "pnl", "regime"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(), action, self.symbol,
                deal_id, direction, f"{price:.3f}", f"{qty:.2f}", f"{notional:.2f}", 
                so_count, f"{pnl:.2f}", self.current_regime
            ])
            
    def start(self):
        """Main trading loop."""
        self._running = True
        
        print(f"\nPaper Trading Bot started!")
        print(f"  Symbol: {self.symbol} | TF: {self.timeframe} | Capital: ${self.capital:.2f}")
        
        send_telegram(
            f"Paper Trading Bot Started\n"
            f"{self.symbol} | {self.timeframe}\n"
            f"Capital: ${self.capital:.2f}\n"
            f"Profile: {PROFILES[self.profile_manager.get_active_profile()]['name']}"
        )
        
        while self._running:
            try:
                self.cycle_count += 1
                
                # Check API connectivity
                if not self.api.ping():
                    print(f"API ping failed, retrying...")
                    time.sleep(10)
                    continue
                    
                # Detect regime
                self.current_regime = self.detect_regime()
                
                # Check for profile changes
                profile_changed = self.profile_manager.refresh_allocation()
                active_profile = self.profile_manager.get_active_profile()
                params = self.profile_manager.get_profile_params(active_profile)
                
                # Get regime allocation
                long_alloc, short_alloc = REGIME_ALLOC.get(self.current_regime, (0.5, 0.5))
                
                # Handle EXTREME regime per profile
                if self.current_regime == "EXTREME":
                    extreme_alloc = params["extreme_allocation"]
                    long_alloc, short_alloc = extreme_alloc[0] / 100, extreme_alloc[1] / 100
                    
                # Directional awareness
                if self.current_regime in DIRECTIONAL_REGIMES and not self._trend_bullish:
                    long_alloc, short_alloc = short_alloc, long_alloc
                    
                # Apply max directional bias
                max_bias = params["max_directional_bias"]
                max_long, max_short = max_bias[0] / 100, max_bias[1] / 100
                if long_alloc > max_long:
                    excess = long_alloc - max_long
                    long_alloc = max_long
                    short_alloc = min(short_alloc + excess, 1.0)
                if short_alloc > max_short:
                    excess = short_alloc - max_short
                    short_alloc = max_short  
                    long_alloc = min(long_alloc + excess, 1.0)
                
                trend_dir = "^" if self._trend_bullish else "v"
                long_str = f"L#{self.deal_manager.long_deal.deal_id}({self.deal_manager.long_deal.safety_orders_filled}SO)" if self.deal_manager.long_deal else "—"
                short_str = f"S#{self.deal_manager.short_deal.deal_id}({self.deal_manager.short_deal.safety_orders_filled}SO)" if self.deal_manager.short_deal else "—"
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ${self.current_price:.3f} | {self.current_regime}{trend_dir} | {PROFILES[active_profile]['name']} | {long_str} {short_str} | alloc L:{long_alloc:.0%}/S:{short_alloc:.0%}")
                
                # Update current price in deal manager
                self.deal_manager.current_price = self.current_price
                
                # Check existing deals for fills
                closed_deals = self.deal_manager.check_deals_for_fills()
                
                # Log closed deals
                for closed in closed_deals:
                    direction, deal_id = closed.split('_')
                    self.log_trade("TP_HIT", self.current_price, 0, 0, direction=direction, deal_id=int(deal_id))
                
                # Open new deals based on allocation (only if we have valid price)
                if self.current_price > 0:
                    if not self.deal_manager.long_deal and long_alloc > 0:
                        print(f"    Opening LONG deal (alloc: {long_alloc:.0%})")
                        self.deal_manager.open_deal("LONG", active_profile, long_alloc)
                        if self.deal_manager.long_deal:
                            self.log_trade("OPEN", self.deal_manager.long_deal.entry_price, 
                                         self.deal_manager.long_deal.entry_qty,
                                         self.deal_manager.long_deal.entry_cost,
                                         direction="LONG", deal_id=self.deal_manager.long_deal.deal_id)
                    
                    if not self.deal_manager.short_deal and short_alloc > 0:
                        print(f"    Opening SHORT deal (alloc: {short_alloc:.0%})")
                        self.deal_manager.open_deal("SHORT", active_profile, short_alloc)
                        if self.deal_manager.short_deal:
                            self.log_trade("OPEN", self.deal_manager.short_deal.entry_price,
                                         self.deal_manager.short_deal.entry_qty,
                                         self.deal_manager.short_deal.entry_cost,
                                         direction="SHORT", deal_id=self.deal_manager.short_deal.deal_id)
                else:
                    print(f"    Waiting for price data to open new deals...")
                
                # Write status
                self.write_status()
                
                time.sleep(30)  # 30-second cycle like live bot
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                time.sleep(15)
                
        self._shutdown()
        
    def _shutdown(self):
        """Graceful shutdown."""
        self._running = False
        self.write_status()
        print("Paper trading bot stopped")
        send_telegram("Paper Trading Bot Stopped")
        
    def stop(self):
        self._running = False