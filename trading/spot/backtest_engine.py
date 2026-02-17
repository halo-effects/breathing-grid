"""Spot DCA Backtest Engine with Scale-Out Exits.

Simulates a spot DCA strategy where safety orders are sold in REVERSE order
(largest/cheapest lots first) with per-lot take-profit targets. Adaptive
TP and deviation are driven by ATR and regime detection.
"""
import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..regime_detector import classify_regime_v2
from ..indicators import atr as compute_atr, atr_pct as compute_atr_pct

logger = logging.getLogger(__name__)

# ── Fee schedules ──────────────────────────────────────────────────────────

EXCHANGE_FEES: Dict[str, Dict[str, float]] = {
    "aster":       {"maker": 0.0001, "taker": 0.00035},
    "hyperliquid": {"maker": 0.0004, "taker": 0.0007},
}

# ── Regime multipliers (ported from aster_trader.py) ───────────────────────

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

# Regimes that block new deal entries
BLOCKED_REGIMES = {"EXTREME"}

# Bearish regime = wider spacing multiplier
BEARISH_SPACING_MULT = 1.4


# ── Risk profiles ──────────────────────────────────────────────────────────

@dataclass
class RiskProfile:
    name: str
    max_safety_orders: int
    base_order_pct: float       # fraction of capital for base order
    tp_min: float               # minimum TP %
    tp_max: float               # maximum TP %
    tp_baseline: float          # baseline TP %
    deviation_min: float        # minimum SO deviation %
    deviation_max: float        # maximum SO deviation %
    deviation_baseline: float   # baseline deviation %
    max_drawdown_pct: float     # halt threshold
    so_size_multiplier: float = 2.0
    atr_baseline_pct: float = 0.8  # calibration baseline

PROFILES: Dict[str, RiskProfile] = {
    "low": RiskProfile(
        name="low", max_safety_orders=5, base_order_pct=0.03,
        tp_min=1.5, tp_max=2.5, tp_baseline=2.0,
        deviation_min=3.0, deviation_max=4.0, deviation_baseline=3.5,
        max_drawdown_pct=15.0,
    ),
    "medium": RiskProfile(
        name="medium", max_safety_orders=8, base_order_pct=0.04,
        tp_min=1.0, tp_max=2.0, tp_baseline=1.5,
        deviation_min=2.0, deviation_max=3.0, deviation_baseline=2.5,
        max_drawdown_pct=25.0,
    ),
    "high": RiskProfile(
        name="high", max_safety_orders=12, base_order_pct=0.05,
        tp_min=0.8, tp_max=1.5, tp_baseline=1.0,
        deviation_min=1.5, deviation_max=2.5, deviation_baseline=2.0,
        max_drawdown_pct=35.0,
    ),
}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class Lot:
    """A single buy lot (base order or safety order)."""
    lot_id: int                 # 0 = base order, 1+ = safety orders
    buy_price: float
    qty: float
    cost_usd: float             # qty * buy_price + fee
    buy_fee: float
    buy_time: str               # ISO timestamp
    sell_price: Optional[float] = None
    sell_fee: float = 0.0
    sell_time: Optional[str] = None
    tp_target: float = 0.0      # price at which this lot should sell
    pnl: float = 0.0

    @property
    def is_sold(self) -> bool:
        return self.sell_price is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Deal:
    """A complete deal: one base order + 0..N safety orders, each sold individually."""
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
    def is_open(self) -> bool:
        return not self.is_complete

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
    def total_fees(self) -> float:
        return sum(l.buy_fee + l.sell_fee for l in self.lots)

    @property
    def capital_deployed(self) -> float:
        """Current USD tied up in unsold lots."""
        return sum(l.cost_usd for l in self.unsold_lots)

    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id,
            "symbol": self.symbol,
            "lots": [l.to_dict() for l in self.lots],
            "open_time": self.open_time,
            "close_time": self.close_time,
            "regime_at_open": self.regime_at_open,
            "total_invested": self.total_invested,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "is_complete": self.is_complete,
        }


@dataclass
class TradeLogEntry:
    timestamp: str
    action: str          # "BUY" or "SELL"
    deal_id: int
    lot_id: int
    price: float
    qty: float
    cost_usd: float
    fee: float
    pnl: float = 0.0
    regime: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    # Summary metrics
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    deals_per_day: float = 0.0
    avg_profit_per_deal_usd: float = 0.0
    avg_profit_per_deal_pct: float = 0.0
    avg_hold_time_hours: float = 0.0
    capital_utilization_pct: float = 0.0
    win_rate: float = 0.0
    largest_single_loss: float = 0.0
    total_fees_paid: float = 0.0
    total_deals_completed: int = 0
    total_deals_open: int = 0
    initial_capital: float = 0.0
    final_equity: float = 0.0

    # Per-SO-level breakdown
    per_lot_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Full logs
    trade_log: List[dict] = field(default_factory=list)
    deals: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)

    # Config
    profile: str = ""
    symbol: str = ""
    timeframe: str = ""
    exchange: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Engine ─────────────────────────────────────────────────────────────────

class SpotBacktestEngine:
    """Simulate spot DCA with scale-out exits on historical OHLCV data."""

    def __init__(
        self,
        profile: str = "medium",
        capital: float = 10000.0,
        exchange: str = "aster",
        symbol: str = "BTC/USDT",
        timeframe: str = "5m",
    ):
        self.profile = PROFILES[profile.lower()]
        self.initial_capital = capital
        self.cash = capital
        self.exchange = exchange.lower()
        self.symbol = symbol
        self.timeframe = timeframe

        fees = EXCHANGE_FEES.get(self.exchange, EXCHANGE_FEES["aster"])
        self.taker_fee = fees["taker"]
        self.maker_fee = fees["maker"]

        self.deals: List[Deal] = []
        self.completed_deals: List[Deal] = []
        self.trade_log: List[TradeLogEntry] = []
        self.equity_snapshots: List[dict] = []

        self._deal_counter = 0
        self._halted = False
        self._current_regime = "UNKNOWN"
        self._current_atr_pct = 0.0
        self._trend_bullish = True

        # Capital utilization tracking
        self._utilization_samples: List[float] = []

    # ── Adaptive parameter calculation ─────────────────────────────────

    def _adaptive_tp(self, regime: str, atr_pct: float) -> float:
        """Calculate adaptive TP% based on ATR and regime."""
        p = self.profile
        if atr_pct <= 0:
            return p.tp_baseline
        atr_ratio = atr_pct / p.atr_baseline_pct
        tp = p.tp_baseline * atr_ratio
        tp *= REGIME_TP_MULT.get(regime, 1.0)
        return max(p.tp_min, min(p.tp_max, round(tp, 3)))

    def _adaptive_deviation(self, regime: str, atr_pct: float, current_tp: float) -> float:
        """Calculate adaptive SO deviation% based on ATR and regime."""
        p = self.profile
        if atr_pct <= 0:
            return p.deviation_baseline
        atr_ratio = atr_pct / p.atr_baseline_pct
        dev = p.deviation_baseline * atr_ratio
        dev *= REGIME_DEV_MULT.get(regime, 1.0)
        dev = max(p.deviation_min, min(p.deviation_max, dev))
        # Deviation must exceed TP * 1.5
        dev = max(dev, current_tp * 1.5)
        return min(p.deviation_max, round(dev, 3))

    def _so_trigger_price(self, base_price: float, so_index: int, deviation: float) -> float:
        """Price at which safety order N triggers (cumulative deviation)."""
        # Each SO is deeper: SO1 = -dev%, SO2 = -2*dev%, etc.
        total_drop = deviation * so_index / 100.0
        return base_price * (1.0 - total_drop)

    def _so_cost(self, base_cost: float, so_index: int) -> float:
        """Cost of safety order N (2× multiplier per level)."""
        return base_cost * (self.profile.so_size_multiplier ** so_index)

    # ── Core simulation ────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on OHLCV DataFrame. Expects columns: timestamp, open, high, low, close, volume."""
        if len(df) < 100:
            logger.warning("Not enough data for backtest (%d rows)", len(df))
            return BacktestResult()

        # Pre-compute regimes and ATR for all rows
        logger.info("Computing regimes and indicators...")
        regimes = classify_regime_v2(df, self.timeframe)
        atr_pct_series = compute_atr_pct(df, 14)

        # Trend direction: price vs SMA50
        sma50 = df["close"].rolling(50).mean()

        logger.info("Running backtest: %s %s, capital=$%.0f, profile=%s",
                     self.symbol, self.timeframe, self.initial_capital, self.profile.name)

        peak_equity = self.initial_capital

        for i in range(100, len(df)):  # skip warmup
            row = df.iloc[i]
            ts = str(row["timestamp"])
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            regime = regimes.iloc[i] if i < len(regimes) else "UNKNOWN"
            self._current_regime = regime
            self._current_atr_pct = float(atr_pct_series.iloc[i]) if not pd.isna(atr_pct_series.iloc[i]) else 0.0
            self._trend_bullish = price >= float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else True

            tp_pct = self._adaptive_tp(regime, self._current_atr_pct)
            dev_pct = self._adaptive_deviation(regime, self._current_atr_pct, tp_pct)

            if self._halted:
                # Still check exits for open deals
                self._check_exits(high, low, price, ts, regime)
            else:
                # Check safety order fills (price went low enough)
                self._check_safety_order_fills(low, price, ts, regime, dev_pct, tp_pct)

                # Check exits (price went high enough to hit TP targets)
                self._check_exits(high, low, price, ts, regime)

                # Open new deal if no active deal and regime allows
                if not self.deals and regime not in BLOCKED_REGIMES:
                    self._open_deal(price, ts, regime, tp_pct)

            # Track equity
            equity = self._equity(price)
            self.equity_snapshots.append({"timestamp": ts, "equity": equity, "cash": self.cash, "price": price})

            # Utilization
            deployed = sum(d.capital_deployed for d in self.deals)
            self._utilization_samples.append(deployed / self.initial_capital * 100 if self.initial_capital > 0 else 0)

            # Drawdown halt check
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd >= self.profile.max_drawdown_pct and not self._halted:
                logger.warning("Drawdown halt triggered: %.1f%% >= %.1f%%", dd, self.profile.max_drawdown_pct)
                self._halted = True

        # Force-close remaining open deals at last price
        last_price = float(df.iloc[-1]["close"])
        last_ts = str(df.iloc[-1]["timestamp"])
        for deal in list(self.deals):
            self._force_close_deal(deal, last_price, last_ts)

        return self._compile_results(df)

    # ── Deal management ────────────────────────────────────────────────

    def _open_deal(self, price: float, ts: str, regime: str, tp_pct: float):
        """Open a new deal with a base order buy."""
        base_cost = self.initial_capital * self.profile.base_order_pct
        if base_cost > self.cash:
            return  # insufficient funds

        fee = base_cost * self.taker_fee  # market buy = taker
        qty = (base_cost - fee) / price

        self._deal_counter += 1
        lot = Lot(
            lot_id=0, buy_price=price, qty=qty,
            cost_usd=base_cost, buy_fee=fee, buy_time=ts,
            tp_target=price * (1 + tp_pct / 100),
        )
        deal = Deal(
            deal_id=self._deal_counter, symbol=self.symbol,
            lots=[lot], open_time=ts, regime_at_open=regime,
        )
        self.deals.append(deal)
        self.cash -= base_cost

        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action="BUY", deal_id=deal.deal_id, lot_id=0,
            price=price, qty=qty, cost_usd=base_cost, fee=fee, regime=regime,
        ))
        logger.debug("Deal %d opened at $%.2f (regime=%s)", deal.deal_id, price, regime)

    def _check_safety_order_fills(self, low: float, close: float, ts: str,
                                   regime: str, dev_pct: float, tp_pct: float):
        """Check if price dropped enough to fill the next safety order."""
        for deal in self.deals:
            filled_sos = len(deal.lots) - 1  # lots[0] is base order
            if filled_sos >= self.profile.max_safety_orders:
                continue
            if regime in BLOCKED_REGIMES:
                continue

            next_so = filled_sos + 1
            base_price = deal.lots[0].buy_price

            # Bearish = wider spacing
            spacing_mult = BEARISH_SPACING_MULT if not self._trend_bullish else 1.0
            trigger = self._so_trigger_price(base_price, next_so, dev_pct * spacing_mult)

            if low <= trigger:
                fill_price = trigger  # assume fill at trigger
                base_cost = self.initial_capital * self.profile.base_order_pct
                so_cost = self._so_cost(base_cost, next_so)
                so_cost = min(so_cost, self.cash)  # can't spend more than we have
                if so_cost < 5.0:  # minimum viable order
                    continue

                fee = so_cost * self.taker_fee
                qty = (so_cost - fee) / fill_price

                lot = Lot(
                    lot_id=next_so, buy_price=fill_price, qty=qty,
                    cost_usd=so_cost, buy_fee=fee, buy_time=ts,
                    tp_target=fill_price * (1 + tp_pct / 100),
                )
                deal.lots.append(lot)
                self.cash -= so_cost

                self.trade_log.append(TradeLogEntry(
                    timestamp=ts, action="BUY", deal_id=deal.deal_id, lot_id=next_so,
                    price=fill_price, qty=qty, cost_usd=so_cost, fee=fee, regime=regime,
                ))
                logger.debug("Deal %d SO%d filled at $%.2f", deal.deal_id, next_so, fill_price)

    def _check_exits(self, high: float, low: float, close: float, ts: str, regime: str):
        """Check if any unsold lots hit their TP target. Sell in REVERSE order (largest/cheapest first)."""
        for deal in list(self.deals):
            # Sort unsold lots by lot_id descending (highest SO = cheapest buy = sell first)
            unsold = sorted(deal.unsold_lots, key=lambda l: l.lot_id, reverse=True)

            # Recalculate TP targets based on current regime
            current_tp_pct = self._adaptive_tp(regime, self._current_atr_pct)

            for lot in unsold:
                # Update TP target dynamically
                lot.tp_target = lot.buy_price * (1 + current_tp_pct / 100)

                if high >= lot.tp_target:
                    # Sell at TP target (limit sell = maker fee)
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

                    self.trade_log.append(TradeLogEntry(
                        timestamp=ts, action="SELL", deal_id=deal.deal_id, lot_id=lot.lot_id,
                        price=sell_price, qty=lot.qty, cost_usd=revenue, fee=fee,
                        pnl=pnl, regime=regime,
                    ))
                    logger.debug("Deal %d lot %d sold at $%.2f, pnl=$%.2f",
                                 deal.deal_id, lot.lot_id, sell_price, pnl)

            # If deal complete, move to completed
            if deal.is_complete:
                deal.close_time = ts
                self.completed_deals.append(deal)
                self.deals.remove(deal)
                logger.debug("Deal %d completed, total pnl=$%.2f", deal.deal_id, deal.total_pnl)

    def _force_close_deal(self, deal: Deal, price: float, ts: str):
        """Force-close all unsold lots at market price (end of data)."""
        for lot in deal.unsold_lots:
            revenue = lot.qty * price
            fee = revenue * self.taker_fee  # market sell
            net_revenue = revenue - fee
            pnl = net_revenue - lot.cost_usd

            lot.sell_price = price
            lot.sell_fee = fee
            lot.sell_time = ts
            lot.pnl = pnl
            self.cash += net_revenue

            self.trade_log.append(TradeLogEntry(
                timestamp=ts, action="SELL(FORCE)", deal_id=deal.deal_id, lot_id=lot.lot_id,
                price=price, qty=lot.qty, cost_usd=revenue, fee=fee, pnl=pnl,
            ))

        deal.close_time = ts
        self.completed_deals.append(deal)
        self.deals.remove(deal)

    def _equity(self, current_price: float) -> float:
        """Current equity = cash + market value of unsold lots."""
        unsold_value = sum(
            lot.qty * current_price
            for deal in self.deals
            for lot in deal.unsold_lots
        )
        return self.cash + unsold_value

    # ── Results compilation ────────────────────────────────────────────

    def _compile_results(self, df: pd.DataFrame) -> BacktestResult:
        all_deals = self.completed_deals
        final_equity = self.cash  # all deals force-closed

        # Basic metrics
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Max drawdown from equity curve
        eq = pd.Series([s["equity"] for s in self.equity_snapshots])
        peak = eq.cummax()
        dd = (peak - eq) / peak * 100
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        # Sharpe ratio (daily returns)
        if len(self.equity_snapshots) > 1:
            eq_df = pd.DataFrame(self.equity_snapshots)
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
            daily = eq_df.set_index("timestamp")["equity"].resample("1D").last().dropna()
            daily_ret = daily.pct_change().dropna()
            if len(daily_ret) > 1 and daily_ret.std() > 0:
                sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Time span
        if len(df) > 1:
            t0 = pd.to_datetime(df.iloc[0]["timestamp"])
            t1 = pd.to_datetime(df.iloc[-1]["timestamp"])
            days = max((t1 - t0).total_seconds() / 86400, 1)
        else:
            days = 1

        completed = [d for d in all_deals if d.is_complete or d.close_time is not None]
        n_completed = len(completed)

        # Per-deal stats
        deal_pnls = [d.total_pnl for d in completed]
        deal_pnl_pcts = [d.total_pnl / d.total_invested * 100 if d.total_invested > 0 else 0 for d in completed]
        deal_fees = [d.total_fees for d in completed]

        # Hold times
        hold_times = []
        for d in completed:
            if d.open_time and d.close_time:
                t_open = pd.to_datetime(d.open_time)
                t_close = pd.to_datetime(d.close_time)
                hold_times.append((t_close - t_open).total_seconds() / 3600)

        # Win rate
        wins = sum(1 for p in deal_pnls if p > 0)

        # Per-lot-level breakdown
        lot_stats: Dict[int, Dict[str, float]] = {}
        for d in completed:
            for lot in d.lots:
                lid = lot.lot_id
                if lid not in lot_stats:
                    lot_stats[lid] = {"count": 0, "total_pnl": 0, "wins": 0, "total_cost": 0}
                lot_stats[lid]["count"] += 1
                lot_stats[lid]["total_pnl"] += lot.pnl
                lot_stats[lid]["total_cost"] += lot.cost_usd
                if lot.pnl > 0:
                    lot_stats[lid]["wins"] += 1
        for lid in lot_stats:
            s = lot_stats[lid]
            s["avg_pnl"] = s["total_pnl"] / s["count"] if s["count"] > 0 else 0
            s["win_rate"] = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            s["avg_return_pct"] = s["total_pnl"] / s["total_cost"] * 100 if s["total_cost"] > 0 else 0

        return BacktestResult(
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            deals_per_day=round(n_completed / days, 2),
            avg_profit_per_deal_usd=round(np.mean(deal_pnls), 2) if deal_pnls else 0,
            avg_profit_per_deal_pct=round(np.mean(deal_pnl_pcts), 2) if deal_pnl_pcts else 0,
            avg_hold_time_hours=round(np.mean(hold_times), 2) if hold_times else 0,
            capital_utilization_pct=round(np.mean(self._utilization_samples), 2) if self._utilization_samples else 0,
            win_rate=round(wins / n_completed * 100, 1) if n_completed > 0 else 0,
            largest_single_loss=round(min(deal_pnls), 2) if deal_pnls else 0,
            total_fees_paid=round(sum(deal_fees), 2),
            total_deals_completed=n_completed,
            total_deals_open=len(self.deals),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            per_lot_stats={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in lot_stats.items()},
            trade_log=[t.to_dict() for t in self.trade_log],
            deals=[d.to_dict() for d in all_deals],
            equity_curve=[{"timestamp": s["timestamp"], "equity": round(s["equity"], 2)} for s in self.equity_snapshots[::max(1, len(self.equity_snapshots)//500)]],  # downsample
            profile=self.profile.name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange=self.exchange,
        )
