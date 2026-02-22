"""V12e Live Lifecycle Engine — Wyckoff phase management for live/paper trading.

Phases: DCA → EXIT → MARKDOWN → SPRING → MARKUP
Ported from backtest_engine_v12.py for live execution.
"""
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Lifecycle Phase Enum ───────────────────────────────────────────

class LifecyclePhase(str, Enum):
    DCA = "DCA"
    EXIT = "EXIT"
    MARKDOWN = "MARKDOWN"
    SPRING = "SPRING"
    MARKUP = "MARKUP"


class RebalancingMode(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


REBALANCING_PROFILES = {
    RebalancingMode.CONSERVATIVE: {
        "cooldown_hours": 12,
        "exit_lot_pct": 0.25,
        "max_rebalances_day": 2,
        "markdown_exit_style": "confirmed",
    },
    RebalancingMode.BALANCED: {
        "cooldown_hours": 4,
        "exit_lot_pct": 0.50,
        "max_rebalances_day": 6,
        "markdown_exit_style": "moderate",
    },
    RebalancingMode.AGGRESSIVE: {
        "cooldown_hours": 1,
        "exit_lot_pct": 0.75,
        "max_rebalances_day": 12,
        "markdown_exit_style": "early",
    },
}

# String-key lookup for convenience
REBALANCING_PROFILES_BY_NAME = {k.value: v for k, v in REBALANCING_PROFILES.items()}


# ── Known ATH prices ──────────────────────────────────────────────

KNOWN_ATH = {
    "ETH/USDT": 4878.0, "ETH/USDC": 4878.0,
    "BTC/USDT": 109000.0, "BTC/USDC": 109000.0,
    "SOL/USDT": 260.0, "SOL/USDC": 260.0,
    "HYPE/USDC": 35.0,
}

# ── Risk profile deployment percentages ──────────────────────────

PROFILE_DEPLOY_PCT = {
    "low": 0.50,
    "medium": 0.70,
    "high": 0.90,
}


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ExitLot:
    """A lot being managed by the Exit Engine."""
    lot_id: int
    buy_price: float
    qty: float
    cost_usd: float
    buy_time: str
    unrealized_pnl_pct: float = 0.0
    trailing_stop: float = 0.0
    sold: bool = False
    sell_price: float = 0.0
    sell_time: str = ""
    sell_reason: str = ""
    pnl: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExitLot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ShortPosition:
    """Virtual short position for paper trading."""
    total_qty: float = 0.0
    total_margin: float = 0.0
    avg_entry: float = 0.0
    entries: int = 0
    first_entry_price: float = 0.0
    trail_low: float = float('inf')
    trail_stop: float = 0.0
    sl_price: float = 0.0
    closed: bool = False
    close_price: float = 0.0
    close_time: str = ""
    pnl: float = 0.0
    funding_cost: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if d['trail_low'] == float('inf'):
            d['trail_low'] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ShortPosition":
        if d.get('trail_low') is None:
            d['trail_low'] = float('inf')
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MarkupPosition:
    """Markup long position."""
    qty: float = 0.0
    cost: float = 0.0
    avg_entry: float = 0.0
    trail_high: float = 0.0
    trail_stop: float = 0.0
    adds: int = 0
    last_add_price: float = 0.0
    phase_cash: float = 0.0
    entry_price: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MarkupPosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LifecycleConfig:
    """Configuration for the lifecycle engine."""
    enabled: bool = False
    risk_profile: str = "medium"  # low/medium/high
    exit_threshold: float = 50.0
    commitment_hours: int = 48
    spring_discount_pct: float = 25.0
    markup_breakout_enabled: bool = True
    ath: float = 0.0

    # Exit engine
    initial_trail_pct: float = 3.0
    trail_floor_pct: float = 1.5
    trail_tighten_per_day: float = 0.5
    rally_sell_pct: float = 1.5
    urgency_day_moderate: int = 4
    urgency_day_aggressive: int = 7
    urgency_day_force: int = 14

    # Shorts
    short_enabled: bool = True
    short_tier1_deploy: float = 0.60
    short_tier2_deploy: float = 0.80
    short_tier3_deploy: float = 0.90
    short_tier2_bounce_pct: float = 3.0
    short_tier3_retest_pct: float = 2.0
    short_sl_pct: float = 15.0
    funding_rate_daily: float = 0.0003

    # Spring
    # V12e PARAM SYNC: updated to match backtest parameters (2025-02-22)
    spring_tier1_discount: float = 25.0
    spring_tier2_discount: float = 28.0   # was 35% — synced with backtest
    spring_tier3_discount: float = 35.0   # was 45% — synced with backtest
    spring_tier1_deploy: float = 0.25
    spring_tier2_deploy: float = 0.55
    spring_tier3_deploy: float = 0.75
    spring_tp_pct: float = 15.0           # was 10% — synced with backtest
    spring_false_drop_pct: float = 15.0
    spring_recovery_pct: float = 20.0
    spring_timeout_days: int = 60

    # CFGI thresholds for phase transitions
    cfgi_spring_extreme_fear: float = 15.0
    cfgi_spring_fear: float = 25.0
    cfgi_spring_neutral: float = 40.0
    cfgi_spring_discount_reduced: float = 0.20
    cfgi_spring_discount_normal: float = 0.25
    cfgi_spring_discount_raised: float = 0.30
    cfgi_breakout_min: float = 55.0
    cfgi_exit_fast_threshold: float = 75.0
    cfgi_exit_fast_hours: float = 24.0
    cfgi_exit_invalidate: float = 50.0

    # Markup
    markup_deploy_pct: float = 0.65
    markup_trail_pct: float = 10.0
    markup_pullback_pct: float = 5.0
    markup_pullback_deploy_pct: float = 0.15
    markup_max_adds: int = 3

    # Rebalancing aggressiveness
    rebalancing_mode: str = "balanced"
    auto_rotation: bool = True

    @property
    def deploy_pct(self) -> float:
        return PROFILE_DEPLOY_PCT.get(self.risk_profile, 0.70)


@dataclass
class LifecycleState:
    """Full lifecycle state — serializable for persistence."""
    phase: LifecyclePhase = LifecyclePhase.DCA

    # Exit engine
    exit_price: float = 0.0
    exit_entry_time: str = ""
    exit_candles_elapsed: int = 0
    exit_local_low: float = float('inf')
    exit_lots: List[dict] = field(default_factory=list)
    exit_lots_sold: int = 0
    exit_lots_total: int = 0
    exit_realized_pnl: float = 0.0
    exit_committed: bool = False
    exit_invalidated: bool = False

    # Shorts
    short_position: Optional[dict] = None

    # Markdown
    markdown_entry_price: float = 0.0
    markdown_ath_at_entry: float = 0.0

    # Spring
    spring_entries: List[dict] = field(default_factory=list)
    spring_entry_price: float = 0.0
    spring_deployed: float = 0.0
    spring_phase_cash: float = 0.0
    spring_highest_tier: int = 0
    spring_candle_count: int = 0

    # Markup
    markup_position: Optional[dict] = None

    # Commitment window
    commitment_start: str = ""
    commitment_score_sustained: bool = False

    # CFGI tracking
    last_cfgi: Optional[float] = None
    commitment_cfgi_min: Optional[float] = None

    # Conductor cache
    conductor_price_ath: float = 0.0
    conductor_cached_score: float = 0.0

    # Reversal detector
    rd_gate_active: bool = False
    rd_rolling_high: float = 0.0

    # Rebalancing tracking
    last_rebalance_time: str = ""
    rebalances_today: int = 0
    rebalance_day: str = ""

    # Metrics
    exit_phases: int = 0
    spring_phases: int = 0
    markdown_phases: int = 0
    markup_phases: int = 0
    short_pnl: float = 0.0
    spring_pnl: float = 0.0
    markup_pnl: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "phase": self.phase.value,
            "exit_price": self.exit_price,
            "exit_entry_time": self.exit_entry_time,
            "exit_candles_elapsed": self.exit_candles_elapsed,
            "exit_local_low": self.exit_local_low if self.exit_local_low != float('inf') else None,
            "exit_lots": self.exit_lots,
            "exit_lots_sold": self.exit_lots_sold,
            "exit_lots_total": self.exit_lots_total,
            "exit_realized_pnl": self.exit_realized_pnl,
            "exit_committed": self.exit_committed,
            "exit_invalidated": self.exit_invalidated,
            "short_position": self.short_position,
            "markdown_entry_price": self.markdown_entry_price,
            "markdown_ath_at_entry": self.markdown_ath_at_entry,
            "spring_entries": self.spring_entries,
            "spring_entry_price": self.spring_entry_price,
            "spring_deployed": self.spring_deployed,
            "spring_phase_cash": self.spring_phase_cash,
            "spring_highest_tier": self.spring_highest_tier,
            "spring_candle_count": self.spring_candle_count,
            "markup_position": self.markup_position,
            "commitment_start": self.commitment_start,
            "commitment_score_sustained": self.commitment_score_sustained,
            "last_cfgi": self.last_cfgi,
            "commitment_cfgi_min": self.commitment_cfgi_min,
            "conductor_price_ath": self.conductor_price_ath,
            "conductor_cached_score": self.conductor_cached_score,
            "rd_gate_active": self.rd_gate_active,
            "rd_rolling_high": self.rd_rolling_high,
            "last_rebalance_time": self.last_rebalance_time,
            "rebalances_today": self.rebalances_today,
            "rebalance_day": self.rebalance_day,
            "exit_phases": self.exit_phases,
            "spring_phases": self.spring_phases,
            "markdown_phases": self.markdown_phases,
            "markup_phases": self.markup_phases,
            "short_pnl": self.short_pnl,
            "spring_pnl": self.spring_pnl,
            "markup_pnl": self.markup_pnl,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LifecycleState":
        s = cls()
        s.phase = LifecyclePhase(d.get("phase", "DCA"))
        s.exit_price = d.get("exit_price", 0.0)
        s.exit_entry_time = d.get("exit_entry_time", "")
        s.exit_candles_elapsed = d.get("exit_candles_elapsed", 0)
        el = d.get("exit_local_low")
        s.exit_local_low = el if el is not None else float('inf')
        s.exit_lots = d.get("exit_lots", [])
        s.exit_lots_sold = d.get("exit_lots_sold", 0)
        s.exit_lots_total = d.get("exit_lots_total", 0)
        s.exit_realized_pnl = d.get("exit_realized_pnl", 0.0)
        s.exit_committed = d.get("exit_committed", False)
        s.exit_invalidated = d.get("exit_invalidated", False)
        s.short_position = d.get("short_position")
        s.markdown_entry_price = d.get("markdown_entry_price", 0.0)
        s.markdown_ath_at_entry = d.get("markdown_ath_at_entry", 0.0)
        s.spring_entries = d.get("spring_entries", [])
        s.spring_entry_price = d.get("spring_entry_price", 0.0)
        s.spring_deployed = d.get("spring_deployed", 0.0)
        s.spring_phase_cash = d.get("spring_phase_cash", 0.0)
        s.spring_highest_tier = d.get("spring_highest_tier", 0)
        s.spring_candle_count = d.get("spring_candle_count", 0)
        s.markup_position = d.get("markup_position")
        s.commitment_start = d.get("commitment_start", "")
        s.commitment_score_sustained = d.get("commitment_score_sustained", False)
        s.last_cfgi = d.get("last_cfgi")
        s.commitment_cfgi_min = d.get("commitment_cfgi_min")
        s.conductor_price_ath = d.get("conductor_price_ath", 0.0)
        s.conductor_cached_score = d.get("conductor_cached_score", 0.0)
        s.rd_gate_active = d.get("rd_gate_active", False)
        s.rd_rolling_high = d.get("rd_rolling_high", 0.0)
        s.last_rebalance_time = d.get("last_rebalance_time", "")
        s.rebalances_today = d.get("rebalances_today", 0)
        s.rebalance_day = d.get("rebalance_day", "")
        s.exit_phases = d.get("exit_phases", 0)
        s.spring_phases = d.get("spring_phases", 0)
        s.markdown_phases = d.get("markdown_phases", 0)
        s.markup_phases = d.get("markup_phases", 0)
        s.short_pnl = d.get("short_pnl", 0.0)
        s.spring_pnl = d.get("spring_pnl", 0.0)
        s.markup_pnl = d.get("markup_pnl", 0.0)
        return s


class LifecycleEngine:
    """V12e Lifecycle Engine for live/paper spot trading.
    
    Manages Wyckoff phase transitions: DCA → EXIT → MARKDOWN → SPRING → MARKUP.
    Uses DailyScorerConductor for phase transition signals.
    
    Designed to be composed into LifecycleTrader — not a standalone bot.
    The trader calls lifecycle methods each cycle to get phase-specific actions.
    """

    def __init__(self, config: LifecycleConfig, symbol: str, taker_fee: float = 0.00035):
        self.config = config
        self.symbol = symbol
        self.taker_fee = taker_fee
        self.state = LifecycleState()

        # Resolve ATH
        ath = config.ath or KNOWN_ATH.get(symbol, 0.0)

        # Initialize conductor
        from .backtest_engine_v12 import DailyScorerConductor
        self._conductor = DailyScorerConductor(
            exit_threshold=config.exit_threshold,
            mcap_ath_pct=0.25,
            ath=ath,
            symbol=symbol,
        )
        if ath > 0:
            self._conductor.set_price_ath(ath)

        # 1h candle accumulator for daily resampling
        self._accumulated_1h = None
        self._last_daily_score = 0.0

    # ── Cold-Start Phase Setting ─────────────────────────────────

    def set_initial_phase(self, phase: LifecyclePhase, reason: str):
        """Set the starting phase from the cold-start classifier.
        
        Only takes effect if still in default DCA with no activity.
        """
        if self.state.phase != LifecyclePhase.DCA:
            logger.info("%s: Skipping set_initial_phase (already in %s)", self.symbol, self.state.phase.value)
            return

        self.state.phase = phase
        logger.info("🎯 %s: Initial phase set to %s — %s", self.symbol, phase.value, reason)

        if phase == LifecyclePhase.MARKUP:
            # Set up markup tracking but don't deploy yet — wait for next cycle to confirm
            self.state.markup_phases += 1
        elif phase == LifecyclePhase.EXIT:
            # Begin scoring immediately
            from datetime import datetime, timezone
            self.state.exit_entry_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.state.exit_candles_elapsed = 0
            self.state.exit_phases += 1
        elif phase == LifecyclePhase.SPRING:
            self.state.spring_phases += 1
            self.state.spring_candle_count = 0
        elif phase == LifecyclePhase.MARKDOWN:
            self.state.markdown_phases += 1

    # ── Conductor Integration ─────────────────────────────────────

    def feed_candles_1h(self, df_1h):
        """Feed 1h candles to build daily conductor data.
        Call with accumulated 1h data each cycle."""
        import pandas as pd
        if self._accumulated_1h is None:
            self._accumulated_1h = df_1h.copy()
        else:
            self._accumulated_1h = pd.concat(
                [self._accumulated_1h, df_1h], ignore_index=True
            ).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        self._conductor.prepare(self._accumulated_1h)

    def get_daily_score(self, ts_ms: int, price: float, cfgi_score: float = None) -> float:
        """Get the daily conductor score. Call each cycle."""
        fg = cfgi_score if cfgi_score is not None else 50.0
        score = self._conductor.score_at(ts_ms, price, fg, coin_cfgi=cfgi_score)
        self._last_daily_score = score
        self.state.conductor_cached_score = score
        self.state.conductor_price_ath = self._conductor._price_ath
        return score

    # ── Phase Processing ──────────────────────────────────────────

    def process_cycle(self, price: float, high: float, low: float,
                      ts: str, regime: str, cfgi_score: float,
                      cash: float, deals: dict,
                      send_telegram_fn=None) -> List[dict]:
        """Process one cycle. Returns list of action dicts for the trader to execute.
        
        Actions: {type: "buy"/"sell"/"open_short"/"close_short"/"phase_change", ...}
        """
        if not self.config.enabled:
            return []

        actions = []
        ts_ms = int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() * 1000) if isinstance(ts, str) else int(ts)

        daily_score = self.get_daily_score(ts_ms, price, cfgi_score)
        phase = self.state.phase

        def _tg(msg):
            if send_telegram_fn:
                send_telegram_fn(msg)
            logger.info(msg)

        # ── DCA Phase ─────────────────────────────────────────
        if phase == LifecyclePhase.DCA:
            acts = self._process_dca(price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, deals, _tg)
            actions.extend(acts)

        # ── EXIT Phase ────────────────────────────────────────
        elif phase == LifecyclePhase.EXIT:
            acts = self._process_exit(price, high, low, ts, ts_ms, regime, daily_score, cash, _tg)
            actions.extend(acts)

        # ── MARKDOWN Phase ────────────────────────────────────
        elif phase == LifecyclePhase.MARKDOWN:
            acts = self._process_markdown(price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, _tg)
            actions.extend(acts)

        # ── SPRING Phase ──────────────────────────────────────
        elif phase == LifecyclePhase.SPRING:
            acts = self._process_spring(price, high, low, ts, regime, cash, _tg)
            actions.extend(acts)

        # ── MARKUP Phase ──────────────────────────────────────
        elif phase == LifecyclePhase.MARKUP:
            acts = self._process_markup(price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, _tg)
            actions.extend(acts)

        return actions

    # ── DCA Phase Processing ──────────────────────────────────

    def _process_dca(self, price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, deals, _tg) -> list:
        actions = []

        # Track CFGI for commitment window modulation
        self.state.last_cfgi = cfgi_score

        # Check DCA → EXIT: score >= threshold for commitment_hours
        if self._conductor.should_exit(ts_ms, price, cfgi_score or 50.0):
            if not self.state.commitment_start:
                self.state.commitment_start = ts
                self.state.commitment_score_sustained = True
                self.state.commitment_cfgi_min = cfgi_score
                logger.info("EXIT commitment window started: score=%.1f at %s", daily_score, ts)
            else:
                # Track minimum CFGI during commitment window
                if cfgi_score is not None:
                    if self.state.commitment_cfgi_min is None:
                        self.state.commitment_cfgi_min = cfgi_score
                    else:
                        self.state.commitment_cfgi_min = min(self.state.commitment_cfgi_min, cfgi_score)

                # CFGI invalidation: greed evaporated during commitment
                if (cfgi_score is not None
                        and cfgi_score < self.config.cfgi_exit_invalidate):
                    logger.info("EXIT commitment: CFGI=%.0f, dropped below %.0f — invalidated",
                                cfgi_score, self.config.cfgi_exit_invalidate)
                    self.state.commitment_start = ""
                    self.state.commitment_score_sustained = False
                    self.state.commitment_cfgi_min = None
                else:
                    # Determine effective commitment window
                    effective_hours = self.config.commitment_hours
                    if (cfgi_score is not None
                            and self.state.commitment_cfgi_min is not None
                            and self.state.commitment_cfgi_min > self.config.cfgi_exit_fast_threshold):
                        effective_hours = self.config.cfgi_exit_fast_hours

                    try:
                        start = datetime.fromisoformat(self.state.commitment_start.replace('Z', '+00:00'))
                        now = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        hours = (now - start).total_seconds() / 3600
                    except Exception:
                        hours = 0

                    logger.info("EXIT commitment: CFGI=%s, window=%.0fh",
                                f"{cfgi_score:.0f}" if cfgi_score is not None else "N/A",
                                effective_hours)

                    if hours >= effective_hours:
                        # Transition DCA → EXIT
                        self.state.commitment_cfgi_min = None
                        actions.extend(self._transition_to_exit(price, ts, daily_score, cfgi_score, deals, _tg))
        else:
            # Reset commitment window if score drops
            if self.state.commitment_start:
                logger.info("EXIT commitment window reset: score=%.1f dropped below threshold", daily_score)
                self.state.commitment_start = ""
                self.state.commitment_score_sustained = False
                self.state.commitment_cfgi_min = None

        # Check DCA → MARKUP breakout (if enabled)
        if self.config.markup_breakout_enabled and regime in ("TRENDING", "MILD_TREND"):
            # Simplified breakout check for live: regime is trending + score is low (not topping)
            if daily_score < 30 and self._accumulated_1h is not None and len(self._accumulated_1h) > 200:
                # CFGI sentiment confirmation for breakout
                if cfgi_score is not None:
                    if cfgi_score > self.config.cfgi_breakout_min:
                        logger.info("BREAKOUT check: CFGI=%.0f, confirmed", cfgi_score)
                        # Full breakout logic would go here
                    else:
                        logger.info("BREAKOUT check: CFGI=%.0f, vetoed", cfgi_score)
                else:
                    logger.info("BREAKOUT check: CFGI=N/A, no sentiment filter")
                    # Full breakout logic would go here (backward compatible)

        return actions

    # ── EXIT Phase Processing ─────────────────────────────────

    def _process_exit(self, price, high, low, ts, ts_ms, regime, daily_score, cash, _tg) -> list:
        actions = []
        self.state.exit_candles_elapsed += 1
        hours_elapsed = self.state.exit_candles_elapsed
        days_elapsed = hours_elapsed / 24.0

        if low < self.state.exit_local_low:
            self.state.exit_local_low = low

        # Commitment check
        if not self.state.exit_committed and hours_elapsed >= self.config.commitment_hours:
            self.state.exit_committed = True
            logger.info("EXIT COMMITTED after %dh", hours_elapsed)

        # False signal check (first commitment_hours only)
        if not self.state.exit_committed:
            if daily_score <= 0:
                logger.info("EXIT INVALIDATED: score=0 within %dh", hours_elapsed)
                # Force sell all exit lots and return to DCA
                actions.extend(self._force_sell_exit_lots(price, ts, "invalidated", _tg))
                actions.extend(self._close_short_actions(price, ts, "invalidation", _tg))
                self._reset_to_dca(ts, _tg)
                return actions

        # Update trailing stops and check for sells
        exit_lots = [ExitLot.from_dict(d) for d in self.state.exit_lots]
        open_lots = [l for l in exit_lots if not l.sold]

        trail_pct = max(
            self.config.trail_floor_pct,
            self.config.initial_trail_pct - days_elapsed * self.config.trail_tighten_per_day
        )

        for lot in open_lots:
            new_trail = price * (1 - trail_pct / 100)
            if new_trail > lot.trailing_stop:
                lot.trailing_stop = new_trail
            lot.unrealized_pnl_pct = (price - lot.buy_price) / lot.buy_price * 100

        # Trailing stop hits
        for lot in open_lots:
            if low <= lot.trailing_stop and not lot.sold:
                lot.sold = True
                lot.sell_price = lot.trailing_stop
                lot.sell_time = ts
                lot.sell_reason = "trailing_stop"
                lot.pnl = (lot.trailing_stop - lot.buy_price) * lot.qty
                self.state.exit_lots_sold += 1
                self.state.exit_realized_pnl += lot.pnl
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": lot.qty, "price": lot.trailing_stop,
                    "reason": f"EXIT trailing_stop (lot {lot.lot_id})",
                })

        # Rally selling — scaled by rebalancing exit_lot_pct
        if self.state.exit_local_low < float('inf'):
            bounce_pct = (high - self.state.exit_local_low) / self.state.exit_local_low * 100
            if bounce_pct >= self.config.rally_sell_pct:
                profitable = sorted(
                    [l for l in open_lots if not l.sold],
                    key=lambda l: l.unrealized_pnl_pct, reverse=True
                )
                if profitable:
                    # Determine how many lots to sell based on exit_lot_pct
                    exit_lot_pct = self.rebalancing_profile["exit_lot_pct"]
                    n_to_sell = max(1, int(len(profitable) * exit_lot_pct + 0.5))
                    candidates = profitable[:n_to_sell]
                    for best in candidates:
                        if best.unrealized_pnl_pct > 0 or days_elapsed >= self.config.urgency_day_moderate:
                            best.sold = True
                            best.sell_price = high
                            best.sell_time = ts
                            best.sell_reason = "rally_sell"
                            best.pnl = (high - best.buy_price) * best.qty
                            self.state.exit_lots_sold += 1
                            self.state.exit_realized_pnl += best.pnl
                            actions.append({
                                "type": "sell", "symbol": self.symbol,
                                "amount": best.qty, "price": high,
                                "reason": f"EXIT rally_sell (lot {best.lot_id})",
                            })
                    self.state.exit_local_low = price

        # Time-based urgency
        still_open = [l for l in open_lots if not l.sold]
        if days_elapsed >= self.config.urgency_day_force and still_open:
            for lot in still_open:
                lot.sold = True
                lot.sell_price = price
                lot.sell_time = ts
                lot.sell_reason = "force_close"
                lot.pnl = (price - lot.buy_price) * lot.qty
                self.state.exit_lots_sold += 1
                self.state.exit_realized_pnl += lot.pnl
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": lot.qty, "price": price,
                    "reason": f"EXIT force_close (lot {lot.lot_id})",
                })
        elif days_elapsed >= self.config.urgency_day_aggressive and still_open:
            for lot in list(still_open):
                if lot.unrealized_pnl_pct >= 0:
                    lot.sold = True
                    lot.sell_price = price
                    lot.sell_time = ts
                    lot.sell_reason = "urgency_aggressive"
                    lot.pnl = (price - lot.buy_price) * lot.qty
                    self.state.exit_lots_sold += 1
                    self.state.exit_realized_pnl += lot.pnl
                    actions.append({
                        "type": "sell", "symbol": self.symbol,
                        "amount": lot.qty, "price": price,
                        "reason": f"EXIT urgency (lot {lot.lot_id})",
                    })

        # Persist updated lots
        self.state.exit_lots = [l.to_dict() for l in exit_lots]

        # Check if all lots sold → MARKDOWN or SPRING (gated by rebalance limits)
        all_sold = all(l.sold for l in exit_lots) if exit_lots else True
        if all_sold:
            if not self.config.auto_rotation:
                logger.info("%s: EXIT complete, auto_rotation OFF — staying idle", self.symbol)
                self._reset_to_dca(ts, _tg)
            else:
                can_rebalance = self._check_rebalance_cooldown() and self._check_rebalance_limit()
                if can_rebalance:
                    self._record_rebalance()
                    if self.config.short_enabled:
                        self._transition_to_markdown(price, ts, cash, _tg)
                    else:
                        self._transition_to_spring(price, ts, cash, _tg)
                else:
                    logger.info("%s: EXIT complete but rebalance cooldown/limit not met, waiting", self.symbol)

        return actions

    # ── MARKDOWN Phase Processing ─────────────────────────────

    def _process_markdown(self, price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, _tg) -> list:
        actions = []

        # Process short position
        short = self._get_short()
        if short and not short.closed:
            # Funding cost
            rate_per_candle = self.config.funding_rate_daily / 24.0
            short.funding_cost += short.total_margin * rate_per_candle

            # Tier 2: add on bounce
            if short.entries == 1:
                bounce_pct = (high - short.first_entry_price) / short.first_entry_price * 100
                if bounce_pct >= self.config.short_tier2_bounce_pct:
                    self._add_short_tier(high, ts, 2, self.config.short_tier2_deploy, cash)
                    actions.append({"type": "open_short", "symbol": self.symbol,
                                    "tier": 2, "price": high, "reason": "MARKDOWN tier2 bounce"})

            # Tier 3: add on retest
            if short.entries == 2:
                retest_pct = abs(high - short.first_entry_price) / short.first_entry_price * 100
                if retest_pct <= self.config.short_tier3_retest_pct:
                    self._add_short_tier(high, ts, 3, self.config.short_tier3_deploy, cash)
                    actions.append({"type": "open_short", "symbol": self.symbol,
                                    "tier": 3, "price": high, "reason": "MARKDOWN tier3 retest"})

            # ATH invalidation
            if self.state.markdown_ath_at_entry > 0 and high > self.state.markdown_ath_at_entry * 1.02:
                acts = self._close_short_actions(high, ts, "ath_invalidation", _tg)
                actions.extend(acts)
                self._reset_to_dca(ts, _tg)
                return actions

            self._save_short(short)

        # Determine CFGI-adjusted spring discount threshold, modulated by markdown_exit_style
        spring_threshold = self._cfgi_spring_threshold(cfgi_score)
        md_style = self.rebalancing_profile["markdown_exit_style"]
        if md_style == "moderate":
            spring_threshold *= 0.85  # 15% reduction
        elif md_style == "early":
            spring_threshold *= 0.70  # 30% reduction

        # Check if short closed → check discount for spring
        short = self._get_short()
        if short is None or short.closed:
            if self.state.exit_price > 0:
                discount = (self.state.exit_price - price) / self.state.exit_price * 100
                logger.info("SPRING entry: price discount=%.1f%%, CFGI=%s (threshold adjusted to %.0f%%, style=%s)",
                            discount, f"{cfgi_score:.0f}" if cfgi_score is not None else "N/A",
                            spring_threshold * 100, md_style)

                # Early style: also allow transition on early weakness (regime-based)
                early_weakness = (md_style == "early" and regime in ("DISTRIBUTION", "CHOPPY", "EXTREME")
                                  and discount >= spring_threshold * 100 * 0.7)

                if not self.config.auto_rotation:
                    logger.info("%s: MARKDOWN discount %.1f%% but auto_rotation OFF — no spring", self.symbol, discount)
                else:
                    can_rebalance = self._check_rebalance_cooldown() and self._check_rebalance_limit()

                    if can_rebalance and (discount >= spring_threshold * 100 or early_weakness):
                        self._record_rebalance()
                        self._transition_to_spring(price, ts, cash, _tg)

                if price >= self.state.markdown_entry_price * 0.95:
                    # Price recovered — false exit
                    _tg(f"\u21a9 {self.symbol}: MARKDOWN\u2192DCA (price recovered, false exit)")
                    self._reset_to_dca(ts, _tg)
        else:
            # Check spring discount while short still open
            if self.state.exit_price > 0:
                discount = (self.state.exit_price - price) / self.state.exit_price * 100
                if not self.config.auto_rotation:
                    logger.info("%s: MARKDOWN+short discount %.1f%% but auto_rotation OFF", self.symbol, discount)
                else:
                    early_weakness = (md_style == "early" and regime in ("DISTRIBUTION", "CHOPPY", "EXTREME")
                                      and discount >= spring_threshold * 100 * 0.7)
                    can_rebalance = self._check_rebalance_cooldown() and self._check_rebalance_limit()
                    if can_rebalance and (discount >= spring_threshold * 100 or early_weakness):
                        acts = self._close_short_actions(price, ts, "spring_transition", _tg)
                        actions.extend(acts)
                        self._record_rebalance()
                        self._transition_to_spring(price, ts, cash, _tg)

        return actions

    # ── SPRING Phase Processing ───────────────────────────────

    def _process_spring(self, price, high, low, ts, regime, cash, _tg) -> list:
        actions = []
        self.state.spring_candle_count += 1

        # Discount-based deployment
        if self.state.spring_phase_cash > 0 and self.state.exit_price > 0:
            discount_pct = (self.state.exit_price - price) / self.state.exit_price * 100

            tier = 0
            target_deploy_pct = 0.0
            if discount_pct >= self.config.spring_tier3_discount:
                tier = 3
                target_deploy_pct = self.config.spring_tier3_deploy
            elif discount_pct >= self.config.spring_tier2_discount:
                tier = 2
                target_deploy_pct = self.config.spring_tier2_deploy
            elif discount_pct >= self.config.spring_tier1_discount:
                tier = 1
                target_deploy_pct = self.config.spring_tier1_deploy

            if tier > 0 and tier > self.state.spring_highest_tier:
                target_amount = self.state.spring_phase_cash * target_deploy_pct
                remaining = target_amount - self.state.spring_deployed
                if remaining > 50 and remaining <= cash:
                    fee = remaining * self.taker_fee
                    qty = (remaining - fee) / price
                    entry = {
                        "tier": tier, "price": price, "qty": qty,
                        "cost": remaining, "fee": fee, "time": ts,
                        "closed": False, "half_cut": False,
                    }
                    self.state.spring_entries.append(entry)
                    self.state.spring_deployed += remaining
                    self.state.spring_highest_tier = tier
                    if self.state.spring_entry_price == 0:
                        self.state.spring_entry_price = price
                    actions.append({
                        "type": "buy", "symbol": self.symbol,
                        "amount": remaining, "price": price,
                        "reason": f"SPRING tier{tier} ({discount_pct:.1f}% discount)",
                    })
                    _tg(f"🌱 {self.symbol}: SPRING T{tier} — ${remaining:.0f} at ${price:.2f} ({discount_pct:.1f}% discount)")

        # Check existing spring entries for TP / false spring
        for entry in self.state.spring_entries:
            if entry.get("closed"):
                continue
            ep = entry["price"]

            # TP
            tp_price = ep * (1 + self.config.spring_tp_pct / 100)
            if high >= tp_price:
                pnl = (tp_price - ep) * entry["qty"]
                entry["closed"] = True
                entry["close_price"] = tp_price
                entry["close_time"] = ts
                entry["pnl"] = pnl
                self.state.spring_pnl += pnl
                self.state.spring_deployed -= entry["cost"]
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": entry["qty"], "price": tp_price,
                    "reason": f"SPRING TP tier{entry['tier']}",
                })
                continue

            # False spring
            drop_pct = (ep - low) / ep * 100
            if drop_pct >= self.config.spring_false_drop_pct * 2:
                pnl = (low - ep) * entry["qty"]
                entry["closed"] = True
                entry["close_price"] = low
                entry["close_time"] = ts
                entry["pnl"] = pnl
                self.state.spring_pnl += pnl
                self.state.spring_deployed -= entry["cost"]
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": entry["qty"], "price": low,
                    "reason": f"FALSE SPRING full cut tier{entry['tier']}",
                })
            elif drop_pct >= self.config.spring_false_drop_pct and not entry.get("half_cut"):
                half_qty = entry["qty"] / 2
                half_cost = entry["cost"] / 2
                pnl = (low - ep) * half_qty
                entry["qty"] = half_qty
                entry["cost"] = half_cost
                entry["half_cut"] = True
                self.state.spring_pnl += pnl
                self.state.spring_deployed -= half_cost
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": half_qty, "price": low,
                    "reason": f"FALSE SPRING half cut tier{entry['tier']}",
                })

        # Check recovery → MARKUP
        if self.state.spring_entry_price > 0:
            recovery_pct = (price - self.state.spring_entry_price) / self.state.spring_entry_price * 100
            if recovery_pct >= self.config.spring_recovery_pct and regime in ("TRENDING", "MILD_TREND", "ACCUMULATION"):
                # Close remaining spring entries
                for entry in self.state.spring_entries:
                    if not entry.get("closed"):
                        pnl = (price - entry["price"]) * entry["qty"]
                        entry["closed"] = True
                        entry["close_price"] = price
                        entry["close_time"] = ts
                        entry["pnl"] = pnl
                        self.state.spring_pnl += pnl
                        self.state.spring_deployed -= entry["cost"]
                        actions.append({
                            "type": "sell", "symbol": self.symbol,
                            "amount": entry["qty"], "price": price,
                            "reason": "SPRING close for MARKUP transition",
                        })
                self._transition_to_markup(price, ts, cash, _tg)
                return actions

        # Timeout
        timeout_candles = self.config.spring_timeout_days * 24  # assuming 1h candles
        if self.state.spring_candle_count >= timeout_candles:
            _tg(f"⏰ {self.symbol}: SPRING TIMEOUT ({self.config.spring_timeout_days}d) → DCA")
            for entry in self.state.spring_entries:
                if not entry.get("closed"):
                    pnl = (price - entry["price"]) * entry["qty"]
                    entry["closed"] = True
                    entry["close_price"] = price
                    entry["close_time"] = ts
                    entry["pnl"] = pnl
                    self.state.spring_pnl += pnl
                    actions.append({
                        "type": "sell", "symbol": self.symbol,
                        "amount": entry["qty"], "price": price,
                        "reason": "SPRING timeout close",
                    })
            self._reset_to_dca(ts, _tg)

        return actions

    # ── MARKUP Phase Processing ───────────────────────────────

    def _process_markup(self, price, high, low, ts, ts_ms, regime, daily_score, cfgi_score, cash, _tg) -> list:
        actions = []
        mp = self._get_markup()
        if mp is None:
            self._reset_to_dca(ts, _tg)
            return actions

        # Track high
        if high > mp.trail_high:
            mp.trail_high = high

        # Check conductor EXIT signal → unwind
        if self._conductor.should_exit(ts_ms, price, cfgi_score or 50.0):
            # Close markup position
            pnl = (price - mp.avg_entry) * mp.qty
            self.state.markup_pnl += pnl
            actions.append({
                "type": "sell", "symbol": self.symbol,
                "amount": mp.qty, "price": price,
                "reason": "MARKUP→EXIT conductor signal",
            })
            _tg(f"🚨 {self.symbol}: MARKUP→EXIT (score: {daily_score:.0f})")
            self.state.markup_position = None
            self.state.markup_phases += 1
            # Transition to EXIT
            # We need deals info but markup has its own position — create exit lots from markup
            self.state.phase = LifecyclePhase.EXIT
            self.state.exit_price = price
            self.state.exit_entry_time = ts
            self.state.exit_candles_elapsed = 0
            self.state.exit_local_low = price
            self.state.exit_lots = []  # No lots to unwind — we just sold the markup
            self.state.exit_lots_sold = 0
            self.state.exit_lots_total = 0
            self.state.exit_committed = False
            self.state.exit_realized_pnl = 0.0
            self.state.exit_phases += 1
            # Since we already sold, go straight to MARKDOWN
            if self.config.short_enabled:
                self._transition_to_markdown(price, ts, cash + price * mp.qty, _tg)
            else:
                self._transition_to_spring(price, ts, cash + price * mp.qty, _tg)
            return actions

        # Pullback additions
        if mp.adds < self.config.markup_max_adds and mp.trail_high > 0:
            pullback_pct = (mp.trail_high - price) / mp.trail_high * 100
            if (pullback_pct >= self.config.markup_pullback_pct
                    and price > mp.avg_entry
                    and price < mp.last_add_price * 0.97):
                add_amount = mp.phase_cash * self.config.markup_pullback_deploy_pct
                if 50 < add_amount <= cash:
                    fee = add_amount * self.taker_fee
                    add_qty = (add_amount - fee) / price
                    total_cost = mp.cost + add_amount
                    total_qty = mp.qty + add_qty
                    mp.avg_entry = total_cost / total_qty if total_qty > 0 else price
                    mp.qty = total_qty
                    mp.cost = total_cost
                    mp.adds += 1
                    mp.last_add_price = price
                    actions.append({
                        "type": "buy", "symbol": self.symbol,
                        "amount": add_amount, "price": price,
                        "reason": f"MARKUP pullback add #{mp.adds}",
                    })
                    _tg(f"📈 {self.symbol}: MARKUP ADD #{mp.adds} — ${add_amount:.0f} at ${price:.2f} (pullback {pullback_pct:.1f}%)")

        self._save_markup(mp)
        return actions

    # ── Phase Transitions ─────────────────────────────────────

    def _transition_to_exit(self, price, ts, daily_score, cfgi_score, deals, _tg) -> list:
        """DCA → EXIT"""
        actions = []
        self.state.phase = LifecyclePhase.EXIT
        self.state.exit_price = price
        self.state.exit_entry_time = ts
        self.state.exit_candles_elapsed = 0
        self.state.exit_local_low = price
        self.state.exit_committed = False
        self.state.exit_invalidated = False
        self.state.exit_realized_pnl = 0.0
        self.state.exit_lots_sold = 0
        self.state.exit_phases += 1
        self.state.commitment_start = ""

        # Extract lots from active deals
        exit_lots = []
        for sym, deal in deals.items():
            if sym != self.symbol:
                continue
            for lot in deal.unsold_lots:
                trail_price = price * (1 - self.config.initial_trail_pct / 100)
                el = ExitLot(
                    lot_id=lot.lot_id,
                    buy_price=lot.buy_price,
                    qty=lot.qty,
                    cost_usd=lot.cost_usd,
                    buy_time=lot.buy_time,
                    unrealized_pnl_pct=(price - lot.buy_price) / lot.buy_price * 100,
                    trailing_stop=trail_price,
                )
                exit_lots.append(el)

        exit_lots.sort(key=lambda l: l.unrealized_pnl_pct, reverse=True)
        self.state.exit_lots = [l.to_dict() for l in exit_lots]
        self.state.exit_lots_total = len(exit_lots)

        cfgi_str = f", CFGI: {cfgi_score:.0f}" if cfgi_score else ""
        _tg(f"🔄 {self.symbol}: DCA → EXIT (score: {daily_score:.0f}{cfgi_str}, {len(exit_lots)} lots)")
        actions.append({"type": "phase_change", "from": "DCA", "to": "EXIT", "price": price})
        return actions

    def _transition_to_markdown(self, price, ts, cash, _tg):
        """EXIT → MARKDOWN"""
        self.state.phase = LifecyclePhase.MARKDOWN
        self.state.markdown_phases += 1
        self.state.markdown_entry_price = price
        self.state.markdown_ath_at_entry = self._conductor._price_ath

        # Deploy tier 1 short
        deploy = cash * self.config.short_tier1_deploy
        if deploy > 100:
            qty = deploy / price
            sl = price * (1 + self.config.short_sl_pct / 100)
            short = ShortPosition(
                total_qty=qty, total_margin=deploy, avg_entry=price,
                entries=1, first_entry_price=price,
                trail_low=price, sl_price=sl,
            )
            self.state.short_position = short.to_dict()

        deploy_pct = int(self.config.deploy_pct * 100)
        _tg(f"📉 {self.symbol}: EXIT → MARKDOWN (deploying {deploy_pct}% shorts at ${price:,.0f})")

    def _transition_to_spring(self, price, ts, cash, _tg):
        """MARKDOWN/EXIT → SPRING"""
        self.state.phase = LifecyclePhase.SPRING
        self.state.spring_phases += 1
        self.state.spring_entries = []
        self.state.spring_entry_price = 0.0
        self.state.spring_deployed = 0.0
        self.state.spring_phase_cash = cash
        self.state.spring_highest_tier = 0
        self.state.spring_candle_count = 0

        discount = (self.state.exit_price - price) / self.state.exit_price * 100 if self.state.exit_price > 0 else 0
        _tg(f"🌱 {self.symbol}: → SPRING ({discount:.0f}% discount from exit)")

    def _transition_to_markup(self, price, ts, cash, _tg):
        """SPRING → MARKUP"""
        deploy_amount = cash * self.config.markup_deploy_pct
        if deploy_amount < 100:
            self._reset_to_dca(ts, _tg)
            return

        fee = deploy_amount * self.taker_fee
        qty = (deploy_amount - fee) / price

        mp = MarkupPosition(
            qty=qty, cost=deploy_amount, avg_entry=price,
            trail_high=price, trail_stop=price * (1 - self.config.markup_trail_pct / 100),
            phase_cash=cash, entry_price=price,
        )
        self.state.markup_position = mp.to_dict()
        self.state.phase = LifecyclePhase.MARKUP
        self.state.markup_phases += 1

        _tg(f"🚀 {self.symbol}: → MARKUP (${deploy_amount:.0f} at ${price:,.2f})")

    def _reset_to_dca(self, ts, _tg):
        """Reset to DCA phase."""
        old_phase = self.state.phase.value
        self.state.phase = LifecyclePhase.DCA
        self.state.exit_lots = []
        self.state.short_position = None
        self.state.spring_entries = []
        self.state.spring_deployed = 0.0
        self.state.spring_highest_tier = 0
        self.state.spring_candle_count = 0
        self.state.markup_position = None
        self.state.commitment_start = ""
        self.state.commitment_score_sustained = False
        logger.info("%s → DCA from %s", self.symbol, old_phase)

    # ── Rebalancing Aggressiveness ────────────────────────────

    @property
    def rebalancing_profile(self) -> dict:
        """Resolve current rebalancing mode to its parameter dict."""
        mode = self.config.rebalancing_mode.lower()
        return REBALANCING_PROFILES_BY_NAME.get(mode, REBALANCING_PROFILES_BY_NAME["balanced"])

    def _check_rebalance_cooldown(self) -> bool:
        """Returns True if cooldown has elapsed since last rebalance."""
        if not self.state.last_rebalance_time:
            return True
        try:
            last = datetime.fromisoformat(self.state.last_rebalance_time.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            hours = (now - last).total_seconds() / 3600
            return hours >= self.rebalancing_profile["cooldown_hours"]
        except Exception:
            return True

    def _check_rebalance_limit(self) -> bool:
        """Returns True if under daily rebalance max."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.rebalance_day != today:
            return True  # new day, counter will reset on record
        return self.state.rebalances_today < self.rebalancing_profile["max_rebalances_day"]

    def _record_rebalance(self):
        """Record a rebalance event — updates time and daily counter."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if self.state.rebalance_day != today:
            self.state.rebalance_day = today
            self.state.rebalances_today = 0
        self.state.last_rebalance_time = now.isoformat(timespec="seconds")
        self.state.rebalances_today += 1
        logger.info("Rebalance recorded: %d/%d today (mode=%s)",
                     self.state.rebalances_today,
                     self.rebalancing_profile["max_rebalances_day"],
                     self.config.rebalancing_mode)

    def rebalance_cooldown_remaining(self) -> float:
        """Hours remaining until next rebalance allowed. 0 if ready."""
        if not self.state.last_rebalance_time:
            return 0.0
        try:
            last = datetime.fromisoformat(self.state.last_rebalance_time.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            elapsed = (now - last).total_seconds() / 3600
            remaining = self.rebalancing_profile["cooldown_hours"] - elapsed
            return max(0.0, remaining)
        except Exception:
            return 0.0

    # ── CFGI Helpers ─────────────────────────────────────────

    def _cfgi_spring_threshold(self, cfgi_score: Optional[float]) -> float:
        """Return spring discount threshold (as fraction) adjusted by CFGI."""
        if cfgi_score is None:
            return self.config.cfgi_spring_discount_normal
        if cfgi_score <= self.config.cfgi_spring_extreme_fear:
            return self.config.cfgi_spring_discount_reduced
        if cfgi_score <= self.config.cfgi_spring_fear:
            return self.config.cfgi_spring_discount_normal
        if cfgi_score > self.config.cfgi_spring_neutral:
            return self.config.cfgi_spring_discount_raised
        return self.config.cfgi_spring_discount_normal

    # ── Short Position Helpers ────────────────────────────────

    def _get_short(self) -> Optional[ShortPosition]:
        if self.state.short_position:
            return ShortPosition.from_dict(self.state.short_position)
        return None

    def _save_short(self, short: ShortPosition):
        self.state.short_position = short.to_dict()

    def _add_short_tier(self, price, ts, tier, target_deploy_pct, cash):
        short = self._get_short()
        if not short:
            return
        total_target = (cash + short.total_margin) * target_deploy_pct
        additional = total_target - short.total_margin
        if additional < 50 or additional > cash:
            additional = min(additional, cash - 100)
            if additional < 50:
                return
        add_qty = additional / price
        short.total_qty += add_qty
        short.total_margin += additional
        short.avg_entry = short.total_margin / short.total_qty if short.total_qty > 0 else price
        short.entries = tier
        self._save_short(short)

    def _close_short_actions(self, price, ts, reason, _tg) -> list:
        actions = []
        short = self._get_short()
        if short and not short.closed:
            pnl = (short.avg_entry - price) * short.total_qty - short.funding_cost
            short.closed = True
            short.close_price = price
            short.close_time = ts
            short.pnl = pnl
            self.state.short_pnl += pnl
            self._save_short(short)
            actions.append({
                "type": "close_short", "symbol": self.symbol,
                "price": price, "pnl": pnl, "reason": reason,
            })
            _tg(f"📈 {self.symbol}: SHORT CLOSE ({reason}) PnL=${pnl:.2f}")
        return actions

    # ── Markup Position Helpers ───────────────────────────────

    def _get_markup(self) -> Optional[MarkupPosition]:
        if self.state.markup_position:
            return MarkupPosition.from_dict(self.state.markup_position)
        return None

    def _save_markup(self, mp: MarkupPosition):
        self.state.markup_position = mp.to_dict()

    # ── Exit Lot Helpers ──────────────────────────────────────

    def _force_sell_exit_lots(self, price, ts, reason, _tg) -> list:
        actions = []
        for lot_d in self.state.exit_lots:
            if not lot_d.get("sold"):
                lot_d["sold"] = True
                lot_d["sell_price"] = price
                lot_d["sell_time"] = ts
                lot_d["sell_reason"] = reason
                lot_d["pnl"] = (price - lot_d["buy_price"]) * lot_d["qty"]
                self.state.exit_realized_pnl += lot_d["pnl"]
                actions.append({
                    "type": "sell", "symbol": self.symbol,
                    "amount": lot_d["qty"], "price": price,
                    "reason": f"EXIT {reason} (lot {lot_d['lot_id']})",
                })
        return actions

    # ── Equity Contribution ───────────────────────────────────

    def unrealized_value(self, price: float) -> float:
        """Calculate the market value of all lifecycle positions (for equity calc)."""
        value = 0.0

        # Open exit lots
        for lot_d in self.state.exit_lots:
            if not lot_d.get("sold"):
                value += lot_d["qty"] * price

        # Open spring entries
        for entry in self.state.spring_entries:
            if not entry.get("closed"):
                value += entry["qty"] * price

        # Markup position
        mp = self._get_markup()
        if mp and mp.qty > 0:
            value += mp.qty * price

        # Short position (margin + unrealized)
        short = self._get_short()
        if short and not short.closed:
            value += short.total_margin + (short.avg_entry - price) * short.total_qty - short.funding_cost

        return value
