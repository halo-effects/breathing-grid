"""Spot DCA Backtest Engine V12 — Three-Engine Lifecycle Architecture.

Three purpose-built engines with clean handoffs, one conductor (daily TA scorer):

  1. DCA Engine (V8/V9 base) — Accumulation/markup. Normal grid trading.
  2. Exit Engine (NEW) — Distribution/markdown. Methodical unwind:
     - Rally selling (sell most profitable lot on bounces)
     - Trailing stops that tighten over time
     - Time-based urgency escalation
     - Optional short rotation after 50% lots unwound
  3. Spring Engine (NEW) — Bottom detection. Tiered capital deployment:
     - Tier 1 (score 60-74): Deploy 25% capital
     - Tier 2 (score 75-89): Deploy 50% capital
     - Tier 3 (score 90+): Deploy 75% capital

Conductor: Daily TA scorer (resampled from 1h candles).
  - 1h never fires EXIT (max 42-46 at tops)
  - Daily fires EXIT at both known ETH tops (scores 61-65)
  - RSI divergence only appears on daily timeframe

See: projects/ait-product/v12-lifecycle-engine-spec.md
"""
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .backtest_engine_v9 import SpotBacktestEngineV9
from .backtest_engine_v3 import BacktestResult, Lot, TradeLogEntry
from .distribution_scorer import DistributionPhase, DistributionResult
from .ta_top_scorer import TATopScorer
from .reversal_detector import ReversalDetector, ReversalDetectorConfig
from ..regime_detector import classify_regime_v2
from ..indicators import compute_all as compute_all_indicators

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  V12 Lifecycle Phases
# ═══════════════════════════════════════════════════════════════════

class LifecyclePhase(str, Enum):
    DCA = "DCA"           # Normal grid trading — ranging/choppy (Engine 1)
    EXIT = "EXIT"         # Methodical unwind — distribution (Engine 2)
    MARKDOWN = "MARKDOWN" # Ride shorts down — confirmed decline (Engine 3a)
    SPRING = "SPRING"     # Bottom deployment — accumulation (Engine 3b)
    MARKUP = "MARKUP"     # Trend riding — confirmed bull after spring (Engine 4)


# ═══════════════════════════════════════════════════════════════════
#  Daily TA Scorer Conductor
# ═══════════════════════════════════════════════════════════════════

class DailyScorerConductor:
    """Runs TA scorer on daily-resampled data. Triggers phase transitions.
    
    Caches daily candles and only re-scores when a new daily bar completes.
    """
    
    def __init__(self, exit_threshold: float = 50.0, mcap_ath_pct: float = 0.25, ath: float = 0.0,
                 symbol: str = ""):
        self._ta_scorer = TATopScorer(min_lookback=50)  # 50 daily bars
        self._exit_threshold = exit_threshold
        self._mcap_ath_pct = mcap_ath_pct
        self._symbol = symbol
        self._known_historical_ath = ath  # Fixed historical ATH — never updated by price action
        self._in_price_discovery = False  # Set by should_exit when price > known ATH
        
        # Reversal detector (V12e/f) — post-peak drop detection
        self._reversal_detector = ReversalDetector(
            ReversalDetectorConfig(ath=ath if ath > 0 else 1e12)
        )
        
        # CFGI per-coin history (V12f)
        self._cfgi_history: Dict[str, float] = self._load_cfgi_history(symbol)
        
        # Daily data (built from accumulated 1h)
        self._daily_df: Optional[pd.DataFrame] = None
        self._daily_regimes: Optional[pd.Series] = None
        self._daily_ready: bool = False
        
        # Caching: only re-score when new daily bar appears
        self._last_scored_daily_idx: int = -1
        self._cached_score: float = 0.0
        self._cached_result: Optional[DistributionResult] = None
        self._cached_fg_exit_score: float = 0.0
        
        # Price ATH tracking
        self._price_ath: float = 0.0
    
    def _load_cfgi_history(self, symbol: str) -> dict:
        """Load cached CFGI history for symbol. Returns {date_str: score} or empty dict."""
        import json
        from pathlib import Path
        # Extract token from symbol like "ETH/USDT" -> "ETH"
        token = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        cache_file = Path(__file__).resolve().parent / "data" / "cfgi_cache" / f"{token}_cfgi_daily.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                logger.info("Loaded %d CFGI entries for %s", len(data), token)
                return data
            except Exception as e:
                logger.warning("Failed to load CFGI cache for %s: %s", token, e)
        return {}

    def set_price_ath(self, ath: float):
        self._price_ath = ath
    
    def prepare(self, df_1h: pd.DataFrame):
        """Resample accumulated 1h data to daily. Call before each chunk."""
        df = df_1h.copy()
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("dt")
        
        daily = df.resample("1D").agg({
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["timestamp"]).reset_index(drop=True)
        
        if len(daily) > 50:
            daily = compute_all_indicators(daily)
            self._daily_regimes = classify_regime_v2(daily, "1h")
            self._daily_df = daily
            self._daily_ready = True
            # Reset scorer cache (new data)
            self._ta_scorer._cache_valid = False
            logger.info("  Conductor: Prepared %d daily candles", len(daily))
        else:
            self._daily_ready = False
            logger.warning("  Conductor: Not enough daily candles (%d)", len(daily))
    
    def _score_pi_cycle(self, daily_idx: int) -> float:
        """Pi Cycle Top: 111-SMA crosses above 2×350-SMA → 25 pts.
        Gap <2% (approaching cross) → 15 pts.
        BTC-specific but works as supplemental for any coin."""
        if self._daily_df is None or daily_idx < 350:
            return 0.0
        closes = self._daily_df["close"].values
        sma111 = float(np.mean(closes[daily_idx - 110:daily_idx + 1]))
        sma350 = float(np.mean(closes[daily_idx - 349:daily_idx + 1]))
        if sma350 == 0:
            return 0.0
        threshold = 2.0 * sma350
        if sma111 >= threshold:
            return 25.0  # Cross confirmed — strong top signal
        gap_pct = (threshold - sma111) / threshold * 100
        if gap_pct < 2.0:
            return 15.0  # Approaching cross — early warning
        if gap_pct < 5.0:
            return 5.0   # Getting close
        return 0.0
    
    def score_at(self, ts_1h_ms: int, current_price: float, fg_value: float = 0.0,
                 coin_cfgi: float = None) -> float:
        """Get daily TA score for the given 1h timestamp.
        
        Only re-computes when a new daily bar has completed.
        Returns the cached score otherwise.
        
        Args:
            coin_cfgi: Per-coin CFGI score from cfgi.io (0-100). If provided,
                       used INSTEAD of market-wide fg_value for exit scoring.
                       Live trading only — backtests use fg_value from historical files.
        """
        if not self._daily_ready or self._daily_df is None:
            return 0.0
        
        # Update price ATH
        if current_price > self._price_ath:
            self._price_ath = current_price
        
        # Find which daily bar this 1h candle belongs to
        dt = pd.Timestamp(ts_1h_ms, unit="ms", tz="UTC")
        day_start_ms = int(dt.normalize().timestamp() * 1000)
        
        # Find daily index
        daily_ts = self._daily_df["timestamp"].values
        diffs = np.abs(daily_ts - day_start_ms)
        daily_idx = int(np.argmin(diffs))
        
        if diffs[daily_idx] > 86400000:  # > 24h away
            return self._cached_score
        
        # Only re-score if new daily bar
        if daily_idx == self._last_scored_daily_idx:
            return self._cached_score
        
        if daily_idx < 50:
            return 0.0
        
        self._last_scored_daily_idx = daily_idx
        regime = self._daily_regimes.iloc[daily_idx] if daily_idx < len(self._daily_regimes) else "UNKNOWN"
        
        result = self._ta_scorer.score(
            self._daily_df, daily_idx, regime, fg_value, self._daily_regimes
        )
        
        # Add Pi Cycle score (supplemental — especially useful for BTC)
        pi_score = self._score_pi_cycle(daily_idx)
        
        # F&G / CFGI scoring for distribution exits (V12e/f: graduated tiers)
        # V12f: Auto-lookup per-coin CFGI from cached history
        if coin_cfgi is None and self._cfgi_history:
            date_str = dt.strftime("%Y-%m-%d")
            if date_str in self._cfgi_history:
                coin_cfgi = self._cfgi_history[date_str]
        
        # If per-coin CFGI is available, use it instead of market-wide F&G
        if coin_cfgi is not None:
            _sentiment_val = coin_cfgi
            _sentiment_src = "CFGI(coin)"
        else:
            _sentiment_val = fg_value
            _sentiment_src = "F&G(market)"
        
        fg_exit_score = 0
        if _sentiment_val >= 90:
            fg_exit_score = 25   # Extreme greed — very strong exit signal
        elif _sentiment_val >= 80:
            fg_exit_score = 15   # Greed — moderate exit signal
        elif _sentiment_val >= 75:
            fg_exit_score = 10   # V12e: High greed — mild exit signal
        elif _sentiment_val >= 70:
            fg_exit_score = 5    # V12e: Elevated greed — slight exit signal
        elif _sentiment_val <= 10:
            fg_exit_score = -20  # Panic — strongly suppress exits
        elif _sentiment_val <= 20:
            fg_exit_score = -10  # Extreme fear — suppress false exit signals
        
        logger.debug("  Conductor: sentiment=%s val=%.0f exit_score=%d", _sentiment_src, _sentiment_val, fg_exit_score)
        
        # V12e: Reversal detector — post-peak drop detection
        reversal_score = self._reversal_detector.score(current_price, ts_1h_ms, fg_value)
        
        # ATH proximity bonus (graduated — separate from ATH gate)
        ath_proximity_score = 0.0
        if self._price_ath > 0 and current_price > 0:
            distance_from_ath = (self._price_ath - current_price) / self._price_ath
            if distance_from_ath <= 0.10:    # Within 10% of ATH
                ath_proximity_score = 10.0
            elif distance_from_ath <= 0.15:  # Within 15% of ATH
                ath_proximity_score = 5.0
        
        total_score = result.score + pi_score + fg_exit_score + reversal_score + ath_proximity_score
        
        self._cached_score = total_score
        self._cached_result = result
        self._cached_fg_exit_score = fg_exit_score
        
        if total_score >= 30:
            logger.info("  Conductor: daily_score=%.0f (TA=%.0f pi=%.0f fg=%.0f rev=%.0f ath_prox=%.0f RSI=%.0f vol=%.0f wick=%.0f mom=%.0f) at daily_idx=%d",
                       total_score, result.score, pi_score, fg_exit_score, reversal_score, ath_proximity_score,
                       result.rsi_divergence_score, result.volume_exhaustion_score,
                       result.upper_wick_rejection_score, result.momentum_stall_score, daily_idx)
        
        return total_score
    
    def should_exit(self, ts_1h_ms: int, current_price: float, fg_value: float = 0.0) -> bool:
        """Check if daily scorer triggers EXIT phase transition."""
        score = self.score_at(ts_1h_ms, current_price, fg_value)
        
        if score < self._exit_threshold:
            return False
        
        # Price discovery detection: if current price > known historical ATH,
        # the ATH gate is meaningless (price IS the ATH). In this case,
        # we flag that weekly confirmation is REQUIRED (handled by caller).
        known_ath = self._known_historical_ath  # Set from KNOWN_ATH dict, never updated
        if known_ath > 0 and current_price > known_ath:
            # In price discovery — ATH gate is useless, signal that weekly must confirm
            self._in_price_discovery = True
            logger.info("  📈 PRICE DISCOVERY: %s at $%.2f > known ATH $%.0f — weekly confirmation required",
                       self._symbol, current_price, known_ath)
            return True  # Score passed, but caller must check weekly
        
        self._in_price_discovery = False
        
        # ATH gate: price must be within mcap_ath_pct of ATH
        if self._price_ath > 0:
            distance = (self._price_ath - current_price) / self._price_ath
            if distance > self._mcap_ath_pct:
                return False
        
        return True
    
    def weekly_confirms_exit(self, df_1h: pd.DataFrame, direction: str) -> bool:
        """Check if weekly timeframe confirms an EXIT signal.
        
        Args:
            df_1h: Accumulated 1h candle data
            direction: 'distribution' (top → markdown) or 'breakout' (bottom → markup)
        
        Returns:
            True if weekly confirms, False if weekly vetoes the exit.
        """
        if df_1h is None or len(df_1h) < 168:  # Need at least 1 week
            return True  # Not enough data, don't veto
        
        try:
            df = df_1h.copy()
            df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("dt")
            
            weekly = df.resample("1W").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index(drop=True)
            
            if len(weekly) < 14:
                return True  # Not enough weekly bars
            
            closes = weekly["close"].values
            
            # Weekly RSI-14
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss == 0:
                weekly_rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                weekly_rsi = 100 - (100 / (1 + rs))
            
            # Weekly SMA50 direction
            if len(closes) >= 50:
                sma50_now = np.mean(closes[-50:])
                sma50_prev = np.mean(closes[-51:-1]) if len(closes) >= 51 else sma50_now
                sma50_rising = sma50_now > sma50_prev
                price_above_sma50 = closes[-1] > sma50_now
            else:
                sma50_rising = True
                price_above_sma50 = True
            
            # Weekly momentum (ROC-4: 4-week rate of change)
            if len(closes) >= 5:
                weekly_momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            else:
                weekly_momentum = 0.0
            
            # Weekly RSI direction (rising or falling)
            # Calculate RSI for previous week too
            if len(closes) >= 16:
                deltas_prev = np.diff(closes[-16:-1])
                gains_prev = np.where(deltas_prev > 0, deltas_prev, 0)
                losses_prev = np.where(deltas_prev < 0, -deltas_prev, 0)
                avg_gain_prev = np.mean(gains_prev[-14:])
                avg_loss_prev = np.mean(losses_prev[-14:])
                if avg_loss_prev == 0:
                    weekly_rsi_prev = 100.0
                else:
                    rs_prev = avg_gain_prev / avg_loss_prev
                    weekly_rsi_prev = 100 - (100 / (1 + rs_prev))
                rsi_rising = weekly_rsi > weekly_rsi_prev
            else:
                rsi_rising = True
                weekly_rsi_prev = weekly_rsi
            
            if direction == "distribution":
                # For distribution EXIT (entering markdown):
                # Weekly RSI should be >60 AND momentum stalling/negative
                confirms = weekly_rsi > 60 and weekly_momentum < 5.0
                logger.info("  📊 WEEKLY CHECK (distribution): RSI=%.1f, momentum=%.1f%%, confirms=%s",
                           weekly_rsi, weekly_momentum, confirms)
                return confirms
            
            elif direction == "breakout":
                # For breakout EXIT (entering markup from DCA):
                # Weekly RSI should be rising AND price above weekly SMA50
                confirms = rsi_rising and price_above_sma50
                logger.info("  📊 WEEKLY CHECK (breakout): RSI=%.1f (rising=%s), above_SMA50=%s, confirms=%s",
                           weekly_rsi, rsi_rising, price_above_sma50, confirms)
                return confirms
            
            return True
        except Exception as e:
            logger.warning("  Weekly check failed: %s", e)
            return True  # Don't veto on error
    
    @property
    def last_score(self) -> float:
        return self._cached_score
    
    @property
    def last_result(self) -> Optional[DistributionResult]:
        return self._cached_result


# ═══════════════════════════════════════════════════════════════════
#  Exit Engine — Lot tracking for methodical unwind
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExitLot:
    """A lot being managed by the Exit Engine."""
    lot_id: int
    buy_price: float
    qty: float
    cost_usd: float
    buy_time: str
    unrealized_pnl_pct: float = 0.0  # Updated each candle
    trailing_stop: float = 0.0       # Price level
    sold: bool = False
    sell_price: float = 0.0
    sell_time: str = ""
    sell_reason: str = ""
    pnl: float = 0.0


@dataclass 
class ShortPosition:
    """Aggressive short position during Exit/Markdown phase.
    Supports tiered entries mirroring spring deployment."""
    total_qty: float = 0.0
    total_margin: float = 0.0
    avg_entry: float = 0.0
    entries: int = 0           # Number of tiers deployed
    first_entry_price: float = 0.0
    trail_low: float = float('inf')  # Lowest price (for trailing stop on upside)
    trail_stop: float = 0.0   # Stop price (above current — loss limit)
    sl_price: float = 0.0     # Hard stop loss
    closed: bool = False
    close_price: float = 0.0
    close_time: str = ""
    pnl: float = 0.0
    funding_cost: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  V12 Engine
# ═══════════════════════════════════════════════════════════════════

class SpotBacktestEngineV12(SpotBacktestEngineV9):
    """V12: Three-Engine Lifecycle Architecture.
    
    Inherits V9 (which inherits V8) for the DCA engine.
    Adds Exit Engine and Spring Engine as new phases.
    Daily TA scorer conducts phase transitions.
    """
    
    # Known ATH prices
    _KNOWN_ATH = {
        "ETH/USDT": 4878.0, "ETH/USDC": 4878.0,
        "BTC/USDT": 73750.0, "BTC/USDC": 73750.0,
        "SOL/USDT": 260.0, "SOL/USDC": 260.0,
        "HYPE/USDC": 35.0,
    }
    
    def __init__(self, **kwargs):
        # V12-specific params
        self._v12_exit_threshold: float = kwargs.pop("v12_exit_threshold", 50.0)
        self._v12_mcap_ath_pct: float = kwargs.pop("v12_mcap_ath_pct", 0.25)
        self._v12_commitment_hours: int = kwargs.pop("v12_commitment_hours", 48)
        
        # Exit Engine params
        self._v12_initial_trail_pct: float = kwargs.pop("v12_initial_trail_pct", 3.0)
        self._v12_trail_floor_pct: float = kwargs.pop("v12_trail_floor_pct", 1.5)
        self._v12_trail_tighten_per_day: float = kwargs.pop("v12_trail_tighten_per_day", 0.5)
        self._v12_rally_sell_pct: float = kwargs.pop("v12_rally_sell_pct", 1.5)
        self._v12_urgency_day_moderate: int = kwargs.pop("v12_urgency_day_moderate", 4)
        self._v12_urgency_day_aggressive: int = kwargs.pop("v12_urgency_day_aggressive", 7)
        self._v12_urgency_day_force: int = kwargs.pop("v12_urgency_day_force", 14)
        
        # Aggressive short deployment (mirror of spring — pile in at confirmed top)
        self._v12_short_enabled: bool = kwargs.pop("v12_short_enabled", True)
        self._v12_short_tier1_deploy: float = kwargs.pop("v12_short_tier1_deploy", 0.60)  # 60% immediately
        self._v12_short_tier2_deploy: float = kwargs.pop("v12_short_tier2_deploy", 0.80)  # 80% on bounce
        self._v12_short_tier3_deploy: float = kwargs.pop("v12_short_tier3_deploy", 0.90)  # 90% on retest
        self._v12_short_tier2_bounce_pct: float = kwargs.pop("v12_short_tier2_bounce_pct", 3.0)  # 3% bounce = add
        self._v12_short_tier3_retest_pct: float = kwargs.pop("v12_short_tier3_retest_pct", 2.0)  # Within 2% of entry = retest
        self._v12_short_trail_pct: float = kwargs.pop("v12_short_trail_pct", 10.0)  # Trailing stop (upside)
        self._v12_short_sl_pct: float = kwargs.pop("v12_short_sl_pct", 15.0)  # Hard stop loss
        self._v12_funding_rate_daily: float = kwargs.pop("v12_funding_rate_daily", 0.0003)
        
        # Spring Engine params — DISCOUNT-BASED (not indicator-based)
        # V12e PARAM SYNC: spring discounts tightened, TP widened (2025-02-22)
        self._v12_spring_tier1_discount: float = kwargs.pop("v12_spring_tier1_discount", 25.0)  # 25% below exit
        self._v12_spring_tier2_discount: float = kwargs.pop("v12_spring_tier2_discount", 28.0)  # was 35% — deploy earlier
        self._v12_spring_tier3_discount: float = kwargs.pop("v12_spring_tier3_discount", 35.0)  # was 45% — deploy earlier
        self._v12_spring_tier1_deploy: float = kwargs.pop("v12_spring_tier1_deploy", 0.25)
        self._v12_spring_tier2_deploy: float = kwargs.pop("v12_spring_tier2_deploy", 0.55)
        self._v12_spring_tier3_deploy: float = kwargs.pop("v12_spring_tier3_deploy", 0.75)
        self._v12_spring_tp_pct: float = kwargs.pop("v12_spring_tp_pct", 15.0)   # was 10% — wider TP for markup hold
        self._v12_spring_false_drop_pct: float = kwargs.pop("v12_spring_false_drop_pct", 15.0)
        self._v12_spring_recovery_pct: float = kwargs.pop("v12_spring_recovery_pct", 20.0)
        
        # Markup Engine params
        self._v12_markup_deploy_pct: float = kwargs.pop("v12_markup_deploy_pct", 0.65)       # Deploy 65% of cash on entry
        self._v12_markup_trail_pct: float = kwargs.pop("v12_markup_trail_pct", 10.0)          # Wide trailing stop
        self._v12_markup_trail_tighten_score: float = kwargs.pop("v12_markup_trail_tighten_score", 30.0)  # Tighten trail when daily score rises
        self._v12_markup_trail_tight_pct: float = kwargs.pop("v12_markup_trail_tight_pct", 5.0)  # Tightened trail %
        self._v12_markup_pullback_pct: float = kwargs.pop("v12_markup_pullback_pct", 5.0)     # Add on 5% pullback from high
        self._v12_markup_pullback_deploy_pct: float = kwargs.pop("v12_markup_pullback_deploy_pct", 0.15)  # Deploy 15% more on pullback
        self._v12_markup_max_adds: int = kwargs.pop("v12_markup_max_adds", 3)                 # Max pullback additions
        self._v12_markup_entry_from_spring: bool = kwargs.pop("v12_markup_entry_from_spring", True)  # Only enter from spring phase
        
        # Weekly veto on distribution exits (coin-configurable, default off for ETH compatibility)
        self._v12_weekly_dist_veto: bool = kwargs.pop("v12_weekly_dist_veto", False)
        
        # V12e PARAM SYNC: realistic fees + slippage for lifecycle trades (2025-02-22)
        # Slippage applied to market-like fills only (not limit/TP orders)
        self._v12_slippage_pct: float = kwargs.pop("v12_slippage_pct", 0.05)  # 5 bps default
        
        # Override V9's exit threshold to be unreachable (V12 conductor handles exits)
        kwargs.setdefault("dist_exit_threshold", 999)
        
        super().__init__(**kwargs)
        
        # ── Conductor ──
        ath = self._KNOWN_ATH.get(kwargs.get("symbol", ""), 0.0)
        self._conductor = DailyScorerConductor(
            exit_threshold=self._v12_exit_threshold,
            mcap_ath_pct=self._v12_mcap_ath_pct,
            ath=ath,
            symbol=kwargs.get("symbol", ""),
        )
        if ath > 0:
            self._conductor.set_price_ath(ath)
        
        # ── Lifecycle state ──
        self._lifecycle_phase: LifecyclePhase = LifecyclePhase.DCA
        
        # ── Exit Engine state ──
        self._exit_lots: List[ExitLot] = []
        self._exit_entry_time_ms: int = 0        # When EXIT phase started
        self._exit_entry_price: float = 0.0      # Price when EXIT fired (spring reference)
        self._exit_candles_elapsed: int = 0
        self._exit_local_low: float = float('inf')
        self._exit_lots_sold: int = 0
        self._exit_lots_total: int = 0
        self._exit_realized_pnl: float = 0.0
        self._exit_short: Optional[ShortPosition] = None
        self._exit_invalidated: bool = False       # Daily score dropped in first 48h
        self._exit_committed: bool = False         # Past commitment window
        
        # ── Spring Engine state ──
        self._spring_entries: List[dict] = []      # Spring lots placed
        self._spring_entry_price: float = 0.0
        self._spring_deployed: float = 0.0
        self._spring_phase_cash: float = 0.0       # Cash at spring entry
        self._spring_highest_tier: int = 0
        
        # ── Markdown Engine state ──
        self._markdown_entry_price: float = 0.0     # Price when markdown started (exit price)
        self._markdown_ath_at_entry: float = 0.0    # ATH when markdown started (for invalidation)
        
        # ── DCA freeze state (natural unwind on breakout) ──
        self._dca_frozen: bool = False              # When True, no new DCA deals/SOs
        self._dca_frozen_lots: List[dict] = []      # DCA lots unwinding naturally via TP
        
        # ── Markup Engine state ──
        self._markup_position_qty: float = 0.0     # Total qty held
        self._markup_position_cost: float = 0.0     # Total cost basis
        self._markup_avg_entry: float = 0.0         # Weighted avg entry price
        self._markup_trail_high: float = 0.0        # Highest price since entry
        self._markup_trail_stop: float = 0.0        # Current trailing stop level
        self._markup_adds: int = 0                   # Pullback additions made
        self._markup_last_add_price: float = 0.0    # Price of last addition
        self._markup_phase_cash: float = 0.0        # Cash at markup entry
        self._markup_entry_price: float = 0.0       # Price at markup entry
        
        # ── V12 metrics ──
        self._v12_exit_phases: int = 0
        self._v12_spring_phases: int = 0
        self._v12_rally_sells: int = 0
        self._v12_trail_stops: int = 0
        self._v12_urgency_closes: int = 0
        self._v12_short_pnl: float = 0.0
        self._v12_spring_pnl: float = 0.0
        self._v12_spring_deploys: int = 0
        self._v12_false_springs: int = 0
        self._v12_exit_pnl_preserved: float = 0.0  # How much PnL the exit engine saved
        self._v12_markdown_phases: int = 0
        self._v12_markup_phases: int = 0
        self._v12_markup_pnl: float = 0.0
        self._v12_markup_adds_total: int = 0
        self._v12_markup_trail_exits: int = 0
        self._v12_markup_conductor_exits: int = 0
        self._v12_breakout_entries: int = 0          # DCA→MARKUP breakout transitions
        self._v12_daily_scores: List[dict] = []     # For diagnostics
    
    @property
    def v12_params(self) -> dict:
        return {
            "v12_exit_threshold": self._v12_exit_threshold,
            "v12_mcap_ath_pct": self._v12_mcap_ath_pct,
            "v12_commitment_hours": self._v12_commitment_hours,
            "v12_initial_trail_pct": self._v12_initial_trail_pct,
            "v12_trail_floor_pct": self._v12_trail_floor_pct,
            "v12_rally_sell_pct": self._v12_rally_sell_pct,
            "v12_short_enabled": self._v12_short_enabled,
            "v12_short_tier1_deploy": self._v12_short_tier1_deploy,
            "v12_short_trail_pct": self._v12_short_trail_pct,
            "v12_short_sl_pct": self._v12_short_sl_pct,
            "v12_markup_deploy_pct": self._v12_markup_deploy_pct,
            "v12_markup_trail_pct": self._v12_markup_trail_pct,
            "v12_markup_pullback_pct": self._v12_markup_pullback_pct,
            "v12_markup_max_adds": self._v12_markup_max_adds,
            "v12_spring_tier1_discount": self._v12_spring_tier1_discount,
            "v12_spring_tier2_discount": self._v12_spring_tier2_discount,
            "v12_spring_tier3_discount": self._v12_spring_tier3_discount,
            "v12_spring_tp_pct": self._v12_spring_tp_pct,
            "v12_slippage_pct": self._v12_slippage_pct,
        }
    
    # ── State serialization ──────────────────────────────────────
    
    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["v12_lifecycle_phase"] = self._lifecycle_phase.value
        state["v12_exit_lots"] = [vars(l) for l in self._exit_lots]
        state["v12_exit_entry_time_ms"] = self._exit_entry_time_ms
        state["v12_exit_entry_price"] = self._exit_entry_price
        state["v12_exit_candles_elapsed"] = self._exit_candles_elapsed
        state["v12_exit_local_low"] = self._exit_local_low
        state["v12_exit_lots_sold"] = self._exit_lots_sold
        state["v12_exit_lots_total"] = self._exit_lots_total
        state["v12_exit_realized_pnl"] = self._exit_realized_pnl
        state["v12_exit_short"] = {
            "total_qty": self._exit_short.total_qty,
            "total_margin": self._exit_short.total_margin,
            "avg_entry": self._exit_short.avg_entry,
            "entries": self._exit_short.entries,
            "first_entry_price": self._exit_short.first_entry_price,
            "trail_low": self._exit_short.trail_low,
            "trail_stop": self._exit_short.trail_stop,
            "sl_price": self._exit_short.sl_price,
            "closed": self._exit_short.closed,
            "close_price": self._exit_short.close_price,
            "close_time": self._exit_short.close_time,
            "pnl": self._exit_short.pnl,
            "funding_cost": self._exit_short.funding_cost,
        } if self._exit_short else None
        state["v12_exit_invalidated"] = self._exit_invalidated
        state["v12_exit_committed"] = self._exit_committed
        state["v12_spring_entries"] = self._spring_entries
        state["v12_spring_entry_price"] = self._spring_entry_price
        state["v12_spring_deployed"] = self._spring_deployed
        state["v12_spring_phase_cash"] = self._spring_phase_cash
        state["v12_spring_highest_tier"] = self._spring_highest_tier
        # Markdown state
        state["v12_markdown_entry_price"] = self._markdown_entry_price
        state["v12_markdown_ath_at_entry"] = self._markdown_ath_at_entry
        state["v12_markdown_phases"] = self._v12_markdown_phases
        # Markup state
        state["v12_markup_position_qty"] = self._markup_position_qty
        state["v12_markup_position_cost"] = self._markup_position_cost
        state["v12_markup_avg_entry"] = self._markup_avg_entry
        state["v12_markup_trail_high"] = self._markup_trail_high
        state["v12_markup_trail_stop"] = self._markup_trail_stop
        state["v12_markup_adds"] = self._markup_adds
        state["v12_markup_last_add_price"] = self._markup_last_add_price
        state["v12_markup_phase_cash"] = self._markup_phase_cash
        state["v12_markup_entry_price"] = self._markup_entry_price
        # Metrics
        state["v12_dca_frozen"] = self._dca_frozen
        state["v12_markup_phases"] = self._v12_markup_phases
        state["v12_markup_pnl"] = self._v12_markup_pnl
        state["v12_markup_adds_total"] = self._v12_markup_adds_total
        state["v12_markup_trail_exits"] = self._v12_markup_trail_exits
        state["v12_markup_conductor_exits"] = self._v12_markup_conductor_exits
        state["v12_breakout_entries"] = self._v12_breakout_entries
        state["v12_exit_phases"] = self._v12_exit_phases
        state["v12_spring_phases"] = self._v12_spring_phases
        state["v12_rally_sells"] = self._v12_rally_sells
        state["v12_trail_stops"] = self._v12_trail_stops
        state["v12_urgency_closes"] = self._v12_urgency_closes
        state["v12_short_pnl"] = self._v12_short_pnl
        state["v12_spring_pnl"] = self._v12_spring_pnl
        state["v12_spring_deploys"] = self._v12_spring_deploys
        state["v12_false_springs"] = self._v12_false_springs
        state["v12_exit_pnl_preserved"] = self._v12_exit_pnl_preserved
        # Conductor state
        state["v12_conductor_price_ath"] = self._conductor._price_ath
        state["v12_conductor_last_scored_idx"] = self._conductor._last_scored_daily_idx
        state["v12_conductor_cached_score"] = self._conductor._cached_score
        # V12e: Reversal detector state
        rd = self._conductor._reversal_detector
        state["v12e_rd_gate_active"] = rd._gate_active
        state["v12e_rd_last_near_ath_ts"] = rd._last_near_ath_ts
        state["v12e_rd_rolling_high"] = rd._rolling_high
        state["v12e_rd_rolling_high_ts"] = rd._rolling_high_ts
        state["v12e_rd_fg_history"] = rd._fg_history
        return state
    
    def restore_state(self, state: dict):
        super().restore_state(state)
        self._lifecycle_phase = LifecyclePhase(state.get("v12_lifecycle_phase", "DCA"))
        # Restore exit lots
        self._exit_lots = []
        for ld in state.get("v12_exit_lots", []):
            self._exit_lots.append(ExitLot(**ld))
        self._exit_entry_time_ms = state.get("v12_exit_entry_time_ms", 0)
        self._exit_entry_price = state.get("v12_exit_entry_price", 0.0)
        self._exit_candles_elapsed = state.get("v12_exit_candles_elapsed", 0)
        self._exit_local_low = state.get("v12_exit_local_low", float('inf'))
        self._exit_lots_sold = state.get("v12_exit_lots_sold", 0)
        self._exit_lots_total = state.get("v12_exit_lots_total", 0)
        self._exit_realized_pnl = state.get("v12_exit_realized_pnl", 0.0)
        # Restore short
        sp = state.get("v12_exit_short")
        self._exit_short = ShortPosition(**sp) if sp else None
        self._exit_invalidated = state.get("v12_exit_invalidated", False)
        self._exit_committed = state.get("v12_exit_committed", False)
        # Spring
        self._spring_entries = state.get("v12_spring_entries", [])
        self._spring_entry_price = state.get("v12_spring_entry_price", 0.0)
        self._spring_deployed = state.get("v12_spring_deployed", 0.0)
        self._spring_phase_cash = state.get("v12_spring_phase_cash", 0.0)
        self._spring_highest_tier = state.get("v12_spring_highest_tier", 0)
        # Markdown state
        self._markdown_entry_price = state.get("v12_markdown_entry_price", 0.0)
        self._markdown_ath_at_entry = state.get("v12_markdown_ath_at_entry", 0.0)
        self._v12_markdown_phases = state.get("v12_markdown_phases", 0)
        # Markup state
        self._markup_position_qty = state.get("v12_markup_position_qty", 0.0)
        self._markup_position_cost = state.get("v12_markup_position_cost", 0.0)
        self._markup_avg_entry = state.get("v12_markup_avg_entry", 0.0)
        self._markup_trail_high = state.get("v12_markup_trail_high", 0.0)
        self._markup_trail_stop = state.get("v12_markup_trail_stop", 0.0)
        self._markup_adds = state.get("v12_markup_adds", 0)
        self._markup_last_add_price = state.get("v12_markup_last_add_price", 0.0)
        self._markup_phase_cash = state.get("v12_markup_phase_cash", 0.0)
        self._markup_entry_price = state.get("v12_markup_entry_price", 0.0)
        # Metrics
        self._dca_frozen = state.get("v12_dca_frozen", False)
        self._v12_markup_phases = state.get("v12_markup_phases", 0)
        self._v12_markup_pnl = state.get("v12_markup_pnl", 0.0)
        self._v12_markup_adds_total = state.get("v12_markup_adds_total", 0)
        self._v12_markup_trail_exits = state.get("v12_markup_trail_exits", 0)
        self._v12_markup_conductor_exits = state.get("v12_markup_conductor_exits", 0)
        self._v12_breakout_entries = state.get("v12_breakout_entries", 0)
        self._v12_exit_phases = state.get("v12_exit_phases", 0)
        self._v12_spring_phases = state.get("v12_spring_phases", 0)
        self._v12_rally_sells = state.get("v12_rally_sells", 0)
        self._v12_trail_stops = state.get("v12_trail_stops", 0)
        self._v12_urgency_closes = state.get("v12_urgency_closes", 0)
        self._v12_short_pnl = state.get("v12_short_pnl", 0.0)
        self._v12_spring_pnl = state.get("v12_spring_pnl", 0.0)
        self._v12_spring_deploys = state.get("v12_spring_deploys", 0)
        self._v12_false_springs = state.get("v12_false_springs", 0)
        self._v12_exit_pnl_preserved = state.get("v12_exit_pnl_preserved", 0.0)
        # Conductor
        self._conductor._price_ath = state.get("v12_conductor_price_ath", 0.0)
        self._conductor._last_scored_daily_idx = state.get("v12_conductor_last_scored_idx", -1)
        self._conductor._cached_score = state.get("v12_conductor_cached_score", 0.0)
        # V12e: Reversal detector state
        rd = self._conductor._reversal_detector
        rd._gate_active = state.get("v12e_rd_gate_active", False)
        rd._last_near_ath_ts = state.get("v12e_rd_last_near_ath_ts", None)
        rd._rolling_high = state.get("v12e_rd_rolling_high", 0.0)
        rd._rolling_high_ts = state.get("v12e_rd_rolling_high_ts", None)
        rd._fg_history = state.get("v12e_rd_fg_history", [])
    
    # ══════════════════════════════════════════════════════════════
    #  STEPPING INTERFACE (for shared-capital multi-coin backtest)
    # ══════════════════════════════════════════════════════════════

    def get_cash(self) -> float:
        """Get current cash balance."""
        return self.cash

    def set_cash(self, amount: float):
        """Set current cash balance (used by shared capital engine)."""
        self.cash = amount

    def get_equity(self, price: float) -> float:
        """Get current equity at given price."""
        return self._equity(price)

    def get_deployed(self) -> float:
        """Get total capital currently deployed in positions."""
        deployed = sum(d.capital_deployed for d in self.deals)
        # Exit lots
        for lot in self._exit_lots:
            if not lot.sold:
                deployed += lot.cost_usd
        # Spring entries
        for entry in self._spring_entries:
            if not entry.get("closed"):
                deployed += entry["cost"]
        # Markup position
        deployed += self._markup_position_cost
        # Short margin
        if self._exit_short and not self._exit_short.closed:
            deployed += self._exit_short.total_margin
        return deployed

    def prepare_step(self, df: pd.DataFrame):
        """Prepare engine for candle-by-candle stepping.
        
        Does all the setup that _run_main_loop does before the for loop:
        indicator computation, conductor prep, etc.
        
        After calling this, call step(i) for i in range(100, len(df)).
        """
        from .backtest_engine_v5 import _stochastic
        from .backtest_engine_v4 import HARD_SNAPBACK_REGIMES, SOFT_SNAPBACK_REGIMES
        from .backtest_engine_v6 import (
            DONCHIAN_LOOKBACK, DONCHIAN_RANGE_MAX_PCT,
        )
        from .backtest_engine_v3 import BLOCKED_REGIMES
        from ..indicators import (
            atr_pct as compute_atr_pct,
            atr as compute_atr,
            compute_all as compute_all_indicators,
            bollinger_band_width,
            volume_sma,
        )
        
        # Accumulate 1h data for daily conductor
        if not hasattr(self, '_accumulated_1h'):
            self._accumulated_1h = df.copy()
        else:
            self._accumulated_1h = pd.concat([self._accumulated_1h, df], ignore_index=True)
            self._accumulated_1h = self._accumulated_1h.drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)
        
        # Prepare daily conductor
        self._conductor.prepare(self._accumulated_1h)
        
        # Compute indicators and store on self for step() access
        df = compute_all_indicators(df)
        self._step_df = df
        self._step_regimes = classify_regime_v2(df, self.timeframe)
        self._step_atr_pct = compute_atr_pct(df, 14)
        self._step_atr_abs = compute_atr(df, 14)
        self._step_sma50 = df["close"].rolling(50).mean()
        
        bbw = df["bbw"] if "bbw" in df.columns else bollinger_band_width(df["close"], 20)
        self._step_bbw = bbw
        self._step_bbw_median = bbw.rolling(100, min_periods=20).median()
        self._step_vol = df["volume"]
        self._step_vol_avg = volume_sma(df, 20)
        self._step_stoch = _stochastic(df, 14, 3, 3)
        self._step_adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(np.nan, index=df.index)
        
        donchian_high = df["high"].rolling(DONCHIAN_LOOKBACK).max()
        donchian_low = df["low"].rolling(DONCHIAN_LOOKBACK).min()
        donchian_range_pct = (donchian_high - donchian_low) / donchian_low.replace(0, np.nan) * 100
        price_series = df["close"]
        self._step_in_range = (
            (price_series >= donchian_low)
            & (price_series <= donchian_high)
            & (donchian_range_pct < DONCHIAN_RANGE_MAX_PCT)
        )
        
        self._step_peak_equity = self.initial_capital
        if self.equity_snapshots:
            self._step_peak_equity = max(self._step_peak_equity, max(s["equity"] for s in self.equity_snapshots))
        
        if not hasattr(self, '_candle_timeline'):
            self._candle_timeline = []
        
        if not hasattr(self, '_step_prev_phase'):
            self._step_prev_phase = self._dd_phase if hasattr(self, '_dd_phase') else 1

    def step(self, i: int):
        """Process candle at index i using pre-computed indicators from prepare_step().
        
        This is the body of the for loop in _run_main_loop, extracted into a method.
        """
        from .backtest_engine_v4 import HARD_SNAPBACK_REGIMES, SOFT_SNAPBACK_REGIMES
        from .backtest_engine_v3 import BLOCKED_REGIMES
        from .distribution_scorer import DistributionPhase
        
        df = self._step_df
        regimes = self._step_regimes
        atr_pct_series = self._step_atr_pct
        atr_abs_series = self._step_atr_abs
        sma50 = self._step_sma50
        bbw = self._step_bbw
        bbw_median = self._step_bbw_median
        vol = self._step_vol
        vol_avg = self._step_vol_avg
        stoch = self._step_stoch
        adx_series = self._step_adx
        in_range_series = self._step_in_range
        
        row = df.iloc[i]
        ts = str(row["timestamp"])
        ts_ms = int(row["timestamp"])
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        
        regime = regimes.iloc[i] if i < len(regimes) else "UNKNOWN"
        self._current_regime = regime
        self._current_high = high
        self._current_low = low
        self._current_ts = ts
        self._current_price = price
        self._current_atr_pct = float(atr_pct_series.iloc[i]) if not pd.isna(atr_pct_series.iloc[i]) else 0.0
        self._current_atr_abs = float(atr_abs_series.iloc[i]) if not pd.isna(atr_abs_series.iloc[i]) else 0.0
        sma50_val = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else None
        self._trend_bullish = price >= sma50_val if sma50_val is not None else True
        
        # Store indicator values
        self._cur_vol = float(vol.iloc[i]) if not pd.isna(vol.iloc[i]) else 0.0
        self._cur_vol_avg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) and vol_avg.iloc[i] > 0 else 1.0
        self._cur_stoch_k = float(stoch["stoch_k"].iloc[i]) if not pd.isna(stoch["stoch_k"].iloc[i]) else np.nan
        self._cur_stoch_d = float(stoch["stoch_d"].iloc[i]) if not pd.isna(stoch["stoch_d"].iloc[i]) else np.nan
        self._cur_adx = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else np.nan
        self._cur_adx_prev = float(adx_series.iloc[i-1]) if i > 0 and not pd.isna(adx_series.iloc[i-1]) else np.nan
        self._cur_bbw = float(bbw.iloc[i]) if not pd.isna(bbw.iloc[i]) else 0.0
        self._cur_bbw_prev = float(bbw.iloc[i-1]) if i > 0 and not pd.isna(bbw.iloc[i-1]) else np.nan
        self._cur_bbw_med = float(bbw_median.iloc[i]) if not pd.isna(bbw_median.iloc[i]) else self._cur_bbw
        
        fg_value = self._get_fg_for_candle(df, i)
        if fg_value is None:
            fg_value = 50
        self._cur_fg = fg_value
        
        # Equity tracking
        equity = self._equity(price)
        if equity > self._step_peak_equity:
            self._step_peak_equity = equity
        peak_equity = self._step_peak_equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        
        # DD phase
        if dd < self._v8_phase1_dd_max:
            self._dd_phase = 1
        elif dd < self._v8_phase2_dd_max:
            self._dd_phase = 2
        else:
            self._dd_phase = 3
        
        if self._dd_phase != self._step_prev_phase:
            if self._dd_phase == 2:
                self._v8_cash_at_phase2_entry = self.cash
            elif self._dd_phase == 3:
                self._v8_cash_at_phase3_entry = self.cash
            self._step_prev_phase = self._dd_phase
        self._v8_phase_candles[self._dd_phase] = self._v8_phase_candles.get(self._dd_phase, 0) + 1
        
        # Daily conductor
        daily_score = self._conductor.score_at(ts_ms, price, fg_value)
        
        # DCA → EXIT transition
        if self._lifecycle_phase == LifecyclePhase.DCA:
            if self._conductor.should_exit(ts_ms, price, fg_value):
                # Price discovery: weekly confirmation is ALWAYS required
                if self._conductor._in_price_discovery:
                    if self._conductor.weekly_confirms_exit(
                            getattr(self, '_accumulated_1h', None), "distribution"):
                        self._transition_to_exit(price, ts, ts_ms, daily_score)
                    else:
                        logger.info("  🚫 PRICE DISCOVERY VETO: EXIT vetoed — weekly doesn't confirm")
                elif self._v12_weekly_dist_veto:
                    if self._conductor.weekly_confirms_exit(
                            getattr(self, '_accumulated_1h', None), "distribution"):
                        self._transition_to_exit(price, ts, ts_ms, daily_score)
                    else:
                        logger.info("  🚫 WEEKLY VETO: distribution EXIT vetoed")
                else:
                    self._transition_to_exit(price, ts, ts_ms, daily_score)
            elif not self._dca_frozen and self._check_breakout(df, i, price, regime):
                if self._conductor.weekly_confirms_exit(
                        getattr(self, '_accumulated_1h', None), "breakout"):
                    self._breakout_to_markup(df, i, price, ts, ts_ms, regime)
                else:
                    logger.info("  🚫 WEEKLY VETO: breakout MARKUP vetoed, staying in DCA")
        
        # Dispatch to active engine
        if self._lifecycle_phase == LifecyclePhase.DCA:
            self._run_dca_candle(
                df, i, row, ts, price, high, low, regime, dd,
                sma50_val, vol, vol_avg, stoch, bbw, bbw_median,
                adx_series, in_range_series, fg_value, regimes,
                atr_pct_series, peak_equity,
            )
        elif self._lifecycle_phase == LifecyclePhase.EXIT:
            self._run_exit_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
        elif self._lifecycle_phase == LifecyclePhase.MARKDOWN:
            self._run_markdown_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
        elif self._lifecycle_phase == LifecyclePhase.SPRING:
            self._run_spring_candle(price, high, low, ts, regime, fg_value)
        elif self._lifecycle_phase == LifecyclePhase.MARKUP:
            self._run_markup_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
        
        # Record equity
        equity = self._equity(price)
        self.equity_snapshots.append({
            "timestamp": ts, "equity": equity, "cash": self.cash, "price": price
        })
        deployed = sum(d.capital_deployed for d in self.deals)
        self._utilization_samples.append(
            deployed / self.initial_capital * 100 if self.initial_capital > 0 else 0
        )
        if equity > self._step_peak_equity:
            self._step_peak_equity = equity
        
        # Timeline
        self._candle_timeline.append({
            "timestamp": ts,
            "price": round(price, 4),
            "regime": regime,
            "dd_pct": round(dd, 2),
            "dd_phase": self._dd_phase,
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "lifecycle": self._lifecycle_phase.value,
            "daily_score": round(daily_score, 1),
            "spring_score": round(getattr(self, '_spring_score', 0.0), 1),
            "layers_filled": sum(len(d.lots) for d in self.deals),
            "exit_lots_remaining": len([l for l in self._exit_lots if not l.sold]),
            "markup_qty": round(self._markup_position_qty, 6),
        })

    # ══════════════════════════════════════════════════════════════
    #  MAIN LOOP OVERRIDE
    # ══════════════════════════════════════════════════════════════
    
    def _run_main_loop(self, df: pd.DataFrame):
        """V12 main loop: DCA with daily conductor triggering lifecycle transitions."""
        from .backtest_engine_v5 import _stochastic
        from .backtest_engine_v4 import HARD_SNAPBACK_REGIMES, SOFT_SNAPBACK_REGIMES
        from .backtest_engine_v6 import (
            DONCHIAN_LOOKBACK, DONCHIAN_RANGE_MAX_PCT,
        )
        from .backtest_engine_v3 import BLOCKED_REGIMES
        from ..indicators import (
            atr_pct as compute_atr_pct,
            atr as compute_atr,
            compute_all as compute_all_indicators,
            bollinger_band_width,
            volume_sma,
        )
        
        # Accumulate 1h data for daily conductor
        if not hasattr(self, '_accumulated_1h'):
            self._accumulated_1h = df.copy()
        else:
            self._accumulated_1h = pd.concat([self._accumulated_1h, df], ignore_index=True)
            self._accumulated_1h = self._accumulated_1h.drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)
        
        # Prepare daily conductor with ALL accumulated data
        self._conductor.prepare(self._accumulated_1h)
        
        # Standard indicator computation
        df = compute_all_indicators(df)
        regimes = classify_regime_v2(df, self.timeframe)
        atr_pct_series = compute_atr_pct(df, 14)
        atr_abs_series = compute_atr(df, 14)
        sma50 = df["close"].rolling(50).mean()
        
        bbw = df["bbw"] if "bbw" in df.columns else bollinger_band_width(df["close"], 20)
        bbw_median = bbw.rolling(100, min_periods=20).median()
        vol = df["volume"]
        vol_avg = volume_sma(df, 20)
        stoch = _stochastic(df, 14, 3, 3)
        adx_series = df["adx_14"] if "adx_14" in df.columns else pd.Series(np.nan, index=df.index)
        
        donchian_high = df["high"].rolling(DONCHIAN_LOOKBACK).max()
        donchian_low = df["low"].rolling(DONCHIAN_LOOKBACK).min()
        donchian_range_pct = (donchian_high - donchian_low) / donchian_low.replace(0, np.nan) * 100
        price_series = df["close"]
        in_range_series = (
            (price_series >= donchian_low)
            & (price_series <= donchian_high)
            & (donchian_range_pct < DONCHIAN_RANGE_MAX_PCT)
        )
        
        peak_equity = self.initial_capital
        if self.equity_snapshots:
            peak_equity = max(peak_equity, max(s["equity"] for s in self.equity_snapshots))
        
        if not hasattr(self, '_candle_timeline'):
            self._candle_timeline = []
        
        prev_phase = self._dd_phase if hasattr(self, '_dd_phase') else 1
        
        for i in range(100, len(df)):
            row = df.iloc[i]
            ts = str(row["timestamp"])
            ts_ms = int(row["timestamp"])
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            
            regime = regimes.iloc[i] if i < len(regimes) else "UNKNOWN"
            self._current_regime = regime
            self._current_high = high
            self._current_low = low
            self._current_ts = ts
            self._current_price = price
            self._current_atr_pct = float(atr_pct_series.iloc[i]) if not pd.isna(atr_pct_series.iloc[i]) else 0.0
            self._current_atr_abs = float(atr_abs_series.iloc[i]) if not pd.isna(atr_abs_series.iloc[i]) else 0.0
            sma50_val = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else None
            self._trend_bullish = price >= sma50_val if sma50_val is not None else True
            
            # Store indicator values for spring scoring
            self._cur_vol = float(vol.iloc[i]) if not pd.isna(vol.iloc[i]) else 0.0
            self._cur_vol_avg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) and vol_avg.iloc[i] > 0 else 1.0
            self._cur_stoch_k = float(stoch["stoch_k"].iloc[i]) if not pd.isna(stoch["stoch_k"].iloc[i]) else np.nan
            self._cur_stoch_d = float(stoch["stoch_d"].iloc[i]) if not pd.isna(stoch["stoch_d"].iloc[i]) else np.nan
            self._cur_adx = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else np.nan
            self._cur_adx_prev = float(adx_series.iloc[i-1]) if i > 0 and not pd.isna(adx_series.iloc[i-1]) else np.nan
            self._cur_bbw = float(bbw.iloc[i]) if not pd.isna(bbw.iloc[i]) else 0.0
            self._cur_bbw_prev = float(bbw.iloc[i-1]) if i > 0 and not pd.isna(bbw.iloc[i-1]) else np.nan
            self._cur_bbw_med = float(bbw_median.iloc[i]) if not pd.isna(bbw_median.iloc[i]) else self._cur_bbw
            
            fg_value = self._get_fg_for_candle(df, i)
            if fg_value is None:
                fg_value = 50  # neutral default when F&G data unavailable
            self._cur_fg = fg_value
            
            # Equity tracking
            equity = self._equity(price)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
            
            # DD phase (V8) — still used during DCA phase
            if dd < self._v8_phase1_dd_max:
                self._dd_phase = 1
            elif dd < self._v8_phase2_dd_max:
                self._dd_phase = 2
            else:
                self._dd_phase = 3
            
            if self._dd_phase != prev_phase:
                if self._dd_phase == 2:
                    self._v8_cash_at_phase2_entry = self.cash
                elif self._dd_phase == 3:
                    self._v8_cash_at_phase3_entry = self.cash
                prev_phase = self._dd_phase
            self._v8_phase_candles[self._dd_phase] = self._v8_phase_candles.get(self._dd_phase, 0) + 1
            
            # ════════════════════════════════════════════════════
            #  DAILY CONDUCTOR: Check for phase transitions
            # ════════════════════════════════════════════════════
            
            daily_score = self._conductor.score_at(ts_ms, price, fg_value)
            
            # DCA → EXIT transition (distribution top detected)
            if self._lifecycle_phase == LifecyclePhase.DCA:
                if self._conductor.should_exit(ts_ms, price, fg_value):
                    # Price discovery: weekly confirmation is ALWAYS required
                    if self._conductor._in_price_discovery:
                        if self._conductor.weekly_confirms_exit(
                                getattr(self, '_accumulated_1h', None), "distribution"):
                            self._transition_to_exit(price, ts, ts_ms, daily_score)
                        else:
                            logger.info("  🚫 PRICE DISCOVERY VETO: EXIT vetoed — weekly doesn't confirm")
                    elif self._v12_weekly_dist_veto:
                        if self._conductor.weekly_confirms_exit(
                                getattr(self, '_accumulated_1h', None), "distribution"):
                            self._transition_to_exit(price, ts, ts_ms, daily_score)
                        else:
                            logger.info("  🚫 WEEKLY VETO: distribution EXIT vetoed")
                    else:
                        self._transition_to_exit(price, ts, ts_ms, daily_score)
                # DCA → MARKUP breakout transition (regime shifts bullish)
                elif not self._dca_frozen and self._check_breakout(df, i, price, regime):
                    # Weekly validation: confirm breakout
                    if self._conductor.weekly_confirms_exit(
                            getattr(self, '_accumulated_1h', None), "breakout"):
                        self._breakout_to_markup(df, i, price, ts, ts_ms, regime)
                    else:
                        logger.info("  🚫 WEEKLY VETO: breakout MARKUP vetoed, staying in DCA")
            
            # EXIT → SPRING transition (handled inside _run_exit_engine)
            # SPRING → DCA transition (handled inside _run_spring_engine)
            
            # ════════════════════════════════════════════════════
            #  DISPATCH TO ACTIVE ENGINE
            # ════════════════════════════════════════════════════
            
            if self._lifecycle_phase == LifecyclePhase.DCA:
                self._run_dca_candle(
                    df, i, row, ts, price, high, low, regime, dd,
                    sma50_val, vol, vol_avg, stoch, bbw, bbw_median,
                    adx_series, in_range_series, fg_value, regimes,
                    atr_pct_series, peak_equity,
                )
            elif self._lifecycle_phase == LifecyclePhase.EXIT:
                self._run_exit_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
            elif self._lifecycle_phase == LifecyclePhase.MARKDOWN:
                self._run_markdown_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
            elif self._lifecycle_phase == LifecyclePhase.SPRING:
                self._run_spring_candle(price, high, low, ts, regime, fg_value)
            elif self._lifecycle_phase == LifecyclePhase.MARKUP:
                self._run_markup_candle(price, high, low, ts, ts_ms, regime, fg_value, daily_score)
            
            # Record equity
            equity = self._equity(price)
            self.equity_snapshots.append({
                "timestamp": ts, "equity": equity, "cash": self.cash, "price": price
            })
            deployed = sum(d.capital_deployed for d in self.deals)
            self._utilization_samples.append(
                deployed / self.initial_capital * 100 if self.initial_capital > 0 else 0
            )
            if equity > peak_equity:
                peak_equity = equity
            
            # Timeline
            self._candle_timeline.append({
                "timestamp": ts,
                "price": round(price, 4),
                "regime": regime,
                "dd_pct": round(dd, 2),
                "dd_phase": self._dd_phase,
                "cash": round(self.cash, 2),
                "equity": round(equity, 2),
                "lifecycle": self._lifecycle_phase.value,
                "daily_score": round(daily_score, 1),
                "spring_score": round(getattr(self, '_spring_score', 0.0), 1),
                "layers_filled": sum(len(d.lots) for d in self.deals),
                "exit_lots_remaining": len([l for l in self._exit_lots if not l.sold]),
                "markup_qty": round(self._markup_position_qty, 6),
            })
            
            if i % 500 == 0:
                logger.info("  [%d/%d] eq=$%.0f dd=%.1f%% phase=%s daily=%.0f cash=$%.0f deals=%d",
                            i, len(df), equity, dd, self._lifecycle_phase.value,
                            daily_score, self.cash, len(self.deals))
    
    # ══════════════════════════════════════════════════════════════
    #  ENGINE 1: DCA (delegates to V8/V9 logic)
    # ══════════════════════════════════════════════════════════════
    
    def _run_dca_candle(self, df, i, row, ts, price, high, low, regime, dd,
                        sma50_val, vol, vol_avg, stoch, bbw, bbw_median,
                        adx_series, in_range_series, fg_value, regimes,
                        atr_pct_series, peak_equity):
        """Run one candle of DCA engine (V8 logic with V9 TIGHTEN/WIND_DOWN overlays)."""
        from .backtest_engine_v4 import HARD_SNAPBACK_REGIMES, SOFT_SNAPBACK_REGIMES
        from .backtest_engine_v3 import BLOCKED_REGIMES
        
        # Distribution scoring on 1h (for TIGHTEN/WIND_DOWN — NOT for EXIT, conductor handles that)
        dist_result = self._dist_scorer.score(df, i, regime, fg_value, regimes)
        dist_phase = dist_result.phase
        # Never let the 1h scorer trigger EXIT — conductor does that via daily
        if dist_phase == DistributionPhase.EXIT:
            dist_phase = DistributionPhase.WIND_DOWN
        
        self._v9_phase_candles[dist_phase.value] = self._v9_phase_candles.get(dist_phase.value, 0) + 1
        
        # Dwell compression (Phase 1 only)
        cur_in_range = bool(in_range_series.iloc[i]) if not pd.isna(in_range_series.iloc[i]) else False
        vol_val = float(vol.iloc[i]) if not pd.isna(vol.iloc[i]) else 0.0
        vol_avg_val = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) and vol_avg.iloc[i] > 0 else 1.0
        vol_spike = vol_val > vol_avg_val * 2.0
        
        if self._dd_phase == 1:
            if regime in HARD_SNAPBACK_REGIMES or vol_spike:
                self._snap_back(hard=True)
            elif regime in SOFT_SNAPBACK_REGIMES:
                self._snap_back(hard=False)
            else:
                if self._dwell_cooldown_remaining > 0:
                    self._dwell_cooldown_remaining -= 1
                if self._dwell_cooldown_remaining <= 0:
                    if cur_in_range:
                        self._dwell_candle_count += 1
                    else:
                        self._dwell_candle_count = 0
            self._dwell_decay = self._calculate_dwell_decay()
            conv_score_raw = self._last_conviction.score if self._last_conviction else 0.0
            self._dwell_decay = self._apply_conviction_gate(self._dwell_decay, conv_score_raw)
        else:
            self._dwell_decay = 1.0
            self._conviction_gate = "disabled"
        
        # Spring scoring
        stoch_k = float(stoch["stoch_k"].iloc[i]) if not pd.isna(stoch["stoch_k"].iloc[i]) else np.nan
        stoch_d = float(stoch["stoch_d"].iloc[i]) if not pd.isna(stoch["stoch_d"].iloc[i]) else np.nan
        adx_cur = float(adx_series.iloc[i]) if not pd.isna(adx_series.iloc[i]) else np.nan
        adx_prev = float(adx_series.iloc[i-1]) if i > 0 and not pd.isna(adx_series.iloc[i-1]) else np.nan
        cur_bbw = float(bbw.iloc[i]) if not pd.isna(bbw.iloc[i]) else 0.0
        bbw_prev_val = float(bbw.iloc[i-1]) if i > 0 and not pd.isna(bbw.iloc[i-1]) else np.nan
        cur_bbw_med = float(bbw_median.iloc[i]) if not pd.isna(bbw_median.iloc[i]) else cur_bbw
        
        self._spring_score = self._compute_spring_score(
            vol_val, vol_avg_val, stoch_k, stoch_d, fg_value,
            adx_cur, adx_prev, cur_bbw, bbw_prev_val, cur_bbw_med,
        )
        
        # Adaptive TP/deviation
        exit_mode = self._get_exit_mode(regime)
        self._mode_candle_counts[exit_mode.name] = self._mode_candle_counts.get(exit_mode.name, 0) + 1
        tp_pct = self._adaptive_tp(regime, self._current_atr_pct)
        dev_pct = self._adaptive_deviation_v4(regime, self._current_atr_pct, tp_pct)
        
        # Distribution phase TP overrides
        if dist_phase == DistributionPhase.TIGHTEN:
            tp_pct = min(tp_pct, 0.8)
        elif dist_phase == DistributionPhase.WIND_DOWN:
            tp_pct = min(tp_pct, 0.5)
        
        conv_score = 0.0
        if self.conviction_mode:
            try:
                from ..indicators import regime_transition_signals
                regime_trans = regime_transition_signals(df, regimes)
                conviction = self._compute_conviction(df, i, price, regime, regime_trans)
                tp_pct, dev_pct = self._apply_conviction_to_params(tp_pct, dev_pct, conviction)
                conv_score = self._last_conviction.score if self._last_conviction else 0.0
            except (ImportError, AttributeError):
                pass
        
        self._spring_bypass = False
        
        # If DCA is frozen (breakout unwind), only check TPs — no new deals or SOs
        if self._dca_frozen:
            exit_mode_frozen = self._get_exit_mode(regime)
            self._check_exits(high, low, price, ts, regime, exit_mode_frozen)
            # Check if all deals have closed naturally
            if not self.deals:
                self._dca_frozen = False
                logger.info("  ✅ DCA UNWIND COMPLETE: all deals closed naturally, cash=$%.0f", self.cash)
            return
        
        # Phase-based trading logic (same as V8/V9)
        if self._dd_phase == 1:
            if dist_phase == DistributionPhase.WIND_DOWN:
                self._check_exits(high, low, price, ts, regime, exit_mode)
            elif dist_phase == DistributionPhase.TIGHTEN:
                self._check_tightened_so_fills(low, price, ts, regime, dev_pct, tp_pct)
                self._check_exits(high, low, price, ts, regime, exit_mode)
                if not self.deals and regime not in BLOCKED_REGIMES:
                    adx_for_trend = adx_cur if not np.isnan(adx_cur) else 0.0
                    sma50_for_trend = sma50_val if sma50_val is not None else price
                    if not self._should_block_deal_for_trend(price, sma50_for_trend, adx_for_trend, conv_score):
                        self._open_deal_tightened(price, ts, regime, tp_pct)
            else:
                self._check_safety_order_fills_v8(low, price, ts, regime, dev_pct, tp_pct)
                self._check_exits(high, low, price, ts, regime, exit_mode)
                if not self.deals and regime not in BLOCKED_REGIMES:
                    adx_for_trend = adx_cur if not np.isnan(adx_cur) else 0.0
                    sma50_for_trend = sma50_val if sma50_val is not None else price
                    if not self._should_block_deal_for_trend(price, sma50_for_trend, adx_for_trend, conv_score):
                        self._open_deal(price, ts, regime, tp_pct)
        elif self._dd_phase == 2:
            if self._v8_phase2_allow_tp:
                self._check_exits(high, low, price, ts, regime, exit_mode)
        elif self._dd_phase == 3:
            self._check_exits(high, low, price, ts, regime, exit_mode)
            if (self._spring_score >= self._v8_spring_score_threshold
                    and self.deals
                    and self._spring_entries_this_deal < self._v8_spring_max_entries):
                self._spring_bypass = True
                self._place_spring_entry(
                    self.deals[0], price, ts, regime, tp_pct, self._spring_score
                )
        
        if not self.deals:
            self._spring_entries_this_deal = 0
    
    # ══════════════════════════════════════════════════════════════
    #  PHASE TRANSITIONS
    # ══════════════════════════════════════════════════════════════
    
    def _transition_to_exit(self, price: float, ts: str, ts_ms: int, daily_score: float):
        """DCA → EXIT: Transfer open lots directly to Exit Engine (no force-close).
        
        Lots are extracted from DCA deals and managed by the Exit Engine.
        Cash held by deals is NOT returned — it stays deployed in the lots.
        The Exit Engine sells lots methodically, returning cash as it goes.
        """
        logger.info("  🚨 V12 DCA→EXIT: daily_score=%.0f, price=$%.2f, %d open deals",
                    daily_score, price, len(self.deals))
        
        self._lifecycle_phase = LifecyclePhase.EXIT
        self._v12_exit_phases += 1
        self._exit_entry_time_ms = ts_ms
        self._exit_entry_price = price
        self._exit_candles_elapsed = 0
        self._exit_local_low = price
        self._exit_lots_sold = 0
        self._exit_realized_pnl = 0.0
        self._exit_short = None
        self._exit_invalidated = False
        self._exit_committed = False
        
        # Extract lots directly from deals — no force-close
        self._exit_lots = []
        for deal in self.deals:
            for lot in deal.lots:
                if lot.sell_price is None:  # Open lot
                    trail_price = price * (1 - self._v12_initial_trail_pct / 100)
                    el = ExitLot(
                        lot_id=lot.lot_id,
                        buy_price=lot.buy_price,
                        qty=lot.qty,
                        cost_usd=lot.cost_usd,
                        buy_time=lot.buy_time,
                        unrealized_pnl_pct=(price - lot.buy_price) / lot.buy_price * 100,
                        trailing_stop=trail_price,
                    )
                    self._exit_lots.append(el)
        
        self._exit_lots_total = len(self._exit_lots)
        
        # Sort by profitability (most profitable first — sell these on rallies)
        self._exit_lots.sort(key=lambda l: l.unrealized_pnl_pct, reverse=True)
        
        # Clear DCA deals WITHOUT force-closing (lots are now managed by Exit Engine)
        # The capital is still deployed in the lots — Exit Engine returns it when selling
        self.deals.clear()
        
        open_count = len(self._exit_lots)
        total_invested = sum(l.cost_usd for l in self._exit_lots)
        total_value = sum(l.qty * price for l in self._exit_lots)
        logger.info("  📋 Exit Engine: managing %d lots, invested=$%.0f, value=$%.0f, cash=$%.0f",
                    open_count, total_invested, total_value, self.cash)
    
    def _transition_to_spring(self, price: float, ts: str):
        """EXIT → SPRING: All lots unwound, enter spring detection mode."""
        logger.info("  🌱 V12 EXIT→SPRING: price=$%.2f, exit_pnl=$%.0f, short_pnl=$%.0f, cash=$%.0f",
                    price, self._exit_realized_pnl, self._v12_short_pnl, self.cash)
        
        self._lifecycle_phase = LifecyclePhase.SPRING
        self._v12_spring_phases += 1
        self._spring_entries = []
        self._spring_entry_price = 0.0
        self._spring_deployed = 0.0
        self._spring_phase_cash = self.cash
        self._spring_highest_tier = 0
    
    def _transition_to_dca(self, price: float, ts: str):
        """SPRING → DCA: Recovery confirmed, resume normal trading."""
        logger.info("  ✅ V12 SPRING→DCA: price=$%.2f, spring_pnl=$%.0f, cash=$%.0f",
                    price, self._v12_spring_pnl, self.cash)
        
        self._lifecycle_phase = LifecyclePhase.DCA
        # Reset spring state
        self._spring_entries = []
        self._spring_deployed = 0.0
        self._spring_highest_tier = 0
        # Reset exit state
        self._exit_lots = []
        self._exit_short = None
    
    # ══════════════════════════════════════════════════════════════
    #  ENGINE 2: EXIT
    # ══════════════════════════════════════════════════════════════
    
    def _run_exit_candle(self, price: float, high: float, low: float, 
                         ts: str, ts_ms: int, regime: str, fg_value: float,
                         daily_score: float):
        """Run one candle of Exit Engine."""
        self._exit_candles_elapsed += 1
        hours_elapsed = self._exit_candles_elapsed  # 1h candles = 1 candle per hour
        days_elapsed = hours_elapsed / 24.0
        
        # Track local low for rally detection
        if low < self._exit_local_low:
            self._exit_local_low = low
        
        # ── Commitment check (first 48h) ──
        if not self._exit_committed and hours_elapsed >= self._v12_commitment_hours:
            self._exit_committed = True
            logger.info("  ⏰ V12 EXIT COMMITTED after %dh — no going back", hours_elapsed)
        
        # ── False signal check (first 48h only) ──
        if not self._exit_committed:
            if daily_score <= 0:
                # Full invalidation — return to DCA
                logger.info("  ❌ V12 EXIT INVALIDATED: daily_score=0 within %dh", hours_elapsed)
                self._exit_invalidated = True
                self._close_exit_short(price, ts, "invalidation")
                # Sell remaining exit lots at market and return to DCA
                self._force_sell_all_exit_lots(price, ts, "exit_invalidated")
                self._transition_to_dca(price, ts)
                return
            elif daily_score < self._v12_exit_threshold:
                # Score dropped but not to 0 — pause urgency, stay in EXIT
                pass  # Continue Exit Engine but don't escalate urgency
        
        # ── Update trailing stops ──
        open_lots = [l for l in self._exit_lots if not l.sold]
        trail_pct = max(
            self._v12_trail_floor_pct,
            self._v12_initial_trail_pct - days_elapsed * self._v12_trail_tighten_per_day
        )
        for lot in open_lots:
            # Trail only tightens, never loosens
            new_trail = price * (1 - trail_pct / 100)
            if new_trail > lot.trailing_stop:
                lot.trailing_stop = new_trail
            # Update unrealized PnL
            lot.unrealized_pnl_pct = (price - lot.buy_price) / lot.buy_price * 100
        
        # ── Check trailing stop hits ──
        for lot in open_lots:
            if low <= lot.trailing_stop:
                self._sell_exit_lot(lot, lot.trailing_stop, ts, "trailing_stop")
                self._v12_trail_stops += 1
        
        # ── Rally selling: sell most profitable lot on bounce ──
        if self._exit_local_low < float('inf'):
            bounce_pct = (high - self._exit_local_low) / self._exit_local_low * 100
            if bounce_pct >= self._v12_rally_sell_pct:
                # Find most profitable unsold lot
                profitable = sorted(
                    [l for l in self._exit_lots if not l.sold],
                    key=lambda l: l.unrealized_pnl_pct,
                    reverse=True
                )
                if profitable:
                    best = profitable[0]
                    # Sell into the rally — use the high as sell price (limit order on the bounce)
                    if best.unrealized_pnl_pct > 0 or days_elapsed >= self._v12_urgency_day_moderate:
                        self._sell_exit_lot(best, high, ts, "rally_sell")
                        self._v12_rally_sells += 1
                        # Reset local low after a sell
                        self._exit_local_low = price
        
        # ── Time-based urgency ──
        open_lots = [l for l in self._exit_lots if not l.sold]
        if days_elapsed >= self._v12_urgency_day_force and open_lots:
            # Force close everything
            for lot in open_lots:
                self._sell_exit_lot(lot, price, ts, "force_close_14d")
                self._v12_urgency_closes += 1
        elif days_elapsed >= self._v12_urgency_day_aggressive and open_lots:
            # Close any lot still in profit
            for lot in list(open_lots):
                if lot.unrealized_pnl_pct >= 0:
                    self._sell_exit_lot(lot, price, ts, "urgency_aggressive")
                    self._v12_urgency_closes += 1
        elif days_elapsed >= self._v12_urgency_day_moderate and open_lots:
            # Lower TP expectations — sell lots with > 50% of their peak PnL
            pass  # Trailing stops handle this naturally via tightening
        
        # ── Check if all lots are sold → transition to Markdown (shorts) or Spring ──
        all_sold = all(l.sold for l in self._exit_lots) if self._exit_lots else True
        if all_sold:
            if self._v12_short_enabled:
                self._transition_to_markdown(price, ts)
            else:
                self._transition_to_spring(price, ts)
    
    def _sell_exit_lot(self, lot: ExitLot, sell_price: float, ts: str, reason: str):
        """Sell an exit lot and return cash (market value minus fees/slippage)."""
        # V12e PARAM SYNC: apply slippage to market-like sells (not trailing stop limit fills)
        is_market_like = reason in ("force_close_14d", "urgency_aggressive", "exit_invalidated", "backtest_end")
        if is_market_like:
            sell_price = sell_price * (1 - self._v12_slippage_pct / 100)
        
        lot.sold = True
        lot.sell_price = sell_price
        lot.sell_time = ts
        lot.sell_reason = reason
        
        # V12e PARAM SYNC: apply taker fee to all exit sells
        gross_proceeds = lot.qty * sell_price
        fee = gross_proceeds * self.taker_fee
        proceeds = gross_proceeds - fee
        lot.pnl = proceeds - lot.cost_usd
        
        self.cash += proceeds
        self._exit_lots_sold += 1
        self._exit_realized_pnl += lot.pnl
        self._v12_exit_pnl_preserved += lot.pnl
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action=f"EXIT_SELL_{reason.upper()}", deal_id=0,
            lot_id=lot.lot_id, price=sell_price, qty=lot.qty,
            cost_usd=lot.cost_usd, fee=fee, pnl=lot.pnl,
            regime=self._current_regime, sell_reason=reason,
        ))
        
        remaining = len([l for l in self._exit_lots if not l.sold])
        logger.info("  📤 EXIT SELL (%s): lot %d at $%.2f, pnl=$%.2f (%.1f%%), %d lots remaining",
                    reason, lot.lot_id, sell_price, lot.pnl, lot.unrealized_pnl_pct, remaining)
    
    def _force_sell_all_exit_lots(self, price: float, ts: str, reason: str):
        """Force sell all remaining exit lots."""
        for lot in self._exit_lots:
            if not lot.sold:
                self._sell_exit_lot(lot, price, ts, reason)
    
    def _open_aggressive_short(self, price: float, ts: str):
        """Open aggressive short — deploy 60% of cash immediately (tier 1)."""
        deploy = self.cash * self._v12_short_tier1_deploy
        if deploy < 100:
            return
        
        # V12e PARAM SYNC: apply taker fee + slippage to short open (market order)
        slipped_price = price * (1 + self._v12_slippage_pct / 100)  # worse fill for shorts = higher price
        fee = deploy * self.taker_fee
        effective_deploy = deploy - fee
        qty = effective_deploy / slipped_price  # 1x leverage — margin = notional
        sl = slipped_price * (1 + self._v12_short_sl_pct / 100)
        
        self._exit_short = ShortPosition(
            total_qty=qty, total_margin=deploy, avg_entry=slipped_price,
            entries=1, first_entry_price=slipped_price,
            trail_low=slipped_price, trail_stop=slipped_price * (1 + self._v12_short_trail_pct / 100),
            sl_price=sl,
        )
        self.cash -= deploy
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action="SHORT_TIER1", deal_id=0,
            lot_id=0, price=slipped_price, qty=qty, cost_usd=deploy,
            fee=fee, regime=self._current_regime,
        ))
        logger.info("  📉 SHORT TIER 1: $%.0f (60%%) at $%.2f, SL=$%.2f, cash=$%.0f",
                    deploy, price, sl, self.cash)
    
    def _add_short_tier(self, price: float, ts: str, tier: int, target_deploy_pct: float):
        """Add to short position (tier 2 or 3)."""
        s = self._exit_short
        # Calculate how much more to deploy
        total_target = (self.cash + s.total_margin) * target_deploy_pct  # target % of total capital
        additional = total_target - s.total_margin
        if additional < 50 or additional > self.cash:
            additional = min(additional, self.cash - 100)  # Keep $100 reserve
            if additional < 50:
                return
        
        # V12e PARAM SYNC: apply taker fee + slippage to short tier add
        slipped_price = price * (1 + self._v12_slippage_pct / 100)
        fee = additional * self.taker_fee
        effective = additional - fee
        add_qty = effective / slipped_price
        total_qty = s.total_qty + add_qty
        total_margin = s.total_margin + additional
        s.avg_entry = total_margin / total_qty if total_qty > 0 else slipped_price
        s.total_qty = total_qty
        s.total_margin = total_margin
        s.entries = tier
        self.cash -= additional
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action=f"SHORT_TIER{tier}", deal_id=0,
            lot_id=tier, price=slipped_price, qty=add_qty, cost_usd=additional,
            fee=fee, regime=self._current_regime,
        ))
        logger.info("  📉 SHORT TIER %d: +$%.0f at $%.2f, total=$%.0f, cash=$%.0f",
                    tier, additional, price, total_margin, self.cash)
    
    def _process_aggressive_short(self, price: float, high: float, low: float, ts: str):
        """Process aggressive short: tiered adds, funding. No trail stops — 
        spring signal closes the position (same philosophy as markup: hold until signal)."""
        s = self._exit_short
        if s is None or s.closed:
            return
        
        # ── Funding cost ──
        rate_per_candle = self._v12_funding_rate_daily / 24.0
        cost = s.total_margin * rate_per_candle
        s.funding_cost += cost
        if cost <= self.cash:
            self.cash -= cost
        
        # ── Tier 2: add on bounce (price bounces up from entry then comes back) ──
        if s.entries == 1:
            bounce_pct = (high - s.first_entry_price) / s.first_entry_price * 100
            if bounce_pct >= self._v12_short_tier2_bounce_pct:
                # Price bounced — add on the bounce (selling high)
                self._add_short_tier(high, ts, 2, self._v12_short_tier2_deploy)
        
        # ── Tier 3: add on retest of entry (price comes back near our first entry) ──
        if s.entries == 2:
            retest_pct = abs(high - s.first_entry_price) / s.first_entry_price * 100
            if retest_pct <= self._v12_short_tier3_retest_pct:
                self._add_short_tier(high, ts, 3, self._v12_short_tier3_deploy)
        
        # ── New ATH invalidation — if price breaks above the ATH at markdown entry ──
        if self._markdown_ath_at_entry > 0 and high > self._markdown_ath_at_entry * 1.02:
            # Price made new ATH (2% above previous) — false exit, close shorts
            logger.info("  🚨 MARKDOWN ATH INVALIDATION: new ATH $%.0f > previous $%.0f, closing shorts",
                       high, self._conductor._price_ath)
            self._close_aggressive_short(high, ts, "ath_invalidation")
            return
    
    def _close_aggressive_short(self, close_price: float, ts: str, reason: str):
        """Close the aggressive short position."""
        s = self._exit_short
        if s is None or s.closed:
            return
        
        # V12e PARAM SYNC: apply taker fee + slippage to short close (market order)
        slipped_price = close_price * (1 + self._v12_slippage_pct / 100)  # worse fill = higher buy-back price
        close_fee = s.total_qty * slipped_price * self.taker_fee
        pnl = (s.avg_entry - slipped_price) * s.total_qty - s.funding_cost - close_fee
        s.closed = True
        s.close_price = slipped_price
        s.close_time = ts
        s.pnl = pnl
        
        # Return margin + PnL
        self.cash += s.total_margin + pnl
        self._v12_short_pnl += pnl
        
        pnl_pct = (s.avg_entry - slipped_price) / s.avg_entry * 100 if s.avg_entry > 0 else 0
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action=f"SHORT_CLOSE_{reason.upper()}", deal_id=0,
            lot_id=0, price=slipped_price, qty=s.total_qty,
            cost_usd=s.total_margin, fee=close_fee, pnl=pnl,
            regime=self._current_regime, sell_reason=reason,
        ))
        logger.info("  📈 SHORT CLOSE (%s): pnl=$%.2f (%.1f%%), %d tiers, funding=$%.2f",
                    reason, pnl, pnl_pct, s.entries, s.funding_cost)
    
    def _close_exit_short(self, price: float, ts: str, reason: str):
        """Force close exit short."""
        if self._exit_short is None or self._exit_short.closed:
            return
        self._close_aggressive_short(price, ts, reason)
    
    # ══════════════════════════════════════════════════════════════
    #  ENGINE 3: SPRING
    # ══════════════════════════════════════════════════════════════
    
    def _run_spring_candle(self, price: float, high: float, low: float,
                           ts: str, regime: str, fg_value: float):
        """Run one candle of Spring Engine.
        
        V12 spring logic: deploy based on DISCOUNT FROM EXIT PRICE.
        We confirmed the top (exit engine fired), so the signal is simple:
        the deeper the discount, the more we deploy. No fancy indicators needed.
        
        Tiers based on discount from exit price:
          Tier 1: 25%+ discount → deploy tier1_deploy % of spring_phase_cash
          Tier 2: 35%+ discount → deploy tier2_deploy % (cumulative)
          Tier 3: 45%+ discount → deploy tier3_deploy % (cumulative)
        Always keep 25% reserve.
        """
        self._spring_score = 0.0  # Not using indicator-based scoring
        
        # ── Discount-based deployment ──
        if self._spring_phase_cash > 0 and self._exit_realized_pnl is not None:
            # Use the price at which exit engine started as reference
            exit_ref_price = self._exit_entry_price
            if exit_ref_price > 0:
                discount_pct = (exit_ref_price - price) / exit_ref_price * 100
                
                tier = 0
                target_deploy_pct = 0.0
                if discount_pct >= self._v12_spring_tier3_discount:
                    tier = 3
                    target_deploy_pct = self._v12_spring_tier3_deploy
                elif discount_pct >= self._v12_spring_tier2_discount:
                    tier = 2
                    target_deploy_pct = self._v12_spring_tier2_deploy
                elif discount_pct >= self._v12_spring_tier1_discount:
                    tier = 1
                    target_deploy_pct = self._v12_spring_tier1_deploy
                
                if tier > 0 and tier > self._spring_highest_tier:
                    # Deploy up to target_deploy_pct of initial spring cash
                    target_amount = self._spring_phase_cash * target_deploy_pct
                    remaining = target_amount - self._spring_deployed
                    if remaining > 50:
                        self._deploy_spring(price, ts, tier, remaining, discount_pct)
                
                # Log periodically
                if not hasattr(self, '_spring_diag_count'):
                    self._spring_diag_count = 0
                self._spring_diag_count += 1
                if self._spring_diag_count % 24 == 0:
                    logger.info("  📊 SPRING: price=$%.0f, discount=%.1f%%, tier=%d, deployed=$%.0f/$%.0f, cash=$%.0f",
                               price, discount_pct, self._spring_highest_tier,
                               self._spring_deployed, self._spring_phase_cash, self.cash)
        
        # ── Check existing spring entries ──
        for entry in self._spring_entries:
            if entry.get("closed"):
                continue
            
            entry_price = entry["price"]
            
            # TP check — limit order fill, no slippage, maker fee
            tp_price = entry_price * (1 + self._v12_spring_tp_pct / 100)
            if high >= tp_price:
                gross = entry["qty"] * tp_price
                # V12e PARAM SYNC: maker fee for TP limit fill
                fee = gross * self.maker_fee
                proceeds = gross - fee
                pnl = proceeds - entry["cost"]
                entry["closed"] = True
                entry["close_price"] = tp_price
                entry["close_time"] = ts
                entry["pnl"] = pnl
                self.cash += proceeds
                self._v12_spring_pnl += pnl
                self._spring_deployed -= entry["cost"]
                logger.info("  🌱 SPRING TP: tier %d, pnl=$%.2f, fee=$%.2f, cash=$%.0f",
                           entry["tier"], pnl, fee, self.cash)
                continue
            
            # False spring check: drop below entry — market-like sells with slippage
            drop_pct = (entry_price - low) / entry_price * 100
            if drop_pct >= self._v12_spring_false_drop_pct:
                # V12e PARAM SYNC: slippage + taker fee on false spring sells
                slipped_low = low * (1 - self._v12_slippage_pct / 100)
                # Cut 50% on first breach, full on 2× threshold
                if drop_pct >= self._v12_spring_false_drop_pct * 2:
                    # Full close
                    gross = entry["qty"] * slipped_low
                    fee = gross * self.taker_fee
                    proceeds = gross - fee
                    pnl = proceeds - entry["cost"]
                    entry["closed"] = True
                    entry["close_price"] = slipped_low
                    entry["close_time"] = ts
                    entry["pnl"] = pnl
                    self.cash += proceeds
                    self._v12_spring_pnl += pnl
                    self._spring_deployed -= entry["cost"]
                    self._v12_false_springs += 1
                    logger.info("  ❌ FALSE SPRING (full cut): tier %d, pnl=$%.2f",
                               entry["tier"], pnl)
                elif not entry.get("half_cut"):
                    # Cut 50%
                    half_qty = entry["qty"] / 2
                    half_cost = entry["cost"] / 2
                    gross = half_qty * slipped_low
                    fee = gross * self.taker_fee
                    proceeds = gross - fee
                    pnl = proceeds - half_cost
                    entry["qty"] = half_qty
                    entry["cost"] = half_cost
                    entry["half_cut"] = True
                    self.cash += proceeds
                    self._v12_spring_pnl += pnl
                    self._spring_deployed -= half_cost
                    logger.info("  ⚠️ FALSE SPRING (half cut): tier %d, pnl=$%.2f",
                               entry["tier"], pnl)
        
        # ── Check for recovery → transition to MARKUP (not DCA) ──
        if self._spring_entry_price > 0:
            recovery_pct = (price - self._spring_entry_price) / self._spring_entry_price * 100
            if recovery_pct >= self._v12_spring_recovery_pct and regime in ("TRENDING", "MILD_TREND", "MARKUP", "ACCUMULATION"):
                # Close spring entries, collect proceeds (market sell with fee+slippage)
                for entry in self._spring_entries:
                    if not entry.get("closed"):
                        # V12e PARAM SYNC: taker fee + slippage on spring recovery close
                        slipped = price * (1 - self._v12_slippage_pct / 100)
                        gross = entry["qty"] * slipped
                        fee = gross * self.taker_fee
                        proceeds = gross - fee
                        pnl = proceeds - entry["cost"]
                        entry["closed"] = True
                        entry["close_price"] = slipped
                        entry["close_time"] = ts
                        entry["pnl"] = pnl
                        self.cash += proceeds
                        self._v12_spring_pnl += pnl
                        self._spring_deployed -= entry["cost"]
                
                # Confirmed spring recovery → enter Markup Engine
                self._transition_to_markup(price, ts)
                return
        
        # ── Fallback: 60 days in spring with no signal → return to DCA ──
        spring_candles = sum(1 for _ in self._candle_timeline 
                           if len(self._candle_timeline) > 0 
                           and self._lifecycle_phase == LifecyclePhase.SPRING)
        # Simpler: track candles in spring
        if not hasattr(self, '_spring_candle_count'):
            self._spring_candle_count = 0
        self._spring_candle_count += 1
        
        if self._spring_candle_count >= 1440:  # 60 days on 1h
            logger.info("  ⏰ SPRING TIMEOUT (60d): returning to DCA, cash=$%.0f", self.cash)
            for entry in self._spring_entries:
                if not entry.get("closed"):
                    # V12e PARAM SYNC: taker fee + slippage on timeout close
                    slipped = price * (1 - self._v12_slippage_pct / 100)
                    gross = entry["qty"] * slipped
                    fee = gross * self.taker_fee
                    proceeds = gross - fee
                    pnl = proceeds - entry["cost"]
                    entry["closed"] = True
                    entry["close_price"] = slipped
                    entry["close_time"] = ts
                    entry["pnl"] = pnl
                    self.cash += proceeds
                    self._v12_spring_pnl += pnl
            self._transition_to_dca(price, ts)
    
    def _deploy_spring(self, price: float, ts: str, tier: int, amount: float, score: float):
        """Deploy capital at spring signal."""
        if amount > self.cash:
            amount = self.cash
        if amount < 50:
            return
        
        fee = amount * self.taker_fee
        qty = (amount - fee) / price
        
        entry = {
            "tier": tier,
            "price": price,
            "qty": qty,
            "cost": amount,
            "fee": fee,
            "time": ts,
            "score": score,
            "closed": False,
            "half_cut": False,
        }
        self._spring_entries.append(entry)
        self.cash -= amount
        self._spring_deployed += amount
        self._spring_highest_tier = tier
        self._v12_spring_deploys += 1
        
        if self._spring_entry_price == 0:
            self._spring_entry_price = price
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action=f"SPRING_BUY_T{tier}", deal_id=0,
            lot_id=len(self._spring_entries) - 1, price=price, qty=qty,
            cost_usd=amount, fee=fee, regime=self._current_regime,
        ))
        logger.info("  🌱 SPRING DEPLOY T%d: $%.0f at $%.2f, score=%.0f, cash=$%.0f",
                    tier, amount, price, score, self.cash)
    
    # ══════════════════════════════════════════════════════════════
    #  ENGINE 3a: MARKDOWN (aggressive shorts riding the decline)
    # ══════════════════════════════════════════════════════════════
    
    def _transition_to_markdown(self, price: float, ts: str):
        """EXIT → MARKDOWN: All longs unwound, now ride the decline with shorts."""
        logger.info("  📉 V12 EXIT→MARKDOWN: price=$%.2f, cash=$%.0f", price, self.cash)
        self._lifecycle_phase = LifecyclePhase.MARKDOWN
        self._v12_markdown_phases += 1
        self._markdown_entry_price = price
        self._markdown_ath_at_entry = self._conductor._price_ath  # Snapshot ATH for invalidation
        self._exit_short = None  # Fresh start
        
        # Deploy tier 1 short immediately (60% of cash)
        self._open_aggressive_short(price, ts)
    
    def _run_markdown_candle(self, price: float, high: float, low: float,
                             ts: str, ts_ms: int, regime: str, fg_value: float,
                             daily_score: float):
        """Run one candle of Markdown Engine — ride shorts down, transition to spring at bottom."""
        
        # Process the short position (tiered adds, trail, stops)
        if self._exit_short and not self._exit_short.closed:
            self._process_aggressive_short(price, high, low, ts)
        
        # If short was closed (trail stop or SL), check discount for spring transition
        if self._exit_short is None or self._exit_short.closed:
            # Check if we've declined enough for spring deployment
            if self._exit_entry_price > 0:
                discount = (self._exit_entry_price - price) / self._exit_entry_price * 100
                if discount >= self._v12_spring_tier1_discount:
                    # Deep enough — transition to spring
                    self._transition_to_spring(price, ts)
                    return
                else:
                    # Not deep enough — reopen short if price resumes decline
                    # Only if we're still below the exit entry (still in downtrend)
                    if price < self._markdown_entry_price * 0.95:
                        self._open_aggressive_short(price, ts)
                    else:
                        # Price recovered — false exit, go back to DCA
                        logger.info("  ↩ MARKDOWN→DCA: price recovered to $%.2f (exit was $%.2f), false exit",
                                   price, self._markdown_entry_price)
                        self._transition_to_dca(price, ts)
                        return
        
        # Also check for spring discount while short is still open
        if self._exit_entry_price > 0:
            discount = (self._exit_entry_price - price) / self._exit_entry_price * 100
            if discount >= self._v12_spring_tier1_discount:
                # Close shorts and transition to spring
                if self._exit_short and not self._exit_short.closed:
                    self._close_aggressive_short(price, ts, "spring_transition")
                self._transition_to_spring(price, ts)
                return
    
    # ══════════════════════════════════════════════════════════════
    #  DCA → MARKUP BREAKOUT TRANSITION
    # ══════════════════════════════════════════════════════════════
    
    def _check_breakout(self, df: pd.DataFrame, i: int, price: float, regime: str) -> bool:
        """Check if a breakout warrants DCA → MARKUP transition.
        
        Conditions (all must be true):
        1. Regime is TRENDING, MARKUP, or MILD_TREND
        2. Price above 50-day (1200-candle on 1h) high
        3. ADX > 25 (trending confirmation)
        4. Volume above average
        """
        # Must be in a bullish regime
        if regime not in ("TRENDING", "MARKUP", "MILD_TREND"):
            return False
        
        # Need enough history for 50-day high (1200 1h candles)
        lookback = min(1200, i)
        if lookback < 200:  # Need at least ~8 days of data
            return False
        
        # Price must break above 50-day high
        high_50d = float(df["high"].iloc[i - lookback:i].max())
        if price < high_50d:
            return False
        
        # ADX must confirm trend (>25)
        adx = getattr(self, '_cur_adx', 0.0)
        if np.isnan(adx) or adx < 25:
            return False
        
        # Volume above average
        vol = getattr(self, '_cur_vol', 0.0)
        vol_avg = getattr(self, '_cur_vol_avg', 1.0)
        if vol < vol_avg:
            return False
        
        # Must have some open DCA deals to close (otherwise nothing to transition from)
        if len(self.deals) == 0 and self.cash < 500:
            return False
        
        return True
    
    def _breakout_to_markup(self, df: pd.DataFrame, i: int, price: float, 
                            ts: str, ts_ms: int, regime: str):
        """DCA → MARKUP BREAKOUT: Natural unwind — freeze DCA grid, deploy available cash.
        
        Instead of force-closing DCA positions (which crystallizes losses at breakout),
        we freeze the grid (no new deals/SOs) and let existing deals close via their TPs.
        Prices are rising in a breakout so TPs hit fast. Available cash goes into markup
        position immediately, and as DCA TPs free more cash, we add to markup up to cap.
        """
        logger.info("  🚀 V12 DCA→MARKUP BREAKOUT: regime=%s, price=$%.2f, ADX=%.1f, %d deals",
                    regime, price, getattr(self, '_cur_adx', 0), len(self.deals))
        
        # Freeze DCA — no new deals or safety orders
        self._dca_frozen = True
        self._v12_breakout_entries += 1
        
        open_lots = sum(1 for d in self.deals for l in d.lots if l.sell_price is None)
        invested = sum(l.cost_usd for d in self.deals for l in d.lots if l.sell_price is None)
        logger.info("  🚀 BREAKOUT: Froze DCA grid (%d lots, $%.0f invested), deploying cash=$%.0f into markup",
                    open_lots, invested, self.cash)
        
        # Transition to MARKUP — deploys available cash (DCA deals unwind in background)
        self._transition_to_markup(price, ts)
    
    # ══════════════════════════════════════════════════════════════
    #  ENGINE 4: MARKUP (trend riding after confirmed spring)
    # ══════════════════════════════════════════════════════════════
    
    def _transition_to_markup(self, price: float, ts: str):
        """SPRING → MARKUP: Confirmed recovery, deploy big and ride the trend."""
        logger.info("  🚀 V12 SPRING→MARKUP: price=$%.2f, cash=$%.0f", price, self.cash)
        
        self._lifecycle_phase = LifecyclePhase.MARKUP
        self._v12_markup_phases += 1
        self._markup_phase_cash = self.cash
        self._markup_entry_price = price
        self._markup_trail_high = price
        self._markup_adds = 0
        self._markup_last_add_price = price
        
        # Deploy large initial position
        deploy_amount = self.cash * self._v12_markup_deploy_pct
        if deploy_amount < 100:
            # Not enough cash — fall back to DCA
            self._transition_to_dca(price, ts)
            return
        
        fee = deploy_amount * self.taker_fee
        qty = (deploy_amount - fee) / price
        
        self._markup_position_qty = qty
        self._markup_position_cost = deploy_amount
        self._markup_avg_entry = price
        self.cash -= deploy_amount
        
        # Set initial trailing stop
        self._markup_trail_stop = price * (1 - self._v12_markup_trail_pct / 100)
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action="MARKUP_ENTRY", deal_id=0,
            lot_id=0, price=price, qty=qty, cost_usd=deploy_amount,
            fee=fee, regime=self._current_regime,
        ))
        logger.info("  🚀 MARKUP DEPLOY: $%.0f (%.0f%%) at $%.2f, trail=$%.2f, cash=$%.0f",
                    deploy_amount, self._v12_markup_deploy_pct * 100, price,
                    self._markup_trail_stop, self.cash)
    
    def _run_markup_candle(self, price: float, high: float, low: float,
                           ts: str, ts_ms: int, regime: str, fg_value: float,
                           daily_score: float):
        """Run one candle of Markup Engine — HOLD until EXIT fires. No trail stops.
        Spot = no liquidation risk. Ride the full wave. Conductor decides when to exit.
        
        If DCA is frozen (breakout entry), also process DCA deal TPs and deploy
        freed cash into the markup position up to the profile cap.
        """
        
        # ── Process frozen DCA deals (natural unwind via TPs) ──
        if self._dca_frozen and self.deals:
            # Use the existing exit logic — check TPs on all lots
            exit_mode_frozen = self._get_exit_mode(regime)
            self._check_exits(high, low, price, ts, regime, exit_mode_frozen)
            
            if not self.deals:
                self._dca_frozen = False
                logger.info("  ✅ DCA UNWIND COMPLETE during markup, cash=$%.0f", self.cash)
        
        # ── Deploy freed cash into markup (up to profile cap) ──
        # Profile caps: low=50%, medium=70%, high=90% of total capital
        if self._markup_position_qty > 0:
            total_equity = self._equity(price)
            markup_value = self._markup_position_qty * price
            markup_pct = markup_value / total_equity if total_equity > 0 else 0
            
            # Determine cap based on deploy_pct (which is profile-scaled)
            cap_pct = self._v12_markup_deploy_pct  # 0.50/0.70/0.90 per profile
            
            if markup_pct < cap_pct and self.cash > 200:
                # Deploy more cash — up to cap
                target_value = total_equity * cap_pct
                additional = min(target_value - markup_value, self.cash - 100)  # Keep $100 reserve
                if additional > 100:
                    fee = additional * self.taker_fee
                    add_qty = (additional - fee) / price
                    
                    total_cost = self._markup_position_cost + additional
                    total_qty = self._markup_position_qty + add_qty
                    self._markup_avg_entry = total_cost / total_qty if total_qty > 0 else price
                    self._markup_position_qty = total_qty
                    self._markup_position_cost = total_cost
                    self.cash -= additional
                    
                    self.trade_log.append(TradeLogEntry(
                        timestamp=ts, action="MARKUP_DEPLOY_FREED", deal_id=0,
                        lot_id=0, price=price, qty=add_qty, cost_usd=additional,
                        fee=fee, regime=regime,
                    ))
                    logger.info("  📈 MARKUP DEPLOY FREED: +$%.0f at $%.2f (%.0f%% of equity), cash=$%.0f",
                               additional, price, markup_pct * 100, self.cash)
        
        # ── Track high water mark ──
        if high > self._markup_trail_high:
            self._markup_trail_high = high
        
        # ── Check conductor EXIT signal (top detected → unwind) ──
        if self._conductor.should_exit(ts_ms, price, fg_value):
            # Determine if weekly confirmation is needed
            need_weekly = self._v12_weekly_dist_veto or self._conductor._in_price_discovery
            if need_weekly:
                if self._conductor.weekly_confirms_exit(
                        getattr(self, '_accumulated_1h', None), "distribution"):
                    logger.info("  🚨 MARKUP→EXIT: daily_score=%.0f, closing markup position%s",
                               daily_score, " (price discovery)" if self._conductor._in_price_discovery else "")
                    self._close_markup_position(price, ts, "conductor_exit")
                    self._v12_markup_conductor_exits += 1
                    self._transition_to_exit(price, ts, ts_ms, daily_score)
                    return
                else:
                    logger.info("  🚫 %s: MARKUP→EXIT vetoed",
                               "PRICE DISCOVERY VETO" if self._conductor._in_price_discovery else "WEEKLY VETO")
            else:
                logger.info("  🚨 MARKUP→EXIT: daily_score=%.0f, closing markup position", daily_score)
                self._close_markup_position(price, ts, "conductor_exit")
                self._v12_markup_conductor_exits += 1
                self._transition_to_exit(price, ts, ts_ms, daily_score)
                return
        
        # ── Pullback additions (buy the dip during uptrend) ──
        if (self._markup_adds < self._v12_markup_max_adds 
                and self._markup_trail_high > 0):
            pullback_pct = (self._markup_trail_high - price) / self._markup_trail_high * 100
            
            # Only add if price pulled back enough AND is above our avg entry
            # (we're adding to a winner, not averaging down)
            if (pullback_pct >= self._v12_markup_pullback_pct 
                    and price > self._markup_avg_entry
                    and price < self._markup_last_add_price * 0.97):  # At least 3% below last add
                
                add_amount = self._markup_phase_cash * self._v12_markup_pullback_deploy_pct
                if add_amount > 50 and add_amount <= self.cash:
                    fee = add_amount * self.taker_fee
                    add_qty = (add_amount - fee) / price
                    
                    # Update position
                    total_cost = self._markup_position_cost + add_amount
                    total_qty = self._markup_position_qty + add_qty
                    self._markup_avg_entry = total_cost / total_qty if total_qty > 0 else price
                    self._markup_position_qty = total_qty
                    self._markup_position_cost = total_cost
                    self.cash -= add_amount
                    self._markup_adds += 1
                    self._markup_last_add_price = price
                    self._v12_markup_adds_total += 1
                    
                    self.trade_log.append(TradeLogEntry(
                        timestamp=ts, action="MARKUP_ADD", deal_id=0,
                        lot_id=self._markup_adds, price=price, qty=add_qty,
                        cost_usd=add_amount, fee=fee, regime=self._current_regime,
                    ))
                    logger.info("  📈 MARKUP ADD #%d: $%.0f at $%.2f (pullback %.1f%%), total_qty=%.4f, cash=$%.0f",
                               self._markup_adds, add_amount, price, pullback_pct, total_qty, self.cash)
        
        # Periodic logging
        if not hasattr(self, '_markup_log_count'):
            self._markup_log_count = 0
        self._markup_log_count += 1
        if self._markup_log_count % 48 == 0:
            unrealized = (price - self._markup_avg_entry) * self._markup_position_qty
            unrealized_pct = (price - self._markup_avg_entry) / self._markup_avg_entry * 100 if self._markup_avg_entry > 0 else 0
            logger.info("  📊 MARKUP: price=$%.0f, entry=$%.0f, unreal=$%.0f (%.1f%%), high=$%.0f, adds=%d, cash=$%.0f",
                       price, self._markup_avg_entry, unrealized, unrealized_pct,
                       self._markup_trail_high, self._markup_adds, self.cash)
    
    def _close_markup_position(self, sell_price: float, ts: str, reason: str):
        """Close the markup position and return proceeds."""
        if self._markup_position_qty <= 0:
            return
        
        # V12e PARAM SYNC: apply taker fee + slippage to markup close (market order)
        slipped_price = sell_price * (1 - self._v12_slippage_pct / 100)
        gross_proceeds = self._markup_position_qty * slipped_price
        fee = gross_proceeds * self.taker_fee
        proceeds = gross_proceeds - fee
        pnl = proceeds - self._markup_position_cost
        
        self.cash += proceeds
        self._v12_markup_pnl += pnl
        
        self.trade_log.append(TradeLogEntry(
            timestamp=ts, action=f"MARKUP_CLOSE_{reason.upper()}", deal_id=0,
            lot_id=0, price=slipped_price, qty=self._markup_position_qty,
            cost_usd=self._markup_position_cost, fee=fee, pnl=pnl,
            regime=self._current_regime, sell_reason=reason,
        ))
        logger.info("  💰 MARKUP CLOSE (%s): sold %.4f at $%.2f, pnl=$%.2f (%.1f%%), proceeds=$%.0f",
                    reason, self._markup_position_qty, sell_price, pnl,
                    (sell_price - self._markup_avg_entry) / self._markup_avg_entry * 100 if self._markup_avg_entry > 0 else 0,
                    proceeds)
        
        # Reset
        self._markup_position_qty = 0.0
        self._markup_position_cost = 0.0
        self._markup_avg_entry = 0.0
        self._markup_trail_high = 0.0
        self._markup_trail_stop = 0.0
    
    # ══════════════════════════════════════════════════════════════
    #  EQUITY CALCULATION (include exit lots + spring entries)
    # ══════════════════════════════════════════════════════════════
    
    def _equity(self, price: float) -> float:
        """Override equity to include exit lots, spring entries, and short positions."""
        # Base: cash + open DCA deal values (from parent)
        base = super()._equity(price)
        
        # Add market value of open exit lots (capital is deployed, not in cash)
        for lot in self._exit_lots:
            if not lot.sold:
                base += lot.qty * price
        
        # Add market value of open spring entries
        for entry in self._spring_entries:
            if not entry.get("closed"):
                base += entry["qty"] * price
        
        # Add markup position market value
        if self._markup_position_qty > 0:
            base += self._markup_position_qty * price
        
        # Add short position: margin + unrealized PnL
        if self._exit_short and not self._exit_short.closed:
            s = self._exit_short
            base += s.total_margin + (s.avg_entry - price) * s.total_qty - s.funding_cost
        
        return base
    
    # ══════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════
    
    def run(self, df: pd.DataFrame) -> BacktestResult:
        if len(df) < 100:
            logger.warning("Not enough data (%d rows)", len(df))
            return BacktestResult()
        
        logger.info("Running V12 backtest (3-engine lifecycle): %s %s, $%.0f, "
                     "exit_thresh=%.0f, mcap_ath=%.0f%%, trail=%.1f%%→%.1f%%",
                     self.symbol, self.timeframe, self.initial_capital,
                     self._v12_exit_threshold, self._v12_mcap_ath_pct * 100,
                     self._v12_initial_trail_pct, self._v12_trail_floor_pct)
        
        self._candle_timeline = []
        self._spring_candle_count = 0
        self._run_main_loop(df)
        
        # Force close any remaining positions
        last_price = float(df.iloc[-1]["close"])
        last_ts = str(df.iloc[-1]["timestamp"])
        
        # Close remaining DCA deals
        for deal in list(self.deals):
            self._force_close_deal(deal, last_price, last_ts)
        
        # Close markup position
        if self._markup_position_qty > 0:
            self._close_markup_position(last_price, last_ts, "backtest_end")
        
        # Close remaining exit lots
        self._force_sell_all_exit_lots(last_price, last_ts, "backtest_end")
        
        # Close exit short
        self._close_exit_short(last_price, last_ts, "backtest_end")
        
        # Close spring entries (with fees)
        for entry in self._spring_entries:
            if not entry.get("closed"):
                slipped = last_price * (1 - self._v12_slippage_pct / 100)
                gross = entry["qty"] * slipped
                fee = gross * self.taker_fee
                proceeds = gross - fee
                pnl = proceeds - entry["cost"]
                entry["closed"] = True
                entry["close_price"] = slipped
                entry["close_time"] = last_ts
                entry["pnl"] = pnl
                self.cash += proceeds
                self._v12_spring_pnl += pnl
        
        result = self._compile_results(df)
        result.variant = "v12_lifecycle_engine"
        
        result.extra = {
            "v12_params": self.v12_params,
            "v8_spring_buys": self._v8_spring_buys,
            "v8_phase_candles": self._v8_phase_candles,
            "v9_dist_phase_candles": self._v9_phase_candles,
            "v9_force_exits": self._v9_force_exits,
            # V12 metrics
            "v12_exit_phases": self._v12_exit_phases,
            "v12_spring_phases": self._v12_spring_phases,
            "v12_rally_sells": self._v12_rally_sells,
            "v12_trail_stops": self._v12_trail_stops,
            "v12_urgency_closes": self._v12_urgency_closes,
            "v12_short_pnl": round(self._v12_short_pnl, 2),
            "v12_spring_pnl": round(self._v12_spring_pnl, 2),
            "v12_spring_deploys": self._v12_spring_deploys,
            "v12_false_springs": self._v12_false_springs,
            "v12_exit_pnl_preserved": round(self._v12_exit_pnl_preserved, 2),
            "v12_markdown_phases": self._v12_markdown_phases,
            "v12_markup_phases": self._v12_markup_phases,
            "v12_markup_pnl": round(self._v12_markup_pnl, 2),
            "v12_markup_adds": self._v12_markup_adds_total,
            "v12_markup_trail_exits": self._v12_markup_trail_exits,
            "v12_markup_conductor_exits": self._v12_markup_conductor_exits,
            "v12_breakout_entries": self._v12_breakout_entries,
        }
        
        return result
