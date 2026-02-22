"""Cold-start phase classifier for V12e lifecycle engine.

Determines the correct starting Wyckoff phase for a coin based on:
- Distance from ATH
- Market regime (from recent candles)
- Trend direction (SMA50)
- Crypto Fear & Greed Index
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from .lifecycle_engine import LifecyclePhase

logger = logging.getLogger(__name__)


def _get_trend_and_regime(candle_db, symbol: str, df: Optional[pd.DataFrame] = None) -> Tuple[Optional[str], Optional[bool]]:
    """Get regime and trend from candle data.
    
    If a DataFrame is provided directly, uses that (preferred for cold-start).
    Otherwise falls back to candle_db lookup.
    
    Uses last 200 candles for regime detection and SMA50 trend.
    
    Returns: (regime, is_bullish) or (None, None) if insufficient data.
    """
    # Prefer direct DataFrame over DB lookup
    if df is not None and len(df) >= 50:
        df_full = df
    elif candle_db is not None:
        df_full = candle_db.get_candles(symbol, "1h", limit=1200)
        if df_full is None or len(df_full) < 50:
            return None, None
    else:
        return None, None

    # Trend: SMA50 on full history
    sma50 = df_full["close"].rolling(50).mean()
    current_price = float(df_full["close"].iloc[-1])
    sma50_val = float(sma50.iloc[-1])
    is_bullish = current_price >= sma50_val if not pd.isna(sma50_val) else None

    # Regime: use last 200 candles
    df_regime_src = df_full.tail(200).copy().reset_index(drop=True)
    regime = None
    if len(df_regime_src) >= 100:
        try:
            from ..regime_detector import classify_regime_v2
            # Need datetime timestamps for regime detector
            if pd.api.types.is_numeric_dtype(df_regime_src["timestamp"]):
                df_regime_src["timestamp"] = pd.to_datetime(df_regime_src["timestamp"], unit="ms")
            regimes = classify_regime_v2(df_regime_src, "1h")
            regime = regimes.iloc[-1] if len(regimes) > 0 else None
        except Exception as e:
            logger.warning("Regime detection failed for %s: %s", symbol, e)

    return regime, is_bullish


def classify_phase(
    symbol: str,
    current_price: float,
    ath: float,
    candle_db=None,
    cfgi_value: Optional[float] = None,
    candles_df: Optional[pd.DataFrame] = None,
) -> Tuple[LifecyclePhase, str]:
    """Determine the correct starting phase for a coin.
    
    Args:
        candles_df: Pre-fetched DataFrame of candles (preferred over candle_db).
                    Should have 200+ rows for regime detection, 50+ for trend.
    
    Returns: (LifecyclePhase, reason_string)
    """
    if ath <= 0 or current_price <= 0:
        reason = f"invalid data (price={current_price}, ath={ath})"
        return LifecyclePhase.DCA, reason

    ath_distance = (ath - current_price) / ath  # fraction below ATH
    ath_pct = ath_distance * 100

    regime, is_bullish = _get_trend_and_regime(candle_db, symbol, df=candles_df)

    regime_str = regime or "UNKNOWN"
    trend_str = "bullish" if is_bullish else ("bearish" if is_bullish is False else "unknown")
    cfgi_str = f"{cfgi_value:.0f}" if cfgi_value is not None else "N/A"

    detail = f"{ath_pct:.1f}% below ATH, regime={regime_str}, trend={trend_str}, CFGI={cfgi_str}"

    # EXIT: very near ATH with greed/distribution signals
    if ath_distance < 0.10:
        if (cfgi_value is not None and cfgi_value > 75) or regime == "DISTRIBUTION":
            return LifecyclePhase.EXIT, f"EXIT ({detail})"

    # MARKUP: near ATH and trending up
    if ath_distance < 0.15 and regime in ("TRENDING", "MILD_TREND") and is_bullish:
        return LifecyclePhase.MARKUP, f"MARKUP ({detail})"

    # SPRING: deep discount with accumulation signs
    if ath_distance > 0.50 and regime == "ACCUMULATION" and is_bullish:
        return LifecyclePhase.SPRING, f"SPRING ({detail})"

    # MARKDOWN: mid-range decline, bearish
    if 0.15 < ath_distance < 0.50 and is_bullish is False and regime != "ACCUMULATION":
        return LifecyclePhase.MARKDOWN, f"MARKDOWN ({detail})"

    # DCA: significant discount or ranging/accumulation
    if ath_distance > 0.25 or regime in ("RANGING", "CHOPPY", "ACCUMULATION"):
        return LifecyclePhase.DCA, f"DCA ({detail})"

    # Default
    return LifecyclePhase.DCA, f"DCA/default ({detail})"
