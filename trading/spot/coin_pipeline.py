"""Coin Pipeline: Connects scanner results → spot trader via commands.json.

Reads scanner output (scanner_recommendation.json / scanner_t1.json),
evaluates which coins should be traded based on score + lifecycle phase,
and writes add/remove commands for the spot trader to pick up.

Works for both live and paper modes — just point output_dir at the
trader's paper_dir or live_dir.

Design principles:
- Scanner runs independently (4h cron) and writes JSON
- Pipeline reads scanner JSON + trader status.json
- Pipeline writes commands.json → trader picks up next cycle
- No direct imports of trader — pure file-based coordination
- CFGI mandatory: coins without CFGI data are excluded
- Minimum hold time enforced: don't churn coins
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

# Scanner output locations
SCANNER_DIR = Path(__file__).resolve().parent.parent / "live"
SCANNER_RECOMMENDATION = SCANNER_DIR / "scanner_recommendation.json"
SCANNER_T1 = SCANNER_DIR / "scanner_t1.json"

# CFGI cache directory
CFGI_CACHE_DIR = Path(__file__).parent / "data" / "cfgi_cache"

# Pipeline parameters
MIN_SCORE = 25.0             # minimum scanner score to be considered
ROTATION_THRESHOLD = 0.20    # 20% improvement required to replace a coin
MIN_HOLD_HOURS = 24          # minimum hours before a coin can be rotated out
STALE_SCANNER_HOURS = 12     # ignore scanner results older than this
MIN_SCORE_TO_KEEP = 15.0     # below this score, coin gets removed even without replacement


def has_cfgi_data(symbol: str) -> bool:
    """Check if a coin has CFGI data cached (mandatory for certification)."""
    token = symbol.split("/")[0].upper()
    cache_path = CFGI_CACHE_DIR / f"{token}_cfgi_daily.json"
    if not cache_path.exists():
        return False
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not data or len(data) < 30:  # need at least 30 days
            return False
        # Check freshness — last entry should be within 7 days
        last_date = data[-1].get("date", "")
        if last_date:
            last_dt = datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_dt).days > 7:
                logger.warning("CFGI data stale for %s (last: %s)", symbol, last_date)
                return False
        return True
    except Exception:
        return False


def _read_scanner_results() -> Optional[dict]:
    """Read the latest scanner recommendation."""
    if not SCANNER_RECOMMENDATION.exists():
        logger.warning("No scanner recommendation found at %s", SCANNER_RECOMMENDATION)
        return None
    try:
        data = json.loads(SCANNER_RECOMMENDATION.read_text(encoding="utf-8"))
        # Check staleness
        ts_str = data.get("timestamp", "")
        if ts_str:
            ts = datetime.fromisoformat(ts_str)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_hours > STALE_SCANNER_HOURS:
                logger.warning("Scanner results stale (%.1fh old), skipping", age_hours)
                return None
        return data
    except Exception as e:
        logger.error("Failed to read scanner results: %s", e)
        return None


def _read_trader_status(output_dir: Path) -> Optional[dict]:
    """Read the trader's current status.json."""
    status_path = output_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read trader status: %s", e)
        return None


def _get_coin_start_times(output_dir: Path) -> Dict[str, datetime]:
    """Get when each coin was added (from state.json or status.json)."""
    state_path = output_dir / "state.json"
    start_times = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            coin_starts = state.get("coin_start_times", {})
            for sym, ts_str in coin_starts.items():
                try:
                    start_times[sym] = datetime.fromisoformat(ts_str)
                except Exception:
                    pass
        except Exception:
            pass
    return start_times


def _write_commands(output_dir: Path, commands: List[dict]):
    """Write commands.json for the trader to pick up."""
    cmd_path = output_dir / "commands.json"
    # Don't overwrite existing unprocessed commands — append
    existing = []
    if cmd_path.exists():
        try:
            existing = json.loads(cmd_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    combined = existing + commands
    cmd_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    logger.info("📋 Wrote %d command(s) to %s", len(commands), cmd_path)


def evaluate_coins(
    output_dir: Path,
    max_coins: int = 3,
    dry_run: bool = False,
) -> List[dict]:
    """Evaluate scanner results against current portfolio and generate commands.
    
    Returns list of commands that were (or would be, if dry_run) written.
    """
    scanner = _read_scanner_results()
    if not scanner:
        logger.info("No actionable scanner results")
        return []

    status = _read_trader_status(output_dir)
    coin_starts = _get_coin_start_times(output_dir)
    now = datetime.now(timezone.utc)

    # Current coins from status
    current_symbols = []
    current_phases = {}
    if status:
        # Get active symbols from status coins dict
        coins_info = status.get("coins", {})
        current_symbols = list(coins_info.keys())
        current_phases = status.get("coin_phases", {})

    # Build scored candidate list (CFGI-certified only)
    top_coins = scanner.get("top_5", [])
    candidates = []
    for coin in top_coins:
        sym = coin["symbol"]
        score = coin.get("score", 0)
        if score < MIN_SCORE:
            continue
        if not has_cfgi_data(sym):
            logger.info("⚠️ %s excluded — no CFGI data", sym)
            continue
        candidates.append({"symbol": sym, "score": score, **coin})

    if not candidates:
        logger.info("No CFGI-certified candidates above minimum score")
        return []

    logger.info("📊 Scanner candidates (CFGI-certified, score>%.0f): %s",
                MIN_SCORE,
                [(c["symbol"], c["score"]) for c in candidates])
    logger.info("📊 Current coins: %s", current_symbols)

    commands = []

    # Phase 1: Check if any current coins should be removed
    # (score dropped below threshold, or not in scanner top results)
    candidate_syms = {c["symbol"] for c in candidates}
    candidate_scores = {c["symbol"]: c["score"] for c in candidates}

    for sym in current_symbols:
        score = candidate_scores.get(sym, 0)
        if score >= MIN_SCORE_TO_KEEP:
            continue  # Still good

        # Check minimum hold time
        start = coin_starts.get(sym)
        if start:
            held_hours = (now - start).total_seconds() / 3600
            if held_hours < MIN_HOLD_HOURS:
                logger.info("⏳ %s score=%.1f below threshold but held only %.1fh (min %dh)",
                            sym, score, held_hours, MIN_HOLD_HOURS)
                continue

        # Check if coin is in active deal — only remove if no active position
        phase = current_phases.get(sym, "DCA")
        if phase in ("EXIT", "MARKDOWN"):
            logger.info("🔄 %s score=%.1f dropping but in %s — letting lifecycle complete",
                        sym, score, phase)
            continue

        logger.info("❌ %s score=%.1f below keep threshold %.1f — removing",
                    sym, score, MIN_SCORE_TO_KEEP)
        commands.append({
            "action": "remove_coin",
            "symbol": sym,
            "reason": f"score={score:.1f} < {MIN_SCORE_TO_KEEP}",
            "timestamp": now.isoformat(),
        })

    # Phase 2: Check if better coins should replace weaker ones
    # (only if at max_coins and a candidate is significantly better)
    remaining_symbols = [s for s in current_symbols
                         if not any(c["action"] == "remove_coin" and c["symbol"] == s for c in commands)]

    if len(remaining_symbols) >= max_coins:
        # Find the weakest current coin
        weakest_sym = None
        weakest_score = float("inf")
        for sym in remaining_symbols:
            score = candidate_scores.get(sym, 0)
            # Don't replace coins in EXIT/MARKDOWN
            phase = current_phases.get(sym, "DCA")
            if phase in ("EXIT", "MARKDOWN"):
                continue
            # Check hold time
            start = coin_starts.get(sym)
            if start:
                held_hours = (now - start).total_seconds() / 3600
                if held_hours < MIN_HOLD_HOURS:
                    continue
            if score < weakest_score:
                weakest_score = score
                weakest_sym = sym

        if weakest_sym:
            # Find best candidate not already active
            for cand in candidates:
                if cand["symbol"] in remaining_symbols:
                    continue
                improvement = (cand["score"] - weakest_score) / max(weakest_score, 1)
                if improvement >= ROTATION_THRESHOLD:
                    logger.info("🔄 Rotating: %s (%.1f) → %s (%.1f) [+%.0f%%]",
                                weakest_sym, weakest_score,
                                cand["symbol"], cand["score"],
                                improvement * 100)
                    commands.append({
                        "action": "switch_coin",
                        "from": weakest_sym,
                        "to": cand["symbol"],
                        "reason": f"{cand['symbol']}={cand['score']:.1f} vs {weakest_sym}={weakest_score:.1f} (+{improvement*100:.0f}%)",
                        "timestamp": now.isoformat(),
                    })
                    remaining_symbols.remove(weakest_sym)
                    remaining_symbols.append(cand["symbol"])
                    break  # one rotation per cycle

    # Phase 3: Fill empty slots with best available candidates
    slots_available = max_coins - len(remaining_symbols)
    if slots_available > 0:
        for cand in candidates:
            if slots_available <= 0:
                break
            if cand["symbol"] in remaining_symbols:
                continue
            logger.info("➕ Adding %s (score=%.1f) — %d slot(s) available",
                        cand["symbol"], cand["score"], slots_available)
            commands.append({
                "action": "add_coin",
                "symbol": cand["symbol"],
                "reason": f"score={cand['score']:.1f}, slot available",
                "timestamp": now.isoformat(),
            })
            remaining_symbols.append(cand["symbol"])
            slots_available -= 1

    # Write commands
    if commands:
        logger.info("📋 Generated %d command(s):", len(commands))
        for cmd in commands:
            logger.info("  %s: %s (%s)", cmd["action"],
                        cmd.get("symbol", cmd.get("to", "?")),
                        cmd.get("reason", ""))
        if not dry_run:
            _write_commands(output_dir, commands)
    else:
        logger.info("✅ No changes needed — current portfolio is optimal")

    return commands
