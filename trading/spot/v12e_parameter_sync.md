# V12e Parameter Sync — 2025-02-22

Synchronized live V12e code with backtest engine parameters.  
All changes are backwards compatible. Bots need manual restart.

---

## 1. Compounding (lifecycle_trader.py)

**Problem:** Live used `self.initial_capital` for position sizing every cycle.  
**Fix:** Use current equity at deal open (compounding). Mid-deal SOs scale from the deal's actual base lot cost.

| Location | Before | After |
|---|---|---|
| `_open_deal()` LIVE | `self.initial_capital * self.profile.base_order_pct` | `current_equity * self.profile.base_order_pct` (equity = exchange balance) |
| `_open_deal()` PAPER | `self.initial_capital * self.profile.base_order_pct` | `current_equity * self.profile.base_order_pct` (equity from `_current_equity_for_sizing()`) |
| `_open_deal()` V12f allocations | `compute_phase_allocations(self.initial_capital, ...)` | `compute_phase_allocations(current_equity, ...)` |
| `_check_safety_orders()` | `self.initial_capital * self.profile.base_order_pct` | `deal.lots[0].cost_usd` (scale from deal's actual base) |

**New method:** `_current_equity_for_sizing(prices)` — estimates equity for paper mode.

---

## 2. Spring Parameters (lifecycle_engine.py + backtest_engine_v12.py)

**Problem:** Live and backtest spring discounts/TP didn't match intended values.

| Parameter | Before (both) | After (both) |
|---|---|---|
| `spring_tier2_discount` | 35% | **28%** |
| `spring_tier3_discount` | 45% | **35%** |
| `spring_tp_pct` | 10% | **15%** |
| `spring_tier1_discount` | 25% | 25% (unchanged) |

Files changed:
- `lifecycle_engine.py`: `LifecycleConfig` defaults
- `backtest_engine_v12.py`: `__init__` kwargs defaults

---

## 3. 48h Commitment Window

**Status:** Already implemented in both live and backtest. No changes needed.

- Backtest: `_v12_commitment_hours=48` checked in `_run_exit_candle()`
- Live: `commitment_hours=48` in `LifecycleConfig`, checked in both `_process_dca()` (pre-transition) and `_process_exit()` (post-transition)
- Live additionally has CFGI-modulated fast commitment (24h if CFGI > 75) — backtest doesn't have this (live is stricter)

---

## 4. Realistic Fees + Slippage in Backtest (backtest_engine_v12.py)

**Problem:** Lifecycle trades (EXIT sells, short open/close, markup close, spring TP) charged 0 fees.

**New parameter:** `v12_slippage_pct` (default 0.05% = 5bps). Applied to market-like fills only.

| Trade Type | Fee | Slippage | Notes |
|---|---|---|---|
| EXIT lot sell (trailing_stop, rally_sell) | taker_fee | No | Trailing stop = limit-like |
| EXIT lot sell (force_close, urgency, invalidation) | taker_fee | Yes | Market order |
| Short open (all tiers) | taker_fee | Yes | Market order, slippage = higher entry |
| Short close | taker_fee | Yes | Market order, slippage = higher buy-back |
| Markup close | taker_fee | Yes | Market order |
| Spring TP | **maker_fee** | No | Limit order fills at set price |
| Spring false spring sell | taker_fee | Yes | Panic market sell |
| Spring recovery/timeout close | taker_fee | Yes | Market order |
| Spring deploy buy | taker_fee | No | Already had fee (existing code) |

Exchange fees (from `exchange_adapter` registry):
- Aster: 0% maker / 0.04% taker
- Hyperliquid: 0.02% maker / 0.05% taker

---

## 5. Stale Fee Table (backtest_engine_v3.py)

**Note:** V3 has its own fee handling — not modified (legacy). V12 inherits `self.taker_fee` and `self.maker_fee` from V3's `__init__` which reads from `exchange_adapter.EXCHANGE_REGISTRY` (single source of truth). No duplication issue.

---

## Files Modified

- `trading/spot/lifecycle_trader.py` — compounding (#1)
- `trading/spot/lifecycle_engine.py` — spring params (#2)  
- `trading/spot/backtest_engine_v12.py` — spring params (#2), fees + slippage (#4)
- `trading/spot/v12e_parameter_sync.md` — this file
