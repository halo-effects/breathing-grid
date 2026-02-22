# V2: Conviction-Gated Leverage — Product Specification

*Version: 0.2 (Scoping Draft) | Created: 2026-02-22 | Updated: 2026-02-22 | Status: PROPOSED — Not Committed*

---

## 1. Overview & Philosophy

### 1.1 Context

V12e MVP delivers a complete spot-only Wyckoff lifecycle engine across five phases (DCA → EXIT → MARKDOWN → SPRING → MARKUP) with three risk profiles, short hedging, and conductor-scored phase transitions. It trades at 1x — no leverage, no margin.

V2 introduces **conviction-gated leverage** as an opt-in enhancement for Medium and High risk profiles, covering all three profitable phases: leveraged longs in SPRING and MARKUP, leveraged shorts in MARKDOWN. The core thesis:

> **Leverage is a weapon, not a strategy.** Only deploy when the lifecycle engine has maximum confidence in the phase signal.

### 1.2 Design Principles

1. **Leverage amplifies confirmed signals, never speculative positions.** If you can't explain why leverage is on right now in one sentence, it shouldn't be on.
2. **Light leverage only.** 2-3x maximum. This isn't a margin casino — it's a precision amplifier for high-conviction Wyckoff events.
3. **Binary activation.** No partial leverage. Either the conviction score clears the threshold and leverage snaps on, or it doesn't. No sliding scale, no "a little bit leveraged."
4. **Base stays spot.** The existing 1x spot position is untouched. Leverage applies only to an additional capital allocation via futures/margin — the "conviction bonus."
5. **Fail-safe by default.** Every leverage position has a max duration, tighter trailing stops, auto-delever triggers, and emergency kill switches.

### 1.3 What V2 Is NOT

- Not a perpetual futures trading bot
- Not available to Low risk profiles under any circumstances
- Not active during DCA or EXIT phases
- Not a way to take speculative positions outside the lifecycle engine's phase signals

---

## 2. Leverage Activation Rules

### 2.1 Eligible Phases

| Phase | Leverage Allowed | Rationale |
|-------|-----------------|-----------|
| **SPRING** | ✅ Leveraged long | Deep discount + extreme fear = highest conviction long signal |
| **MARKUP** | ✅ Leveraged long | Breakout confirmed, regime trending, momentum validated |
| **MARKDOWN** | ✅ Leveraged short | Distribution confirmed, greed still high = room to fall |
| **DCA** | ❌ Never | DCA = uncertainty. "We think it'll go up eventually" ≠ conviction. |
| **EXIT** | ❌ Never | Exiting positions, not adding exposure. |

**Decision rationale:** SPRING, MARKUP, and MARKDOWN are the three phases where the conductor, CFGI, regime, and price action converge to produce a directional thesis strong enough to warrant amplification. DCA is structurally uncertain — the whole point is systematic buying despite not knowing the bottom. EXIT is about de-risking, not adding exposure.

### 2.2 SPRING Activation Criteria

All four conditions must be TRUE simultaneously:

| Signal | Threshold | Source |
|--------|-----------|--------|
| CFGI | < 25 (extreme fear) | Crypto Fear & Greed Index |
| ATH discount | > 30% below ATH | `KNOWN_ATH` dict / live ATH tracker |
| Conductor score | > 70 | `DailyScorerConductor.score_at()` |
| Spring tier fills | ≥ 2 of 3 tiers filled | Lifecycle engine spring state |

**Why these thresholds:**
- CFGI < 25 historically correlates with capitulation events. The existing engine uses `cfgi_spring_extreme_fear: 15` and `cfgi_spring_fear: 25` — we use the fear boundary.
- 30% ATH discount ensures meaningful reversion potential. The spring tier1 discount is already 25%; requiring 30% adds a buffer.
- Conductor > 70 confirms the daily scoring model sees accumulation signals. The EXIT threshold is 50; requiring 70 for leverage is substantially more selective.
- 2 of 3 tiers filled means we have meaningful capital deployed — this isn't a toe-dip, it's a committed spring position.

### 2.3 MARKUP Activation Criteria

All four conditions must be TRUE simultaneously:

| Signal | Threshold | Source |
|--------|-----------|--------|
| Conductor score | > 80 | `DailyScorerConductor.score_at()` |
| Regime | TRENDING | `classify_regime_v2()` |
| SMA50 | Bullish (price > SMA50) | Computed from 1h candles |
| Phase transition | Confirmed SPRING → MARKUP | Lifecycle engine state |

**Why these thresholds:**
- Conductor > 80 is more selective than spring (> 70) because markup comes after spring — we need stronger confirmation that the breakout is real.
- TRENDING regime (not MILD_TREND) ensures momentum is validated. MILD_TREND allows leverage-free markup; TRENDING gates leverage.
- SMA50 bullish is a simple, explainable trend filter. If price is below the 50-period moving average, the markup thesis is weakened.
- Must have transitioned FROM spring — no leverage on cold-start markups.

### 2.4 MARKDOWN Activation Criteria (Leveraged Shorts)

All four conditions must be TRUE simultaneously:

| Signal | Threshold | Source |
|--------|-----------|--------|
| Conductor score | > 75 | `DailyScorerConductor.score_at()` (bearish reading) |
| Price breakdown | Below EXIT level | Price broke below the level that triggered EXIT → MARKDOWN transition |
| CFGI | > 70 (greed still high) | Crypto Fear & Greed Index |
| Regime | DISTRIBUTION or EXTREME | `classify_regime_v2()` |

**Why these thresholds:**
- Conductor > 75 confirms the daily scoring model sees distribution/markdown signals. Lower than MARKUP's 80 requirement because shorts are inherently riskier (unlimited upside risk) and we want to be selective but not impossibly so.
- Price below EXIT level is structural confirmation — the lifecycle engine already validated the top. This isn't a speculative short, it's amplifying a confirmed breakdown.
- CFGI > 70 is the contrarian bearish signal: when the market is still greedy during a confirmed markdown, there's more room to fall. Extreme greed (> 80) during distribution is historically the strongest short signal. If fear has already set in (CFGI < 50), the easy money in the short is gone.
- DISTRIBUTION regime means the Wyckoff distribution pattern is playing out. EXTREME is also eligible because markdown phases that enter extreme volatility often accelerate downward before capitulation.

**Why shorts are different from longs:**
- Shorts have asymmetric risk: max gain is 100% (price → 0), max loss is unlimited (price → ∞)
- This asymmetry is why the safety rails in §6.6 are tighter than for leveraged longs
- The CFGI signal is inverted: high greed = bullish for shorts (contrarian), vs. high fear = bullish for spring longs

---

## 3. Profile Gating

| Profile | Max Long Leverage | Max Short Leverage | Phases | Eligible Coins | Rationale |
|---------|------------------|-------------------|--------|----------------|-----------|
| **Low** | 1x (none) | 1x (none) | N/A | N/A | Capital preservation is the mandate. Zero leverage, period. |
| **Medium** | 2x | 2x | SPRING, MARKUP, MARKDOWN | Mature only (BTC, ETH, SOL) | Balanced risk-reward. Restricting to mature coins limits liquidity/slippage risk. |
| **High** | 3x | 3x | SPRING, MARKUP, MARKDOWN | All certified coins | Maximum return potential. Accepts thinner order books on certified alts. |

**Decision rationale:** Low profile users chose capital preservation — leverage contradicts their stated preference. Medium gets conservative leverage (2x) on the most liquid, backtested assets. High accepts more risk for more upside, including certified alts with less Wyckoff history. Short leverage uses the same multipliers as longs — the tighter safety rails on shorts (§6.6) compensate for asymmetric risk.

---

## 4. Conviction Scoring System

### 4.1 Formula

The conviction score is a weighted composite of four normalized signals:

```
CONVICTION = (W_cfgi × S_cfgi) + (W_conductor × S_conductor) + (W_ath × S_ath) + (W_regime × S_regime)
```

**Weights:**

| Signal | Weight | Rationale |
|--------|--------|-----------|
| CFGI sentiment | 0.25 | Market-wide fear/greed context |
| Conductor score | 0.35 | Phase transition confidence — highest weight because it's our primary signal |
| ATH distance | 0.20 | Magnitude of discount/opportunity |
| Regime alignment | 0.20 | Technical regime confirmation |

**Total: 1.00**

### 4.2 Signal Normalization (0–100 scale)

**S_cfgi (phase-dependent — sentiment must support the thesis):**
```
if phase == SPRING:
    S_cfgi = max(0, min(100, (50 - CFGI) × 2))    # CFGI 0 → 100, CFGI 25 → 50, CFGI 50 → 0
elif phase == MARKUP:
    S_cfgi = max(0, min(100, (CFGI - 30) × 2))     # CFGI 80 → 100, CFGI 55 → 50, CFGI 30 → 0
elif phase == MARKDOWN:
    S_cfgi = max(0, min(100, (CFGI - 40) × 2))     # CFGI 90 → 100, CFGI 70 → 60, CFGI 40 → 0
```
*Rationale: In SPRING, extreme fear is bullish (contrarian). In MARKUP, rising greed confirms momentum. In MARKDOWN, high greed means the market hasn't capitulated yet — more room to fall. CFGI 70+ scores well because complacent greed during a confirmed distribution is the strongest short signal.*

**S_conductor:**
```
S_conductor = conductor_score    # Already 0–100 from DailyScorerConductor
```

**S_ath (distance from ATH):**
```
ath_discount_pct = (ATH - price) / ATH × 100
if phase == SPRING:
    S_ath = max(0, min(100, (ath_discount_pct - 20) × 2.5))  # 20% discount → 0, 60% → 100
elif phase == MARKUP:
    S_ath = max(0, min(100, 100 - ath_discount_pct × 2))      # Near ATH = high score in markup
elif phase == MARKDOWN:
    S_ath = max(0, min(100, 100 - ath_discount_pct × 1.5))    # Near ATH = more room to fall
```
*In MARKDOWN, proximity to ATH means the drop is early — more downside potential. If price is already 50%+ below ATH, the short opportunity is largely spent.*

**S_regime (phase-dependent):**
```
REGIME_SCORES_LONG = {       # For SPRING and MARKUP
    "TRENDING": 100,
    "MILD_TREND": 70,
    "BREAKOUT_WARNING": 50,
    "ACCUMULATION": 60,
    "RANGING": 30,
    "CHOPPY": 10,
    "DISTRIBUTION": 5,
    "EXTREME": 0,
}
REGIME_SCORES_SHORT = {      # For MARKDOWN
    "DISTRIBUTION": 100,
    "EXTREME": 85,
    "BREAKOUT_WARNING": 60,
    "CHOPPY": 30,
    "RANGING": 20,
    "MILD_TREND": 10,
    "TRENDING": 0,           # Trending up = worst time to short
    "ACCUMULATION": 5,
}
S_regime = REGIME_SCORES_LONG[regime] if phase in (SPRING, MARKUP) else REGIME_SCORES_SHORT[regime]
```

### 4.3 Thresholds

| Conviction Score | Leverage Eligible |
|-----------------|-------------------|
| < 85 | No leverage (1x only) |
| ≥ 85 | 2x eligible (Medium + High profiles) |
| ≥ 92 | 3x eligible (High profile only) |

**Why 85 and 92:**
- At 85, you need strong readings across most signals. Example: conductor 90 (31.5) + CFGI extreme at 80 (20) + 35% ATH discount giving 75 (15) + TRENDING regime (20) = 86.5. That's a genuine high-conviction setup.
- At 92, you need near-perfect alignment. This should fire rarely — a few times per full market cycle. The gap between 85 and 92 is intentionally narrow to make 3x a rare event.
- No partial leverage: binary on/off eliminates complexity and prevents "conviction creep" where 1.7x positions accumulate into unclear risk.

### 4.4 Explainability Requirement

Every leverage activation must produce a human-readable sentence:

> "Leverage LONG (2x): SPRING phase, CFGI=18 (extreme fear), conductor=87, BTC 38% below ATH, regime=ACCUMULATION. Conviction: 88.2"

> "Leverage SHORT (2x): MARKDOWN phase, CFGI=74 (greed), conductor=79, ETH 8% below ATH, regime=DISTRIBUTION. Conviction: 86.1"

If the system can't produce this sentence, leverage stays off. This is enforced in code, not just policy.

---

## 5. Coin Maturity Tiers

### 5.1 Mature Coins (Medium + High profiles)

**Current list:** BTC, ETH, SOL

**Qualification criteria (all must be met):**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Market cap | > $10B | Deep liquidity, institutional flow |
| Trading history | > 4 years | Sufficient Wyckoff cycle data (at least 1 full cycle) |
| 24h volume | > $500M average | Slippage tolerable at 2-3x position sizes |
| CFGI data | **Dedicated index required** | Conviction scoring requires per-coin CFGI — no proxy |
| Backtest win rate | > 70% across lifecycle phases | Proven strategy fit |
| Exchange support | Listed on ≥ 2 supported exchanges | Execution redundancy |

### 5.2 Certified Coins (High profile only)

**Current list:** Full certified pool from scanner (passes validation pipeline).

**Qualification criteria (subset of Mature):**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Market cap | > $500M | Minimum liquidity for leveraged positions |
| Trading history | > 1 year | Enough data for phase classification |
| 24h volume | > $50M average | Acceptable but thinner order books |
| CFGI data | **Dedicated index required** | No proxy — if cfgi.io doesn't track it, it's not certified |
| Backtest win rate | > 60% across lifecycle phases | Lower bar, compensated by diversification |
| Scanner validated | Passes certification pipeline | Fundamental/technical screening |

### 5.3 Key Decisions

- **Mature list is DYNAMIC.** Coins can graduate in/out based on §5.1 criteria. Re-evaluated every scanner cycle (4h). Crypto moves fast — today's alt is tomorrow's blue chip.
- **No CFGI = no certification.** Hard gate. If cfgi.io doesn't track a coin with a dedicated index, it cannot be traded by the lifecycle engine. No proxying BTC CFGI for alts. The conviction scoring formula requires per-coin sentiment data.
- **Certification = scanner validation + CFGI availability.** Both must pass. This is the single gate for all coins across all profiles.
- **HYPE example:** Market cap may qualify, but if CFGI doesn't track HYPE specifically → not certified → not tradeable. No exceptions.

### 5.4 Open Questions

- **Q:** Should maturity re-evaluation trigger immediate position changes, or only affect new deal openings? (Recommend: only new deals — don't yank capital from active lifecycles.)

---

## 6. Safety Rails

### 6.1 Auto-Deleveraging

| Trigger | Action | Profile Thresholds |
|---------|--------|-------------------|
| Portfolio drawdown | Close all leveraged positions immediately | Low: 15%, Medium: 25%, High: 35% |
| EXTREME regime detected | Emergency delever all positions | All profiles |
| Phase transition away from SPRING/MARKUP | Close leverage component | All profiles |
| Conviction score drops below 75 | Close leverage (10-point hysteresis below 85 entry) | All profiles |

**Hysteresis rationale:** Entry at 85, exit at 75 prevents whipsaw. A 10-point buffer means conviction must deteriorate meaningfully before deleveraging.

### 6.2 Duration Limits

- **Max leverage duration:** 7 days per position
- After 7 days, leverage component auto-reduces to 1x (spot position remains)
- Can re-activate if conviction score still qualifies on next evaluation cycle
- **Rationale:** Extended leverage bleeds funding costs and increases tail risk. 7 days covers typical spring/markup confirmation windows.

### 6.3 Position Structure

```
Total position = Base (1x spot) + Conviction Bonus (Nx futures)

Example (Medium, 2x, $10,000 allocation):
  - Base: $10,000 spot long (existing lifecycle engine)
  - Conviction bonus: $10,000 notional via futures (1x additional)
  - Effective exposure: $20,000 (2x)
  - Margin required: ~$5,000 (at 2x futures leverage)
```

**Key principle:** Base position at 1x spot is never touched by leverage logic. Leverage only applies to additional capital deployed as the "conviction bonus" via futures/margin.

### 6.4 Tightened Trailing Stops

When leveraged, trailing stops tighten proportionally:

| Leverage | Standard Trail | Leveraged Trail | Rationale |
|----------|---------------|-----------------|-----------|
| 1x | 15% (default) | 15% | No change |
| 2x | 15% → 7.5% | 7.5% | Same dollar loss at 2x price move |
| 3x | 15% → 5.0% | 5.0% | Same dollar loss at 3x price move |

**Formula:** `leveraged_trail = standard_trail / leverage_multiplier`

This ensures the maximum dollar loss on the leveraged portion equals what the standard trail would produce at 1x.

### 6.5 Capital Reserve Interaction

- The existing 10% capital reserve is **untouchable** by leverage
- Leverage capital comes from the deployable pool only
- If leverage margin requirements would reduce cash below reserve threshold, leverage is denied
- **Emergency reserve:** Additional 5% held back when any leveraged position is active (total 15% reserve)

### 6.6 MARKDOWN Leveraged Short — Additional Safety Rails

Leveraged shorts carry asymmetric risk (unlimited upside = unlimited loss). These rails are **in addition to** the general rails in §6.1–6.5.

**Auto-cover trigger:**
- If price recovers to **95% of the markdown entry price**, all leveraged short positions close immediately
- Rationale: A 5% recovery from the breakdown level suggests the markdown thesis is failing. At 2-3x leverage, waiting for a full reversal is catastrophic.

**Tightened trailing stops for shorts:**

| Leverage | Standard Short Trail | Leveraged Short Trail | Rationale |
|----------|---------------------|----------------------|-----------|
| 1x | 10% (existing `v12_short_trail_pct`) | 10% | No change |
| 2x | 10% → 5.0% | 5.0% | Same dollar loss at 2x |
| 3x | 10% → 3.3% | 3.3% | Same dollar loss at 3x |

**Formula:** `leveraged_short_trail = standard_short_trail / leverage_multiplier`

**Stop-loss override:**
- Existing short SL is 15% (`v12_short_sl_pct`). With leverage:
  - 2x: Hard SL at 7.5% against entry
  - 3x: Hard SL at 5.0% against entry
- This caps maximum loss on the leveraged portion to the same dollar amount as the 1x SL

**Max duration (same as longs):** 7 days, then auto-cover. Can re-enter if conviction still qualifies.

**Phase transition kill switch:** If the lifecycle engine transitions from MARKDOWN to SPRING, all leveraged shorts close immediately — the markdown thesis is invalidated.

**Funding cost awareness:**
- Short positions pay/receive funding. At the existing `funding_rate_daily: 0.0003`, a 7-day leveraged short at 2x costs ~0.42% in funding.
- If funding rate spikes above 0.1% daily (indicating extreme short crowding), leveraged shorts are paused — the trade is too crowded.

---

## 7. Execution Mechanics

### 7.1 Architecture

```
┌─────────────────────┐
│  Lifecycle Engine    │  Phase signals, conductor scores
│  (existing V12e)     │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Conviction Scorer   │  New V2 component
│  (conviction_engine) │  Evaluates 4 signals → score
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Leverage Manager    │  New V2 component
│  (leverage_manager)  │  Position sizing, safety rails
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Exchange Adapter    │  Existing, extended for futures
│  (exchange_adapter)  │  Spot + futures order routing
└─────────────────────┘
```

### 7.2 Exchange Support

| Exchange | Instrument | Status | Notes |
|----------|-----------|--------|-------|
| **Aster** | USDT-M Futures | Primary | Already integrated for spots; futures API similar |
| **Hyperliquid** | Perps | Primary | Native perps exchange, low fees |
| **Bybit** | USDT Perps | Secondary | Wide coin coverage, proven API |
| **Coinbase Derivatives** | Futures | Future | US-regulated option, limited pairs |

### 7.3 Position Sizing

```python
def calculate_leverage_position(base_order_usd, leverage_mult, cash_available, reserve_pct=0.15):
    """
    base_order_usd: existing spot position value
    leverage_mult: 2 or 3 (binary, from conviction score)
    cash_available: current free cash
    reserve_pct: emergency reserve (15% when leveraged)
    """
    bonus_notional = base_order_usd * (leverage_mult - 1)  # Additional exposure
    margin_required = bonus_notional / leverage_mult        # Exchange margin
    
    max_margin = cash_available * (1 - reserve_pct)
    if margin_required > max_margin:
        return None  # Insufficient margin — no leverage
    
    return {
        "notional": bonus_notional,
        "margin": margin_required,
        "instrument": "futures",
        "direction": "long",
    }
```

### 7.4 Order Flow

**Longs (SPRING / MARKUP):**
1. Lifecycle engine signals SPRING or MARKUP phase entry
2. Conviction scorer evaluates 4 signals → composite score
3. If score ≥ 85 (or ≥ 92 for 3x): leverage manager activates
4. Base spot order executes normally via existing engine
5. Conviction bonus order placed as futures long on same asset
6. Both positions tracked independently with separate P&L
7. Leverage position has its own tightened trailing stop
8. On exit: futures position closes first, then spot position follows normal exit logic

**Shorts (MARKDOWN):**
1. Lifecycle engine signals EXIT → MARKDOWN transition
2. Conviction scorer evaluates 4 signals (with short-specific normalization) → composite score
3. If score ≥ 85 (or ≥ 92 for 3x): leverage manager activates short
4. Base 1x short executes normally via existing markdown engine (tiered entries)
5. Conviction bonus short placed as additional futures short on same asset
6. Leveraged short has tighter trail (5%/3.3%), auto-cover at 95% recovery, 7-day max
7. On MARKDOWN → SPRING transition: leveraged short closes immediately, base short follows normal engine logic

---

## 8. Risk Analysis

### 8.1 Backtest-Derived Phase P&L (V12e Extended, $10K start)

Data from `trading/spot/backtest_results/v12_lifecycle/` (v12e_ext runs, Oct 2020 – Feb 2026):

| Coin | Profile | Total P&L % | Markup P&L | Spring P&L | Short P&L | False Springs | Max DD |
|------|---------|------------|------------|------------|-----------|---------------|--------|
| BTC | Low | +166% | $35,828 | -$3,193 | -$12,921 | 0 | 57% |
| BTC | Medium | +120% | $46,953 | -$7,179 | -$21,215 | 0 | 68% |
| BTC | High | +299% | $118,798 | -$20,937 | -$53,931 | 0 | 73% |
| ETH | Medium | +539% | $32,002 | -$18,557 | +$40,658 | 0 | 58% |
| ETH | High | +1,116% | $43,178 | -$17,968 | +$88,052 | 0 | 69% |
| SOL | Medium | +27,461% | $4,286,717 | -$1,427,517 | +$50,561 | 5 | 63% |
| SOL | High | +49,690% | $3,991,720 | +$144,544 | +$1,160,085 | 4 | 65% |

### 8.2 Modeled Leverage Impact

Applying leverage ONLY to markup and spring P&L (the eligible phases):

**BTC Medium (baseline: +120% at 1x):**
- Markup at 2x: $46,953 × 2 = $93,906 (+$46,953 additional)
- Spring at 2x: -$7,179 × 2 = -$14,358 (-$7,179 additional)
- Net leverage benefit: +$39,774
- **Estimated total with V2 leverage: ~+160%** (vs 120% at 1x)

**ETH High (baseline: +1,116% at 1x):**
- Markup at 3x: $43,178 × 3 = $129,534 (+$86,356 additional)
- Spring at 3x: -$17,968 × 3 = -$53,904 (-$35,936 additional)
- Net leverage benefit: +$50,420
- **Estimated total with V2 leverage: ~+1,620%** (vs 1,116% at 1x)

### 8.3 Modeled Leveraged Short Impact

Short P&L from backtests (v12e_ext, $10K start):

| Coin | Profile | Short P&L (1x) | At 2x | At 3x | Notes |
|------|---------|----------------|-------|-------|-------|
| BTC | Low | -$12,921 | -$25,842 | N/A | Shorts unprofitable — conviction gating should prevent most activations |
| BTC | Medium | -$21,215 | -$42,430 | N/A | Same — BTC shorts struggled in backtest period |
| BTC | High | -$53,931 | N/A | -$161,793 | Catastrophic at 3x without conviction gating |
| ETH | Medium | +$40,658 | +$81,316 | N/A | Profitable shorts amplified by 2x |
| ETH | High | +$88,052 | N/A | +$264,156 | Strong short performance, significant upside at 3x |
| SOL | Medium | +$50,561 | +$101,122 | N/A | Profitable shorts, good leverage candidate |
| SOL | High | +$1,160,085 | N/A | +$3,480,255 | Exceptional short performance (SOL's deep markdowns) |

**Critical insight: BTC shorts were unprofitable at 1x.** Conviction gating must prevent leveraged short activation on BTC during these periods. The conviction scoring system's CFGI and regime requirements should naturally filter these out — BTC's markdown phases in the backtest period often lacked the CFGI > 70 + DISTRIBUTION regime combination. This is the entire point of conviction gating: it should reject low-quality short setups even when the phase says MARKDOWN.

**ETH and SOL shorts were highly profitable.** Leverage amplification here is pure signal amplification — exactly the use case V2 is designed for.

### 8.4 Worst Case: Markdown Reversal with Leveraged Shorts

Scenario: Confirmed markdown (conductor > 75, CFGI 74, DISTRIBUTION regime) suddenly reverses — a V-shaped recovery.

**At 2x short leverage:**
- Auto-cover at 95% of markdown entry price limits loss to ~5% on notional × 2 = 10% of leveraged portion
- With the 5% trailing stop, actual loss capped at 5% × 2 = 10% of margin
- Hard SL at 7.5% catches any gap: 7.5% × 2 = 15% max loss on margin

**At 3x short leverage:**
- Auto-cover at 95%: ~5% × 3 = 15% of margin
- Trailing stop at 3.3%: 3.3% × 3 = 10% of margin
- Hard SL at 5%: 5% × 3 = 15% max loss on margin

**Flash crash reversal (worst worst case):**
- Exchange goes down, can't execute stop → price gaps 20% against short
- At 2x: 20% × 2 = 40% loss on margin
- At 3x: 20% × 3 = 60% loss on margin (margin nearly liquidated)
- Mitigation: emergency reserve (15%), cross-exchange redundancy, and the fact that leveraged portion is only the conviction bonus — base position is unaffected

**Net assessment:** The auto-cover at 95% of entry is the primary protection. In a normal reversal, losses are manageable. In a flash-crash gap scenario, losses are severe but bounded by margin — and the base spot position is independent.

### 8.5 Worst Case: False Spring with Leveraged Longs

SOL medium had 5 false springs. Modeling a false spring at 2x:

- Spring false drop threshold: 20% (from config `v12_spring_false_drop_pct`)
- At 2x leverage on the conviction bonus: 20% × 2 = 40% loss on leveraged portion
- But leveraged portion is only the bonus (not base), so effective loss on total position:
  - Base (1x): -20%
  - Bonus (1x additional via futures): -40% of margin
- With tightened trailing stop at 7.5%: loss capped earlier
- **With safety rails active, max loss per false spring ≈ 12-15% of total allocation** (vs 8-10% at 1x)

### 8.6 Comparative Expected Outcomes

| Configuration | Expected Return Range | Max Drawdown | Risk Rating |
|--------------|----------------------|--------------|-------------|
| V12e (1x spot) | +120% – +27,000%* | 57-73% | Baseline |
| V2 Medium (2x longs + shorts) | +180% – +40,000%* | 65-78% | Moderate increase |
| V2 High (3x longs + shorts) | +350% – +65,000%* | 70-85% | Significant increase |

*Range reflects BTC (low end) to SOL (high end). SOL numbers are extreme due to 2020-2024 price action and should not be taken as expected. Leveraged short contribution varies dramatically by coin — ETH/SOL shorts add significant value, BTC shorts may be net negative even with conviction gating.*

### 8.7 Per-Coin Conviction Calibration

**Decision: conviction thresholds should be coin-specific**, calibrated from historical candle data.

Each certified coin gets a **conviction profile** derived from its backtest history:
- **False spring rate**: coins with 0 historical false springs get a lower conviction bar (e.g., 80 instead of 85 for 2x)
- **Typical markdown duration**: shorter markdowns = tighter leverage windows
- **Volatility regime distribution**: coins that spend more time in EXTREME get a conviction penalty
- **Phase transition reliability**: how often does the conductor correctly call transitions?
- **Short profitability**: coins where backtest short_pnl < 0 (e.g., BTC) get leveraged shorts disabled

This calibration runs as part of the certification pipeline — each coin's conviction profile is stored alongside its scanner scores.

### 8.8 Open Questions

- **Q:** Should leverage P&L be modeled net of funding costs? At 0.03%/day × 7 day max = 0.21% per activation — small but cumulative.
- **Q:** The backtest doesn't account for futures-specific slippage and liquidation risk. How much margin of safety to add?
- **Q:** What's the minimum historical data required to build a reliable conviction profile? (Propose: 12 months of 1h candles minimum.)
- **Q:** Should the auto-cover at 95% of markdown entry be tighter (e.g., 97%) for 3x leverage? The 5% recovery at 3x = 15% margin loss.
- **Q:** Leveraged shorts during EXTREME regime carry higher gap risk. Should EXTREME be excluded from short leverage despite being eligible for 1x shorts?

---

## 9. Implementation Phases

### Phase 1: Conviction Scoring Engine
*Ship as dashboard indicator — no leverage yet*

- Build `ConvictionScorer` class implementing §4 formula
- Integrate with existing lifecycle engine cycle
- Display conviction score in dashboard and Telegram alerts
- Collect real-time scoring data for validation
- **Deliverable:** Conviction score visible alongside conductor score in all reporting
- **Duration:** 2-3 weeks
- **Risk:** Zero — read-only, no trading impact

### Phase 2: Paper Trading with Virtual Leverage
*Validate conviction gating without real capital*

- Build `LeverageManager` class implementing §6 safety rails
- Integrate with paper trading mode
- Track virtual leverage P&L alongside actual paper P&L
- Compare: would leverage have improved or worsened outcomes?
- **Deliverable:** Paper trading reports showing V2 impact
- **Duration:** 4-6 weeks (need to observe at least one phase transition)
- **Risk:** Zero — virtual positions only

### Phase 3: Live Leverage — Medium Profile
*Mature coins, 2x max*

- Extend exchange adapter for futures order routing
- Enable on Medium profile with mature coins only (BTC, ETH, SOL)
- Maximum 2x leverage, all safety rails active
- Tight monitoring: daily P&L review for first 30 days
- **Deliverable:** Live V2 Medium on production
- **Duration:** 2-4 weeks development + 30-day monitored rollout
- **Risk:** Moderate — real capital at risk, but gated and monitored

### Phase 4: Live Leverage — High Profile
*All certified coins, 3x max*

- Enable on High profile with full certified pool
- Maximum 3x leverage for conviction > 92
- Extended monitoring period for alt-coin leverage behavior
- **Deliverable:** Full V2 on production
- **Duration:** 2-4 weeks after Phase 3 stabilization
- **Risk:** Higher — thinner liquidity on alts, less historical data

---

## 10. Smart Capital Allocation Integration (V3 Preview)

### 10.1 V2 Impact on Capital Efficiency

V2 leverage partially addresses the "idle capital during markdown" problem:
- During markdown, shorts operate at 1x (no change from V12e)
- During spring/markup, leverage means less absolute capital needed for same exposure
- Freed capital can either remain as reserve or deploy to other coins

### 10.2 V3 Direction

With leverage available, smart capital allocation shifts from "which coins get capital" to "which coins get leverage capital." V3 would:

- Rank coins by conviction score
- Allocate leverage budget to highest-conviction signals first
- Cross-coin capital optimization (leverage on BTC spring, spot-only on ETH DCA)
- Dynamic reserve adjustment based on aggregate portfolio leverage

**This is explicitly V3 territory — V2 treats each coin's leverage independently.**

---

## Appendix A: Configuration Defaults

```python
V2_LEVERAGE_CONFIG = {
    "enabled": False,                    # Opt-in
    "max_leverage": {
        "low": 1,                        # No leverage
        "medium": 2,                     # 2x max
        "high": 3,                       # 3x max
    },
    "conviction_weights": {
        "cfgi": 0.25,
        "conductor": 0.35,
        "ath_distance": 0.20,
        "regime": 0.20,
    },
    "conviction_thresholds": {
        "2x": 85,
        "3x": 92,
        "exit_hysteresis": 75,           # Close leverage below this
    },
    "safety": {
        "max_duration_days": 7,
        "emergency_reserve_pct": 0.15,   # 15% reserve when leveraged
        "trail_divisor": True,           # trail = standard_trail / leverage
    },
    "mature_coins": ["BTC", "ETH", "SOL"],
    "eligible_phases": ["SPRING", "MARKUP", "MARKDOWN"],
    "short_safety": {
        "auto_cover_recovery_pct": 0.95,   # Cover if price recovers to 95% of entry
        "funding_rate_pause": 0.001,        # Pause if daily funding > 0.1%
        "hard_sl_divisor": True,            # SL = standard_sl / leverage
    },
}
```

## Appendix B: Reference Files

| File | Contents |
|------|----------|
| `trading/spot/lifecycle_engine.py` | Phase transitions, conductor scoring, `LifecycleConfig` |
| `trading/spot/lifecycle_trader.py` | `PROFILES` dict, `RiskProfile` dataclass |
| `trading/spot/phase_classifier.py` | Phase classification logic |
| `trading/spot/backtest_results/v12_lifecycle/` | Backtest results with phase-level P&L |
| `projects/ait-product/risk-profiles-spec.md` | V1 risk profile specification |

---

*This document is a scoping specification. Thresholds, weights, and criteria are proposed based on backtest analysis and system design principles. All values are subject to validation during Phase 1-2 implementation. Open questions are flagged inline throughout.*
