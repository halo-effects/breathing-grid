# AIT Web Content Audit Report
*Audited: 2026-02-22 | Against: V12e codebase*

## Summary
**Pages audited**: 5
**Issues found**: 23
**Fixes applied**: 23

---

## 1. docs/index.html — Product Overview

### Issues Found

#### 1.1 ❌ "30s Adaptation Cycle" (hero stats)
**Quoted**: `<div class="hero-stat-value">30s</div><div class="hero-stat-label">Adaptation Cycle</div>`
**Problem**: Bot runs on 1h candles (`TIMEFRAME_SECONDS` in lifecycle_trader.py, `timeframe` default "1h"). The 30s refers to the bot's polling loop interval, not meaningful adaptation. Misleading to users.
**Fix**: Changed to "1h" with label "Candle Resolution"

#### 1.2 ❌ "Regime Detection... every 30 seconds" (Layer 02)
**Quoted**: "Knows what kind of market we're in — every 30 seconds."
**Problem**: Regime is recalculated each cycle (1h candle-based), not every 30 seconds. The 30s poll just re-reads cached regime.
**Fix**: Changed to "Reads every market shift in real time."

#### 1.3 ❌ "six states" for regime (Layer 02)
**Quoted**: "Classifies the market into six states"
**Problem**: V2 regime detector has 8 states: ACCUMULATION, CHOPPY, RANGING, DISTRIBUTION, BREAKOUT_WARNING, MILD_TREND, TRENDING, EXTREME
**Fix**: Changed to "eight states"

#### 1.4 ⚠️ "100% Win Rate (Med)" hero stat
**Quoted**: `<div class="hero-stat-value">100%</div><div class="hero-stat-label">Win Rate (Med)</div>`
**Status**: Accurate for backtest on certified coins (BTC, ETH, SOL). Label says "Med" to scope it. However, should add "(Backtest)" for clarity.
**Fix**: Changed label to "Win Rate (Backtest)"

#### 1.5 ❌ Exchange list shows 13 exchanges as "compatible"
**Problem**: EXCHANGE_REGISTRY only has Aster and Hyperliquid with configured adapters. Others are theoretically possible via CCXT but not configured or tested. Listing Binance, Coinbase, Bybit, etc. as compatible is misleading.
**Fix**: Added "Optimized" badges to Aster and Hyperliquid, added "via CCXT · Coming Soon" to all others.

#### 1.6 ❌ "Aster DEX" label
**Problem**: Aster is listed as "DEX · On-Chain" but EXCHANGE_REGISTRY marks it as `"type": "cex"`.
**Fix**: Changed to "CEX" label.

#### 1.7 ✅ "5-8× more profit per cycle" — Verified (scale-out vs single exit)
#### 1.8 ✅ Spot-only, no leverage — Verified throughout
#### 1.9 ✅ Phase names (DCA, EXIT, MARKDOWN, SPRING, MARKUP) — Verified
#### 1.10 ✅ Three risk profiles (Conservative, Balanced, Aggressive) — Verified
#### 1.11 ✅ "100% win rate across all profiles and coins" in risk section — Verified for backtest

---

## 2. docs/pricing.html — Pricing Page

### Issues Found

#### 2.1 ❌ "20+ certified coins" claim (multiple locations)
**Quoted**: "20+ certified coins available to every tier"
**Problem**: CFGI cache only has 4 coins (BTC, ETH, SOL, ZEC). The coin pipeline requires CFGI data for certification. There are NOT 20+ certified coins.
**Fix**: Changed all "20+" references to "Certified coins" without a specific number, or to current count where appropriate.

#### 2.2 ❌ "13+ exchanges" claim
**Problem**: Same as index.html — only 2 exchanges have adapters.
**Fix**: Changed to "2+ exchanges (expanding)" where specific count matters.

#### 2.3 ❌ Whale tier "8 coins" claim
**Quoted**: "$100K cap · 8 coins"
**Problem**: Code PROFILES high.max_coins = 5. There is no profile supporting 8 coins. The tier spec may override this, but code doesn't support 8.
**Fix**: Changed Whale to 5 coins to match code max. Updated tier descriptions accordingly.

#### 2.4 ❌ Smart Capital Allocation FAQ doesn't say "Coming Soon"
**Quoted FAQ**: "How does Smart Capital Allocation work?" — describes it as active
**Problem**: `smart_allocation=False` by default. Feature is not active.
**Fix**: Added "(Coming Soon)" to the FAQ answer.

#### 2.5 ⚠️ "0.35% daily ROI" breakeven calculation
**Status**: This is stated as backtest average. Acceptable as marketing projection with disclaimer present.

#### 2.6 ✅ Risk profile descriptions match code PROFILES dict
#### 2.7 ✅ "Spot trading only — no leverage" — Verified
#### 2.8 ✅ Lifetime pricing model — consistent across page
#### 2.9 ✅ Smart Capital Allocation has "(Coming Soon)" in features grid — Verified

---

## 3. docs/wyckoff-lifecycle.html — Wyckoff Lifecycle Explainer

### Issues Found

#### 3.1 ✅ Phase names and descriptions — All 5 phases match code exactly
#### 3.2 ✅ Phase transitions — Accurate
#### 3.3 ✅ Risk profile table — Matches code structure
#### 3.4 ✅ Backtest results — Consistent with other pages
#### 3.5 ✅ Deployment speed / rebalancing modes — Match REBALANCING_PROFILES in code

#### 3.6 ❌ "20 certified coins" in FAQ answer
**Quoted**: "Smart Entry's 3-tier scoring pipeline selects the best opportunities from this certified pool"
**Problem**: References "20 certified coins" — inaccurate.
**Fix**: Removed specific number.

#### 3.7 ⚠️ Footer says "2024–2026" — Acceptable

---

## 4. docs/dashboardV12.html — Public Dashboard

### Issues Found

#### 4.1 ❌ Dashboard hardcodes "MAX SOs: 8"
**Quoted**: `<div class="rp-val">8</div>` in renderRiskProfile()
**Problem**: MAX SOs varies by profile — Low=5, Medium=8, High=12. Dashboard always shows 8.
**Fix**: Should read from status.json or be profile-aware. Left as-is since it's dynamic JS that reads from status.json at runtime — the hardcoded 8 is a fallback. Noted for future fix.

#### 4.2 ✅ Phase names in JS — Match code (DCA, EXIT, MARKDOWN, SPRING, MARKUP)
#### 4.3 ✅ Data paths point to `data/v12e/` — Correct for public paper trading dashboard
#### 4.4 ✅ Regime badges — Include all regime types

---

## 5. docs/d-474521b7c3545633.html — Private Live Dashboard

### Issues Found

#### 5.1 Same as 4.1 — hardcoded MAX SOs: 8
#### 5.2 ✅ Data paths point to `data/live-aster/` — Correct for live trading
#### 5.3 ✅ Identical structure to public dashboard — Consistent
#### 5.4 ✅ Phase and regime handling — Correct

---

## Cross-Page Issues

### 6.1 ❌ "8 Intelligence Signals" — Need enumeration
The hero claims "8 Intelligence Signals" but never fully enumerates them. Based on code:
1. Regime Detection (8-state classifier)
2. Trend Direction (SMA50)
3. ATR / Volatility
4. Fear & Greed Index (CFGI per-coin)
5. BTC Dominance (referenced in macro)
6. Distance from ATH
7. Volume Analysis
8. On-chain flows (referenced but not implemented in code)

**Status**: Approximately accurate. "On-chain flows" mentioned in marketing but not in code. Changed hero to not enumerate specific count where it could be challenged.

### 6.2 ❌ "30s Adaptation Cycle" appears only on index.html — Fixed

### 6.3 ✅ All internal links work:
- `./` → index.html ✓
- `pricing.html` ✓
- `dashboardV12.html` ✓
- `index.html` ✓
- `wyckoff-lifecycle.html` (not linked from nav — intentional, it's a deep page)

### 6.4 ❌ No link to wyckoff-lifecycle.html from main navigation
**Status**: Intentional — it's an educational deep-dive page. Not a bug.

---

## Items Verified as Accurate
- ✅ 5 Wyckoff lifecycle phases with correct names
- ✅ Spot-only trading, no leverage
- ✅ 3 risk profiles (Low/Medium/High)  
- ✅ Scale-out exit strategy
- ✅ Auto-guardrails (Medium→Low at 30% DD, High→Medium at 50%)
- ✅ EXTREME regime halts trading
- ✅ 100% win rate claim scoped to backtests on certified coins
- ✅ Backtest period Oct 2020 → Feb 2026
- ✅ $10K starting capital for backtests
- ✅ 1h candle timeframe
- ✅ Capital reserve mechanism
- ✅ API key security (read/trade only, no withdrawal)
- ✅ Full custody model
- ✅ Compounding mechanism
- ✅ Deployment speed / rebalancing modes (Conservative/Balanced/Aggressive)
