# Spot DCA Scale-Out Strategy
**Status:** Draft — Strategy Design Phase
**Created:** 2026-02-17
**Last Updated:** 2026-02-17

---

## Executive Summary

**Intelligent Accumulation** — a spot-based DCA strategy that uses adaptive intelligence to systematically buy assets in layers as price drops, then sells in reverse order (largest lots first) as price recovers. Not day trading. Not swing trading. This is disciplined, regime-aware accumulation for long-term investors who believe in the assets they're buying and want smarter entries and automated profit-taking.

Eliminates funding fees, reduces complexity, and captures significantly more profit per cycle compared to perpetual futures. Designed to shine brightest during bear markets and corrections — exactly when human discipline fails most.

---

## 1. Core Concept

### Current Approach (Perpetual Futures)
- Bidirectional DCA (long + short simultaneously)
- Close entire position at single TP price (avg entry + X%)
- Subject to funding fees every 4 hours
- Net position mode introduces sync/orphan risks

### New Approach (Spot Scale-Out)
- Long-only DCA on spot markets
- Scale-out exit: sell largest lots first as price recovers
- Zero funding fees — you own the actual coin
- No position mode complexity

### Why Switch?
Analysis of 4 days of live trading (23 TPs, $330 capital) showed:
- Gross TP profit: $5.95
- Funding fees estimated: $0.16 (conservative; real impact likely higher on longer holds)
- Orphan position bug: -$4.57 (one-time, but caused by perps complexity)
- 17/23 TPs were tiny $0.08 trades at 0.6% TP floor
- Funding fees on 24h+ holds can eat 50-66% of TP profit

---

## 2. Entry Logic — Adaptive DCA

Same proven DCA/martingale entry system, adapted for spot:

### Base Order
- Percentage of allocated capital (e.g., 4-8%)
- Market or limit buy when conditions are met
- Only opens when regime allows (no entries in EXTREME)

### Safety Orders (SOs)
- Each SO is 2× the previous (martingale multiplier)
- Deviation between SOs is adaptive (ATR + regime-based)
- Maximum 8 SOs (configurable per risk profile)

### Adaptive Entry Spacing
| Regime | Deviation | Rationale |
|--------|-----------|-----------|
| RANGING | 1.5-2.0% | Tight grid, rapid cycles in oscillating markets |
| MILD_TREND (bearish) | 2.5-3.0% | Wider spacing, don't catch falling knife |
| TRENDING (bearish) | 3.0-4.0% | Very wide, only buy at deep discounts |
| ACCUMULATION | 2.0-2.5% | Normal spacing, price consolidating |
| EXTREME | No new entries | Protect capital, wait for clarity |
| TRENDING (bullish) | 1.5-2.0% | Normal entries, price moving in our favor |

### Example Entry (HYPE @ $29.50, 4% base, $330 capital)
```
Base:  0.44 HYPE @ $29.50 = $12.98
SO1:   0.87 HYPE @ $28.76 = $25.02  (2.5% lower)
SO2:   1.71 HYPE @ $28.04 = $47.96  (5.0% lower)
SO3:   3.39 HYPE @ $27.34 = $92.67  (7.5% lower)
─────────────────────────────────────
Total: 6.41 HYPE, avg $27.84, cost $178.63
```

---

## 3. Exit Logic — Scale-Out (Reverse DCA)

**Key Innovation:** Instead of closing everything at one TP price, sell in tranches starting with the largest (cheapest) lots.

### Scale-Out Order
1. **Sell SO3 lot first** (largest, bought lowest) — highest profit margin
2. **Sell SO2 lot** — still good profit
3. **Sell SO1 lot** — moderate profit
4. **Sell base order last** — smallest lot, breakeven or small profit

### Adaptive Exit Targets
Each sell order's TP distance adapts to regime:

| Regime | TP per Tranche | Rationale |
|--------|---------------|-----------|
| RANGING | 1.0-1.5% above each lot's entry | Quick captures on oscillations |
| MILD_TREND (bullish) | 1.5-2.5% | Let it run a bit |
| TRENDING (bullish) | Trail with 2-3% offset | Ride momentum, maximize capture |
| BEARISH recovery | 1.0% conservative | Take profit quickly, don't get greedy |

### Example Exit (continuing from above, price bounces)
```
Sell SO3: 3.39 HYPE @ $28.62 (+4.7% from entry) = +$4.34
Sell SO2: 1.71 HYPE @ $29.10 (+3.8% from entry) = +$1.81
Sell SO1: 0.87 HYPE @ $29.40 (+2.2% from entry) = +$0.56
Sell Base: 0.44 HYPE @ $29.50 (+0.0%)            = +$0.00
───────────────────────────────────────
Total profit: +$6.71
```

### Comparison: Same Price Movement
| Approach | Profit | Fees | Net |
|----------|--------|------|-----|
| Futures (close all at avg+1%) | $1.79 | ~$0.50 funding | ~$1.29 |
| Spot scale-out | $6.71 | ~$0.03 trading | ~$6.68 |
| **Improvement** | | | **~5.2×** |

---

## 4. Regime Detection (Unchanged)

Leverages the existing adaptive intelligence engine:

- **14-period ATR** for volatility measurement
- **SMA50** for trend direction (bullish/bearish)
- **ADX** for trend strength
- **Hurst exponent** for mean-reversion vs momentum classification

Regimes: ACCUMULATION, RANGING, MILD_TREND, TRENDING, DISTRIBUTION, EXTREME

### Directional Awareness
- Bearish detected → widen entry spacing (patient entries at better prices)
- Bullish detected → normal/tight spacing, aggressive scale-out targets
- Regime change → adjust unfilled SO levels and pending sell orders dynamically

---

## 5. Coin Selection Criteria

### For Spot DCA, Prioritize Mature Markets

**Screen Out:**
- 🚫 Parabolic runners (>100% gain in 30 days)
- 🚫 Low liquidity (<$5M daily spot volume)
- 🚫 New coins (<90 days old)
- 🚫 Pure meme coins with no utility
- 🚫 Extreme volatility (ATR >5% daily) — unpredictable

**Screen In:**
- ✅ Top 50 market cap — deep liquidity, tight spreads
- ✅ Established price history (6+ months minimum)
- ✅ Moderate volatility (ATR 1-3% daily) — enough to cycle deals
- ✅ Consistent volume (not spike-driven)
- ✅ Strong ranging behavior (Hurst 0.3-0.5)

### Coin Eligibility by Risk Profile

Critical design decision: **coins are gated by risk tier** to protect users from themselves. Smaller/newer coins may show great backtest returns but carry 95-98% downside from ATH in a real bear market. The DCA strategy assumes the investor believes the asset will recover — that assumption must be sound.

**Low Risk — Blue Chips + Institutional/RWA/Infrastructure:**
| Coin | Category | Rationale |
|------|----------|-----------|
| BTC | Store of value | Survived 4+ bear markets, always recovered |
| ETH | Smart contracts | Backbone of DeFi/NFTs, institutional adoption |
| SOL | L1 Infrastructure | Speed/ecosystem, growing institutional interest |
| LINK | Oracle infrastructure | Critical blockchain connectivity, irreplaceable role |
| XRP | Institutional payments | Settlement/cross-border payments, regulatory clarity |
| AAVE | DeFi infrastructure | Leading lending protocol, real revenue |
| ONDO | RWA tokenization | Tokenized treasuries, institutional backing |
| PLUME | RWA infrastructure | RWA-focused L2, growing ecosystem |

These coins are volatile (may drop 70-80% in a bear) but have **real utility, institutional backing, and strong recovery catalysts**. The intelligent accumulation thesis is sound: buying these at deep discounts during a bear is a fundamentally good trade.

**Medium Risk — Established Ecosystems (everything in Low, plus):**
| Coin | Category | Rationale |
|------|----------|-----------|
| BNB | Exchange/L1 | Binance ecosystem, proven survivor |
| DOGE | Payments/cultural | Massive liquidity, has survived multiple cycles |
| DOT | Interoperability | Parachain ecosystem, institutional backing |
| AVAX | L1/Subnets | Enterprise adoption, subnet architecture |
| NEAR | L1/AI | Sharding tech, AI narrative |
| UNI | DeFi governance | Leading DEX, real volume |
| + Top 20-30 by market cap with proven ecosystems |

**High Risk — Broader Selection:**
- All screened coins including HYPE, ASTER, newer mid-caps
- Higher volatility, less proven track record
- User explicitly accepts these may not recover in a bear
- Best backtest returns but highest permanent loss risk
- Suitable for users with high conviction in specific newer projects

### Ideal Spot Portfolio (Conservative — Low Risk)
| Coin | Allocation | Rationale |
|------|-----------|-----------|
| BTC | 30% | Anchor, lowest risk, deepest liquidity |
| ETH | 25% | Smart contract backbone |
| SOL | 15% | High-performance L1, active ecosystem |
| LINK | 10% | Oracle infrastructure, essential middleware |
| XRP/ONDO/AAVE | 20% | RWA/payments/DeFi diversification |

### Sector-Diversified Portfolio — "Crypto Index" Approach

Rather than random coin selection, portfolios are built around **sector blue chips** — giving users diversified exposure across the crypto ecosystem, similar to how traditional investors diversify across tech, healthcare, finance, etc.

**Crypto Sectors & Blue Chip Representatives:**

| Sector | Blue Chips | Rationale |
|--------|-----------|-----------|
| **Store of Value** | BTC | Digital gold, ultimate safe haven |
| **Smart Contracts / L1** | ETH, SOL, AVAX, NEAR | Core infrastructure, compete for DApp ecosystem |
| **Layer 2 / Scaling** | ARB, OP, MATIC, PLUME | Ethereum scaling, lower fees, growing adoption |
| **AI / Compute** | NEAR, FET, RENDER, TAO | AI narrative + real compute demand |
| **DeFi / Lending** | AAVE, UNI, MKR, CRV | Battle-tested protocols, real revenue |
| **Payments / Settlement** | XRP, XLM, HBAR | Cross-border, institutional payment rails |
| **Infrastructure / Oracles** | LINK, GRT, PYTH | Essential middleware that everything depends on |
| **RWA / Tokenization** | ONDO, PLUME, WLFI | Tokenized real-world assets, institutional wave |
| **Gaming / Metaverse** | IMX, GALA, AXS | Gaming infrastructure, surviving projects |
| **Exchange / Ecosystem** | BNB, CRO | Exchange-backed, ecosystem utility |
| **Meme / Cultural** | DOGE, SHIB | Massive liquidity, cultural staying power |

**Default Portfolio Templates by Risk Profile:**

**Low Risk — "Blue Chip Core" (6-8 sectors):**
```
Store of Value:     BTC   25%
Smart Contracts:    ETH   20%
L1 Infrastructure:  SOL   12%
DeFi:               AAVE   8%
Infrastructure:     LINK  10%
Payments:           XRP   10%
RWA:                ONDO   8%
AI:                 NEAR   7%
```

**Medium Risk — "Sector Balanced" (8-10 sectors):**
```
Store of Value:     BTC   20%
Smart Contracts:    ETH   15%, SOL 8%
Layer 2:            ARB    5%, PLUME 5%
DeFi:               AAVE   6%, UNI 4%
Infrastructure:     LINK   7%
Payments:           XRP    6%, HBAR 4%
RWA:                ONDO   5%, WLFI 3%
AI:                 NEAR   5%, FET 3%
Gaming:             IMX    4%
```

**High Risk — "Full Spectrum" (all sectors + mid-caps):**
Broader exposure including sector #2 and #3 picks, scanner-driven allocation, volatile newcomers

**Product Feature: "Sector View" Dashboard**
Users see their portfolio organized by sector, not just a flat coin list:
- Sector health indicators (which sectors are trending/ranging/declining)
- Over/underweight alerts ("You're 40% DeFi — consider diversifying")
- Sector rotation suggestions based on regime detection per sector
- One-click rebalance to target allocations

**Why This Matters for Sales:**
- "Don't just buy Bitcoin — build a diversified crypto portfolio across 10 sectors"
- "Like a crypto ETF, but smarter — adaptive intelligence times your entries"
- "Sector diversification means one bad sector doesn't sink your portfolio"
- Positions AIT as an **investment platform**, not just a trading bot
- Appeals to traditional finance people who understand sector diversification

### "Already in a Bear" Coins — Ideal Accumulation Targets
Coins that have already corrected 70-80%+ from ATH present the best risk/reward for intelligent accumulation:
- **Downside is limited** — they've already crashed, further downside is more bounded
- **Institutional backing ensures recovery** — NEAR, HBAR, ONDO, PLUME, AAVE, LINK, WLFI all have strong institutional investors
- **Choppy ranging markets = ideal for DCA grid** — low volume oscillations create constant deal cycles
- **First to move on recovery** — institutional-backed coins lead the way when sentiment shifts

These aren't speculative bets — they're fundamentally sound projects trading at deep discounts. The bot's job is to systematically accumulate at the best prices within those discounts.

Examples as of Feb 2026: NEAR (-80% from ATH), HBAR (-75%), PLUME (early stage), ONDO (-60%), LINK (-65%)

### User Coin Selection — Defaults + Customization

**Design Principle:** Smart defaults for beginners, full control for power users, guardrails for everyone.

**Default Recommended Portfolios (pre-selected):**

Low Risk defaults:
| Coin | Allocation | Tag |
|------|-----------|-----|
| BTC | 30% | ✅ Recommended |
| ETH | 25% | ✅ Recommended |
| SOL | 15% | ✅ Recommended |
| LINK | 10% | ✅ Recommended |
| XRP | 10% | ✅ Recommended |
| AAVE | 10% | ✅ Recommended |

Medium Risk defaults: Everything in Low, plus NEAR, ONDO, PLUME, HBAR, WLFI with smaller allocations

High Risk defaults: Scanner-driven, auto-selected based on DCA suitability scores

**User Customization Features:**
- Toggle individual coins on/off
- Adjust allocation % per coin (sliders, must sum to 100%)
- "Add Custom Coin" — search by symbol from exchange's available pairs
- One-click "Reset to Recommended" button
- Save custom portfolio as a template

**Guardrails:**
- Coins not in the recommended list for user's risk profile get a ⚠️ warning tag
- "HYPE is classified as High Risk. Adding it to your Low Risk portfolio increases your risk exposure. Continue?"
- Live "Portfolio Risk Score" that updates as user customizes (green/yellow/red)
- Maximum single-coin allocation cap (e.g., no more than 40% in any non-BTC/ETH coin)
- Minimum allocation per coin enforced ($3K floor per coin — too little isn't worth the gas/fees)
- System validates total allocation = 100% before saving

**UI Concept:**
```
┌─────────────────────────────────────┐
│ Your Portfolio          Risk: LOW 🟢│
├─────────────────────────────────────┤
│ ✅ BTC  [====30%====]  $3,000  [x] │
│ ✅ ETH  [===25%===]    $2,500  [x] │
│ ✅ SOL  [==15%==]      $1,500  [x] │
│ ✅ LINK [=10%=]        $1,000  [x] │
│ ✅ XRP  [=10%=]        $1,000  [x] │
│ ✅ AAVE [=10%=]        $1,000  [x] │
│                                     │
│ [+ Add Coin]  [Reset to Recommended]│
│                                     │
│ Portfolio Risk Score: ████████░░ 82 │
└─────────────────────────────────────┘
```

### Coin Scanner Adaptation
Existing Tier 1/Tier 2 scanner needs spot-specific enhancements:
- Weight ranging behavior higher (Hurst closer to 0.5 = better)
- Weight volume consistency (not just magnitude)
- Penalize parabolic price action heavily
- Add spot spread analysis (tight spreads = less friction)
- **Utility tier classification**: tag coins as Blue Chip / Institutional / DeFi / Meme / Speculative
- **Risk profile gating**: scanner results filtered by profile eligibility before recommendation

---

## 6. Risk Management

### Capital Protection
- **10% global reserve** — never fully deployed
- **Max capital per deal** — capped by tier/risk profile
- **$3K minimum per coin** — ensures meaningful position sizes
- **EXTREME regime = no new entries** — hard stop

### Downside Protection
- **Adaptive spacing in downtrends** — wider SOs = better average entry
- **Max SO limit** — caps total exposure per deal (8 SOs = ~255× base order total)
- **Portfolio stop-loss** — if total unrealized loss exceeds X% of capital, halt new entries
- **Individual coin stop-loss (optional)** — if price drops X% below last SO, consider cutting

### Bag Holding Mitigation
- Regime detection prevents entries in EXTREME markets
- Wider spacing in downtrends means fewer fills at bad prices
- Scale-out means partial recovery still yields profit (don't need full reversal)
- Multi-coin diversification spreads risk

---

## 7. Hybrid Approach (Advanced)

Run both strategies simultaneously for bidirectional coverage:

| Strategy | Market | Direction | Coins | Purpose |
|----------|--------|-----------|-------|---------|
| Spot DCA Scale-Out | Spot | Long only | BTC, ETH, SOL (mature) | Core profit, no funding fees |
| Futures Shorts | Perps | Short only | HYPE, volatile alts | Profit from drops, hedge longs |

### Benefits of Hybrid
- Bidirectional coverage without funding drag on longs
- Short positions tend to close faster (profits from drops are quick)
- Funding fees on shorts are manageable (shorter hold times)
- Natural hedge: if market dumps, shorts profit while spot accumulates at lower prices

### Capital Split (Example)
- 70% → Spot DCA (conservative, core returns)
- 30% → Futures shorts (opportunistic, higher risk/reward)

---

## 8. Exchange Candidates

### Requirements
- Spot trading with API access
- Limit orders (for maker fee benefits)
- Sufficient liquidity for target coins
- Wallet-based / minimal KYC preferred
- Reliable API uptime

### Candidates
| Exchange | Type | KYC | Spot API | HYPE Spot | Notes |
|----------|------|-----|----------|-----------|-------|
| Aster | DEX | No (wallet) | Custom (`/bapi/spot/`) | ❌ No | 50 spot pairs, new spot market |
| Hyperliquid | DEX | No (wallet) | Custom | ✅ Yes | Native HYPE home, best liquidity |
| Bybit | CEX | Yes | Binance-like | ✅ Yes | Easy API port, deep liquidity |
| MEXC | CEX | Yes | Binance-compatible | ✅ Yes | Easiest code port |
| Binance | CEX | Yes | Native | ✅ Yes | Deepest liquidity, geo restrictions |

### Current Recommendation
- **Hyperliquid** for HYPE spot (if we keep trading HYPE)
- **Aster spot** for BNB chain tokens (already connected, wallet-based)
- **Bybit or MEXC** for broadest coin access (requires KYC)

Decision pending — evaluate after current TP floor experiment.

### CCXT — Universal Exchange Layer

Rather than targeting Binance-compatible APIs, we build on **CCXT** (CryptoCurrency eXchange Trading Library) — a universal Python/JS library that wraps 100+ exchanges behind one API. Already installed and working on our system.

**One codebase, every exchange:**
- `create_order()`, `fetch_balance()`, `fetch_ticker()` — same calls everywhere
- CCXT handles per-exchange API translation internally
- Adding a new exchange = changing one config line
- No per-exchange API rewrites

**CCXT-Supported DEXs (No KYC, Wallet-Based):**
| Exchange | Spot | Perps | Chain | Notes |
|----------|------|-------|-------|-------|
| Aster | ✅ | ✅ | BNB Chain | Already connected, live bot running |
| Hyperliquid | ✅ | ✅ | Hyperliquid L1 | HYPE native, Python SDK also available |
| Backpack | ✅ | ✅ | Solana | Growing DEX, SOL ecosystem |
| Paradex | ❌ | ✅ | StarkNet | Perps only |
| dYdX | ❌ | ✅ | Cosmos | Perps only |

**CCXT-Supported CEXs (KYC Required, API Key Auth):**
| Exchange | Spot | Perps | Notes |
|----------|------|-------|-------|
| Binance | ✅ | ✅ | Deepest liquidity globally |
| Bybit | ✅ | ✅ | Very popular, strong API |
| OKX | ✅ | ✅ | Strong derivatives market |
| MEXC | ✅ | ✅ | Binance-compatible, huge coin selection |
| Gate.io | ✅ | ✅ | Massive altcoin catalog |
| KuCoin | ✅ | ✅ | Popular for altcoins |
| Coinbase | ✅ | ❌ | US-friendly, trusted |
| Kraken | ✅ | ✅ | US-friendly, institutional |
| + 90 more | — | — | Full list: github.com/ccxt/ccxt |

**Competitive Advantage:**
- **3Commas**: Supports ~15 exchanges with per-exchange integrations
- **Pionex**: Locked to their own exchange
- **CryptoHopper**: ~12 exchanges, complex setup
- **AIT via CCXT**: 100+ exchanges, one integration, add new ones instantly

**Architecture:**
```
┌─────────────────────────────┐
│   AIT Adaptive Intelligence │
│   (Regime, DCA, Scale-Out)  │
├─────────────────────────────┤
│       CCXT Unified API      │
├──────┬──────┬──────┬────────┤
│Aster │Hyper │Bybit │ +97    │
│      │liquid│      │ more   │
└──────┴──────┴──────┴────────┘
```

Users connect their wallet (DEX) or API key (CEX) and the bot just works. Exchange-universal from day one.

---

## 9. Risk Profiles (Adapted for Spot)

Since spot has no leverage, the risk knobs shift from leverage to capital deployment depth:

| Profile | Max SOs | Base Order | TP Range | Deviation | Coins | Drawdown Halt |
|---------|---------|-----------|----------|-----------|-------|---------------|
| Low | 5 | 3% | 1.5-2.5% | 3-4% (wider) | BTC, ETH only | 15% |
| Medium | 8 | 4% | 1.0-2.0% | 2-3% (adaptive) | BTC, ETH, SOL, BNB | 25% |
| High | 12 | 5% | 0.8-1.5% | 1.5-2.5% (tighter) | Broader incl. mid-caps | 35% |

**What the profiles control:**
1. **How deep** you DCA (number of SOs)
2. **How much** per layer (base order %)
3. **How tight** the grid (deviation %)
4. **Which coins** are eligible (mature vs mid-cap)
5. **When to stop** (drawdown halt threshold)

**Key difference from perps:** On perps, risk = leverage × position size. On spot, risk = DCA depth × capital deployed. No margin calls, no liquidation — worst case is clearly defined (coin goes to zero, you lose what you put in).

### Capital Efficiency: Spot vs Perps
| | Perps | Spot |
|---|---|---|
| $330 capital | ~$250-280 deployable (margin, reserves, both sides) | ~$297 deployable (just 10% reserve) |
| Reserve needed for TP | Yes (margin trap risk) | No (selling coins you own) |
| Liquidation risk | Yes | No |
| Multi-exchange deployment | Impractical (margin efficiency) | Natural fit |

---

## 10. Multi-DEX as a Product Feature

### The Advantage
Spot trading makes multi-DEX deployment practical. With perps, you want everything on one exchange for margin efficiency. With spot, spreading across DEXs **reduces risk** — no single exchange failure wipes you out.

### Product Positioning
**"One bot, any exchange, any wallet"**

Most DCA bots lock users into one exchange (3Commas → Binance, Pionex is their own exchange). AIT offers:
- Connect any wallet, any DEX
- Split capital across exchanges for safety
- Same adaptive intelligence everywhere
- User picks their coins, their exchanges, their risk level

### Tiered Exchange Access
| Tier | Exchanges | Coins | Price |
|------|-----------|-------|-------|
| Starter ($5K) | 1 | 1 | Entry |
| Trader ($10K) | 2 | 2 | Mid |
| Pro ($25K) | 3 | 3 | Mid-high |
| Elite ($50K) | 5 | 5 | Premium |
| Whale ($100K) | Unlimited | 8 | Top |

Exchange count becomes a tier feature alongside coin count — another reason to upgrade.

### The Post-FTX Pitch
> "Why trust all your capital to one exchange? Spread it across multiple DEXs with one dashboard."

Exchange diversification as a **feature**, not just a technical detail. This is a real pain point for crypto traders after FTX, Celsius, etc. Trust is fractured — AIT turns that into a selling point.

### Competitive Differentiator
- **3Commas**: Single exchange per bot
- **Pionex**: Locked to their exchange
- **CryptoHopper**: Multi-exchange but CEX-focused, complex setup
- **AIT**: Multi-DEX native, wallet-based, no KYC required on DEXs

---

## 11. Product Positioning — "Intelligent Accumulation"

### The Reframe
This is not a trading bot for day traders chasing daily ROI. This is an **intelligent accumulation engine** for long-term crypto investors who:
- Already believe in BTC/ETH/SOL long-term
- Are going to DCA anyway (many already do this manually)
- Want better entries than "buy $100 every Monday regardless of price"
- Want automated profit-taking on bounces instead of blindly HODLing
- Need discipline during bear markets when emotions say "don't buy"

That's a much larger market than active traders. Every crypto investor who does manual DCA is a potential customer.

### The Pitch by Market Phase

**Bear Market (strongest pitch):**
> "Stop catching falling knives. Let AI time your entries."
- People are scared to buy but know they should be accumulating
- Human discipline fails at the worst time — bot stays systematic
- Wider entry spacing in downtrends = better average cost basis
- EXTREME regime detection = hands off during capitulation events
- "Even in a 3-month downtrend, the bot kept you disciplined, got you entries 15-20% below where you would've bought manually"

**Recovery / Early Bull:**
> "Look at what our users accumulated at the bottom — now they're selling in tranches on the way up."
- Scale-out sells lock in profits automatically
- Largest lots (bought at lowest prices) sell first = maximum profit capture
- Users who accumulated during the bear are now realizing 3-5× returns

**Bull Market:**
> "Don't just HODL. Take profits automatically while keeping your core position."
- Scale-out captures gains without selling everything
- Adaptive regime detection tightens entries during momentum
- Users keep accumulating, keep taking profits

**Correction / Pullback:**
> "While everyone panics, your bot is buying the dip with a plan."
- Regime-aware spacing prevents overbuying
- Previous profits from scale-out create a buffer
- Systematic approach removes emotional decision-making

### Why This Works in Every Phase
The strategy profits from **volatility and mean reversion**, not from predicting direction. It doesn't need to be right about whether the market goes up or down — it just needs prices to move and eventually revert (partially). This is the nature of crypto markets.

The key insight: **you own the asset on spot**. There are no funding fees bleeding you, no liquidation risk, no margin calls. If BTC drops 30%, you haven't "lost" — you've accumulated more at a 30% discount. When (not if) it recovers, the scale-out captures that recovery.

### Macro Timing — Why Build Now
- BTC reached ATH, now in multi-month correction
- Probably 6-12 months of correction/accumulation before the next real bull run
- **Perfect window to build, backtest, and refine** while the market stress-tests us for free
- Product launches into the bear market when the pitch is strongest
- Early users accumulate at the best prices and become the best testimonials for the recovery phase

### Early Backtest Validation (90-day, Nov 2025 — Feb 2026)
Even during a significant market correction:

| Coin | Profile | Return | Win Rate | Max DD | Deals |
|------|---------|--------|----------|--------|-------|
| ETH | Medium | **+5.68%** | 100% | 10.68% | 14 |
| ETH | Low | **+2.63%** | 100% | 5.42% | 9 |
| BTC | Low | -13.62% | 83.3% | 20.4% | 6 |
| BTC | Medium | -22.4% | 88.9% | 32.76% | 9 |

ETH Medium: +5.68% over 90 days in a downtrend = ~23% annualized with 100% win rate on closed deals. BTC went negative because the downtrend was severe and sustained — but even there, 83-90% of individual deals were profitable. The "losses" are unrealized positions that would recover with time.

**This is the pitch:** "Your bot had an 89% win rate on individual trades, even in a 3-month correction. The unrealized positions? Those are your best entries for the next bull run."

### Target Customer Personas

**The Manual DCAer:**
- Already buys $X of BTC/ETH weekly or monthly
- Knows they should buy more during dips but can't time it
- Would love automation that's smarter than "buy every Monday"

**The HODL-and-Forget Investor:**
- Bought and holds, never takes profits
- Watches gains evaporate in corrections without selling
- Needs automated profit-taking that doesn't require daily attention

**The Bear Market Accumulator:**
- Experienced a cycle before, knows the value of buying the bear
- Wants to be systematic about accumulation
- Needs confidence that the bot won't over-commit during freefall

**The Emotional Trader:**
- Panic sells at the bottom, FOMO buys at the top
- Knows they'd be better off with a system
- Needs the discipline removed from their hands

---

## 12. Implementation Phases

### Phase 1: Strategy Validation (Current)
- [x] Strategy concept designed
- [x] Entry/exit logic defined
- [x] Coin screening criteria established
- [ ] Backtest spot scale-out vs futures on historical data
- [ ] Validate profit projections with real spread/fee data

### Phase 2: Exchange Selection
- [ ] Test Aster spot API capabilities (order types, limits)
- [ ] Evaluate Hyperliquid API for HYPE spot
- [ ] Compare fee structures across candidates
- [ ] Select primary exchange

### Phase 3: Bot Development
- [ ] Build spot API client (new or adapted from futures)
- [ ] Implement scale-out exit engine
- [ ] Adapt regime detection for spot-specific signals
- [ ] Build spot-specific dashboard view
- [ ] Paper trade for validation

### Phase 4: Live Deployment
- [ ] Deploy with minimal capital
- [ ] Monitor for 1-2 weeks
- [ ] Compare performance vs futures bot
- [ ] Scale up if results confirm strategy

---

## 13. Open Questions

1. **Partial fill handling** — What if a sell order partially fills? Hold remainder or adjust?
2. **Reentry logic** — After full scale-out (all lots sold), when do we reenter?
3. **Multi-coin coordination** — How to allocate capital across coins dynamically?
4. **Tax implications** — Spot buy/sell creates taxable events per transaction vs futures settlement
5. **Spread impact on small orders** — Will $13 base orders get good fills on spot?
6. **Aster spot API maturity** — Is it stable enough for bot trading? (Spot appears newer)

---

*This document captures the strategy as designed 2026-02-17. To be refined through backtesting and exchange evaluation.*
