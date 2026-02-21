# Rebalancing Aggressiveness System — Spec v2

## Overview
Controls how quickly and aggressively the bot rotates capital between coins based on Wyckoff lifecycle phase changes. Works alongside (not replacing) the risk profile system.

## Core Principle
**Entry conviction is always high.** Aggressiveness only controls how quickly you *leave* positions and how fast freed capital gets redeployed. The bot never rotates *into* a coin without strong conviction and confirmed phase.

## Three Modes

### Conservative *(Low profile default)*
Slow to exit, patient redeployment. Capital may sit idle longer between rotations. Fewer trades, lower fees.
*"I'd rather miss a move than get whipsawed."*

### Balanced *(Medium profile default)*
Responsive exits, steady redeployment pace. Good trade-off between opportunity cost and overtrading.
*"Keep me in the game without churning."*

### Aggressive *(High profile default)*
Quick exits on early weakness, rapid redeployment as soon as qualified coins appear. More trades, higher fees, but capital rarely sits idle.
*"I'd rather overtrade slightly than miss a rotation."*

## What Aggressiveness Controls

| Parameter | Conservative | Balanced | Aggressive |
|---|---|---|---|
| Rebalance cooldown | 12h | 4h | 1h |
| Exit lot sizing | 25% per step | 50% per step | 75% per step |
| Max rebalances/day | 2 | 6 | 12 |
| MARKDOWN exit speed | Patient (wait for confirmation) | Moderate | Fast (early exit on weakening signals) |

## What Stays Constant (All Modes)
- **Entry conviction threshold**: High (70+) — always
- **Phase confirmation**: Full (2+ confirming signals) — always
- **Entry = same quality bar** regardless of slider position

## Profile Defaults
- **Low** → Conservative
- **Medium** → Balanced
- **High** → Aggressive
- User can override anytime; change takes effect next cycle

## Interaction Rules
1. **Open positions unaffected** — aggressiveness only governs new rebalancing decisions, not existing DCA layers or open deals
2. **MARKDOWN behavior** — aggressive exits markdown coins faster (may miss spring entry), conservative waits longer (may catch spring but holds through more downside). This is a deliberate trade-off that matches investor risk appetite.
3. **Fee estimate** — shown in tooltip per mode (estimated monthly trade count based on historical rebalancing frequency)
4. **Cooldown is the governing lever** — no "capital idle tolerance" metric; cooldown timer between rebalances is sufficient

## How It Interacts with Risk Profiles
- **Risk profile** controls *how much* risk per position (leverage, SO depth, position sizing)
- **Aggressiveness** controls *how fast* capital moves between positions (exit speed, rebalance frequency)
- They are orthogonal — you can run High risk + Conservative rebalancing (big positions, slow rotation) or Low risk + Aggressive rebalancing (small positions, fast rotation)
- Defaults pair them logically but users have full control

## User Experience
- Three-position slider in dashboard settings
- Default set on account creation based on chosen risk profile
- User can change anytime; takes effect on next rebalancing cycle
- Tooltip explains each mode in plain language + estimated monthly fee impact
- All three modes available to all tiers — no lockout

## Decision Log
- Entry conviction/phase confirmation must be high for ALL modes (Brett, 2026-02-21)
- Capital idle tolerance removed as parameter — too complex to measure/enforce (Brett, 2026-02-21)
- Aggressiveness only affects new rebalancing decisions, not open positions (Brett, 2026-02-21)
- Aggressive MARKDOWN exit = may miss spring, acceptable for that risk appetite (Brett, 2026-02-21)
- Fee impact shown as tooltip only, not in main dashboard area (Brett, 2026-02-21)
