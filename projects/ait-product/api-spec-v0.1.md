# AIT Central Server — API Spec (Draft v0.1)
*Created: 2026-02-15*

## Overview
Product model: Lump sum lifetime license for AIT trading bot.
- Customer runs bot locally (their machine/VPS)
- Customer manages their own Aster account, wallet, capital
- Central server provides: scanner, strategy updates, dashboards, licensing

## Base URL
`https://api.ait-trading.com/v1`

## Authentication
All endpoints require `Authorization: Bearer {license_key}` header.

---

## Endpoints

### License & Identity

**`POST /auth/activate`** — First-time activation
```json
// Request
{ "license_key": "AIT-XXXX-XXXX-XXXX", "machine_id": "hash_of_hw_identifiers" }
// Response 200
{ "customer_id": "c_abc123", "tier": "lifetime", "activated_at": "2026-02-15T00:00:00Z", "dashboard_url": "https://dashboard.ait-trading.com/c_abc123" }
// Response 403
{ "error": "license_already_bound", "message": "Contact support to transfer license" }
```

**`GET /auth/heartbeat`** — Periodic license check (every 6h)
```json
// Response 200
{ "valid": true, "tier": "lifetime", "bot_version_latest": "2.4.1", "message": null }
```

### Scanner & Strategy

**`GET /scanner/recommendation`** — Current coin rankings
```json
{
  "timestamp": "2026-02-16T00:07:11Z",
  "rankings": [
    { "symbol": "HYPEUSDT", "composite_score": 50.2, "daily_roi_pct": 0.348, "cycles_per_day": 2.7, "friendly_regime_pct": 62.0 }
  ],
  "rotation_threshold_pct": 20,
  "next_scan": "2026-02-16T04:00:00Z"
}
```

**`GET /strategy/params`** — Latest strategy parameters
```json
{
  "version": "2.4.1",
  "params": {
    "base_order_pct": 4.0, "max_safety_orders": 8, "so_volume_mult": 2.0,
    "timeframe": "5m", "tp_range": [0.6, 2.5], "deviation_range": [1.2, 4.0],
    "atr_period": 14, "atr_baseline_pct": 0.8,
    "regime_multipliers": {
      "RANGING": { "tp": 0.85, "dev": 0.80 },
      "TRENDING": { "tp": 1.20, "dev": 1.30 },
      "EXTREME": { "tp": 0.70, "dev": 1.50 }
    },
    "capital_reserve_pct": 10
  }
}
```

**`GET /strategy/regimes`** — Allocation rules
```json
{
  "allocation_rules": {
    "TRENDING": { "bullish": [75, 25], "bearish": [25, 75] },
    "RANGING": [50, 50],
    "EXTREME": [0, 0]
  },
  "sma_period": 50
}
```

### Data Ingestion (Bot → Central)

**`POST /data/{customer_id}/status`** — Push bot status (every 2 min)
**`POST /data/{customer_id}/trade`** — Push completed trade event

### Updates

**`GET /updates/check`** — Check for bot updates
**`GET /updates/download/{version}`** — Download bot package

### Dashboard (Public, no auth)

**`GET /dashboard/{customer_id}/`** — Dashboard HTML
**`GET /dashboard/{customer_id}/data/status.json`** — Customer status
**`GET /dashboard/{customer_id}/data/trades.csv`** — Trade history
**`GET /dashboard/{customer_id}/data/scanner.json`** — Shared scanner results

## Rate Limits
| Endpoint | Limit |
|----------|-------|
| Status push | 1/min |
| Trade push | 10/min |
| Scanner/params | 1/5min |
| Heartbeat | 1/hr |

---

## Customer Cap & Edge Analysis

### Edge Erosion Factors
- Liquidity crowding at same SO/TP levels
- Front-running from predictable order clusters
- Slippage when many bots hit TP simultaneously

### Natural Protections
- Adaptive params vary by ATR timing per bot
- Different capital sizes = different order sizes
- Coin portfolio variation across customers
- Entry timing differs (deal cycles are asynchronous)
- Net position mode reduces exchange-level footprint

### Capacity Estimates (HYPE/USDT, Aster, ~$3M daily volume)
| Customers | Max Capital/Each | Total Footprint | % of Volume | Risk |
|-----------|-----------------|-----------------|-------------|------|
| 10 × $10K | $100K | ~$30K/day | 1% | None |
| 50 × $25K | $1.25M | ~$375K/day | 12% | Noticeable |
| 50 × $50K | $2.5M | ~$750K/day | 25% | Safe ceiling |
| 100 × $100K | $5M | ~$1.5M/day | 50% | Edge erosion |

### Recommended Caps
- **Conservative**: 50 customers, $50K max → $150K revenue
- **Aggressive**: 100 customers, $100K max → $500K revenue (needs mitigations)

### Mitigations at Scale
1. Staggered scanner delivery (batch rotations over 2-4h)
2. Parameter jitter (±5-10% randomization on TP/deviation per customer)
3. Coin diversification enforcement (max 40% on same coin)
4. Volume monitoring (alert at 20% footprint)
5. Multi-exchange expansion (Binance, Bybit = dramatically more liquidity)

### Key Insight
Real cap = total AIT capital per coin per exchange, not customer count.
Multi-exchange + coin diversification = 200+ customers feasible.

---

## Tiered Licensing Model (Added 2026-02-15)

### Pricing: 10% of Max Capital
| Tier | Max Capital | Price |
|------|-------------|-------|
| Starter | $5K | $500 |
| Trader | $10K | $1,000 |
| Pro | $25K | $2,500 |
| Elite | $50K | $5,000 |
| Whale | $100K | $10,000 |

### Global Pool Cap: $2.5M total reserved capital
- Enforced via capacity reservation at purchase
- All tiers break even in ~29 days at 0.35% daily ROI
- Revenue converges ~$250K regardless of customer mix (10% × $2.5M)
- Upgrades: pay the difference, instant activation
- Enforcement: degrade scanner access if over cap (soft → hard)
- Pool status is public (creates FOMO / scarcity marketing)
- Expansion via new exchanges / coins increases pool cap

### New API Endpoints
- `GET /pool/status` — public capacity meter
- `POST /license/upgrade` — tier upgrade with differential pricing
