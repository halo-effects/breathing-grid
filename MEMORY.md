# MEMORY.md — Long-Term Memory

## Brett
- Direct, no-fluff communicator. Values security/governance deeply.
- Timezone: America/Los_Angeles
- Uses Telegram for personal, Slack for Halo Effects business
- No desktop Slack — browser only via Gmail login
- Quote: "It's about finding the right coin at the right time and running the strategy and getting out with your shirt"

## Adaptive Intelligence Trading (AIT)
- **Product name**: Adaptive Intelligence Trading (AIT) — decided 2026-02-14
- **GitHub**: github.com/halo-effects/adaptive-intelligence-trading (account: halo-effects, geegee@haloeffects.net)
- **Product page**: https://halo-effects.github.io/adaptive-intelligence-trading/ (served from `docs/` on main branch)
- **Live Dashboard**: https://halo-effects.github.io/adaptive-intelligence-trading/dashboard.html (v3.0, data synced every 2 min)
- **Local dashboard**: trading/live/dashboard.html (served on port 8080)
- GitHub PAT: `openclaw-deploy` (repo scope, expires ~Mar 16 2026)
- **Dashboard sync**: Windows Scheduled Task `AIT_DashboardSync` runs every 2 min, pushes status.json/trades.csv to `docs/data/` via `trading/sync_dashboard.ps1`
- GitHub Pages config: Deploy from branch `main`, folder `/docs`

## Aster Trading Bot (HYPEUSDT)
- **Live since 2026-02-13** on Aster DEX (Binance-compatible futures API)
- Dual-tracking bidirectional DCA: simultaneous LONG + SHORT virtual deals
- Net position mode (Aster doesn't support hedge mode) — opposing positions offset on exchange
- Capital: ~$332 USDT (was $292, +$40 deposit 2026-02-14), no leverage (1x)
- Params: TP=1.5%, Deviation=2.5%, MaxSO=8, SO_mult=2.0, 5m timeframe, base_order=4%
- Regime-based allocation: TRENDING=75/25 long/short, CHOPPY/RANGING=50/50, etc.
- EXTREME regime = 0/0 (halt trading)
- Files: `trading/aster_trader.py` (live), `trading/martingale_engine.py` (backtest)
- Runs as Windows Scheduled Tasks: `AsterTradingBot`, `AsterDashboard`
- **Restart procedure**: Stop-Process by PID first, THEN Start-ScheduledTask (task restart alone won't kill python)
- Dashboard: `trading/live/dashboard.html` (auto-refresh, shows both sides)

### Key Technical Lessons
- reduceOnly orders fail in net position mode when opposing side has larger position — don't use reduceOnly
- Base order sizing should NOT scale by allocation % (hits minimums) — allocation only gates open/close
- TP retry logic needed — exchange can reject TP placement, must auto-retry
- 5m timeframe >> 1m for this strategy (less noise)
- 1x leverage >> 2x (counterintuitive — 2x eats capital reducing deal cycling)
- Wider SO deviation (2.5%) strongly preferred over tight (1.5%)
- **Net position mode margin trap**: selling to close a long can flip net short, requiring margin for the flip — must check margin before TP placement
- **TP > SOs priority**: always ensure TP can be placed, cancel deep SOs if needed to free margin
- **Aster fees**: Maker = 0% (free), Taker = 0.04% — limit orders (TP, SO fills) are free

### Adaptive TP/Deviation System (Live since 2026-02-14)
- Dynamic TP: 0.6–2.5% based on 14-period ATR + regime multipliers (baseline 1.5%, ATR_BASELINE=0.8%)
- Dynamic deviation: 1.2–4.0% (baseline 2.5%), floor = TP × 1.5
- Regime multipliers: RANGING=0.85×TP/0.80×DEV, TRENDING=1.20×TP/1.30×DEV, EXTREME=0.70×TP/1.50×DEV
- Margin-aware: 10% capital reserve, skips unaffordable SOs, cancels deep SOs to ensure TP placement
- TP-hit analysis logged with duration, adaptive params, ATR, regime insight

### WhiteHatFX History
- Brett previously traded MT4 Martingale (WhiteHatFX v2) on FTMO 100k prop firms: BTC, currencies, gold, US30
- Dual-engine bidirectional grid — same core concept as Aster
- Key diff: 4x lot multiplier (vs 2x now), fixed params, no equity protection
- Evolution: from static hardcoded → dynamic regime-adaptive ("Breathing Grid")

### Coin Scanner (Two-Tier Architecture) — Built 2026-02-15
- **Tier 1** (`trading/coin_scanner_t1.py`): ADX, ATR%, Hurst, SMA crosses, volume on all 275 Aster pairs (seconds/coin)
- **Tier 2** (`trading/coin_scanner_t2.py`): Full 14-day 5m backtest on shortlisted coins (minutes/coin)
- **Runner**: `trading/run_scanner.py` — ties both tiers, outputs recommendation
- **Maturity filters**: 60+ day age, <120% price swing, <4x volume spike, $1M volume floor
- Latest results: HYPE #1 (52.9), ASTER #2 (46.1), DOGE #3 (41.2), SOL #4, ETH #5
- Cron job: every 4h (ID: b9571b56-5d72-4d25-b125-d834b12ea572)
- Rotation threshold: 20% improvement required
- Output: `trading/live/scanner_t1.json`, `scanner_t2.json`, `scanner_recommendation.json`

### Multi-Coin Portfolio Manager — Built 2026-02-15
- `trading/portfolio_manager.py` — PortfolioManager + CoinSlot classes
- `trading/run_portfolio.py` — entry point (--dry-run, --max-coins, --leverage, --capital)
- Up to 3 coins, scanner-driven, score-proportional allocation, 10% reserve, 15% min per coin
- Graceful wind-down: no new deals, 2h force-close, 4h minimum hold time
- **Not yet live** — bot still running single-coin HYPE via `run_aster.py`
- Capital concern: $335 split 3 ways risks hitting $5 minimum notional; recommend max-coins=2

### Directional Awareness — Added 2026-02-15
- SMA50-based trend direction detection in `detect_regime()`
- Directional regimes (TRENDING, MILD_TREND, DISTRIBUTION) flip long/short allocation when bearish
- status.json: `trend_direction: "bullish"/"bearish"`, log shows ▲/▼

### Legacy Coin Screener (superseded)
- Old: `trading/coin_screener.py` — single-tier, no maturity filters
- HYPE ranked #1 (0.876 fitness) for dual-tracking: low trend (2.7%), good range

## Slack Integration
- Workspace: halo-effects.slack.com
- Channel: C092S0TVA0Z
- Full gateway restart needed for Slack socket (SIGUSR1 insufficient)
- Bot name: "Gee Gee"
- **Socket drops silently** — Brett sends messages I never receive. Recurring issue.
- Agent can't restart gateway directly (commands.restart=true not set) — Brett must run `openclaw gateway restart`
- Consider adding Slack health check to HEARTBEAT.md

## TrustedBusinessReviews.com Migration (Active Project)
- **Phase 1 (active):** WordPress → static HTML migration, review system, admin dashboard, Google schema
- Instructions: `projects/tbr/migration-instructions.md`
- FTP access working (Adeel fixed path 2026-02-14)
- **Malware cleanup in progress** — major compromise found, mostly cleaned, ~1,900 spam pages + 2 plugins still need finishing
- Password changes still recommended (credential exfil was active)
- Public crawl done — ~10 business listings across 5-6 categories, Phoenix AZ focused
- Google Doc trick: append `/mobilebasic` to extract text from Google Docs via browser

## Communication Channels
- **Slack** → Halo Effects business (TBR, ShadowQuery, Adeel)
- **Telegram** → Trading bot, personal projects, everything else
- Slack channel: C092DGXUZFW (#team-)
- Slack user IDs: Brett=U092S0TJK5X, Adeel=U092D6SA0JW

## Deferred Projects
- **AI GEO / ShadowQuery**: Brett moved discussion to Slack with Adeel; TBR migration is prep for this
- Tutorial notes saved: `reference/shadowquery-tutorials.md`

## Embedding/Memory Search
- Not working — no OpenAI/Google/Voyage API key configured for embeddings
