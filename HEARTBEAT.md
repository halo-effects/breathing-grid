# HEARTBEAT.md

## Aster Spot Live Bot (ASTER/USDT)
- Check `trading/spot/live/aster/status.json` for bot health
- Alert if: `running` is false, drawdown > 15%, or regime changes to EXTREME
- Check status.json `last_update` — stale if >10 min old
- Restart: `Start-ScheduledTask -TaskName "AsterSpotLive"`
- Config: ASTER/USDT, Medium profile, 1h timeframe, lifecycle enabled

## V12e Paper Bot (Hyperliquid — ETH/SOL/BTC USDC)
- Check `trading/spot/paper/hyperliquid/status.json` for bot health
- Alert if: process not running or status.json stale (>20 min)
- Coins: ETH/USDC, SOL/USDC, BTC/USDC — 1h timeframe, Medium profile
- Pipeline enabled (scanner → pipeline → trader)
- Restart: `Start-ScheduledTask -TaskName "SpotPaperHyperliquid"`

## Dashboard Sync
- Task: `AIT_DashboardSync` (every 2 min)
- Verify `docs/data/v12e/status.json` and `docs/data/live-aster/status.json` are fresh on GitHub Pages
