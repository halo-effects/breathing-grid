# HEARTBEAT.md

## Aster Trading Bot (Live Futures)
- Check `trading/live/status.json` for bot health
- Alert if: `running` is false, `halted` is true, drawdown > 15%, or regime changes to EXTREME
- Check `trading/live/bot_service.log` tail for recent activity (stale = no logs in 5+ min)
- If bot is down, restart via: `Start-ScheduledTask -TaskName "AsterTradingBot"`
- If dashboard is down, restart via: `Start-ScheduledTask -TaskName "AsterDashboard"`

## Spot Paper Bots
- Check `trading/spot/paper/aster/status.json` and `trading/spot/paper/hyperliquid/status.json`
- Alert if: process not running or status.json stale (>20 min)
- Aster: ETH/USDT Medium 15m, PID check: `Get-Process -Name python -EA SilentlyContinue`
- Hyperliquid: HYPE/USDC Medium 15m
- Restart: `Start-ScheduledTask -TaskName "SpotPaperAster"` / `"SpotPaperHyperliquid"`
