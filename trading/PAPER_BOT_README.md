# AIT Paper Trading Bot — Implementation Summary

## Phase 1: Single-Coin Paper Bot with Risk Profiles ✅

**Files Created:**
- `aster_trader_v2.py` — Single-coin paper trading engine
- `run_paper.py` — Entry point for single-coin bot
- `start_paper.ps1` — PowerShell launcher

**Key Features:**
- **Risk Profiles**: Low, Medium, High risk settings with different parameters
- **Virtual Trading**: Simulates orders using real market data, no actual trades
- **Regime Detection**: Same ADX/ATR/Hurst logic as live bot
- **Dual Tracking**: Simultaneous LONG + SHORT virtual deals
- **Status Output**: Comprehensive JSON with deal tracking
- **Trade Logging**: CSV format compatible with live bot
- **Telegram Notifications**: Prefixed with [PAPER] to distinguish from live

## Phase 2: Multi-Coin Extension ✅

**Files Created:**
- `aster_trader_v3.py` — Multi-coin paper trading engine  
- `run_paper_multicoin.py` — Entry point for multi-coin bot
- `start_paper_multicoin.ps1` — PowerShell launcher

**Key Features:**
- **Scanner Integration**: Reads `scanner_recommendation.json` and `scanner_t2.json`
- **CoinSlot System**: Virtual coin slots manage independent trading per symbol
- **Score-Proportional Allocation**: Capital distributed based on scanner scores
- **Per-Coin Regime Detection**: Each coin has independent market analysis
- **Multi-Coin Deal Tracking**: Separate long/short deals per coin
- **Capital Management**: 10% reserve, 15% minimum allocation per coin

## File Structure

```
trading/
├── aster_trader.py              # Live bot (unchanged)
├── aster_trader_v2.py           # Single-coin paper bot
├── aster_trader_v3.py           # Multi-coin paper bot
├── run_paper.py                 # Single-coin entry point
├── run_paper_multicoin.py       # Multi-coin entry point
├── start_paper.ps1              # Single-coin launcher
├── start_paper_multicoin.ps1    # Multi-coin launcher
└── paper/
    ├── allocation.json          # Risk profile allocation + max_coins
    ├── status.json             # Real-time bot status
    ├── trades.csv              # Trade log
    └── bot_service.log         # Bot logs
```

## Usage

### Single-Coin Paper Bot
```powershell
cd C:\Users\Never\.openclaw\workspace\trading
.\start_paper.ps1

# Or with custom parameters:
python -m trading.run_paper --capital 5000 --profile high
```

### Multi-Coin Paper Bot
```powershell
cd C:\Users\Never\.openclaw\workspace\trading
.\start_paper_multicoin.ps1

# Or with custom parameters:
python -m trading.run_paper_multicoin --capital 10000 --profile medium --max-coins 3
```

## Configuration

### Risk Profiles (`allocation.json`)
```json
{
  "low": 0,
  "medium": 100,     // 100% allocated to Medium Risk profile
  "high": 0,
  "total_capital": 10000,
  "max_coins": 3     // Trade up to 3 coins simultaneously
}
```

### Risk Profile Parameters

| Parameter | Low Risk | Medium Risk | High Risk |
|-----------|----------|-------------|-----------|
| **Leverage** | 1× | 2× | 5× |
| **Max SOs** | 8 | 12 | 16 |
| **SO Volume Mult** | 2.0× | 2.5× | 3.0× |
| **Base Order %** | 4% | 6% | 8% |
| **Capital Reserve** | 10% | 5% | 2% |
| **TP Range** | 0.6–2.5% | 0.4–2.0% | 0.2–1.5% |
| **Deviation Range** | 1.2–4.0% | 0.8–3.0% | 0.5–2.0% |
| **EXTREME Regime** | Halt (0/0) | Reduce (20/20) | Continue (40/40) |

## Status Output

### Multi-Coin Status Structure
```json
{
  "mode": "paper_multicoin",
  "active_profile": "medium",
  "total_capital": 10000,
  "equity": 10000,
  "pnl": 0,
  "active_coins": 3,
  "coins": {
    "HYPEUSDT": {
      "alloc_pct": 36.8,
      "alloc_capital": 3310,
      "scanner_score": 45.8,
      "regime": "ACCUMULATION",
      "long_deal": { ... },
      "short_deal": { ... },
      "pnl": 0
    },
    "ASTERUSDT": { ... },
    "DOGEUSDT": { ... }
  }
}
```

## Scanner Integration

The multi-coin bot automatically:
1. **Reads scanner results** every 4 hours from existing live scanner files
2. **Allocates capital** proportionally to top 3 coins by score
3. **Opens virtual deals** based on each coin's regime and trend
4. **Tracks performance** independently per coin

**Scanner Data Sources:**
- Primary: `trading/live/scanner_recommendation.json`
- Fallback: `trading/live/scanner_t2.json`
- Default: HYPEUSDT if no scanner data

## Testing Verification ✅

**Single-Coin Bot Test:**
- ✅ Starts without errors
- ✅ Detects regime (ACCUMULATION) 
- ✅ Opens LONG/SHORT virtual deals
- ✅ Creates SO levels with correct pricing
- ✅ Writes status.json with all required fields
- ✅ Logs trades to CSV with correct format

**Multi-Coin Bot Test:**
- ✅ Reads scanner results (HYPE, ASTER, DOGE, SOL, ETH)
- ✅ Allocates capital proportionally (36.8%, 33.7%, 29.5%)
- ✅ Opens deals for multiple coins simultaneously
- ✅ Per-coin regime detection (ACCUMULATION, RANGING, EXTREME)
- ✅ Independent deal tracking per symbol
- ✅ Multi-coin status output with per-coin breakdown

## Next Steps

The paper bot is now production-ready for Phase 1 testing. Key capabilities:

- **Risk Profile Testing**: Users can experiment with different risk levels safely
- **Multi-Coin Validation**: Test scanner-based coin selection and allocation
- **Strategy Validation**: Verify regime detection and adaptive parameters work across multiple assets
- **Performance Benchmarking**: Compare paper results vs. live bot performance

**Recommended Testing Approach:**
1. Run multi-coin paper bot for 1-2 weeks
2. Compare results with live single-coin bot
3. Analyze per-coin performance and allocation efficiency  
4. Use results to validate Phase 2 dashboard design

The foundation is solid for building the full AIT risk profile system on top of this paper trading engine.