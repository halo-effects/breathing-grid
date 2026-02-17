import csv, json
from datetime import datetime, timezone

trades = list(csv.DictReader(open('trading/live/trades.csv')))

tps = [t for t in trades if t['action'] == 'TP_HIT']
opens = [t for t in trades if t['action'] == 'OPEN']
sos = [t for t in trades if t['action'] == 'SO_FILL']

print(f'Total rows: {len(trades)}')
print(f'Opens: {len(opens)}, SOs: {len(sos)}, TPs: {len(tps)}')
print()

total_pnl = 0
total_fees = 0
total_notional_all = 0
total_funding_est = 0

for tp in tps:
    deal_id = tp['deal_id']
    direction = tp['direction']
    pnl = float(tp['pnl'])
    so_count = int(tp['so_count'])
    
    # Find all orders for this deal
    deal_trades = [t for t in trades if t['deal_id'] == deal_id and t['direction'] == direction]
    entry_trades = [t for t in deal_trades if t['action'] in ('OPEN', 'SO_FILL')]
    total_entry_notional = sum(float(t['notional']) for t in entry_trades)
    tp_notional = float(tp['notional'])
    
    # Parse timestamps
    open_time = datetime.fromisoformat(entry_trades[0]['timestamp'])
    close_time = datetime.fromisoformat(tp['timestamp'])
    duration_hrs = (close_time - open_time).total_seconds() / 3600
    
    # Fees: entries are taker (0.04%), TP is maker (0%)
    entry_fees = total_entry_notional * 0.0004
    
    # Funding estimate: ~0.01% per 4h on avg position
    avg_position_notional = total_entry_notional * 0.6  # rough avg
    funding_periods = duration_hrs / 4
    est_funding = avg_position_notional * 0.0001 * funding_periods
    
    net = pnl - entry_fees - est_funding
    pnl_pct = (pnl / total_entry_notional * 100) if total_entry_notional else 0
    
    total_pnl += pnl
    total_fees += entry_fees
    total_notional_all += total_entry_notional
    total_funding_est += est_funding
    
    print(f'Deal {deal_id:>3} {direction:>5}: PnL ${pnl:>6.3f} | Notional ${total_entry_notional:>7.2f} | '
          f'SOs={so_count} | {duration_hrs:>5.1f}h | '
          f'Gross {pnl_pct:.2f}% | Fees ${entry_fees:.3f} | Fund ~${est_funding:.3f} | Net ~${net:.3f}')

print(f'\n{"="*80}')
print(f'Gross PnL:        ${total_pnl:.3f}')
print(f'Entry fees:       ${total_fees:.3f}')
print(f'Est funding:      ${total_funding_est:.3f}')
print(f'Est net:          ${total_pnl - total_fees - total_funding_est:.3f}')
print(f'Total notional:   ${total_notional_all:.2f}')
print(f'Avg gross/trade:  ${total_pnl/len(tps):.3f}')
print(f'Avg notional:     ${total_notional_all/len(tps):.2f}')
print(f'Avg gross %:      {total_pnl/total_notional_all*100:.3f}%')

# Check adaptive TP values from status
status = json.load(open('trading/live/status.json'))
print(f'\nCurrent adaptive TP: {status.get("adaptive_tp_pct", "?")}%')
print(f'Current adaptive Dev: {status.get("adaptive_dev_pct", "?")}%')
print(f'Regime: {status.get("regime")}')
print(f'Capital: ${status.get("equity", 0):.2f}')
print(f'Base order %: {status.get("base_order_pct", "?")}%')

# Show active deals
for d in status.get('active_deals', []):
    dur = d.get('duration_str', '?')
    so = d.get('so_count', 0)
    cost = d.get('total_cost', 0)
    entry = d.get('entry_price', 0)
    avg = d.get('avg_entry', 0)
    print(f"\nActive: {d['direction']} #{d['deal_id']} - {so} SOs, "
          f"entry=${entry}, avg=${avg}, cost=${cost:.2f}, duration={dur}")
