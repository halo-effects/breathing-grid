# AIT Risk Profile System — Product Specification v1.0
*Created: 2026-02-16*

## 1. Overview & Philosophy

### 1.1 Risk-Reward Tradeoff Framework

The AIT Risk Profile System applies **modern portfolio theory to algorithmic trading**, enabling users to construct their own risk-return profile through systematic parameter allocation rather than hoping for the best with a single strategy.

**Core Principle**: Different market participants have different risk tolerances, capital sizes, and return expectations. Rather than force all users into one "optimal" configuration, AIT empowers users to consciously choose their position on the risk-reward spectrum.

**Default Position**: 100% Low Risk allocation (current system settings) ensures existing users experience no behavioral changes while gaining access to additional options.

### 1.2 Why Multiple Risk Profiles?

**Strategic Differentiation**
- Most trading bots offer binary choices: "conservative" or "aggressive"
- AIT provides granular control with transparent parameter mapping
- Users understand exactly what each setting does and why it affects risk/reward

**User Empowerment**
- Beginners start conservative, scale up as they gain experience
- Experienced traders can fine-tune their risk exposure
- Portfolio diversification within a single trading system

**Market Reality**
- No single parameter set performs optimally across all market conditions
- Risk tolerance varies with account size, personal circumstances, and market cycle
- Allows users to "dial in" their comfort zone precisely

**Competitive Moat**
- First-mover advantage in portfolio-theory-based bot trading
- Transparent risk parameter mapping vs. black box approaches
- Educational component builds user loyalty and understanding

### 1.3 Product Philosophy

"AIT doesn't pick your risk level for you — it gives you the tools to pick it yourself, with full transparency about the tradeoffs."

---

## 2. Risk Profile Specifications

### 2.1 🟢 Low Risk Profile (Conservative)

**Target User**: Capital preservation focused, beginners, steady compounding seekers

**Performance Targets**
- Daily ROI: 0.3–0.5%
- Maximum Drawdown: 5–15%
- Blow-up Risk: Very Low
- Deal Success Rate: 85–95%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Leverage** | 1× (spot) | No amplification of losses |
| **Max Safety Orders** | 5 | Limited Martingale exposure |
| **SO Volume Multiplier** | 2.0× | Conservative progression |
| **Base Order %** | 3% | Moderate position sizing |
| **Capital Reserve** | 10% | Substantial buffer for volatility |
| **TP Range** | 1.5–2.5% | Patient profit-taking |
| **Deviation Range** | 3.0–4.0% | Wider tolerance for price swings |
| **Max Drawdown** | 15% | Halt threshold |
| **Max Coins** | 2 | Focused allocation |
| **EXTREME Regime** | Halt (0/0) | Complete preservation mode |
| **Max Directional Bias** | 75/25 | Moderate trend following |

**Risk Characteristics**
- Survives extended drawdowns without liquidation
- Slow but steady capital growth
- High probability of profitable exits
- Minimal overnight risk

---

### 2.2 🟡 Medium Risk Profile (Balanced)

**Target User**: Experienced traders comfortable with volatility, balanced growth seekers

**Performance Targets**
- Daily ROI: 0.8–1.5%
- Maximum Drawdown: 15–35%
- Blow-up Risk: Low-Moderate
- Deal Success Rate: 75–85%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Leverage** | 1× (spot) | Spot-only, no leverage |
| **Max Safety Orders** | 8 | Extended Martingale depth |
| **SO Volume Multiplier** | 2.0× | Moderate averaging down |
| **Base Order %** | 4% | Balanced position sizing |
| **Capital Reserve** | 10% | Standard buffer |
| **TP Range** | 1.0–2.0% | Balanced profit realization |
| **Deviation Range** | 2.0–3.0% | Moderate price tolerance |
| **Max Drawdown** | 25% | Halt threshold |
| **Max Coins** | 3 | Diversified allocation |
| **EXTREME Regime** | Reduce (20/20) | Limited activity in chaos |
| **Max Directional Bias** | 85/15 | Strong trend following |

**Risk Characteristics**
- Faster returns but higher volatility
- Moderate leverage increases both gains and losses
- Can handle medium-term adverse moves
- Requires active monitoring during extreme markets

---

### 2.3 🔴 High Risk Profile (Aggressive)

**Target User**: Risk-tolerant traders, maximum return seekers, experienced leverage users

**Performance Targets**
- Daily ROI: 2–5%+
- Maximum Drawdown: 35–70%
- Blow-up Risk: Meaningful
- Deal Success Rate: 60–75%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Leverage** | 1× (spot) | Spot-only, no leverage |
| **Max Safety Orders** | 12 | Deep Martingale progression |
| **SO Volume Multiplier** | 2.0× | Aggressive averaging |
| **Base Order %** | 5% | Larger initial positions |
| **Capital Reserve** | 10% | Standard buffer |
| **TP Range** | 0.8–1.5% | Rapid profit-taking |
| **Deviation Range** | 1.5–2.5% | Tighter price tolerance |
| **Max Drawdown** | 35% | Halt threshold |
| **Max Coins** | 5 | Maximum diversification |
| **EXTREME Regime** | Continue (40/40) | Opportunistic in volatility |
| **Max Directional Bias** | 95/5 | Maximum trend conviction |

**Risk Characteristics**
- High return potential with commensurate risk
- Significant leverage multiplies all movements
- Can achieve rapid gains or losses
- Requires sophisticated risk management understanding
- Potential for account liquidation in adverse conditions

---

## 3. Risk Amplifier Analysis

### 3.1 Parameter Impact Ranking

**1. Leverage (Highest Impact)**
- **Risk**: Multiplies losses proportionally (2× leverage = 2× loss potential)
- **Reward**: Multiplies gains proportionally (2× leverage = 2× profit potential)
- **Mechanism**: Direct position size amplification affects every trade outcome
- **Liquidation Risk**: Higher leverage = lower liquidation threshold

**2. SO Depth + Volume Multiplier (High Impact)**
- **Risk**: Deeper Martingale = larger drawdowns before recovery
- **Reward**: More aggressive averaging = faster recovery when trend reverses
- **Mechanism**: Each SO compounds previous position, creating exponential exposure
- **Capital Efficiency**: More SOs = higher capital utilization but less flexibility

**3. Capital Reserve Reduction (Medium-High Impact)**
- **Risk**: Less buffer for unexpected volatility spikes
- **Reward**: Higher capital utilization = more active trading opportunities
- **Mechanism**: Reserve acts as safety margin; less reserve = more leverage indirectly
- **Liquidity**: Lower reserves = less ability to handle multiple simultaneous deals

**4. EXTREME Regime Behavior (Medium Impact)**
- **Risk**: Trading in extreme volatility increases whipsaw losses
- **Reward**: Volatility creates more profitable mean reversion opportunities
- **Mechanism**: EXTREME markets offer highest profit potential but highest uncertainty
- **Frequency**: Affects 10-20% of trading days but disproportionate P&L impact

**5. Directional Bias Strength (Medium Impact)**
- **Risk**: Strong bias increases exposure to trend reversals
- **Reward**: Catching major trends with conviction multiplies returns
- **Mechanism**: Higher bias = larger positions in trending direction
- **Market Dependency**: More effective in trending markets, dangerous in ranging

**6. TP/Deviation Tightness (Lower Impact)**
- **Risk**: Tighter parameters = more frequent but smaller losses
- **Reward**: Tighter parameters = more frequent trading opportunities
- **Mechanism**: Affects trade frequency and win/loss ratio balance
- **Optimization**: Fine-tuning for specific market conditions and volatility regimes

### 3.2 Compound Risk Effects

Risk amplifiers **multiply each other** rather than simply adding:
- High leverage + deep SOs + low reserves = exponential risk increase
- Conservative settings across all parameters = compound safety
- Mixed profiles allow risk diversification within single account

---

## 4. Capital Allocation System (UX)

### 4.1 Allocation Interface

**Primary Control**: Percentage sliders for each risk profile (must sum to 100%)

**Preset Allocations**
- **Conservative**: 100% Low / 0% Medium / 0% High
- **Balanced**: 60% Low / 30% Medium / 10% High  
- **Aggressive**: 30% Low / 40% Medium / 30% High
- **Custom**: User-defined slider positions

**Minimum Allocation Requirements**
- Minimum $500 per active profile (avoid order size issues)
- Maximum 3 active profiles simultaneously
- Total allocation must equal 100% of available capital

### 4.2 Virtual Sub-Account Architecture

**Independent Tracking**
- Each active profile operates as isolated virtual account
- Separate deal tracking, P&L calculation, parameter application
- Individual performance metrics and statistics
- Independent coin rotation and position management

**Shared Resources**
- Single exchange account and API keys
- Unified balance management and margin calculation  
- Shared scanner recommendations (applied per profile parameters)
- Central risk monitoring and emergency controls

**Capital Isolation**
- Profile allocations are ring-fenced (Low Risk can't access Medium Risk capital)
- Deal sizes calculated from profile-specific allocation only
- Profits retained within originating profile unless rebalanced
- Losses contained within profile allocation

### 4.3 Rebalancing Mechanics

**Allocation Changes**
- New allocations take effect at next deal cycle (no forced liquidations)
- Removed profiles gracefully wind down existing positions
- Added profiles immediately begin trading with new allocation
- Capital transfers between profiles occur at position close

**Rebalancing Triggers**
- User-initiated allocation changes
- Automatic triggers (profit taking, loss limits)
- Schedule-based rebalancing (weekly/monthly)
- Performance-based adjustments (trailing stop on risk escalation)

**Implementation**
- Changes applied during natural position cycling
- No forced position closures (maintains deal integrity)
- Smooth transition preserves trading momentum
- Emergency override available for immediate stops

---

## 5. Implementation Architecture

### 5.1 Bot Integration

**Virtual Account Layer**
```
User Capital ($50,000)
├── Low Risk Profile (60% = $30,000)
│   ├── Deal Tracking Engine
│   ├── Parameter Set: Low Risk
│   └── P&L Calculation
├── Medium Risk Profile (30% = $15,000)  
│   ├── Deal Tracking Engine
│   ├── Parameter Set: Medium Risk
│   └── P&L Calculation
└── High Risk Profile (10% = $5,000)
    ├── Deal Tracking Engine
    ├── Parameter Set: High Risk
    └── P&L Calculation
```

**Shared Components**
- Exchange API interface (single connection)
- Scanner data ingestion (broadcast to all profiles)
- Risk monitoring system (aggregate oversight)
- Emergency controls (kill switches, margin calls)

**Isolation Mechanisms**
- Memory separation between profile engines
- Independent parameter application
- Separate position sizing calculations
- Isolated P&L accounting

### 5.2 Central Server Enhancements

**New API Endpoints**

```
GET /strategy/risk-profiles          # Available risk profile definitions
POST /user/allocation               # Update user's risk allocation
GET /user/allocation                # Current user allocation
GET /user/profiles/performance      # Per-profile performance metrics
POST /user/profiles/rebalance       # Trigger rebalancing
GET /strategy/params/{profile_id}   # Profile-specific parameters
```

**Profile Configuration Management**
- Risk profiles defined centrally, pushed to all bots
- Version control for parameter updates
- A/B testing capability for parameter optimization
- Performance monitoring across user base

**Enhanced Data Collection**
- Per-profile trade reporting
- Allocation change tracking
- Performance attribution analysis
- Risk metric calculation and alerting

### 5.3 Dashboard Modifications

**Multi-Profile Dashboard**
- Unified view showing combined performance
- Individual profile breakdowns with separate P&L
- Allocation visualization (pie charts, performance comparison)
- Risk metrics dashboard (drawdown, volatility, Sharpe ratio by profile)

**Risk Management Interface**
- Real-time risk monitoring across all profiles
- Automatic alert system for drawdown thresholds
- Manual override controls (emergency stops, allocation locks)
- Performance analytics and optimization suggestions

**User Experience Enhancements**
- Profile performance comparison charts
- Risk-adjusted return calculations
- Backtesting interface (simulated historical performance by profile)
- Educational content explaining risk/reward tradeoffs

---

## 6. Risk Disclosures & Guardrails

### 6.1 Mandatory Risk Warnings

**Medium Risk Profile Activation**
```
⚠️ WARNING: Medium Risk Profile
• Uses 2-3× leverage, amplifying both gains and losses
• Target drawdown: 15-35% (vs. 5-15% for Low Risk)
• Requires active monitoring during volatile markets
• Not suitable for capital you cannot afford to lose

[ ] I understand the increased risk and potential for larger losses
[ ] I have experience with leveraged trading
[ ] I will monitor my positions actively

[Continue] [Cancel]
```

**High Risk Profile Activation**
```
🚨 EXTREME RISK WARNING: High Risk Profile
• Uses 5-10× leverage with significant liquidation risk
• Target drawdown: 35-70% - you may lose most of your allocation
• Continues trading during extreme market conditions
• Only for experienced traders comfortable with high volatility
• Account liquidation is possible during adverse conditions

REQUIRED CONFIRMATIONS:
[ ] I understand this profile may result in total loss of allocated capital
[ ] I have extensive experience with high leverage trading (5×+ leverage)
[ ] I can afford to lose 100% of the capital allocated to this profile
[ ] I will not allocate more than 20% of my total capital to High Risk

Type "I ACCEPT THE EXTREME RISK" to continue: ___________

[Continue] [Cancel]
```

### 6.2 Automatic Safeguards

**Drawdown Protection**
- **Medium Risk**: Auto-reduce to Low Risk if drawdown exceeds 30%
- **High Risk**: Auto-reduce to Medium Risk if drawdown exceeds 50%
- **High Risk**: Auto-reduce to Low Risk if drawdown exceeds 60%
- **Emergency**: Complete halt if total account drawdown exceeds 70%

**Allocation Limits**
- Maximum 50% of total capital in Medium + High Risk combined
- Maximum 25% of total capital in High Risk profile
- Minimum 25% must remain in Low Risk (safety buffer)
- Override requires manual confirmation and cooling-off period

**Leverage Monitoring**
- Real-time effective leverage calculation across all profiles
- Margin call alerts at 80% of liquidation threshold
- Automatic position reduction if approaching margin limits
- Emergency deleveraging if exchange margin requirements change

### 6.3 Circuit Breakers

**Market Condition Triggers**
- Automatic risk reduction during flash crashes (>10% move in <1 hour)
- Enhanced monitoring during high volatility periods (VIX equivalent >30)
- Forced profile downgrade during exchange outages or API failures
- Complete trading halt during "black swan" events (manual override only)

**User Behavior Safeguards**
- 24-hour cooling off period before High Risk activation
- Maximum 2 risk escalations per month
- Mandatory performance review before major allocation increases
- Account lock if repeated rapid allocation changes (prevents emotional trading)

---

## 7. Pricing Impact

### 7.1 Tier Integration

**Risk Profiles as Core Feature**
- All risk profiles available to all tier levels (Starter through Whale)
- No additional charges for accessing different risk profiles
- Feature differentiation comes from capital limits, not risk access

**Capital Limit Application**
```
Example: Pro Tier ($25K max capital)
• Can allocate across all three risk profiles
• Total allocation cannot exceed $25K across all profiles
• Example: $15K Low + $7K Medium + $3K High = $25K total
```

**Tier-Specific Considerations**
- **Starter/Trader**: Limited capital makes High Risk profile less relevant
- **Pro/Elite/Whale**: Full risk profile utilization becomes meaningful
- **All Tiers**: Educational value and upgrade path motivation

### 7.2 Revenue Impact

**No Direct Revenue Change**
- Risk profiles are value-added feature, not separate SKU
- Revenue remains 10% of maximum capital across all profiles
- Tier pricing structure unchanged

**Indirect Revenue Benefits**
- Increased customer satisfaction and retention
- Motivation for tier upgrades (higher capital limits)
- Competitive differentiation supporting premium pricing
- Educational component reduces support burden

**Customer Lifetime Value Enhancement**
- Users more likely to stay engaged with customizable risk
- Natural progression path from Conservative → Balanced → Aggressive
- Reduces churn from "one size fits all" limitations

---

## 8. Competitive Advantage

### 8.1 Market Differentiation

**Industry Standard Approach**
- Binary risk settings: "Conservative" or "Aggressive"
- Black box parameter selection
- No transparency into risk/reward tradeoffs
- Single strategy per subscription

**AIT Risk Profile Advantage**
- **Transparency**: Users see exactly what each parameter does
- **Control**: Granular risk allocation rather than binary choices
- **Education**: Understanding builds confidence and loyalty
- **Flexibility**: Portfolio approach within single trading system

### 8.2 Strategic Moat

**First-Mover Advantage**
- First portfolio-theory approach to retail trading bots
- Patent opportunity for risk allocation methodology
- Brand positioning as "intelligent risk management" platform
- Technical complexity barrier for competitors

**Network Effects**
- User-generated backtesting data improves profile optimization
- Community sharing of allocation strategies
- Performance benchmarking across user base
- Educational content library growth

**Switching Costs**
- Users develop expertise in AIT's risk framework
- Portfolio allocation becomes part of user's trading philosophy
- Historical performance data locked to platform
- Relearning cost of switching to simpler systems

### 8.3 Educational Competitive Edge

**Risk Literacy Building**
- Users learn trading risk management through hands-on experience
- Transparent parameter mapping demystifies algorithmic trading
- Performance attribution teaches cause-and-effect relationships
- Creates sophisticated user base less likely to switch

**Trust Through Transparency**
- No hidden "secret sauce" or black box decisions
- Users understand exactly what they're paying for
- Performance attribution is clear and verifiable
- Builds long-term customer relationships vs. transactional usage

---

## 9. Future Considerations

### 9.1 Advanced Features Roadmap

**Backtesting Engine (Q2 2026)**
- Historical performance simulation for each risk profile
- Monte Carlo analysis for drawdown probability estimation
- Stress testing against historical volatility events
- ROI projections based on capital allocation scenarios

**Custom Risk Profiles (Q3 2026)**
- Advanced users can create custom parameter combinations
- Template sharing between users (community-driven)
- Expert-curated profile library
- A/B testing framework for personal optimization

**AI-Powered Risk Assessment (Q4 2026)**
- Risk tolerance questionnaire with behavioral analysis
- Suggested allocation based on user psychology and capital size
- Dynamic rebalancing recommendations based on performance
- Market regime detection with automatic profile adjustment

**Performance Analytics Suite (Q1 2027)**
- Sharpe ratio and risk-adjusted return calculations
- Correlation analysis between profiles and market conditions
- Attribution analysis: which profile contributed to overall returns
- Benchmarking against traditional trading strategies

### 9.2 Community Features

**Anonymous Performance Leaderboard**
- Risk-adjusted performance rankings by profile
- Allocation strategy sharing (without revealing capital amounts)
- Community voting on most effective allocation strategies
- Seasonal challenges and competitions

**Educational Content Platform**
- Video tutorials explaining each risk parameter
- Case studies of successful allocation strategies
- Market commentary with profile-specific recommendations
- Webinar series on advanced risk management

**User-Generated Content**
- Strategy sharing and discussion forums
- User testimonials and case studies
- Community-driven parameter optimization research
- Peer mentorship program for new users

### 9.3 Technical Expansion

**Multi-Exchange Risk Profiles**
- Profile allocation across different exchanges
- Cross-exchange arbitrage opportunities
- Risk diversification through platform spreading
- Enhanced liquidity through multi-venue access

**Multi-Asset Risk Profiles**
- Crypto, forex, and stock market profile applications
- Cross-asset correlation analysis and allocation
- Sector-specific risk profiles (DeFi, gaming tokens, memecoins)
- Traditional finance integration (stocks, bonds, commodities)

**Algorithmic Enhancements**
- Machine learning optimization of profile parameters
- Dynamic profile adjustment based on market conditions
- Predictive modeling for optimal allocation timing
- Advanced risk models incorporating options theory

### 9.4 Regulatory Considerations

**Compliance Preparation**
- Risk disclosure standardization for regulatory approval
- Audit trail capabilities for all allocation changes
- Performance reporting standardization
- User agreement updates for international expansion

**Institutional Readiness**
- Professional risk management features for larger accounts
- Compliance reporting for institutional investors
- Integration with traditional portfolio management systems
- White-label risk profile system for financial advisors

---

## 10. Success Metrics

### 10.1 Product KPIs

**User Engagement**
- Profile adoption rate (% users activating Medium/High Risk)
- Allocation change frequency (optimal: 2-4 changes per quarter)
- Time spent in dashboard (risk monitoring engagement)
- Feature utilization depth (basic vs. advanced features)

**Risk Management Effectiveness**
- Drawdown incidents by profile (vs. predicted ranges)
- User-initiated emergency stops (should be <1% of user base)
- Automatic safeguard triggers (effectiveness validation)
- Account liquidation rate by profile (<0.1% for Low, <5% for High)

**Business Impact**
- Customer retention improvement (target: +15% vs. single-profile system)
- Average tier upgrade rate (risk profiles as upgrade motivation)
- Support ticket reduction (better user understanding)
- Net Promoter Score improvement (+20 points target)

### 10.2 Performance Validation

**Profile Performance Targets**
- Low Risk: 0.3-0.5% daily ROI, <15% max drawdown, >85% deal success
- Medium Risk: 0.8-1.5% daily ROI, <35% max drawdown, >75% deal success  
- High Risk: 2-5% daily ROI, <70% max drawdown, >60% deal success

**Risk-Adjusted Returns**
- Sharpe ratio improvement over single-profile system
- Maximum drawdown vs. expected ranges
- Volatility matching risk profile specifications
- Recovery time from drawdown events

### 10.3 Launch Success Criteria

**Phase 1 (Beta Launch - 50 users)**
- Technical stability: <0.1% system downtime
- User satisfaction: >8.0/10 average rating
- Risk calibration: Performance within 20% of targets
- Safety validation: No unexpected liquidations

**Phase 2 (Limited Release - 200 users)**
- Profile adoption: >60% of users activate multiple profiles
- Performance attribution: Clear profile performance differentiation
- User education: >90% pass risk understanding quiz
- Competitive advantage: Demonstrable differentiation in user feedback

**Phase 3 (Full Release - All Users)**
- Business impact: Customer retention +10% minimum
- Product-market fit: Organic user request for advanced features
- Technical scalability: System handles peak loads without degradation
- Revenue protection: No negative impact on tier upgrade rates

---

## 11. Tier-Based Coin Scaling

### 11.1 Capital Tier Framework

The AIT system scales coin allocation based on user capital tiers, providing broader market diversification as capital increases while maintaining appropriate per-coin minimums.

| Tier | Capital | Max Coins | Per-Coin Floor |
|------|---------|-----------|----------------|
| Starter | $5K | 1 | ~$5K |
| Trader | $10K | 2 | ~$5K |
| Pro | $25K | 3 | ~$8K |
| Elite | $50K | 5 | ~$10K |
| Whale | $100K | 8 | ~$12.5K |

### 11.2 Allocation Rules

**Scanner Integration**
- Scanner runs every 4 hours, identifying top N coins per user's tier
- Coins allocated proportional to scanner scores within tier limits
- Each coin's allocation further splits across active risk profiles (Low/Medium/High)

**Capital Distribution**
- 10% global reserve stays off the table for risk management
- Remaining 90% of capital distributed across selected coins
- Minimum ~$3K per coin floor — if capital can't support a coin above this threshold, reduce max coins
- Higher tiers get broader market coverage, not just more capital per coin

**Rotation Management**
- When a coin rotates out of top recommendations: no new deals initiated
- Graceful wind-down process: 2-hour force-close window, 4-hour minimum hold requirement
- Prevents excessive churn while allowing portfolio optimization

### 11.3 Multi-Coin Risk Distribution

**Risk Profile Integration**
- Each selected coin operates across all active risk profiles
- Low Risk profile parameters applied to Low Risk allocation for each coin
- Medium Risk profile parameters applied to Medium Risk allocation for each coin  
- High Risk profile parameters applied to High Risk allocation for each coin

**Example: $25K Pro Tier User with 60% Low / 30% Medium / 10% High allocation**
```
Total Capital: $25K
Reserve (10%): $2.5K  
Available: $22.5K
Max Coins: 3

Per-Coin Allocation:
Coin A (40% scanner score): $9K → $5.4K Low + $2.7K Medium + $0.9K High
Coin B (35% scanner score): $7.9K → $4.7K Low + $2.4K Medium + $0.8K High  
Coin C (25% scanner score): $5.6K → $3.4K Low + $1.7K Medium + $0.5K High
```

### 11.4 Implementation Benefits

**Diversification Without Dilution**
- Higher capital tiers achieve broader market exposure
- Each coin maintains meaningful allocation sizes
- Risk profiles apply consistently across all coins

**Scalable Architecture**
- Single framework scales from $5K starter accounts to $100K+ whale accounts
- Maintains risk management principles at all capital levels
- Natural upgrade path encourages tier progression

---

## Appendix A: Technical Implementation Notes

### A.1 Database Schema Extensions

```sql
-- Risk profile definitions
CREATE TABLE risk_profiles (
    profile_id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(64),
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT true
);

-- User risk allocations
CREATE TABLE user_risk_allocations (
    user_id VARCHAR(32),
    profile_id VARCHAR(32),
    allocation_pct DECIMAL(5,2),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (profile_id) REFERENCES risk_profiles(profile_id)
);

-- Profile performance tracking
CREATE TABLE profile_performance (
    user_id VARCHAR(32),
    profile_id VARCHAR(32), 
    date DATE,
    pnl_usd DECIMAL(15,2),
    drawdown_pct DECIMAL(5,2),
    trades_count INTEGER,
    success_rate DECIMAL(5,2)
);
```

### A.2 Configuration Management

Risk profiles stored as versioned JSON configurations:

```json
{
  "profile_id": "low_risk_v1",
  "name": "Low Risk",
  "version": "1.0",
  "parameters": {
    "leverage": 1,
    "max_safety_orders": 8,
    "so_volume_mult": 2.0,
    "base_order_pct": 4.0,
    "capital_reserve_pct": 10.0,
    "tp_range": [0.6, 2.5],
    "deviation_range": [1.2, 4.0],
    "extreme_allocation": [0, 0],
    "max_directional_bias": [75, 25]
  },
  "risk_metrics": {
    "target_daily_roi": [0.3, 0.5],
    "max_drawdown_pct": 15,
    "blow_up_risk": "very_low"
  }
}
```

### A.3 Performance Monitoring

```python
# Pseudo-code for risk monitoring
class RiskMonitor:
    def check_profile_limits(self, user_id, profile_id):
        current_drawdown = self.get_current_drawdown(user_id, profile_id)
        profile_limits = self.get_profile_limits(profile_id)
        
        if current_drawdown > profile_limits.auto_reduce_threshold:
            self.trigger_automatic_downgrade(user_id, profile_id)
            self.notify_user(user_id, "profile_downgraded", profile_id)
            
    def aggregate_risk_exposure(self, user_id):
        total_leverage = 0
        for profile in self.get_active_profiles(user_id):
            allocation = self.get_allocation_pct(user_id, profile.id)
            leverage = profile.parameters.leverage
            total_leverage += (allocation / 100) * leverage
        
        return total_leverage
```

---

*End of Specification Document*

**Document Status**: Ready for Product Development  
**Next Steps**: Technical feasibility review, UI/UX design, development sprint planning  
**Estimated Implementation**: 3-4 months (Beta), 6 months (Full Release)