# Funding Rate Arbitrage System — Master Plan

## 1. Executive Summary

An automated, delta-neutral trading system that extracts yield from funding rate differentials across cryptocurrency perpetual futures venues. The system continuously scans funding rates across exchanges, models the opportunity space as a weighted directed graph, identifies optimal arbitrage cycles via negative-cycle detection, executes hedged positions atomically, and manages risk through formally verified invariants — all with minimal human intervention.

**Capital allocation:** $10k–$50k initial, scaling to $200k+
**Target return:** 15–30% APR net of all costs (conservative: 10–15%)
**Infrastructure:** EC2 (or equivalent) running 24/7, ~$30–50/month
**Edge decay assumption:** Strategies are re-evaluated quarterly; alpha half-life estimated at 6–12 months before requiring adaptation

---

## 2. Core Mechanics

### 2.1 Funding Rate Primer

Perpetual futures contracts have no expiry date. To keep the futures price anchored to the spot price, exchanges impose a **funding rate** — a periodic payment (typically every 8 hours) transferred between longs and shorts.

- **Positive funding rate:** Longs pay shorts → the market is net-long (bullish bias)
- **Negative funding rate:** Shorts pay longs → the market is net-short (bearish bias)

### 2.2 The Core Trade

The simplest form: buy spot ETH and simultaneously short ETH perpetual futures on the same exchange.

```
Position:  +1 ETH (spot)  /  -1 ETH (perp short)
Net delta:  0  (price-neutral)
Income:     funding rate × notional, every 8 hours (when rate is positive)
```

Price movements cancel out. The only P&L driver is the funding rate itself. Historically, funding rates for major assets (BTC, ETH) are **positive ~70–80% of the time**, creating a persistent yield.

### 2.3 Cross-Venue Arbitrage

Funding rates diverge significantly across exchanges. If Hyperliquid charges +0.03%/8h and Binance charges +0.005%/8h, a more sophisticated trade emerges:

```
Short ETH-PERP on Hyperliquid  (earn +0.030%/8h)
Long  ETH-PERP on Binance      (pay  -0.005%/8h)
────────────────────────────────────────────────
Net funding income:              +0.025%/8h
Annualized (naive):              ~27% APR gross
After fees, slippage, transfer:  ~15–22% APR net
```

This is pure funding rate arbitrage — no spot leg required, though it introduces **cross-exchange counterparty risk** and **margin fragmentation** (capital must be posted on both venues).

### 2.4 Why This Edge Exists

Funding rate differentials persist for structural reasons that are slow to arbitrage away:

- **Venue fragmentation:** Capital cannot move freely between CEXs and DEXs. Bridging, withdrawal delays, and KYC barriers create friction.
- **Retail bias:** Onchain perpetual traders (Hyperliquid, dYdX) skew aggressively long, pushing funding rates higher than institutional venues.
- **Capital inefficiency:** Most participants don't optimize across >2 venues simultaneously.
- **Latency tolerance:** Unlike HFT arb, funding settles every 8 hours — the window is wide, but most traders lack the infrastructure to monitor it systematically.

---

## 3. Research Domains

### 3.1 Graph-Theoretic Arbitrage Detection

The entire opportunity space is modeled as a weighted directed graph, enabling systematic discovery of the most profitable capital cycles.

#### Graph Construction

```
Nodes:  Each (exchange, instrument, position_type) tuple
        Examples:
          (Binance,     ETH, SPOT_LONG)
          (Binance,     ETH, SHORT_PERP)
          (Hyperliquid, ETH, SHORT_PERP)
          (Aave,        ETH, LENDING)
          (Lido,        ETH, STAKING)
          (Binance,     USDC, CASH)

Edges:  Possible transitions between states, weighted by cost/income per period
        Negative weight = profitable (income exceeds cost)
        Positive weight = costly
```

#### Edge Types and Weights

| Edge Type | From → To | Weight Formula |
|---|---|---|
| Funding (short) | COLLATERAL → SHORT_PERP | −funding_rate + taker_fee |
| Funding (long) | COLLATERAL → LONG_PERP | +funding_rate + taker_fee |
| Spot purchase | CASH → SPOT_LONG | +spread + taker_fee |
| Spot sale | SPOT_LONG → CASH | +spread + taker_fee |
| Cross-exchange transfer | (Ex_A, SPOT) → (Ex_B, COLLATERAL) | +withdrawal_fee + time_cost |
| Lending deposit | SPOT_LONG → LENDING | −lending_apy + protocol_fee |
| Staking | SPOT_LONG → STAKING | −staking_yield + unbonding_cost |
| Borrowing | LENDING → CASH | +borrow_rate |

#### Negative Cycle Detection

An arbitrage opportunity exists **if and only if** the graph contains a **negative-weight cycle** — a closed path where the sum of edge weights is negative (total income exceeds total cost).

**Bellman-Ford** detects these in O(V·E). However, standard Bellman-Ford finds only one negative cycle. For ranking all opportunities, we use an extended approach:

1. Run Bellman-Ford from every node, collecting all negative cycles
2. Deduplicate equivalent cycles (rotation-invariant comparison)
3. For each cycle, compute: net yield per period, capital required, Sharpe-like ratio (yield / variance of constituent rates)
4. Rank by risk-adjusted yield and filter by capacity constraints

#### Capacity-Constrained Optimization

Real markets have finite liquidity. Each edge has a capacity (max position size, available order book depth, lending pool utilization). This transforms the problem from pure negative-cycle detection into a **minimum-cost maximum-flow** problem:

- **Objective:** Maximize total yield across all active cycles, subject to per-edge capacity and per-node capital constraints.
- **Algorithm:** Successive shortest-path algorithm on the residual graph, or LP relaxation for the multi-commodity variant (multiple instruments competing for the same capital).

This is substantially more powerful than greedy cycle selection — it finds the globally optimal capital allocation across all venues and instruments simultaneously.

#### Dynamic Graph Maintenance

The graph is not static. Edge weights change every time funding rates update, order book depth shifts, or lending rates adjust. The system maintains the graph incrementally:

- On each scan tick (every 30–60s), only update edges whose underlying data has changed.
- Re-run negative cycle detection only when the topology or weight of any edge in an active cycle changes by more than a threshold (e.g., 10% relative change).
- Maintain a priority queue of cycles sorted by staleness.

### 3.2 Time Series Analysis — Funding Rate Prediction

Funding rates exhibit exploitable temporal structure. Predicting their trajectory 1–3 periods ahead informs entry/exit timing and position sizing.

#### Observable Correlates

- **Open interest (OI):** Rising OI with rising price → leveraged longs entering → funding rate likely to increase.
- **Long/short ratio:** Direct measure of directional skew. Accessible via exchange APIs.
- **Liquidation cascades:** Large liquidation events reset positioning, often causing funding to mean-revert sharply.
- **Price momentum:** Rapid price appreciation → retail FOMO → elevated funding. Mean-reversion after sharp moves.
- **Time-of-day/week effects:** Funding often spikes around US market hours and dips during Asian session weekends.
- **Cross-asset correlation:** BTC funding often leads altcoin funding by 1–2 periods.

#### Regime Detection

Rather than point-predicting each funding rate, the system classifies the market into regimes:

| Regime | Characteristics | Strategy |
|---|---|---|
| **High-positive** | Funding >0.03%, strong bullish sentiment | Aggressive shorting of perps, full yield harvesting |
| **Low-positive** | Funding 0.005–0.03%, neutral sentiment | Standard delta-neutral, yield-stacking for margin |
| **Near-zero** | Funding ±0.005%, indecisive | Reduce position sizes, tighten spread thresholds |
| **Negative** | Funding < −0.005%, bearish sentiment | Reverse the trade (long perp + short spot/borrow) or exit |

A Hidden Markov Model (HMM) with 4 states, fitted on rolling 90-day windows, provides the regime classification. Transition probabilities directly inform expected position duration and optimal entry timing.

#### Prediction Models

**Tier 1 — Baseline:** ARIMA(p,d,q) on 8-hour funding rates per instrument per venue. Provides a simple mean-reversion signal with confidence intervals.

**Tier 2 — Volatility-aware:** GARCH(1,1) on funding rate residuals. Captures heteroskedasticity (funding volatility clusters after market events). Used for dynamic position sizing — reduce size when predicted funding volatility is high.

**Tier 3 — Feature-rich:** Gradient-boosted trees (XGBoost/LightGBM) with engineered features:
- Rolling OI z-score (8h, 24h, 7d)
- Funding rate momentum (Δrate over last 3 periods)
- Order book imbalance (bid depth / ask depth ratio)
- Liquidation volume (rolling 24h)
- BTC dominance change
- Fear & Greed index delta
- On-chain: exchange net flows (via Glassnode or similar)

Target: Probability that funding remains above break-even threshold (covering all fees) for the next N periods.

#### Survival Analysis for Position Duration

Rather than predicting funding at a point in time, model **how long a position remains profitable** using survival analysis (Cox proportional hazards or Kaplan-Meier). This directly answers the question: "Given current conditions, what is the expected holding period before this trade turns unprofitable?"

This drives:
- **Entry decisions:** Only enter positions with expected survival > 3 funding periods (24h).
- **Kelly-optimal sizing:** Longer expected survival → larger position.
- **Exit scheduling:** Pre-schedule exits at the median survival time unless conditions improve.

### 3.3 Yield Stacking — Capital Efficiency Optimization

The spot leg of a delta-neutral position is idle capital. Yield stacking puts it to work.

#### Tier 1: Lending (Low Risk)

```
Spot ETH → Deposit in Aave/Compound (lending yield ~2–5% APR)
         + Short ETH-PERP (funding income ~10–20% APR)
─────────────────────────────────────────────────────────────
Total yield: 12–25% APR
Additional risk: Smart contract risk (protocol exploit)
```

#### Tier 2: Liquid Staking + Lending (Medium Risk)

```
Spot ETH → Convert to stETH via Lido (~3% staking APR)
         → Deposit stETH in Aave (earn supply APR + staking)
         → Borrow USDC against stETH (pay borrow rate ~3–6%)
         → Use USDC as collateral for short ETH-PERP
─────────────────────────────────────────────────────────────
Total yield: staking (3%) + funding (10–20%) − borrow cost (3–6%) = 7–17% APR
Additional risk: stETH depeg, liquidation risk, smart contract risk (compounded across protocols)
```

#### Tier 3: Recursive Leverage (High Risk — Research Only)

```
Spot ETH → stETH → Deposit in Aave → Borrow ETH → Convert to stETH → Redeposit
Repeat N times (practical limit: 3–4 loops before LTV ceiling)
Final staking yield ≈ base_yield × leverage_multiplier
         + Short ETH-PERP on top
─────────────────────────────────────────────────────────────
Total yield: Potentially 30–50% APR
Additional risk: Extremely sensitive to stETH depeg, cascading liquidation, gas costs
```

Tier 3 is modeled and backtested but only deployed with explicit manual approval and hard position size caps.

#### Yield Stack Selection as an Optimization Problem

At any given time, the system evaluates all available yield stacks and selects the optimal combination via linear programming:

- **Objective:** Maximize expected yield across all yield stacks.
- **Constraints:** Total capital = portfolio value. Per-protocol exposure ≤ 30% of total. Per-chain exposure ≤ 50%. Leverage ratio ≤ max_allowed. Smart contract risk budget ≤ threshold (scored by protocol age, TVL, audit count).

### 3.4 Risk Management — Formally Verified Invariants

The risk engine is the system's immune system. It operates at two levels: a real-time soft layer (fast checks in Python) and a formal verification layer (Z3-based proofs that the soft layer is correct).

#### Invariants

The system must maintain the following properties at all times, proven exhaustively over the state space:

| Invariant | Formal Expression | Purpose |
|---|---|---|
| **Delta neutrality** | \|Σ position_delta\| / portfolio_value ≤ ε | No directional exposure |
| **Position concentration** | ∀ position: notional ≤ α × total_equity | No single trade can destroy the portfolio |
| **Exchange concentration** | ∀ exchange: equity_at_exchange ≤ β × total_equity | Counterparty risk cap |
| **Collateral safety** | ∀ exchange: equity / margin_used ≥ γ | Never approach liquidation |
| **Drawdown circuit breaker** | drawdown_from_peak ≤ δ | Emergency exit |
| **Gross leverage** | Σ \|notional\| / total_equity ≤ λ | Leverage cap |

Default parameters: ε = 0.02, α = 0.20, β = 0.30, γ = 2.0, δ = 0.05, λ = 3.0.

#### Z3 Formal Verification

Z3 is used not for live trading decisions, but to **prove that the risk-checking code correctly enforces the invariants under all possible inputs:**

```python
from z3 import *

# Prove: the position sizing function never exceeds MAX_POSITION_PCT
equity = Real('equity')
rate = Real('rate')
spread = Real('spread')

# Model the sizing function
size = If(rate > spread * 2,
          equity * 0.20,   # max size at best rate
          equity * (rate / (spread * 10)))  # scaled size

# Prove the invariant
s = Solver()
s.add(equity > 0)
s.add(rate > 0)
s.add(spread > 0)
s.add(size > equity * 0.20)  # try to violate

assert s.check() == unsat  # PROVEN: no input can violate the cap
```

This catches logic bugs that unit tests miss — particularly around edge cases in position sizing, rebalancing, and emergency exit code.

#### Dynamic Risk Adjustment

Beyond static invariants, the risk engine adjusts parameters based on market conditions:

- **Volatility regime:** When realized volatility (30-min EWMA) exceeds the 90th percentile of its 30-day distribution, automatically tighten ε (delta tolerance), increase γ (collateral ratio), and reduce α (position concentration limit).
- **Correlation breakdown:** Monitor the correlation between spot and perp prices. If the rolling 1-hour correlation drops below 0.95, trigger alert and tighten positions. This signals potential market microstructure stress (exchange lag, API issues, liquidity crises).
- **Funding rate volatility:** Use GARCH-predicted funding volatility to adjust position sizes. High-volatility funding environments warrant smaller positions even if the expected rate is attractive.

#### Value at Risk (VaR) and Expected Shortfall

While the portfolio is delta-neutral in theory, real-world execution introduces residual risks. The system computes:

- **Parametric VaR** (1-day, 99%) on the residual delta exposure.
- **Historical simulation VaR** using the last 180 days of funding rate and basis data.
- **Expected Shortfall (CVaR)** at 99% — the expected loss in the worst 1% of scenarios.

These feed into position sizing via a modified Kelly criterion: `f* = (edge / variance) × kelly_fraction`, where `kelly_fraction` starts at 0.25 (quarter-Kelly) for conservatism.

---

## 4. Technology Stack

### 4.1 Core Components

| Component | Technology | Rationale |
|---|---|---|
| Core engine | Python 3.12+ | Best exchange API support (ccxt), fastest iteration |
| Async I/O | asyncio + aiohttp | Sub-second polling; latency requirements are seconds, not microseconds |
| Graph computation | NetworkX + NumPy + SciPy | Bellman-Ford, min-cost flow, sparse matrix operations |
| ML / prediction | scikit-learn, XGBoost, statsmodels | ARIMA, GARCH, gradient boosting, survival analysis |
| Data storage | SQLite (operational state) + Parquet (time series) | SQLite for ACID state; Parquet for columnar analytics |
| Risk engine | Python + Z3 (formal proofs) | Invariant verification at build time; fast Python checks at runtime |
| Deployment | EC2 t3.medium (or Graviton) | ~$25–40/month, sufficient for 10–30s polling intervals |
| Monitoring | Telegram bot + Prometheus + Grafana | Real-time alerts + historical dashboards |
| Configuration | TOML files + environment variables | Human-readable config, secrets in env vars |
| Testing | pytest + hypothesis (property-based) | Hypothesis generates edge cases the developer wouldn't think of |

### 4.2 Exchange Integrations

**ccxt** (CryptoCurrency eXchange Trading Library) provides a unified API across 100+ exchanges. Primary venues:

| Exchange | Type | Why |
|---|---|---|
| **Binance** | CEX | Deepest liquidity, lowest fees, most reliable API, broadest instrument coverage |
| **Hyperliquid** | DEX (L1) | Frequently elevated funding rates due to retail long bias; fully onchain, no KYC |
| **Bybit** | CEX | Strong perp market, competitive fees, good API stability |
| **OKX** | CEX | Large derivatives market, portfolio margin mode |
| **dYdX v4** | DEX (Cosmos) | Different funding mechanism (hourly), worth monitoring for divergence |

### 4.3 Data Architecture

#### Real-Time Pipeline (every 30–60 seconds)

- Funding rates: Per exchange, per instrument (via ccxt unified API)
- Order book snapshots: Top 5 levels of bid/ask (for spread and slippage estimation)
- Open interest: Aggregated and per-exchange
- Account state: Balances, positions, margin usage, unrealized P&L per exchange

#### Periodic Pipeline (every 8 hours, at funding settlement)

- Funding payments: Actual funding received/paid vs. expected (track execution quality)
- P&L attribution: Decompose returns into funding income, basis change, fee drag, slippage cost
- Delta rebalancing: If net delta has drifted beyond tolerance, execute corrective trades
- Yield stack rebalancing: Check lending rates, staking yields; rotate if better options available

#### Historical Pipeline (daily)

- Funding rate history: From Coinglass API and direct exchange endpoints (redundancy)
- Trade log: Every order placed, filled, cancelled — with timestamps, prices, fees
- Risk metrics: Daily VaR, CVaR, max drawdown, Sharpe ratio, funding rate Sharpe

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│                  (async main loop, 60s tick)                      │
│                                                                  │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐              │
│  │  SCANNER  │───>│  OPTIMIZER  │───>│  EXECUTOR  │              │
│  │           │    │            │    │            │              │
│  │ Fetch all │    │ Build      │    │ Atomic     │              │
│  │ rates,    │    │ weighted   │    │ multi-leg  │              │
│  │ books,    │    │ digraph    │    │ order      │              │
│  │ OI, bal-  │    │            │    │ execution  │              │
│  │ ances     │    │ Bellman-   │    │ via ccxt   │              │
│  │ from all  │    │ Ford +     │    │            │              │
│  │ venues    │    │ min-cost   │    │ Leg sync + │              │
│  │           │    │ flow       │    │ emergency  │              │
│  │           │    │            │    │ unwind     │              │
│  └───────────┘    └────────────┘    └────────────┘              │
│        │                │                 │                      │
│        v                v                 v                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                   RISK MANAGER                        │       │
│  │                                                       │       │
│  │  • Delta neutrality enforcement                       │       │
│  │  • Per-exchange collateral ratio monitor               │       │
│  │  • Position concentration limits                      │       │
│  │  • Gross leverage cap                                 │       │
│  │  • Drawdown circuit breaker                           │       │
│  │  • Correlation breakdown detector                     │       │
│  │  • VaR / CVaR computation                             │       │
│  │  • Dynamic parameter adjustment (volatility-aware)    │       │
│  └──────────────────────────────────────────────────────┘       │
│        │                                                         │
│        v                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ REBALANCER │  │  P&L       │  │  ALERTER   │                │
│  │            │  │  TRACKER   │  │            │                │
│  │ Every 8h:  │  │            │  │ Telegram   │                │
│  │ delta      │  │ Per-trade  │  │ alerts on: │                │
│  │ correction │  │ P&L attri- │  │ new pos,   │                │
│  │ + yield    │  │ bution,    │  │ risk event,│                │
│  │ stack      │  │ fee drag,  │  │ P&L, exit  │                │
│  │ rotation   │  │ daily      │  │            │                │
│  │            │  │ summary    │  │            │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└──────────────────────────────────────────────────────────────────┘

        │                 │                 │
        v                 v                 v
  ┌───────────┐   ┌──────────────┐   ┌───────────┐
  │ EXCHANGE  │   │   DATABASE   │   │ TELEGRAM  │
  │   APIs    │   │              │   │   BOT     │
  │           │   │ state.db     │   │           │
  │ Binance   │   │ trades.db    │   │ /status   │
  │ Hyper-    │   │ funding.db   │   │ /pnl      │
  │ liquid    │   │ rates.pq     │   │ /close    │
  │ Bybit     │   │ risk.pq      │   │ /pause    │
  │ OKX       │   │              │   │ /config   │
  │ dYdX      │   │              │   │           │
  └───────────┘   └──────────────┘   └───────────┘
```

### 5.1 Scanner

```python
class FundingScanner:
    """
    Polls all configured exchanges for funding rates, order book snapshots,
    open interest, and account state. Returns a unified snapshot.
    """
    exchanges: list[Exchange]  # [Binance, Hyperliquid, Bybit, OKX, dYdX]
    instruments: list[str]     # Top 20 perps by open interest

    async def scan(self) -> MarketSnapshot:
        """
        Concurrent fetch across all exchanges.
        Returns a unified snapshot with staleness metadata.
        """
        tasks = [self._fetch_exchange(ex) for ex in self.exchanges]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshot = MarketSnapshot(timestamp=datetime.utcnow())
        for ex, result in zip(self.exchanges, results):
            if isinstance(result, Exception):
                snapshot.mark_stale(ex, error=result)
                logger.warning(f"Scan failed for {ex.name}: {result}")
            else:
                snapshot.update(ex, result)

        return snapshot

    async def _fetch_exchange(self, ex: Exchange) -> ExchangeData:
        """Fetch rates, book, OI, and balances for one exchange."""
        rates, books, oi, balances = await asyncio.gather(
            ex.fetch_funding_rates(self.instruments),
            ex.fetch_order_books(self.instruments, depth=5),
            ex.fetch_open_interest(self.instruments),
            ex.fetch_balances(),
        )
        return ExchangeData(
            rates=rates, books=books,
            open_interest=oi, balances=balances,
            fetched_at=datetime.utcnow()
        )
```

### 5.2 Optimizer (Graph Model)

```python
class ArbitrageOptimizer:
    """
    Builds and maintains the weighted directed graph.
    Finds optimal capital allocation across all venues and instruments.
    """

    def build_graph(self, snapshot: MarketSnapshot) -> nx.DiGraph:
        G = nx.DiGraph()

        for instrument in snapshot.instruments:
            for exchange in snapshot.exchanges:
                data = snapshot.get(exchange, instrument)
                if data is None or data.is_stale:
                    continue

                # Funding edge: collateral → short perp (earn funding)
                G.add_edge(
                    (exchange, instrument, 'COLLATERAL'),
                    (exchange, instrument, 'SHORT_PERP'),
                    weight=-data.funding_rate + data.taker_fee,
                    capacity=min(data.max_position, data.available_margin),
                    edge_type='funding_short'
                )

                # Reverse funding edge: collateral → long perp (pay funding)
                G.add_edge(
                    (exchange, instrument, 'COLLATERAL'),
                    (exchange, instrument, 'LONG_PERP'),
                    weight=+data.funding_rate + data.taker_fee,
                    capacity=min(data.max_position, data.available_margin),
                    edge_type='funding_long'
                )

                # Spot purchase
                G.add_edge(
                    (exchange, instrument, 'CASH'),
                    (exchange, instrument, 'SPOT_LONG'),
                    weight=data.spread + data.taker_fee,
                    capacity=data.book_depth_usd,
                    edge_type='spot_buy'
                )

            # Cross-exchange transfer edges
            for ex_a, ex_b in combinations(snapshot.exchanges, 2):
                fee = self.withdrawal_fees.get((ex_a, instrument), float('inf'))
                G.add_edge(
                    (ex_a, instrument, 'SPOT_LONG'),
                    (ex_b, instrument, 'COLLATERAL'),
                    weight=fee + self.time_cost(ex_a, ex_b),
                    capacity=self.transfer_limits.get((ex_a, ex_b), float('inf')),
                    edge_type='transfer'
                )

            # Lending edges (Aave, Compound)
            for protocol in self.lending_protocols:
                lending_data = snapshot.get_lending(protocol, instrument)
                if lending_data:
                    G.add_edge(
                        (protocol, instrument, 'SPOT_LONG'),
                        (protocol, instrument, 'LENDING'),
                        weight=-lending_data.supply_apy + lending_data.protocol_fee,
                        capacity=lending_data.available_liquidity,
                        edge_type='lending'
                    )

        return G

    def find_opportunities(self, G: nx.DiGraph) -> list[Opportunity]:
        """
        Phase 1: Bellman-Ford negative cycle detection (all cycles).
        Phase 2: Capacity-constrained optimization via min-cost flow.
        Returns ranked opportunities with allocation sizes.
        """
        # Phase 1: Enumerate negative cycles
        raw_cycles = self._find_all_negative_cycles(G)

        # Phase 2: Solve capacity-constrained allocation
        if raw_cycles:
            allocations = self._min_cost_flow_allocation(G, raw_cycles)
            return sorted(allocations, key=lambda a: a.risk_adjusted_yield, reverse=True)

        return []

    def _find_all_negative_cycles(self, G: nx.DiGraph) -> list[Cycle]:
        """
        Extended Bellman-Ford: run from every node, collect and
        deduplicate all negative cycles (rotation-invariant).
        """
        cycles = []
        for source in G.nodes:
            cycle = self._bellman_ford_negative_cycle(G, source)
            if cycle and not self._is_duplicate(cycle, cycles):
                cycles.append(cycle)
        return cycles

    def _min_cost_flow_allocation(
        self, G: nx.DiGraph, cycles: list[Cycle]
    ) -> list[Opportunity]:
        """
        Given candidate cycles and edge capacities,
        solve the min-cost max-flow problem to find
        globally optimal capital allocation.
        """
        # Build flow network from candidate cycles
        # Solve using scipy.optimize.linprog or ortools
        ...
```

### 5.3 Risk Manager

```python
@dataclass
class RiskParameters:
    max_delta_pct: float = 0.02          # 2% net delta tolerance
    max_position_pct: float = 0.20       # 20% of equity per position
    max_exchange_pct: float = 0.30       # 30% of equity per exchange
    min_collateral_ratio: float = 2.0    # 2x equity/margin
    max_drawdown: float = 0.05           # 5% drawdown → emergency exit
    max_gross_leverage: float = 3.0      # 3x total notional / equity
    correlation_floor: float = 0.95      # spot-perp correlation minimum
    kelly_fraction: float = 0.25         # quarter-Kelly for sizing

class RiskManager:
    def __init__(self, params: RiskParameters):
        self.params = params
        self.vol_estimator = EWMAVolatility(halflife_minutes=30)
        self.var_engine = HistoricalVaR(lookback_days=180, confidence=0.99)

    def check_invariants(self, portfolio: Portfolio) -> list[Violation]:
        violations = []

        # Delta neutrality
        net_delta = sum(p.delta_usd for p in portfolio.positions)
        delta_pct = abs(net_delta) / portfolio.total_equity
        if delta_pct > self.params.max_delta_pct:
            violations.append(DeltaDrift(delta_pct, net_delta))

        # Position concentration
        for pos in portfolio.positions:
            pos_pct = pos.notional_usd / portfolio.total_equity
            if pos_pct > self.params.max_position_pct:
                violations.append(PositionConcentration(pos, pos_pct))

        # Exchange concentration
        for ex_name, equity in portfolio.equity_by_exchange.items():
            ex_pct = equity / portfolio.total_equity
            if ex_pct > self.params.max_exchange_pct:
                violations.append(ExchangeConcentration(ex_name, ex_pct))

        # Collateral ratios
        for ex_name, margin_state in portfolio.margin_by_exchange.items():
            ratio = margin_state.equity / max(margin_state.used, 1e-8)
            if ratio < self.params.min_collateral_ratio:
                violations.append(LowCollateral(ex_name, ratio))

        # Gross leverage
        gross_notional = sum(abs(p.notional_usd) for p in portfolio.positions)
        leverage = gross_notional / portfolio.total_equity
        if leverage > self.params.max_gross_leverage:
            violations.append(ExcessiveLeverage(leverage))

        # Drawdown
        if portfolio.drawdown_from_peak > self.params.max_drawdown:
            violations.append(EmergencyDrawdown(portfolio.drawdown_from_peak))

        return violations

    def calculate_position_size(self, opp: Opportunity, portfolio: Portfolio) -> float:
        """
        Kelly-criterion-based sizing with vol adjustment.
        """
        edge = opp.expected_net_yield_per_period
        variance = opp.yield_variance
        if variance <= 0 or edge <= 0:
            return 0.0

        # Kelly fraction
        kelly_size = (edge / variance) * self.params.kelly_fraction

        # Cap by concentration limit
        max_by_concentration = portfolio.total_equity * self.params.max_position_pct

        # Cap by exchange headroom
        max_by_exchange = (
            portfolio.total_equity * self.params.max_exchange_pct
            - portfolio.equity_at(opp.exchange)
        )

        # Cap by leverage headroom
        current_notional = sum(abs(p.notional_usd) for p in portfolio.positions)
        max_by_leverage = (
            portfolio.total_equity * self.params.max_gross_leverage
            - current_notional
        )

        return max(0, min(kelly_size, max_by_concentration, max_by_exchange, max_by_leverage))

    def adjust_for_regime(self, vol_regime: str):
        """Tighten or loosen parameters based on volatility regime."""
        if vol_regime == 'high':
            self.params.max_delta_pct *= 0.5
            self.params.min_collateral_ratio *= 1.5
            self.params.max_position_pct *= 0.6
        elif vol_regime == 'low':
            # Restore defaults (or slightly relax)
            self.params = RiskParameters()
```

### 5.4 Executor

```python
class TradeExecutor:
    """
    Handles atomic multi-leg execution with emergency unwind.
    """

    async def open_arb_position(self, opp: Opportunity, size: float):
        """
        Execute both legs as close to simultaneously as possible.
        If one leg fails, immediately unwind the other.
        """
        # Pre-flight checks
        if not self.risk_manager.check_pre_trade(opp, size):
            logger.warning(f"Pre-trade risk check failed for {opp}")
            return None

        # Execute both legs concurrently
        try:
            results = await asyncio.gather(
                self._execute_leg(opp.leg_a, size),
                self._execute_leg(opp.leg_b, size),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None

        leg_a_result, leg_b_result = results

        # Check for partial fills
        if isinstance(leg_a_result, Exception) or isinstance(leg_b_result, Exception):
            await self._emergency_unwind(leg_a_result, leg_b_result)
            return None

        if not (leg_a_result.is_filled and leg_b_result.is_filled):
            await self._emergency_unwind(leg_a_result, leg_b_result)
            return None

        # Verify execution quality
        actual_spread = abs(leg_a_result.avg_price - leg_b_result.avg_price)
        expected_spread = opp.expected_spread
        if actual_spread > expected_spread * 1.5:
            logger.warning(
                f"Slippage alert: expected {expected_spread}, got {actual_spread}"
            )

        # Register position
        position = ArbitragePosition(
            leg_a=leg_a_result,
            leg_b=leg_b_result,
            entry_funding_rate=opp.net_rate,
            entry_spread=actual_spread,
            opened_at=datetime.utcnow()
        )
        self.portfolio.add_position(position)
        self.alerter.notify_new_position(position)

        return position

    async def _execute_leg(self, leg: TradeLeg, size: float) -> OrderResult:
        """
        Execute a single leg with retry logic.
        Uses limit orders with aggressive pricing (cross the spread).
        Falls back to market order after timeout.
        """
        # Place limit order at best ask + 1 tick (for buys)
        order = await leg.exchange.create_limit_order(
            symbol=leg.symbol,
            side=leg.side,
            amount=size,
            price=leg.aggressive_price,
        )

        # Wait for fill with timeout
        filled = await self._wait_for_fill(leg.exchange, order, timeout_s=5)
        if not filled:
            await leg.exchange.cancel_order(order.id)
            # Fallback: market order
            order = await leg.exchange.create_market_order(
                symbol=leg.symbol, side=leg.side, amount=size
            )

        return await self._get_order_result(leg.exchange, order)

    async def _emergency_unwind(self, leg_a_result, leg_b_result):
        """Immediately close any filled leg to prevent naked exposure."""
        logger.critical("EMERGENCY UNWIND triggered")
        tasks = []
        if not isinstance(leg_a_result, Exception) and leg_a_result.is_filled:
            tasks.append(self._close_leg(leg_a_result))
        if not isinstance(leg_b_result, Exception) and leg_b_result.is_filled:
            tasks.append(self._close_leg(leg_b_result))
        if tasks:
            await asyncio.gather(*tasks)
        self.alerter.notify_emergency_unwind(leg_a_result, leg_b_result)
```

---

## 6. Phased Roadmap

### Phase 1: Manual Validation (Week 1)

**Objective:** Prove the thesis works with real money before writing any code.

- Set up accounts on Binance + Hyperliquid (KYC, API keys, funding)
- Manually identify funding rate divergences via Coinglass
- Execute 3–5 manual delta-neutral trades (spot + short perp)
- Carefully measure: actual funding received, actual fees paid, slippage on entry/exit, gas costs
- Compute realized net APR vs. theoretical

**Go/No-Go:** If realized net yield after all costs < 5% APR → re-evaluate the strategy or venue selection before proceeding.

### Phase 2: Data Pipeline + Backtesting (Weeks 2–3)

**Objective:** Build the data foundation and validate graph-based strategy on historical data.

- Implement `FundingScanner` with ccxt, polling all target exchanges
- Build local database: SQLite for state, Parquet for time series
- Ingest 6–12 months of historical funding rates (via Coinglass API + exchange endpoints)
- Build and backtest the graph model (Bellman-Ford cycle detection on historical snapshots)
- Backtest top 5 strategies: compute net APR, Sharpe ratio, max drawdown, number of trades

**Deliverable:** Backtest report with strategy comparison, fee sensitivity analysis, and recommended configuration.

### Phase 3: Execution Engine + Paper Trading (Weeks 4–5)

**Objective:** Build the live execution system and validate against real market data without real capital.

- Implement `TradeExecutor` with ccxt (paper trading mode: simulated fills against live order book)
- Implement `RiskManager` with all invariant checks
- Implement Telegram alerter (new positions, risk events, P&L summaries)
- Run for 1 week in paper trading mode, comparing simulated P&L to theoretical

**Deliverable:** 7-day paper trading P&L report with execution quality analysis (slippage, fill rates, timing).

### Phase 4: Live Deployment — Small Capital (Weeks 6–8)

**Objective:** Validate with real capital in production conditions.

- Deploy to EC2 (t3.medium, us-east-1 for low-latency to Binance)
- Start with $2–5k (acceptable loss if everything goes wrong)
- Run for 2–3 weeks, measuring: realized yield, fee drag, API reliability, rebalancing frequency
- Identify and fix edge cases: API downtime, rate limit handling, fee schedule changes, rebalancing under volatility

**Go/No-Go:** If realized yield is within ±30% of backtest projections → proceed to scale. If > 30% divergence → diagnose and iterate.

### Phase 5: Scale + Optimize (Weeks 9–12)

**Objective:** Increase capital and add sophistication.

- Scale to full capital allocation ($10–50k)
- Add remaining exchanges (Bybit, OKX, dYdX)
- Expand instrument coverage (BTC, SOL, ARB, and other top-20 by OI)
- Implement yield stacking (Tier 1: Aave lending on spot leg)
- Deploy prediction models (ARIMA baseline, then regime detection)
- Implement dynamic risk parameter adjustment

### Phase 6: Advanced Features (Months 4–6)

**Objective:** Maximize alpha and robustness.

- Capacity-constrained optimization (min-cost flow)
- Survival analysis for position duration
- ML-based funding rate prediction (XGBoost with full feature set)
- Tier 2 yield stacking (liquid staking + lending)
- Prometheus + Grafana monitoring dashboard
- Formal verification of risk engine with Z3
- Automated strategy rebalancing (quarterly alpha decay check)

---

## 7. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Exchange hack or insolvency** | Low | Catastrophic | Max 30% of equity per exchange; prefer self-custody where possible; monitor exchange proof-of-reserves |
| **Funding rate sign flip** | Medium | Low per event | Automated exit when rate goes negative; survival analysis for expected duration; tight monitoring |
| **API downtime during rebalance** | Medium | Medium | Redundant WebSocket + REST connections; exponential backoff; exchange-specific failover logic |
| **Smart contract exploit (DEX/DeFi)** | Low–Medium | High | Cap DeFi protocol exposure at 30%; prefer blue-chip protocols (Aave v3, Lido); monitor audit status |
| **Slippage on exit (low liquidity)** | Low–Medium | Medium | Limit orders with staggered exit; position size capped by available book depth; avoid illiquid instruments |
| **Regulatory action** | Medium | High | Avoid US-based exchanges for perps; comply with EU MiCA; maintain exit-ready posture |
| **Delta drift between legs** | Medium | Medium | 2% delta tolerance with automatic rebalancing; tight correlation monitoring; emergency circuit breaker |
| **Stale data / stalled scanner** | Low | Medium | Staleness metadata on every data point; auto-close if data age > 5 minutes; Telegram alert on scan failure |
| **Cascading liquidation (yield stacking)** | Low | High | Conservative LTV (max 50% of allowed); real-time health factor monitoring; Tier 3 requires manual approval |
| **Alpha decay** | High (long-term) | Medium | Quarterly strategy review; expand to new venues/instruments; keep overhead low to maintain profitability at lower yields |

---

## 8. Future Development Roadmap

### 8.1 Cross-Chain Funding Arbitrage

The same asset traded on different L2s (Arbitrum, Optimism, Base, zkSync) can exhibit different funding rates due to fragmented liquidity. As bridging infrastructure matures (sub-minute finality, lower costs), cross-L2 funding arbitrage becomes viable.

**Technical requirements:** Integration with canonical bridges and fast bridge protocols (Across, Stargate). Monitoring of bridge liquidity and latency. Gas cost modeling per chain.

**Expected timeline:** 3–6 months, contingent on bridge reliability improvements.

### 8.2 Options Overlay for Tail Risk Hedging

The delta-neutral strategy has bounded downside in normal conditions, but extreme events (exchange outages during flash crashes, depegs) can cause losses. A cheap options overlay provides insurance:

- **Strategy:** Buy deep OTM puts on the underlying (e.g., 20-delta puts, 30-day expiry) to cap downside from delta drift during extreme moves.
- **Cost:** Typically 1–3% APR, reducing net yield but dramatically improving the tail risk profile.
- **Implementation:** Use Deribit or onchain options protocols (Lyra, Hegic). Automate rolling before expiry.
- **Portfolio-level hedging:** Rather than hedging each position individually, buy portfolio-level puts on BTC (as a proxy for overall crypto market risk) — cheaper and more capital-efficient.

### 8.3 Dynamic Lending Protocol Rotation

Lending rates fluctuate significantly across protocols and chains. A lending optimizer continuously monitors rates across Aave (multiple deployments), Compound, Morpho, Spark, and others:

- **Objective:** Maximize supply APY on the spot hedge leg.
- **Constraints:** Gas costs for reallocation, protocol risk budget, withdrawal delays.
- **Algorithm:** Threshold-based reallocation: switch only when the rate differential exceeds gas cost amortized over expected holding period.
- **Advanced:** Use Morpho's peer-to-peer matching for higher base rates. Use Yearn-style aggregators as a fallback.

### 8.4 On-Chain Signal Integration

On-chain data provides leading indicators for funding rate movements:

- **Whale wallet tracking:** Large deposits to exchanges (via Arkham, Nansen) often precede selling pressure, which compresses funding rates.
- **Liquidation heatmaps:** Concentrations of liquidation levels (via Coinglass, Kingfisher) predict where cascades will occur, resetting funding.
- **Exchange net flows:** Net inflows to derivatives exchanges signal increased speculation (higher funding).
- **Stablecoin flows:** USDT/USDC minting → capital inflow → bullish → higher funding rates.

These signals feed into the prediction model as additional features, improving entry/exit timing.

### 8.5 Multi-Asset Portfolio Optimization

As the system scales beyond a single instrument, cross-asset correlation becomes critical:

- **Correlation matrix:** Maintain a rolling correlation matrix of funding rates across all instruments. Avoid concentrating in highly correlated positions (e.g., ETH and SOL funding rates often move together).
- **Markowitz-style allocation:** Optimize the portfolio for maximum funding Sharpe ratio subject to correlation constraints.
- **Sector diversification:** Spread across L1s (ETH, SOL, AVAX), L2 tokens (ARB, OP), DeFi tokens (AAVE, UNI), and meme coins (higher funding but higher vol).

### 8.6 MEV-Aware Execution on DEXs

On DEX venues (Hyperliquid, dYdX), order execution is subject to MEV (Maximal Extractable Value) — front-running, sandwich attacks, and other adversarial behavior:

- **Private transaction submission:** Use Flashbots Protect (on Ethereum-based DEXs) or equivalent MEV-protection mechanisms.
- **Order splitting:** Break large orders into smaller chunks submitted across multiple blocks.
- **Timing optimization:** Submit orders during low-MEV periods (higher block utilization, lower searcher activity).

### 8.7 Institutional-Grade Reporting

As capital scales, reporting requirements increase:

- **Daily P&L statement:** Broken down by strategy, instrument, venue.
- **Risk report:** VaR, CVaR, Greeks (delta, gamma of the portfolio), exposure by chain/protocol.
- **Tax reporting:** Per-trade P&L with cost basis tracking (FIFO/LIFO), exportable in formats compatible with crypto tax software (Koinly, CoinTracker).
- **Audit trail:** Every decision logged with full context (why a position was opened/closed, what the graph looked like, what the risk state was).

### 8.8 Decentralized Execution via Smart Contracts

Long-term: move execution logic onchain for trustless, non-custodial operation:

- **Smart contract vault:** Holds collateral, executes trades on onchain perp DEXs, earns funding — all without trusting a centralized server.
- **Keeper network:** Off-chain keepers (similar to Gelato or Chainlink Automation) trigger rebalancing and position management.
- **Composability:** Direct integration with lending protocols for yield stacking without bridging or transfers.
- **Challenge:** Only viable for DEX-to-DEX strategies; CEX integration still requires centralized custody.

---

## 9. Success Metrics and KPIs

| Metric | Target (Phase 4) | Target (Phase 6) |
|---|---|---|
| Net APR (after all costs) | 10–15% | 15–25% |
| Sharpe ratio (funding rate returns) | > 1.5 | > 2.0 |
| Max drawdown | < 3% | < 5% |
| Uptime | > 99% | > 99.5% |
| Average position duration | > 24h | > 48h |
| Execution slippage vs. expected | < 20% | < 10% |
| Delta drift (max observed) | < 2% | < 1% |
| Time to detect and act on opportunity | < 120s | < 60s |
