"""Graph-based and convex arbitrage optimizer using CVXPY and network flows."""
from __future__ import annotations

import logging
from itertools import combinations

import cvxpy as cp
import networkx as nx
import numpy as np

from .config import OptimizerConfig
from .models import (
    Cycle,
    EdgeType,
    GraphNode,
    MarketSnapshot,
    Opportunity,
    OrderSide,
    PositionType,
    TradeLeg,
)

logger = logging.getLogger(__name__)


# Default fee schedule (taker fees per exchange, can be overridden)
DEFAULT_TAKER_FEES: dict[str, float] = {
    "binance": 0.0004,
    "hyperliquid": 0.0005,
    "bybit": 0.0006,
    "okx": 0.0005,
    "dydx": 0.0005,
}

# Withdrawal fees per exchange per base asset (in USD equivalent)
DEFAULT_WITHDRAWAL_FEES: dict[str, float] = {
    "binance": 5.0,
    "hyperliquid": 2.0,
    "bybit": 5.0,
    "okx": 5.0,
}

# Estimated transfer time cost (opportunity cost as a fraction)
DEFAULT_TIME_COST: float = 0.0001  # 1 bps for transfer delay


class ArbitrageOptimizer:
    """Builds weighted directed graph and finds optimal arbitrage distribution via convex optimization."""

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()
        self.taker_fees = dict(DEFAULT_TAKER_FEES)
        self.withdrawal_fees = dict(DEFAULT_WITHDRAWAL_FEES)
        self.time_cost = DEFAULT_TIME_COST
        # Slippage penalty coefficient (higher means more conservative)
        self.slippage_lambda = 0.5

    def build_graph(self, snapshot: MarketSnapshot) -> nx.DiGraph:
        """Build weighted directed graph from market snapshot."""
        G = nx.DiGraph()

        for instrument in snapshot.instruments:
            self._add_instrument_edges(G, snapshot, instrument)
            self._add_transfer_edges(G, snapshot, instrument)

        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def _add_instrument_edges(
        self, G: nx.DiGraph, snapshot: MarketSnapshot, instrument: str
    ) -> None:
        """Add funding, spot buy/sell edges for an instrument on each exchange."""
        for exchange in snapshot.exchanges:
            data = snapshot.get(exchange, instrument, max_pos=self.config.max_position_usd)
            if data is None or data.is_stale:
                continue

            taker_fee = self.taker_fees.get(exchange, 0.0006)
            depth = data.book_depth_usd if data.book_depth_usd > 0 else 1000.0

            # Funding edge: collateral -> short perp (earn funding)
            collateral = GraphNode(exchange, instrument, PositionType.COLLATERAL)
            short_perp = GraphNode(exchange, instrument, PositionType.SHORT_PERP)
            G.add_edge(
                collateral, short_perp,
                weight=-data.funding_rate + taker_fee,
                net_yield=-data.funding_rate - taker_fee, # used for convex
                capacity=min(data.max_position, data.available_margin),
                depth=depth,
                edge_type=EdgeType.FUNDING_SHORT,
            )

            # Reverse: collateral -> long perp (pay funding)
            long_perp = GraphNode(exchange, instrument, PositionType.LONG_PERP)
            G.add_edge(
                collateral, long_perp,
                weight=data.funding_rate + taker_fee,
                net_yield=data.funding_rate - taker_fee,
                capacity=min(data.max_position, data.available_margin),
                depth=depth,
                edge_type=EdgeType.FUNDING_LONG,
            )

            # Spot purchase: cash -> spot long
            cash = GraphNode(exchange, instrument, PositionType.CASH)
            spot = GraphNode(exchange, instrument, PositionType.SPOT_LONG)
            G.add_edge(
                cash, spot,
                weight=data.spread + taker_fee,
                net_yield=-(data.spread + taker_fee),
                capacity=data.book_depth_usd,
                depth=depth,
                edge_type=EdgeType.SPOT_BUY,
            )

            # Spot sale: spot long -> cash
            G.add_edge(
                spot, cash,
                weight=data.spread + taker_fee,
                net_yield=-(data.spread + taker_fee),
                capacity=data.book_depth_usd,
                depth=depth,
                edge_type=EdgeType.SPOT_SELL,
            )

            # Spot -> collateral (use spot as margin)
            G.add_edge(
                spot, collateral,
                weight=0.0,
                net_yield=0.0,
                capacity=data.max_position,
                depth=depth * 10, # high liquidity
                edge_type=EdgeType.SPOT_BUY,
            )

            # Short perp -> collateral (closing returns to collateral state)
            G.add_edge(
                short_perp, collateral,
                weight=taker_fee,
                net_yield=-taker_fee,
                capacity=data.max_position,
                depth=depth,
                edge_type=EdgeType.SPOT_SELL,
            )

            # Long perp -> collateral (closing)
            G.add_edge(
                long_perp, collateral,
                weight=taker_fee,
                net_yield=-taker_fee,
                capacity=data.max_position,
                depth=depth,
                edge_type=EdgeType.SPOT_SELL,
            )

    def _add_transfer_edges(
        self, G: nx.DiGraph, snapshot: MarketSnapshot, instrument: str
    ) -> None:
        """Add cross-exchange transfer edges."""
        exchanges = snapshot.exchanges
        for ex_a, ex_b in combinations(exchanges, 2):
            fee_a = self.withdrawal_fees.get(ex_a, 10.0)
            fee_b = self.withdrawal_fees.get(ex_b, 10.0)

            # Normalize fee to fractional cost (assume $10k reference notional)
            ref_notional = 10_000.0
            cost_a = (fee_a / ref_notional) + self.time_cost
            cost_b = (fee_b / ref_notional) + self.time_cost

            # A -> B
            spot_a = GraphNode(ex_a, instrument, PositionType.SPOT_LONG)
            collateral_b = GraphNode(ex_b, instrument, PositionType.COLLATERAL)
            G.add_edge(
                spot_a, collateral_b,
                weight=cost_a,
                net_yield=-cost_a,
                capacity=ref_notional,
                depth=ref_notional * 10,
                edge_type=EdgeType.TRANSFER,
            )

            # B -> A
            spot_b = GraphNode(ex_b, instrument, PositionType.SPOT_LONG)
            collateral_a = GraphNode(ex_a, instrument, PositionType.COLLATERAL)
            G.add_edge(
                spot_b, collateral_a,
                weight=cost_b,
                net_yield=-cost_b,
                capacity=ref_notional,
                depth=ref_notional * 10,
                edge_type=EdgeType.TRANSFER,
            )

    def find_opportunities(self, G: nx.DiGraph) -> list[Opportunity]:
        """Solve the convex network flow problem for optimal arbitrage allocation."""
        if G.number_of_edges() == 0:
            return []

        nodes = list(G.nodes())
        edges = list(G.edges(data=True))

        n_nodes = len(nodes)
        n_edges = len(edges)

        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # c: net yield vector
        c = np.array([e[2].get("net_yield", -e[2]["weight"]) for e in edges])

        # u: capacity vector
        u = np.array([e[2].get("capacity", 10000.0) for e in edges])

        # gamma: slippage matrix (diagonal)
        gamma = np.array([
            self.slippage_lambda / max(e[2].get("depth", 1000.0), 1.0)
            for e in edges
        ])

        # A: Incidence matrix (n_nodes x n_edges)
        A = np.zeros((n_nodes, n_edges))
        for j, (u_node, v_node, _data) in enumerate(edges):
            A[node_to_idx[u_node], j] = -1  # Flow out
            A[node_to_idx[v_node], j] = 1   # Flow in

        # Optimization variables
        x = cp.Variable(n_edges)

        # Objective: max yield - quadratic slippage penalty
        objective = cp.Maximize(c @ x - cp.quad_form(x, np.diag(gamma)))

        # Constraints
        constraints = [
            A @ x == 0,      # Flow conservation (delta neutrality / circulation)
            x >= 0,          # Non-negative flow
            x <= u,          # Capacity limits
        ]

        # Solve using Clarabel
        prob = cp.Problem(objective, constraints)
        try:
            # We try to use Clarabel if available, otherwise default
            solver_args = {}
            if cp.CLARABEL in cp.installed_solvers():
                solver_args["solver"] = cp.CLARABEL

            prob.solve(**solver_args, verbose=False)
        except Exception as e:
            logger.error(f"Convex solver failed: {e}")
            return []

        if prob.status not in ["optimal", "feasible"]:
            logger.warning(f"Solver status: {prob.status}")
            return []

        if x.value is None:
            return []

        return self._flow_to_opportunities(G, edges, x.value)

    def _flow_to_opportunities(self, G: nx.DiGraph, edges: list[tuple], x_val: np.ndarray) -> list[Opportunity]:
        """Convert the optimal flow vector into executable cycles via decomposition."""
        opportunities = []

        # 1. Collect all significant flows
        active_flows = []
        for i, (u, v, data) in enumerate(edges):
            flow = x_val[i]
            if flow > 10.0:
                active_flows.append({"u": u, "v": v, "data": data, "flow": flow})

        # 2. Heuristic: Group flows by instrument to find offsetting legs
        # This is a simplified version of cycle decomposition.
        by_instrument: dict[str, list[dict]] = {}
        for f in active_flows:
            inst = f["u"].instrument
            if inst not in by_instrument:
                by_instrument[inst] = []
            by_instrument[inst].append(f)

        for inst, flows in by_instrument.items():
            shorts = [f for f in flows if f["data"]["edge_type"] == EdgeType.FUNDING_SHORT]
            longs = [f for f in flows if f["data"]["edge_type"] == EdgeType.FUNDING_LONG]

            # Match shorts with longs for cross-exchange funding arb
            for s in shorts:
                for long_flow in longs:
                    trade_flow = min(s["flow"], long_flow["flow"])
                    if trade_flow < 10.0:
                        continue

                    net_rate = s["data"]["net_yield"] + long_flow["data"]["net_yield"]
                    if net_rate <= 0:
                        continue

                    # Create legs for the executor
                    leg_a = TradeLeg(
                        exchange=s["u"].exchange,
                        symbol=inst,
                        side=OrderSide.SELL, # Short perp
                        aggressive_price=0.0
                    )
                    leg_b = TradeLeg(
                        exchange=long_flow["u"].exchange,
                        symbol=inst,
                        side=OrderSide.BUY, # Long perp
                        aggressive_price=0.0
                    )

                    opp = Opportunity(
                        cycle=Cycle(nodes=[s["u"], s["v"], long_flow["u"], long_flow["v"]], total_weight=-net_rate),
                        expected_net_yield_per_period=net_rate,
                        yield_variance=1e-6,
                        capital_required=trade_flow,
                        risk_adjusted_yield=net_rate * (trade_flow ** 0.5),
                        exchange=s["u"].exchange,
                        net_rate=net_rate,
                        leg_a=leg_a,
                        leg_b=leg_b
                    )
                    opportunities.append(opp)

                    # Deduct used flow
                    s["flow"] -= trade_flow
                    long_flow["flow"] -= trade_flow
        opportunities.sort(key=lambda o: o.risk_adjusted_yield, reverse=True)
        return opportunities[:self.config.max_cycles_to_evaluate]

    def find_simple_opportunities(self, snapshot: MarketSnapshot) -> list[dict]:
        """Find simple cross-exchange funding rate differentials without full graph."""
        opportunities = []

        for instrument in snapshot.instruments:
            rates: list[tuple[str, float]] = []
            for exchange in snapshot.exchanges:
                data = snapshot.get(exchange, instrument)
                if data and not data.is_stale:
                    rates.append((exchange, data.funding_rate))

            if len(rates) < 2:
                continue

            rates.sort(key=lambda x: x[1], reverse=True)

            for i in range(len(rates)):
                for j in range(i + 1, len(rates)):
                    high_ex, high_rate = rates[i]
                    low_ex, low_rate = rates[j]
                    spread = high_rate - low_rate

                    high_fee = self.taker_fees.get(high_ex, 0.0006)
                    low_fee = self.taker_fees.get(low_ex, 0.0006)
                    total_fees = high_fee + low_fee

                    net_yield = spread - total_fees

                    if net_yield * 10_000 >= self.config.min_net_yield_bps:
                        opportunities.append({
                            "instrument": instrument,
                            "short_exchange": high_ex,
                            "long_exchange": low_ex,
                            "short_rate": high_rate,
                            "long_rate": low_rate,
                            "spread": spread,
                            "total_fees": total_fees,
                            "net_yield_per_period": net_yield,
                            "annualized_yield": net_yield * 3 * 365,
                        })

        opportunities.sort(key=lambda o: o["net_yield_per_period"], reverse=True)
        return opportunities
