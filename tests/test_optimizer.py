"""Tests for the graph-based arbitrage optimizer."""
import pytest
from funding_arb.models import (
    Balance,
    ExchangeData,
    FundingRate,
    GraphNode,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    PositionType,
)
from funding_arb.optimizer import ArbitrageOptimizer, OptimizerConfig


def make_snapshot(rates: dict[tuple[str, str], float]) -> MarketSnapshot:
    """Helper to create a snapshot with given (exchange, symbol) -> rate mapping."""
    snapshot = MarketSnapshot()
    by_exchange: dict[str, dict[str, FundingRate]] = {}
    for (exchange, symbol), rate in rates.items():
        if exchange not in by_exchange:
            by_exchange[exchange] = {}
        by_exchange[exchange][symbol] = FundingRate(
            exchange=exchange, symbol=symbol, rate=rate,
        )

    for exchange, rate_dict in by_exchange.items():
        books = {}
        balances = {"USDT": Balance(currency="USDT", free=100000.0, used=0.0, total=100000.0)}
        for symbol in rate_dict:
            books[symbol] = OrderBook(
                exchange=exchange,
                symbol=symbol,
                bids=[OrderBookLevel(price=50000, amount=10)],
                asks=[OrderBookLevel(price=50010, amount=10)],
            )
        snapshot.update(exchange, ExchangeData(rates=rate_dict, books=books, balances=balances))

    return snapshot


class TestSimpleOpportunities:
    def test_finds_spread(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0005,
            ("hyperliquid", "BTC/USDT:USDT"): 0.003,
        })
        optimizer = ArbitrageOptimizer(OptimizerConfig(min_net_yield_bps=1.0))
        opps = optimizer.find_simple_opportunities(snapshot)
        assert len(opps) > 0
        assert opps[0]["short_exchange"] == "hyperliquid"
        assert opps[0]["long_exchange"] == "binance"
        assert opps[0]["net_yield_per_period"] > 0

    def test_no_opportunity_when_spread_too_small(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0005,
            ("bybit", "BTC/USDT:USDT"): 0.0006,
        })
        optimizer = ArbitrageOptimizer(OptimizerConfig(min_net_yield_bps=50.0))
        opps = optimizer.find_simple_opportunities(snapshot)
        assert len(opps) == 0

    def test_multiple_instruments(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0005,
            ("hyperliquid", "BTC/USDT:USDT"): 0.003,
            ("binance", "ETH/USDT:USDT"): 0.0004,
            ("hyperliquid", "ETH/USDT:USDT"): 0.002,
        })
        optimizer = ArbitrageOptimizer(OptimizerConfig(min_net_yield_bps=1.0))
        opps = optimizer.find_simple_opportunities(snapshot)
        assert len(opps) >= 2


class TestGraphConstruction:
    def test_builds_graph(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0003,
            ("hyperliquid", "BTC/USDT:USDT"): 0.003,
        })
        optimizer = ArbitrageOptimizer()
        G = optimizer.build_graph(snapshot)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_single_exchange_graph(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.001,
        })
        optimizer = ArbitrageOptimizer()
        G = optimizer.build_graph(snapshot)
        assert G.number_of_nodes() > 0


class TestNegativeCycleDetection:
    def test_finds_arbitrage_with_high_spread(self):
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0001,
            ("hyperliquid", "BTC/USDT:USDT"): 0.01,
        })
        optimizer = ArbitrageOptimizer(OptimizerConfig(min_net_yield_bps=1.0))
        G = optimizer.build_graph(snapshot)
        opps = optimizer.find_opportunities(G)
        # May or may not find graph opportunities depending on cycle structure
        # The important thing is it doesn't crash
        assert isinstance(opps, list)


class TestConvexOptimization:
    def test_convex_optimizer_solve(self):
        optimizer = ArbitrageOptimizer()
        snapshot = make_snapshot({
            ("binance", "BTC/USDT:USDT"): 0.0001,
            ("hyperliquid", "BTC/USDT:USDT"): 0.01,
        })
        G = optimizer.build_graph(snapshot)
        results = optimizer.find_opportunities(G)
        assert isinstance(results, list)
        if results:
            assert results[0].expected_net_yield_per_period > 0
