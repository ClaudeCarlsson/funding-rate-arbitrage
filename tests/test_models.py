"""Tests for core data models."""
import pytest
from funding_arb.models import (
    ArbitragePosition,
    Balance,
    Cycle,
    FundingRate,
    GraphNode,
    MarginState,
    MarketSnapshot,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    OrderResult,
    OrderSide,
    Portfolio,
    PositionType,
    ExchangeData,
)


class TestFundingRate:
    def test_annualized(self):
        rate = FundingRate(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003)
        expected = 0.0003 * 3 * 365
        assert abs(rate.annualized - expected) < 1e-10

    def test_annualized_negative(self):
        rate = FundingRate(exchange="binance", symbol="BTC/USDT:USDT", rate=-0.0001)
        assert rate.annualized < 0


class TestOrderBook:
    def test_mid_price(self):
        book = OrderBook(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            bids=[OrderBookLevel(price=50000, amount=1.0)],
            asks=[OrderBookLevel(price=50010, amount=1.0)],
        )
        assert book.mid_price == 50005.0

    def test_spread(self):
        book = OrderBook(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            bids=[OrderBookLevel(price=50000, amount=1.0)],
            asks=[OrderBookLevel(price=50010, amount=1.0)],
        )
        expected = 10 / 50005
        assert abs(book.spread - expected) < 1e-10

    def test_empty_book(self):
        book = OrderBook(exchange="binance", symbol="BTC/USDT:USDT")
        assert book.mid_price == 0.0
        assert book.spread == float("inf")

    def test_depth(self):
        book = OrderBook(
            exchange="binance",
            symbol="BTC/USDT:USDT",
            bids=[
                OrderBookLevel(price=50000, amount=1.0),
                OrderBookLevel(price=49990, amount=2.0),
            ],
            asks=[OrderBookLevel(price=50010, amount=0.5)],
        )
        assert book.bid_depth_usd == 50000 * 1.0 + 49990 * 2.0
        assert book.ask_depth_usd == 50010 * 0.5


class TestGraphNode:
    def test_hash_equality(self):
        a = GraphNode("binance", "BTC", PositionType.SHORT_PERP)
        b = GraphNode("binance", "BTC", PositionType.SHORT_PERP)
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_inequality(self):
        a = GraphNode("binance", "BTC", PositionType.SHORT_PERP)
        b = GraphNode("bybit", "BTC", PositionType.SHORT_PERP)
        assert a != b

    def test_set_membership(self):
        nodes = {
            GraphNode("binance", "BTC", PositionType.SHORT_PERP),
            GraphNode("binance", "BTC", PositionType.SHORT_PERP),
        }
        assert len(nodes) == 1


class TestCycle:
    def test_net_yield(self):
        cycle = Cycle(
            nodes=[
                GraphNode("binance", "BTC", PositionType.COLLATERAL),
                GraphNode("binance", "BTC", PositionType.SHORT_PERP),
            ],
            total_weight=-0.002,
        )
        assert cycle.net_yield_per_period == 0.002

    def test_canonical_form(self):
        nodes_a = [
            GraphNode("binance", "BTC", PositionType.COLLATERAL),
            GraphNode("bybit", "BTC", PositionType.SHORT_PERP),
        ]
        nodes_b = [
            GraphNode("bybit", "BTC", PositionType.SHORT_PERP),
            GraphNode("binance", "BTC", PositionType.COLLATERAL),
        ]
        cycle_a = Cycle(nodes=nodes_a, total_weight=-0.002)
        cycle_b = Cycle(nodes=nodes_b, total_weight=-0.002)
        assert cycle_a.canonical_form() == cycle_b.canonical_form()


class TestPortfolio:
    def test_total_equity(self):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 5000, "bybit": 3000}
        )
        assert portfolio.total_equity == 8000

    def test_drawdown(self):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 9000},
            peak_equity=10000,
        )
        assert abs(portfolio.drawdown_from_peak - 0.1) < 1e-10

    def test_no_drawdown(self):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 10000},
            peak_equity=10000,
        )
        assert portfolio.drawdown_from_peak == 0.0

    def test_update_peak(self):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 12000},
            peak_equity=10000,
        )
        portfolio.update_peak()
        assert portfolio.peak_equity == 12000


class TestArbitragePosition:
    def test_delta_neutral(self):
        pos = ArbitragePosition(
            id="test",
            leg_a=OrderResult(
                order_id="a", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.BUY, amount=1.0, avg_price=50000, fee=0,
                is_filled=True,
            ),
            leg_b=OrderResult(
                order_id="b", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.SELL, amount=1.0, avg_price=50000, fee=0,
                is_filled=True,
            ),
        )
        assert pos.delta_usd == 0.0
        assert pos.notional_usd == 50000.0

    def test_is_open(self):
        pos = ArbitragePosition(id="test")
        assert pos.is_open
        from datetime import datetime, timezone
        pos.closed_at = datetime.now(timezone.utc)
        assert not pos.is_open


class TestMarketSnapshot:
    def test_instruments(self):
        snapshot = MarketSnapshot()
        snapshot.update("binance", ExchangeData(
            rates={"BTC/USDT:USDT": FundingRate(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003)},
        ))
        snapshot.update("bybit", ExchangeData(
            rates={"ETH/USDT:USDT": FundingRate(exchange="bybit", symbol="ETH/USDT:USDT", rate=0.0001)},
        ))
        assert snapshot.instruments == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
        assert set(snapshot.exchanges) == {"binance", "bybit"}

    def test_get_none(self):
        snapshot = MarketSnapshot()
        assert snapshot.get("non-existent", "BTC") is None

    def test_get_no_rate(self):
        snapshot = MarketSnapshot()
        snapshot.update("binance", ExchangeData(rates={}))
        assert snapshot.get("binance", "BTC") is None

    def test_mark_stale(self):
        snapshot = MarketSnapshot()
        snapshot.mark_stale("binance", "error")
        assert snapshot.stale_exchanges["binance"] == "error"


class TestModelsEdgeCases:
    def test_graph_node_eq_invalid(self):
        node = GraphNode("binance", "BTC", PositionType.SHORT_PERP)
        assert node != "not a node"

    def test_cycle_canonical_form_empty(self):
        cycle = Cycle(nodes=[], total_weight=0)
        assert cycle.canonical_form() == ()

    def test_arbitrage_position_notional_none_leg_a(self):
        pos = ArbitragePosition(leg_a=None)
        assert pos.notional_usd == 0.0

    def test_margin_state_ratio(self):
        ms = MarginState(equity=1000, used=500)
        assert ms.ratio == 2.0
        ms_zero = MarginState(equity=1000, used=0)
        assert ms_zero.ratio == 1000 / 1e-8

    def test_portfolio_drawdown_zero_peak(self):
        p = Portfolio(peak_equity=0.0)
        assert p.drawdown_from_peak == 0.0

    def test_portfolio_equity_at_none(self):
        p = Portfolio(equity_by_exchange={})
        assert p.equity_at("non-existent") == 0.0

    def test_portfolio_add_position(self):
        p = Portfolio()
        pos = ArbitragePosition(id="test")
        p.add_position(pos)
        assert len(p.positions) == 1
        assert p.positions[0].id == "test"
