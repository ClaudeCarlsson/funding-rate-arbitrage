"""Core data models for the funding rate arbitrage system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any


class PositionType(Enum):
    SPOT_LONG = auto()
    SHORT_PERP = auto()
    LONG_PERP = auto()
    CASH = auto()
    COLLATERAL = auto()


class EdgeType(Enum):
    FUNDING_SHORT = auto()
    FUNDING_LONG = auto()
    SPOT_BUY = auto()
    SPOT_SELL = auto()
    TRANSFER = auto()


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class FundingRate:
    exchange: str
    symbol: str
    rate: float  # per-period rate (typically 8h)
    next_funding_time: datetime | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def annualized(self) -> float:
        """Annualize assuming 3 funding periods per day."""
        return self.rate * 3 * 365


@dataclass
class OrderBookLevel:
    price: float
    amount: float


@dataclass
class OrderBook:
    exchange: str
    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return float("inf")
        return (self.asks[0].price - self.bids[0].price) / self.mid_price

    @property
    def bid_depth_usd(self) -> float:
        return sum(lvl.price * lvl.amount for lvl in self.bids)

    @property
    def ask_depth_usd(self) -> float:
        return sum(lvl.price * lvl.amount for lvl in self.asks)


@dataclass
class OpenInterest:
    exchange: str
    symbol: str
    oi_contracts: float
    oi_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Balance:
    currency: str
    free: float
    used: float
    total: float


@dataclass
class ExchangeData:
    rates: dict[str, FundingRate] = field(default_factory=dict)
    books: dict[str, OrderBook] = field(default_factory=dict)
    open_interest: dict[str, OpenInterest] = field(default_factory=dict)
    balances: dict[str, Balance] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class InstrumentData:
    funding_rate: float
    taker_fee: float
    spread: float
    max_position: float
    available_margin: float
    book_depth_usd: float
    is_stale: bool = False


@dataclass
class MarketSnapshot:
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    exchange_data: dict[str, ExchangeData] = field(default_factory=dict)
    stale_exchanges: dict[str, str] = field(default_factory=dict)

    @property
    def exchanges(self) -> list[str]:
        return list(self.exchange_data.keys())

    @property
    def instruments(self) -> set[str]:
        instruments: set[str] = set()
        for data in self.exchange_data.values():
            instruments.update(data.rates.keys())
        return instruments

    def update(self, exchange: str, data: ExchangeData) -> None:
        self.exchange_data[exchange] = data

    def mark_stale(self, exchange: str, error: str) -> None:
        self.stale_exchanges[exchange] = error

    def get(self, exchange: str, instrument: str, max_pos: float = 10000.0) -> InstrumentData | None:
        """Helper to extract relevant data for an instrument."""
        ex_data = self.exchange_data.get(exchange)
        if ex_data is None:
            return None
        rate = ex_data.rates.get(instrument)
        book = ex_data.books.get(instrument)
        if rate is None:
            return None
        return InstrumentData(
            funding_rate=rate.rate,
            taker_fee=0.0006,  # default, overridden per exchange
            spread=book.spread if book else 0.001,
            max_position=max_pos,
            available_margin=sum(b.free for b in ex_data.balances.values()),
            book_depth_usd=book.bid_depth_usd if book else 0.0,
            is_stale=exchange in self.stale_exchanges,
        )


@dataclass
class GraphNode:
    exchange: str
    instrument: str
    position_type: PositionType

    def __hash__(self) -> int:
        return hash((self.exchange, self.instrument, self.position_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return (
            self.exchange == other.exchange
            and self.instrument == other.instrument
            and self.position_type == other.position_type
        )

    def __repr__(self) -> str:
        return f"({self.exchange}, {self.instrument}, {self.position_type.name})"


@dataclass
class Cycle:
    nodes: list[GraphNode]
    total_weight: float
    edge_types: list[EdgeType] = field(default_factory=list)

    @property
    def net_yield_per_period(self) -> float:
        return -self.total_weight

    def canonical_form(self) -> tuple[GraphNode, ...]:
        """Rotation-invariant representation for deduplication."""
        if not self.nodes:
            return ()
        min_idx = self.nodes.index(min(self.nodes, key=repr))
        rotated = self.nodes[min_idx:] + self.nodes[:min_idx]
        return tuple(rotated)


@dataclass
class Opportunity:
    cycle: Cycle
    expected_net_yield_per_period: float
    yield_variance: float
    capital_required: float
    risk_adjusted_yield: float
    exchange: str = ""
    expected_spread: float = 0.0
    net_rate: float = 0.0
    leg_a: TradeLeg | None = None
    leg_b: TradeLeg | None = None


@dataclass
class TradeLeg:
    exchange: str
    symbol: str
    side: OrderSide
    aggressive_price: float = 0.0


@dataclass
class OrderResult:
    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    amount: float
    avg_price: float
    fee: float
    is_filled: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitragePosition:
    id: str = ""
    leg_a: OrderResult | None = None
    leg_b: OrderResult | None = None
    entry_funding_rate: float = 0.0
    entry_spread: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    realized_pnl: float = 0.0
    funding_collected: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def notional_usd(self) -> float:
        if self.leg_a is None:
            return 0.0
        return self.leg_a.amount * self.leg_a.avg_price

    @property
    def delta_usd(self) -> float:
        delta = 0.0
        if self.leg_a:
            sign = 1.0 if self.leg_a.side == OrderSide.BUY else -1.0
            delta += sign * self.leg_a.amount * self.leg_a.avg_price
        if self.leg_b:
            sign = 1.0 if self.leg_b.side == OrderSide.BUY else -1.0
            delta += sign * self.leg_b.amount * self.leg_b.avg_price
        return delta


@dataclass
class MarginState:
    equity: float
    used: float

    @property
    def ratio(self) -> float:
        return self.equity / max(self.used, 1e-8)


@dataclass
class Portfolio:
    positions: list[ArbitragePosition] = field(default_factory=list)
    equity_by_exchange: dict[str, float] = field(default_factory=dict)
    margin_by_exchange: dict[str, MarginState] = field(default_factory=dict)
    peak_equity: float = 0.0

    @property
    def total_equity(self) -> float:
        return sum(self.equity_by_exchange.values())

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.total_equity) / self.peak_equity

    def equity_at(self, exchange: str) -> float:
        return self.equity_by_exchange.get(exchange, 0.0)

    def add_position(self, position: ArbitragePosition) -> None:
        self.positions.append(position)

    def update_peak(self) -> None:
        current = self.total_equity
        if current > self.peak_equity:
            self.peak_equity = current


class ViolationType(Enum):
    DELTA_DRIFT = auto()
    POSITION_CONCENTRATION = auto()
    EXCHANGE_CONCENTRATION = auto()
    LOW_COLLATERAL = auto()
    EXCESSIVE_LEVERAGE = auto()
    EMERGENCY_DRAWDOWN = auto()
    CORRELATION_BREAKDOWN = auto()


@dataclass
class Violation:
    type: ViolationType
    message: str
    severity: str = "warning"  # warning, critical
    details: dict[str, Any] = field(default_factory=dict)
