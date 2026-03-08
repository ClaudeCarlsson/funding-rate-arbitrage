"""Backtesting framework for funding rate arbitrage strategies."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import OptimizerConfig
from .models import (
    ExchangeData,
    FundingRate,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
)
from .optimizer import ArbitrageOptimizer
from .risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 12, 31, tzinfo=UTC))
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0006  # taker fee per side
    min_net_yield_bps: float = 5.0
    funding_period_hours: int = 8
    max_open_positions: int = 5
    exit_on_negative_rate: bool = True
    # Simulated spread/slippage
    simulated_spread_bps: float = 2.0  # 0.02%
    simulated_slippage_bps: float = 1.0  # 0.01%


@dataclass
class BacktestTrade:
    """Record of a single backtested trade."""
    entry_time: datetime
    exit_time: datetime | None = None
    instrument: str = ""
    short_exchange: str = ""
    long_exchange: str = ""
    size_usd: float = 0.0
    entry_rate_spread: float = 0.0
    periods_held: int = 0
    total_funding_earned: float = 0.0
    total_fees_paid: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        if not self.equity_curve:
            return 0.0
        return (self.equity_curve[-1][1] - self.config.initial_capital) / self.config.initial_capital

    @property
    def total_return_pct(self) -> float:
        return self.total_return * 100

    @property
    def annualized_return(self) -> float:
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        if days <= 0:
            return 0.0
        return (1 + self.total_return) ** (365.0 / days) - 1

    @property
    def annualized_return_pct(self) -> float:
        return self.annualized_return * 100

    @property
    def sharpe_ratio(self) -> float:
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0
        returns = np.array(self.daily_returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        return (mean * 365**0.5) / std  # annualized

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_drawdown_pct(self) -> float:
        return self.max_drawdown * 100

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.net_pnl > 0)
        return winners / len(self.trades)

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.net_pnl for t in self.trades) / len(self.trades)

    @property
    def avg_holding_periods(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.periods_held for t in self.trades) / len(self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.total_fees_paid for t in self.trades)

    @property
    def total_funding(self) -> float:
        return sum(t.total_funding_earned for t in self.trades)

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}",
            f"Initial capital: ${self.config.initial_capital:,.2f}",
            f"Final equity: ${self.equity_curve[-1][1]:,.2f}" if self.equity_curve else "Final equity: N/A",
            "",
            f"Total return: {self.total_return_pct:.2f}%",
            f"Annualized return: {self.annualized_return_pct:.2f}%",
            f"Sharpe ratio: {self.sharpe_ratio:.2f}",
            f"Max drawdown: {self.max_drawdown_pct:.2f}%",
            "",
            f"Total trades: {len(self.trades)}",
            f"Win rate: {self.win_rate*100:.1f}%",
            f"Avg trade P&L: ${self.avg_trade_pnl:.2f}",
            f"Avg holding periods: {self.avg_holding_periods:.1f}",
            "",
            f"Total funding earned: ${self.total_funding:,.2f}",
            f"Total fees paid: ${self.total_fees:,.2f}",
            f"Net P&L: ${self.total_funding - self.total_fees:,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class Backtester:
    """Replays historical funding rate data to evaluate strategy performance."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.optimizer = ArbitrageOptimizer(
            OptimizerConfig(min_net_yield_bps=self.config.min_net_yield_bps)
        )
        self.risk_manager = RiskManager()

        # Integrate Queue Simulator for realistic paper trading queue modeling
        from .queue_simulator import FIFOQueueSimulator
        self.queue_sim = FIFOQueueSimulator()

    def run(self, funding_data: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical funding rate data.

        Args:
            funding_data: DataFrame with columns:
                - timestamp (datetime)
                - exchange (str)
                - symbol (str)
                - rate (float) -- funding rate per period

        Returns:
            BacktestResult with full trade log and performance metrics.
        """
        result = BacktestResult(config=self.config)

        # Validate input
        required_cols = {"timestamp", "exchange", "symbol", "rate"}
        if not required_cols.issubset(funding_data.columns):
            missing = required_cols - set(funding_data.columns)
            raise ValueError(f"Missing columns: {missing}")

        # Sort and group by timestamp
        funding_data = funding_data.sort_values("timestamp").copy()
        funding_data["timestamp"] = pd.to_datetime(funding_data["timestamp"], utc=True)

        # Filter date range
        mask = (
            (funding_data["timestamp"] >= self.config.start_date) &
            (funding_data["timestamp"] <= self.config.end_date)
        )
        funding_data = funding_data[mask]

        if funding_data.empty:
            logger.warning("No data in date range")
            return result

        # Initialize state
        equity = self.config.initial_capital
        open_trades: list[BacktestTrade] = []
        prev_date = None

        # Group by funding period (timestamp)
        for timestamp, period_data in funding_data.groupby("timestamp"):
            ts = pd.Timestamp(timestamp).to_pydatetime()

            # Track daily returns
            current_date = ts.date()
            if prev_date is not None and current_date != prev_date and result.equity_curve:
                prev_equity = result.equity_curve[-1][1]
                if prev_equity > 0:
                    daily_ret = (equity - prev_equity) / prev_equity
                    result.daily_returns.append(daily_ret)
            prev_date = current_date

            # Build snapshot from period data
            snapshot = self._build_snapshot(period_data, ts)

            # 1. Update open trades -- collect funding or exit
            trades_to_close: list[int] = []
            for i, trade in enumerate(open_trades):
                # Find current rate spread for this trade
                short_rate = self._get_rate(period_data, trade.short_exchange, trade.instrument)
                long_rate = self._get_rate(period_data, trade.long_exchange, trade.instrument)

                if short_rate is not None and long_rate is not None:
                    net_rate = short_rate - long_rate
                    period_funding = net_rate * trade.size_usd
                    trade.total_funding_earned += period_funding
                    trade.periods_held += 1
                    equity += period_funding

                    # Exit conditions
                    if self.config.exit_on_negative_rate and net_rate < 0:
                        trade.exit_reason = "negative_rate"
                        trades_to_close.append(i)
                else:
                    # Data missing -- close for safety
                    trade.exit_reason = "missing_data"
                    trades_to_close.append(i)

            # Close trades (reverse order to preserve indices)
            for i in sorted(trades_to_close, reverse=True):
                trade = open_trades.pop(i)
                # Exit fees
                exit_fees = trade.size_usd * self.config.fee_rate * 2  # both legs
                exit_slippage = trade.size_usd * self.config.simulated_slippage_bps / 10_000
                trade.total_fees_paid += exit_fees + exit_slippage
                equity -= exit_fees + exit_slippage

                trade.exit_time = ts
                trade.net_pnl = trade.total_funding_earned - trade.total_fees_paid
                result.trades.append(trade)

            # 2. Find new opportunities
            if len(open_trades) < self.config.max_open_positions:
                opps = self.optimizer.find_simple_opportunities(snapshot)
                for opp in opps:
                    if len(open_trades) >= self.config.max_open_positions:
                        break

                    # Skip if we already have a position in this instrument on these exchanges
                    already_in = any(
                        t.instrument == opp["instrument"]
                        and t.short_exchange == opp["short_exchange"]
                        and t.long_exchange == opp["long_exchange"]
                        for t in open_trades
                    )
                    if already_in:
                        continue

                    # Size the position
                    size = min(
                        equity * self.config.initial_capital / max(equity, 1) * 0.20,  # max 20% of current equity
                        equity * 0.20,
                        opp.get("net_yield_per_period", 0) / max(opp.get("net_yield_per_period", 0.0001), 0.0001) * 1000,
                    )
                    size = min(size, equity * 0.20)
                    if size < 100:  # minimum $100 position
                        continue

                    # Entry fees + slippage
                    entry_fees = size * self.config.fee_rate * 2
                    entry_slippage = size * self.config.simulated_slippage_bps / 10_000
                    entry_spread_cost = size * self.config.simulated_spread_bps / 10_000

                    total_entry_cost = entry_fees + entry_slippage + entry_spread_cost
                    equity -= total_entry_cost

                    trade = BacktestTrade(
                        entry_time=ts,
                        instrument=opp["instrument"],
                        short_exchange=opp["short_exchange"],
                        long_exchange=opp["long_exchange"],
                        size_usd=size,
                        entry_rate_spread=opp["spread"],
                        total_fees_paid=total_entry_cost,
                    )
                    open_trades.append(trade)

            # Record equity
            result.equity_curve.append((ts, equity))

        # Close any remaining open trades at end
        for trade in open_trades:
            exit_fees = trade.size_usd * self.config.fee_rate * 2
            trade.total_fees_paid += exit_fees
            equity -= exit_fees
            trade.exit_time = self.config.end_date
            trade.exit_reason = "end_of_backtest"
            trade.net_pnl = trade.total_funding_earned - trade.total_fees_paid
            result.trades.append(trade)

        if result.equity_curve:
            result.equity_curve.append((self.config.end_date, equity))

        return result

    def _build_snapshot(self, period_data: pd.DataFrame, ts: datetime) -> MarketSnapshot:
        """Build a MarketSnapshot from a single period's data."""
        snapshot = MarketSnapshot(timestamp=ts)
        by_exchange: dict[str, dict[str, FundingRate]] = {}

        for _, row in period_data.iterrows():
            ex = row["exchange"]
            sym = row["symbol"]
            if ex not in by_exchange:
                by_exchange[ex] = {}
            by_exchange[ex][sym] = FundingRate(
                exchange=ex, symbol=sym, rate=row["rate"], timestamp=ts,
            )

        for exchange, rates in by_exchange.items():
            books = {}
            for symbol in rates:
                books[symbol] = OrderBook(
                    exchange=exchange, symbol=symbol,
                    bids=[OrderBookLevel(price=50000, amount=100)],
                    asks=[OrderBookLevel(price=50010, amount=100)],
                    timestamp=ts,
                )
            snapshot.update(exchange, ExchangeData(rates=rates, books=books, fetched_at=ts))

        return snapshot

    def _get_rate(
        self, period_data: pd.DataFrame, exchange: str, symbol: str
    ) -> float | None:
        """Get funding rate for a specific exchange/symbol from period data."""
        mask = (period_data["exchange"] == exchange) & (period_data["symbol"] == symbol)
        rows = period_data[mask]
        if rows.empty:
            return None
        return float(rows.iloc[0]["rate"])

    @staticmethod
    def generate_synthetic_data(
        exchanges: list[str] | None = None,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        base_rate: float = 0.0003,
        volatility: float = 0.0005,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic funding rate data for testing.

        Creates realistic-looking funding rate data with:
        - Mean-reverting rates around base_rate
        - Different rate levels per exchange (simulating real divergences)
        - 8-hour funding periods
        """
        rng = np.random.default_rng(seed)

        if exchanges is None:
            exchanges = ["binance", "hyperliquid", "bybit"]
        if symbols is None:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        if start_date is None:
            start_date = datetime(2024, 1, 1, tzinfo=UTC)
        if end_date is None:
            end_date = datetime(2024, 12, 31, tzinfo=UTC)

        # Generate timestamps (every 8 hours)
        timestamps = pd.date_range(start=start_date, end=end_date, freq="8h", tz=UTC)

        rows = []
        # Exchange-specific rate multipliers (simulates structural differences)
        exchange_multipliers = {ex: 1.0 + rng.uniform(-0.5, 1.5) for ex in exchanges}

        for symbol in symbols:
            # Symbol-specific base rate
            sym_base = base_rate * (1 + rng.uniform(-0.3, 0.3))

            for exchange in exchanges:
                rate = sym_base * exchange_multipliers[exchange]
                for ts in timestamps:
                    # Mean-reverting random walk
                    innovation = rng.normal(0, volatility)
                    mean_reversion = 0.1 * (sym_base * exchange_multipliers[exchange] - rate)
                    rate = rate + innovation + mean_reversion

                    rows.append({
                        "timestamp": ts,
                        "exchange": exchange,
                        "symbol": symbol,
                        "rate": rate,
                    })

        return pd.DataFrame(rows)
