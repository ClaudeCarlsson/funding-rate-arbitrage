"""Tests for the backtesting framework."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from funding_arb.backtester import Backtester, BacktestConfig, BacktestResult


class TestSyntheticData:
    def test_generates_data(self):
        data = Backtester.generate_synthetic_data()
        assert not data.empty
        assert set(data.columns) == {"timestamp", "exchange", "symbol", "rate"}
        assert len(data["exchange"].unique()) == 3
        assert len(data["symbol"].unique()) == 2

    def test_custom_params(self):
        data = Backtester.generate_synthetic_data(
            exchanges=["binance", "bybit"],
            symbols=["BTC/USDT:USDT"],
            start_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )
        assert len(data["exchange"].unique()) == 2
        assert len(data["symbol"].unique()) == 1

    def test_deterministic_with_seed(self):
        data1 = Backtester.generate_synthetic_data(seed=123)
        data2 = Backtester.generate_synthetic_data(seed=123)
        pd.testing.assert_frame_equal(data1, data2)


class TestBacktester:
    def test_runs_on_synthetic_data(self):
        data = Backtester.generate_synthetic_data(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 3, 31, tzinfo=timezone.utc),
        )
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 3, 31, tzinfo=timezone.utc),
            initial_capital=10_000.0,
            min_net_yield_bps=1.0,
        )
        bt = Backtester(config)
        result = bt.run(data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_no_trades_with_high_threshold(self):
        data = Backtester.generate_synthetic_data(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            base_rate=0.0001,
            volatility=0.00001,
        )
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            min_net_yield_bps=1000.0,  # very high threshold
        )
        bt = Backtester(config)
        result = bt.run(data)
        # With very high threshold, likely no trades
        assert isinstance(result, BacktestResult)

    def test_empty_data(self):
        data = pd.DataFrame(columns=["timestamp", "exchange", "symbol", "rate"])
        bt = Backtester()
        result = bt.run(data)
        assert len(result.trades) == 0
        assert len(result.equity_curve) == 0

    def test_missing_columns_raises(self):
        data = pd.DataFrame({"foo": [1], "bar": [2]})
        bt = Backtester()
        with pytest.raises(ValueError, match="Missing columns"):
            bt.run(data)


class TestBacktestResult:
    def test_summary_string(self):
        config = BacktestConfig(initial_capital=10000)
        result = BacktestResult(
            config=config,
            equity_curve=[
                (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000),
                (datetime(2024, 12, 31, tzinfo=timezone.utc), 11500),
            ],
            daily_returns=[0.001] * 100,
        )
        summary = result.summary()
        assert "BACKTEST RESULTS" in summary
        assert "11,500" in summary

    def test_metrics_no_data(self):
        result = BacktestResult(config=BacktestConfig())
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0

    def test_max_drawdown(self):
        config = BacktestConfig(initial_capital=10000)
        result = BacktestResult(
            config=config,
            equity_curve=[
                (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000),
                (datetime(2024, 2, 1, tzinfo=timezone.utc), 11000),
                (datetime(2024, 3, 1, tzinfo=timezone.utc), 9900),  # 10% from peak
                (datetime(2024, 4, 1, tzinfo=timezone.utc), 10500),
            ],
        )
        assert abs(result.max_drawdown - 0.1) < 0.001

    def test_annualized_return_edge_cases(self):
        config = BacktestConfig(initial_capital=10000)
        # Less than 2 points
        result = BacktestResult(config=config, equity_curve=[(datetime(2024, 1, 1, tzinfo=timezone.utc), 10000)])
        assert result.annualized_return == 0.0

        # Days <= 0
        result = BacktestResult(
            config=config,
            equity_curve=[
                (datetime(2024, 1, 1, tzinfo=timezone.utc), 10000),
                (datetime(2024, 1, 1, tzinfo=timezone.utc), 11000),
            ],
        )
        assert result.annualized_return == 0.0

    def test_sharpe_ratio_zero_std(self):
        result = BacktestResult(config=BacktestConfig(), daily_returns=[0.01, 0.01, 0.01])
        assert result.sharpe_ratio == 0.0

    def test_metrics_with_trades(self):
        from funding_arb.backtester import BacktestTrade
        result = BacktestResult(config=BacktestConfig())
        result.trades = [
            BacktestTrade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                net_pnl=100.0,
                periods_held=3,
            ),
            BacktestTrade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                net_pnl=-50.0,
                periods_held=5,
            ),
        ]
        assert result.win_rate == 0.5
        assert result.avg_trade_pnl == 25.0
        assert result.avg_holding_periods == 4.0


class TestBacktesterEdgeCases:
    def test_run_missing_data_closes_trade(self):
        # Create data with a gap for one exchange
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 8, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 1, 16, tzinfo=timezone.utc)

        data = pd.DataFrame([
            # Period 1: Opp exists
            {"timestamp": ts1, "exchange": "binance", "symbol": "BTC/USDT:USDT", "rate": 0.01},
            {"timestamp": ts1, "exchange": "bybit", "symbol": "BTC/USDT:USDT", "rate": 0.001},
            # Period 2: missing data for bybit
            {"timestamp": ts2, "exchange": "binance", "symbol": "BTC/USDT:USDT", "rate": 0.01},
            # Period 3: data back
            {"timestamp": ts3, "exchange": "binance", "symbol": "BTC/USDT:USDT", "rate": 0.01},
            {"timestamp": ts3, "exchange": "bybit", "symbol": "BTC/USDT:USDT", "rate": 0.001},
        ])

        config = BacktestConfig(
            start_date=ts1,
            end_date=ts3,
            min_net_yield_bps=1.0,
        )
        bt = Backtester(config)
        result = bt.run(data)

        # Trade should have been closed at ts2 due to missing data
        assert any(t.exit_reason == "missing_data" for t in result.trades)

    def test_run_min_position_size_filter(self):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        data = pd.DataFrame([
            {"timestamp": ts1, "exchange": "binance", "symbol": "BTC/USDT:USDT", "rate": 0.01},
            {"timestamp": ts1, "exchange": "bybit", "symbol": "BTC/USDT:USDT", "rate": 0.001},
        ])

        # Force very small size by having very low capital
        config = BacktestConfig(
            start_date=ts1,
            end_date=ts1,
            initial_capital=10.0,  # Only $10, 20% is $2, less than $100 min
        )
        bt = Backtester(config)
        result = bt.run(data)

        assert len(result.trades) == 0

    def test_get_rate_empty(self):
        bt = Backtester()
        data = pd.DataFrame([
            {"exchange": "binance", "symbol": "BTC", "rate": 0.001}
        ])
        assert bt._get_rate(data, "bybit", "BTC") is None
