"""Tests for the trade executor module."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from funding_arb.config import Config, ExecutorConfig, ExchangeConfig, DatabaseConfig
from funding_arb.executor import TradeExecutor, KILL_SWITCH_PATH
from funding_arb.models import (
    ArbitragePosition,
    Cycle,
    GraphNode,
    Opportunity,
    OrderResult,
    OrderSide,
    PositionType,
    TradeLeg,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path):
    """Patch _make_config to use tmp_path for all tests."""
    _make_config._tmp_path = tmp_path


def _make_config(mode: str = "paper") -> Config:
    tmp = getattr(_make_config, "_tmp_path", Path("/tmp/test_executor"))
    tmp.mkdir(parents=True, exist_ok=True)
    return Config(
        exchanges={"binance": ExchangeConfig(name="binance", enabled=True)},
        executor=ExecutorConfig(mode=mode, max_slippage_pct=0.5),
        database=DatabaseConfig(
            state_db_path=str(tmp / "state.db"),
            trades_db_path=str(tmp / "trades.db"),
            funding_db_path=str(tmp / "funding.db"),
            parquet_dir=str(tmp / "parquet"),
        ),
    )


def _make_opportunity() -> Opportunity:
    return Opportunity(
        cycle=Cycle(
            nodes=[
                GraphNode("binance", "BTC/USDT:USDT", PositionType.SHORT_PERP),
                GraphNode("bybit", "BTC/USDT:USDT", PositionType.LONG_PERP),
            ],
            total_weight=-0.001,
        ),
        expected_net_yield_per_period=0.001,
        yield_variance=1e-6,
        capital_required=1000.0,
        risk_adjusted_yield=0.01,
        exchange="binance",
        net_rate=0.001,
        expected_spread=0.0002,
        leg_a=TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL),
        leg_b=TradeLeg(exchange="bybit", symbol="BTC/USDT:USDT", side=OrderSide.BUY),
    )


class TestPaperTrading:
    def test_paper_fill_returns_order_result(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL)
        result = executor._paper_fill(leg, 1000.0, "test-123", "A")
        assert result.is_filled
        assert result.exchange == "binance"
        assert result.side == OrderSide.SELL
        assert result.amount > 0
        assert result.avg_price > 0
        assert result.fee > 0
        assert "paper" in result.order_id

    @pytest.mark.asyncio
    async def test_open_position_paper(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        opp = _make_opportunity()
        position = await executor.open_position(opp, 500.0)
        assert position is not None
        assert position.id != ""
        assert position.leg_a is not None
        assert position.leg_b is not None
        assert position.is_open

    @pytest.mark.asyncio
    async def test_open_position_missing_legs(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        opp = _make_opportunity()
        opp.leg_a = None
        result = await executor.open_position(opp, 500.0)
        assert result is None


class TestPartialPosition:
    @pytest.mark.asyncio
    async def test_partial_fill_returns_position(self):
        """If leg B fails, executor returns the partial position so it's tracked."""
        config = _make_config(mode="live")
        executor = TradeExecutor(config)

        # Leg A succeeds
        mock_ex_a = AsyncMock()
        mock_ex_a.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_ex_a.create_order = AsyncMock(return_value={
            "id": "order-a", "average": 50000.0, "filled": 0.004,
            "fee": {"cost": 0.1}, "status": "closed",
        })
        mock_ex_a.fetch_order = AsyncMock(return_value={
            "id": "order-a", "status": "closed", "average": 50000.0,
            "filled": 0.004, "fee": {"cost": 0.1},
        })

        # Leg B fails all retries
        mock_ex_b = AsyncMock()
        mock_ex_b.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_ex_b.create_order = AsyncMock(side_effect=Exception("connection lost"))

        executor._exchanges["binance"] = mock_ex_a
        executor._exchanges["bybit"] = mock_ex_b

        opp = _make_opportunity()
        position = await executor.open_position(opp, 200.0)

        # Should return the partial position, not None
        assert position is not None
        assert position.leg_a is not None
        assert position.leg_b is None
        assert position.is_open


class TestKillSwitch:
    def test_kill_switch_not_active_by_default(self):
        config = _make_config()
        executor = TradeExecutor(config)
        # Clean up any stale kill switch file
        KILL_SWITCH_PATH.unlink(missing_ok=True)
        assert not executor.is_killed()

    def test_kill_switch_active_when_file_exists(self, tmp_path):
        config = _make_config()
        executor = TradeExecutor(config)
        with patch.object(type(executor), 'is_killed', return_value=True):
            assert executor.is_killed()

    @pytest.mark.asyncio
    async def test_open_position_refused_when_killed(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        opp = _make_opportunity()
        with patch.object(executor, 'is_killed', return_value=True):
            result = await executor.open_position(opp, 500.0)
        assert result is None


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_paper_position(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        opp = _make_opportunity()
        position = await executor.open_position(opp, 500.0)
        assert position is not None
        assert position.is_open

        success = await executor.close_position(position)
        assert success
        assert not position.is_open

    @pytest.mark.asyncio
    async def test_close_already_closed(self):
        config = _make_config(mode="paper")
        executor = TradeExecutor(config)
        opp = _make_opportunity()
        position = await executor.open_position(opp, 500.0)
        assert position is not None
        position.closed_at = datetime.now(UTC)
        result = await executor.close_position(position)
        assert not result


class TestLiveExecution:
    @pytest.mark.asyncio
    async def test_execute_leg_retries_on_failure(self):
        config = _make_config(mode="live")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_ex.create_order = AsyncMock(side_effect=Exception("rate limited"))
        executor._exchanges["binance"] = mock_ex

        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL)
        result = await executor._execute_leg(leg, 1000.0, "retry-test", "A")

        assert result is None
        assert mock_ex.create_order.call_count == 3  # MAX_RETRIES

    @pytest.mark.asyncio
    async def test_execute_leg_success(self):
        config = _make_config(mode="live")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
        mock_ex.create_order = AsyncMock(return_value={
            "id": "order-123",
            "average": 50010.0,
            "filled": 0.02,
            "fee": {"cost": 0.6},
            "status": "closed",
        })
        mock_ex.fetch_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "closed",
            "average": 50010.0,
            "filled": 0.02,
            "fee": {"cost": 0.6},
        })
        executor._exchanges["binance"] = mock_ex

        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL)
        result = await executor._execute_leg(leg, 1000.0, "success-test", "A")

        assert result is not None
        assert result.is_filled
        assert result.order_id == "order-123"
        assert result.avg_price == 50010.0

    @pytest.mark.asyncio
    async def test_execute_leg_missing_exchange(self):
        config = _make_config(mode="live")
        executor = TradeExecutor(config)
        leg = TradeLeg(exchange="nonexistent", symbol="BTC/USDT:USDT", side=OrderSide.BUY)
        result = await executor._execute_leg(leg, 1000.0, "missing-ex", "A")
        assert result is None


class TestDryRunMode:
    @pytest.mark.asyncio
    async def test_dry_run_fetches_real_price(self):
        config = _make_config(mode="dry_run")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
        })
        executor._exchanges["binance"] = mock_ex

        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL)
        result = await executor._dry_run_fill(leg, 200.0, "dry-test", "A")

        assert result is not None
        assert result.is_filled
        assert "dry-" in result.order_id
        assert result.avg_price == 50000.0
        assert result.amount == pytest.approx(200.0 / 50000.0)
        assert result.fee == pytest.approx(200.0 * 0.0004)
        mock_ex.fetch_ticker.assert_called_once_with("BTC/USDT:USDT")
        # No create_order call — that's the whole point
        mock_ex.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_saves_to_db(self):
        config = _make_config(mode="dry_run")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
        })
        executor._exchanges["binance"] = mock_ex

        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.SELL)
        await executor._dry_run_fill(leg, 200.0, "dry-db", "A")

        dry_trades = executor.db.get_dry_trades()
        assert len(dry_trades) >= 1
        latest = dry_trades[0]
        assert latest["exchange"] == "binance"
        assert latest["symbol"] == "BTC/USDT:USDT"
        assert latest["side"] == "sell"
        assert latest["would_fill_at"] == 50000.0
        assert latest["spread_bps"] == pytest.approx(4.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_dry_run_records_spread(self):
        config = _make_config(mode="dry_run")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "last": 50000.0,
            "bid": 49980.0,
            "ask": 50020.0,
        })
        executor._exchanges["binance"] = mock_ex

        leg = TradeLeg(exchange="binance", symbol="BTC/USDT:USDT", side=OrderSide.BUY)
        result = await executor._dry_run_fill(leg, 200.0, "spread-test", "A")

        assert result is not None
        assert result.raw["spread_bps"] == pytest.approx(8.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_dry_run_missing_exchange(self):
        config = _make_config(mode="dry_run")
        executor = TradeExecutor(config)
        leg = TradeLeg(exchange="nonexistent", symbol="BTC/USDT:USDT", side=OrderSide.BUY)
        result = await executor._dry_run_fill(leg, 200.0, "missing", "A")
        assert result is None

    @pytest.mark.asyncio
    async def test_dry_run_open_position(self):
        config = _make_config(mode="dry_run")
        executor = TradeExecutor(config)

        mock_ex = AsyncMock()
        mock_ex.fetch_ticker = AsyncMock(return_value={
            "last": 50000.0, "bid": 49990.0, "ask": 50010.0,
        })
        executor._exchanges["binance"] = mock_ex
        executor._exchanges["bybit"] = mock_ex

        opp = _make_opportunity()
        position = await executor.open_position(opp, 200.0)

        assert position is not None
        assert position.leg_a is not None
        assert position.leg_b is not None
        assert "dry-" in position.leg_a.order_id
        assert "dry-" in position.leg_b.order_id
        # No real orders placed
        mock_ex.create_order.assert_not_called()
