"""Tests for the orchestrator module."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from funding_arb.config import Config, DatabaseConfig, OptimizerConfig, RiskConfig, ScannerConfig
from funding_arb.models import (
    Cycle,
    ExchangeData,
    FundingRate,
    GraphNode,
    MarketSnapshot,
    Opportunity,
    Portfolio,
    PositionType,
    Violation,
    ViolationType,
)
from funding_arb.orchestrator import Orchestrator, main
from funding_arb.prediction import FundingRegime, RegimeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path) -> Config:
    """Build a Config whose database paths point into tmp_path."""
    return Config(
        scanner=ScannerConfig(poll_interval_s=0.01),
        optimizer=OptimizerConfig(),
        risk=RiskConfig(),
        database=DatabaseConfig(
            state_db_path=str(tmp_path / "state.db"),
            trades_db_path=str(tmp_path / "trades.db"),
            funding_db_path=str(tmp_path / "funding.db"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )


def _make_snapshot(
    *,
    exchanges: dict[str, dict[str, float]] | None = None,
    stale: dict[str, str] | None = None,
) -> MarketSnapshot:
    """Build a MarketSnapshot.

    exchanges: mapping of exchange_name -> {symbol: rate}.
    stale: mapping of exchange_name -> error string.
    """
    snapshot = MarketSnapshot(timestamp=datetime.now(timezone.utc))
    exchanges = exchanges or {}
    for ex_name, rates_map in exchanges.items():
        ex_data = ExchangeData(
            rates={
                symbol: FundingRate(exchange=ex_name, symbol=symbol, rate=rate)
                for symbol, rate in rates_map.items()
            },
        )
        snapshot.update(ex_name, ex_data)
    for ex_name, error in (stale or {}).items():
        snapshot.mark_stale(ex_name, error)
    return snapshot


def _make_opportunity(yield_val: float = 0.001, variance: float = 0.0001) -> Opportunity:
    """Build a minimal Opportunity for sizing tests."""
    node = GraphNode(exchange="binance", instrument="BTC/USDT:USDT", position_type=PositionType.SHORT_PERP)
    cycle = Cycle(nodes=[node], total_weight=-yield_val)
    return Opportunity(
        cycle=cycle,
        expected_net_yield_per_period=yield_val,
        yield_variance=variance,
        capital_required=10_000,
        risk_adjusted_yield=yield_val,
        exchange="binance",
        net_rate=yield_val,
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_init_with_config(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        assert orch.config is config
        assert orch._running is False
        assert orch._iteration == 0
        assert orch._funding_history == {}
        assert isinstance(orch.portfolio, Portfolio)

    def test_init_defaults_to_log_alerter(self, tmp_path):
        """Without Telegram env vars, alerter should be LogAlerter."""
        config = _make_config(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            orch = Orchestrator(config)
        from funding_arb.alerter import LogAlerter
        assert isinstance(orch.alerter, LogAlerter)

    def test_init_uses_telegram_when_env_set(self, tmp_path):
        """With both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, alerter should be TelegramAlerter."""
        config = _make_config(tmp_path)
        env = {"TELEGRAM_BOT_TOKEN": "tok123", "TELEGRAM_CHAT_ID": "chat456"}
        with patch.dict("os.environ", env, clear=True):
            orch = Orchestrator(config)
        from funding_arb.alerter import TelegramAlerter
        assert isinstance(orch.alerter, TelegramAlerter)

    def test_init_loads_config_when_none(self, tmp_path):
        """When config=None, load_config() is called."""
        fake_config = _make_config(tmp_path)
        with patch("funding_arb.orchestrator.load_config", return_value=fake_config):
            orch = Orchestrator(config=None)
        assert orch.config is fake_config


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_runs_ticks_then_stops(self, tmp_path):
        """start() should call _tick repeatedly until _running is False."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.close = AsyncMock()
        orch.alerter = MagicMock()
        orch.alerter.notify_system_start = AsyncMock()
        orch.alerter.notify_system_stop = AsyncMock()

        tick_count = 0

        async def fake_tick():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 2:
                orch._running = False

        orch._tick = AsyncMock(side_effect=fake_tick)

        await orch.start()

        assert tick_count == 2
        orch.scanner.initialize.assert_awaited_once()
        orch.alerter.notify_system_start.assert_awaited_once()
        # stop() is called in the finally block
        orch.alerter.notify_system_stop.assert_awaited_once()
        orch.scanner.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_handles_keyboard_interrupt(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.close = AsyncMock()
        orch.alerter = MagicMock()
        orch.alerter.notify_system_start = AsyncMock()
        orch.alerter.notify_system_stop = AsyncMock()
        orch._tick = AsyncMock(side_effect=KeyboardInterrupt)

        await orch.start()

        orch.scanner.close.assert_awaited_once()
        orch.alerter.notify_system_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_handles_unexpected_exception(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.close = AsyncMock()
        orch.alerter = MagicMock()
        orch.alerter.notify_system_start = AsyncMock()
        orch.alerter.notify_system_stop = AsyncMock()
        orch._tick = AsyncMock(side_effect=RuntimeError("boom"))

        await orch.start()

        orch.scanner.close.assert_awaited_once()
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_stop(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch._running = True
        orch.alerter = MagicMock()
        orch.alerter.notify_system_stop = AsyncMock()
        orch.scanner = MagicMock()
        orch.scanner.close = AsyncMock()

        await orch.stop()

        assert orch._running is False
        orch.alerter.notify_system_stop.assert_awaited_once_with("normal shutdown")
        orch.scanner.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# _tick  --  happy path
# ---------------------------------------------------------------------------


class TestTickHappyPath:
    @pytest.mark.asyncio
    async def test_tick_scans_and_processes(self, tmp_path):
        """Full happy-path tick: scan, save, find opportunities, update peak."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={
                "binance": {"BTC/USDT:USDT": 0.001},
                "bybit": {"BTC/USDT:USDT": -0.0005},
            },
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)

        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()

        # Spy on database.save_funding_rates_batch
        orch.database.save_funding_rates_batch = MagicMock()

        # Stub optimizer returns
        simple_opp = {
            "instrument": "BTC/USDT:USDT",
            "short_exchange": "binance",
            "long_exchange": "bybit",
            "short_rate": 0.001,
            "long_rate": -0.0005,
            "spread": 0.0015,
            "total_fees": 0.001,
            "net_yield_per_period": 0.0005,
            "annualized_yield": 0.5475,
        }
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[simple_opp])

        import networkx as nx
        mock_graph = nx.DiGraph()
        orch.optimizer.build_graph = MagicMock(return_value=mock_graph)
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        assert orch._iteration == 1
        orch.scanner.scan.assert_awaited_once()
        orch.database.save_funding_rates_batch.assert_called_once()
        saved_rates = orch.database.save_funding_rates_batch.call_args[0][0]
        assert len(saved_rates) == 2
        orch.alerter.notify_opportunity.assert_awaited_once_with(simple_opp)
        orch.alerter.notify_scan_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_increments_iteration(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=_make_snapshot())
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])
        orch.database.save_funding_rates_batch = MagicMock()

        await orch._tick()
        await orch._tick()

        assert orch._iteration == 2

    @pytest.mark.asyncio
    async def test_tick_only_notifies_first_simple_opp(self, tmp_path):
        """Only the first simple opportunity triggers an alert."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={"binance": {"BTC/USDT:USDT": 0.002}},
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        opp1 = {"instrument": "BTC", "short_exchange": "a", "long_exchange": "b",
                 "short_rate": 0.01, "long_rate": 0.001, "net_yield_per_period": 0.005,
                 "annualized_yield": 5.0}
        opp2 = {"instrument": "ETH", "short_exchange": "a", "long_exchange": "b",
                 "short_rate": 0.005, "long_rate": 0.001, "net_yield_per_period": 0.003,
                 "annualized_yield": 3.0}
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[opp1, opp2])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        orch.alerter.notify_opportunity.assert_awaited_once_with(opp1)

    @pytest.mark.asyncio
    async def test_tick_sizes_graph_opportunities(self, tmp_path):
        """When graph opportunities exist and sizing > 0, the path is exercised."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])

        opp = _make_opportunity()
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[opp])

        # Give portfolio equity so Kelly sizing returns > 0
        orch.portfolio.equity_by_exchange = {"binance": 100_000}
        orch.risk_manager.calculate_position_size = MagicMock(return_value=500.0)

        await orch._tick()

        orch.risk_manager.calculate_position_size.assert_called_once_with(opp, orch.portfolio)

    @pytest.mark.asyncio
    async def test_tick_no_rates_skips_db_save(self, tmp_path):
        """When snapshot has no rates, save_funding_rates_batch is NOT called."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot()  # no exchanges, no rates
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        orch.database.save_funding_rates_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_updates_portfolio_peak(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.portfolio.equity_by_exchange = {"binance": 50_000}

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.0001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        assert orch.portfolio.peak_equity == 50_000


# ---------------------------------------------------------------------------
# _tick  --  stale exchanges
# ---------------------------------------------------------------------------


class TestTickStaleExchanges:
    @pytest.mark.asyncio
    async def test_tick_alerts_on_stale_exchange(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={"binance": {"BTC/USDT:USDT": 0.001}},
            stale={"bybit": "Connection timeout"},
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        orch.alerter.notify_scan_failure.assert_awaited_once_with("bybit", "Connection timeout")

    @pytest.mark.asyncio
    async def test_tick_alerts_multiple_stale_exchanges(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            stale={"bybit": "timeout", "okx": "rate limited"},
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        assert orch.alerter.notify_scan_failure.await_count == 2


# ---------------------------------------------------------------------------
# _tick  --  critical violations
# ---------------------------------------------------------------------------


class TestTickCriticalViolations:
    @pytest.mark.asyncio
    async def test_tick_skips_trading_on_critical_violation(self, tmp_path):
        """When a critical violation exists, tick returns early before finding opportunities."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        critical_violation = Violation(
            type=ViolationType.EMERGENCY_DRAWDOWN,
            message="Drawdown exceeded",
            severity="critical",
        )
        orch.risk_manager.check_invariants = MagicMock(return_value=[critical_violation])

        # These should NOT be called because tick returns early
        orch.optimizer.find_simple_opportunities = MagicMock()
        orch.optimizer.build_graph = MagicMock()
        orch.optimizer.find_opportunities = MagicMock()

        await orch._tick()

        orch.alerter.notify_risk_violation.assert_awaited_once_with(critical_violation)
        orch.optimizer.find_simple_opportunities.assert_not_called()
        orch.optimizer.build_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_continues_on_warning_violation(self, tmp_path):
        """Non-critical (warning) violations are reported but do not block trading."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        warning_violation = Violation(
            type=ViolationType.POSITION_CONCENTRATION,
            message="Position too large",
            severity="warning",
        )
        orch.risk_manager.check_invariants = MagicMock(return_value=[warning_violation])
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        orch.alerter.notify_risk_violation.assert_awaited_once_with(warning_violation)
        # Trading pipeline was NOT short-circuited
        orch.optimizer.find_simple_opportunities.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_mixed_violations_critical_blocks(self, tmp_path):
        """If any violation is critical (even among warnings), trading is blocked."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        violations = [
            Violation(type=ViolationType.POSITION_CONCENTRATION, message="warn", severity="warning"),
            Violation(type=ViolationType.LOW_COLLATERAL, message="critical", severity="critical"),
        ]
        orch.risk_manager.check_invariants = MagicMock(return_value=violations)
        orch.optimizer.find_simple_opportunities = MagicMock()

        await orch._tick()

        assert orch.alerter.notify_risk_violation.await_count == 2
        orch.optimizer.find_simple_opportunities.assert_not_called()


# ---------------------------------------------------------------------------
# _tick  --  regime detection
# ---------------------------------------------------------------------------


class TestTickRegimeDetection:
    @pytest.mark.asyncio
    async def test_tick_negative_regime_tightens_risk(self, tmp_path):
        """When dominant regime is NEGATIVE, risk is adjusted to 'high'."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": -0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        # Pre-seed 6 entries of history so regime detection fires
        orch._funding_history["binance:BTC/USDT:USDT"] = [-0.001] * 6

        orch.predictor.classify_regime = MagicMock(
            return_value=RegimeState(current_regime=FundingRegime.NEGATIVE)
        )
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        orch.risk_manager.adjust_for_regime.assert_called_with("high")

    @pytest.mark.asyncio
    async def test_tick_high_positive_regime_loosens_risk(self, tmp_path):
        """When dominant regime is HIGH_POSITIVE, risk is adjusted to 'low'."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.005}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        orch._funding_history["binance:BTC/USDT:USDT"] = [0.005] * 6

        orch.predictor.classify_regime = MagicMock(
            return_value=RegimeState(current_regime=FundingRegime.HIGH_POSITIVE)
        )
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        orch.risk_manager.adjust_for_regime.assert_called_with("low")

    @pytest.mark.asyncio
    async def test_tick_near_zero_regime_normal_risk(self, tmp_path):
        """NEAR_ZERO and LOW_POSITIVE regimes map to 'normal'."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.00001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        orch._funding_history["binance:BTC/USDT:USDT"] = [0.00001] * 6

        orch.predictor.classify_regime = MagicMock(
            return_value=RegimeState(current_regime=FundingRegime.NEAR_ZERO)
        )
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        orch.risk_manager.adjust_for_regime.assert_called_with("normal")

    @pytest.mark.asyncio
    async def test_tick_low_positive_regime_normal_risk(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.0001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        orch._funding_history["binance:BTC/USDT:USDT"] = [0.0001] * 6

        orch.predictor.classify_regime = MagicMock(
            return_value=RegimeState(current_regime=FundingRegime.LOW_POSITIVE)
        )
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        orch.risk_manager.adjust_for_regime.assert_called_with("normal")

    @pytest.mark.asyncio
    async def test_tick_regime_requires_min_history(self, tmp_path):
        """Regime detection only runs when there are >= 6 history entries for a key."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        # Only 4 history entries -- below the >= 6 threshold
        orch._funding_history["binance:BTC/USDT:USDT"] = [0.001] * 4

        orch.predictor.classify_regime = MagicMock()
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        # After tick, history has 5 entries (4 pre-seeded + 1 from scan).
        # Still below 6, so classify_regime should NOT be called.
        orch.predictor.classify_regime.assert_not_called()
        orch.risk_manager.adjust_for_regime.assert_not_called()


# ---------------------------------------------------------------------------
# _tick  --  funding history management
# ---------------------------------------------------------------------------


class TestTickFundingHistory:
    @pytest.mark.asyncio
    async def test_tick_appends_to_funding_history(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.002}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        assert "binance:BTC/USDT:USDT" in orch._funding_history
        assert orch._funding_history["binance:BTC/USDT:USDT"] == [0.002]

    @pytest.mark.asyncio
    async def test_tick_caps_funding_history_at_100(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.999}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        # Pre-fill with 100 entries
        orch._funding_history["binance:BTC/USDT:USDT"] = [0.001] * 100

        await orch._tick()

        history = orch._funding_history["binance:BTC/USDT:USDT"]
        assert len(history) == 100
        # The newest entry (0.999) should be at the end
        assert history[-1] == 0.999
        # The oldest original entry was trimmed
        assert history[0] == 0.001


# ---------------------------------------------------------------------------
# _tick  --  exception handling
# ---------------------------------------------------------------------------


class TestTickExceptionHandling:
    @pytest.mark.asyncio
    async def test_tick_catches_scan_exception(self, tmp_path, caplog):
        """If scanner.scan() raises, the tick is caught and logged, not propagated."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(side_effect=RuntimeError("network down"))
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()

        with caplog.at_level(logging.ERROR, logger="funding_arb.orchestrator"):
            await orch._tick()

        assert orch._iteration == 1
        assert "Tick 1 failed" in caplog.text

    @pytest.mark.asyncio
    async def test_tick_catches_optimizer_exception(self, tmp_path, caplog):
        """If optimizer raises mid-tick, the exception is caught."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(
            side_effect=ValueError("graph construction failed")
        )

        with caplog.at_level(logging.ERROR, logger="funding_arb.orchestrator"):
            await orch._tick()

        assert "Tick 1 failed" in caplog.text


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_run_once_returns_results(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={
                "binance": {"BTC/USDT:USDT": 0.003, "ETH/USDT:USDT": 0.001},
                "bybit": {"BTC/USDT:USDT": 0.0005},
            },
        )
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.scanner.close = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        result = await orch.run_once()

        assert result["snapshot"] is snapshot
        assert isinstance(result["simple_opportunities"], list)
        assert isinstance(result["graph_opportunities"], list)
        assert isinstance(result["violations"], list)
        orch.scanner.initialize.assert_awaited_once()
        orch.scanner.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_once_closes_scanner_on_error(self, tmp_path):
        """Scanner is closed even when scan() raises."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.scan = AsyncMock(side_effect=RuntimeError("scan fail"))
        orch.scanner.close = AsyncMock()

        with pytest.raises(RuntimeError, match="scan fail"):
            await orch.run_once()

        orch.scanner.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_once_saves_rates_to_db(self, tmp_path):
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={"binance": {"BTC/USDT:USDT": 0.002}},
        )
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.scanner.close = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        await orch.run_once()

        orch.database.save_funding_rates_batch.assert_called_once()
        saved = orch.database.save_funding_rates_batch.call_args[0][0]
        assert len(saved) == 1
        assert saved[0].exchange == "binance"
        assert saved[0].rate == 0.002

    @pytest.mark.asyncio
    async def test_run_once_empty_snapshot_skips_save(self, tmp_path):
        """No rates means save_funding_rates_batch is not called."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot()
        orch.scanner = MagicMock()
        orch.scanner.initialize = AsyncMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.scanner.close = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()

        result = await orch.run_once()

        orch.database.save_funding_rates_batch.assert_not_called()
        assert result["simple_opportunities"] == []
        assert result["graph_opportunities"] == []


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    @pytest.mark.asyncio
    async def test_main_creates_and_starts_orchestrator(self, tmp_path):
        fake_config = _make_config(tmp_path)

        with (
            patch("funding_arb.orchestrator.load_config", return_value=fake_config),
            patch("funding_arb.orchestrator.Orchestrator") as MockOrch,
        ):
            mock_instance = MockOrch.return_value
            mock_instance.start = AsyncMock()

            await main()

            MockOrch.assert_called_once_with(fake_config)
            mock_instance.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# Edge cases and integration-like scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_tick_multiple_exchanges_multiple_instruments(self, tmp_path):
        """Several exchanges each with several instruments produces correct history keys."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(
            exchanges={
                "binance": {"BTC/USDT:USDT": 0.001, "ETH/USDT:USDT": 0.002},
                "bybit": {"BTC/USDT:USDT": -0.0005, "SOL/USDT:USDT": 0.003},
            },
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        expected_keys = {
            "binance:BTC/USDT:USDT",
            "binance:ETH/USDT:USDT",
            "bybit:BTC/USDT:USDT",
            "bybit:SOL/USDT:USDT",
        }
        assert set(orch._funding_history.keys()) == expected_keys

        # Each key should have exactly one entry from the single tick
        for key in expected_keys:
            assert len(orch._funding_history[key]) == 1

    @pytest.mark.asyncio
    async def test_tick_no_simple_no_graph_opps(self, tmp_path):
        """When there are zero opportunities of any kind, no alerts fire for opportunities."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.0001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        await orch._tick()

        orch.alerter.notify_opportunity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tick_graph_opp_size_zero_no_log(self, tmp_path):
        """When position sizing returns 0, the 'Would open position' path is skipped."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        snapshot = _make_snapshot(exchanges={"binance": {"BTC/USDT:USDT": 0.001}})
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())

        opp = _make_opportunity(yield_val=0.0, variance=0.0)
        orch.optimizer.find_opportunities = MagicMock(return_value=[opp])
        orch.risk_manager.calculate_position_size = MagicMock(return_value=0.0)

        await orch._tick()

        orch.risk_manager.calculate_position_size.assert_called_once()

    @pytest.mark.asyncio
    async def test_regime_detection_dominant_by_count(self, tmp_path):
        """Dominant regime is the one with the most instrument-keys."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        # Two instruments: one NEGATIVE, one HIGH_POSITIVE
        snapshot = _make_snapshot(
            exchanges={
                "binance": {"BTC/USDT:USDT": -0.001, "ETH/USDT:USDT": 0.005},
                "bybit": {"BTC/USDT:USDT": -0.002},
            },
        )
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        # Pre-seed all three keys with >= 6 entries
        orch._funding_history["binance:BTC/USDT:USDT"] = [-0.001] * 6
        orch._funding_history["binance:ETH/USDT:USDT"] = [0.005] * 6
        orch._funding_history["bybit:BTC/USDT:USDT"] = [-0.002] * 6

        # Two of three keys are NEGATIVE, so NEGATIVE is dominant
        def side_effect_classify(series):
            val = float(series.iloc[-1])
            if val < -0.00005:
                return RegimeState(current_regime=FundingRegime.NEGATIVE)
            return RegimeState(current_regime=FundingRegime.HIGH_POSITIVE)

        orch.predictor.classify_regime = MagicMock(side_effect=side_effect_classify)
        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        # NEGATIVE has 2 keys, HIGH_POSITIVE has 1 --> dominant is NEGATIVE --> "high"
        orch.risk_manager.adjust_for_regime.assert_called_with("high")

    @pytest.mark.asyncio
    async def test_no_regime_adjustment_when_no_regime_counts(self, tmp_path):
        """If regime_counts is empty (no keys with >= 6 entries), no adjustment is made."""
        config = _make_config(tmp_path)
        orch = Orchestrator(config)

        # Empty snapshot means no funding history entries are added
        snapshot = _make_snapshot()
        orch.scanner = MagicMock()
        orch.scanner.scan = AsyncMock(return_value=snapshot)
        orch.alerter = MagicMock()
        orch.alerter.notify_scan_failure = AsyncMock()
        orch.alerter.notify_risk_violation = AsyncMock()
        orch.alerter.notify_opportunity = AsyncMock()
        orch.database.save_funding_rates_batch = MagicMock()
        orch.optimizer.find_simple_opportunities = MagicMock(return_value=[])
        import networkx as nx
        orch.optimizer.build_graph = MagicMock(return_value=nx.DiGraph())
        orch.optimizer.find_opportunities = MagicMock(return_value=[])

        orch.risk_manager.adjust_for_regime = MagicMock()

        await orch._tick()

        orch.risk_manager.adjust_for_regime.assert_not_called()
