"""Tests for the __main__ CLI entry point."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from funding_arb.__main__ import main, _handle_kill_switch
from funding_arb.executor import KILL_SWITCH_PATH
from funding_arb.models import MarketSnapshot


def _make_scan_result(
    num_exchanges=2,
    num_simple=1,
    num_graph=0,
    num_violations=0,
):
    """Build a mock run_once() return dict."""
    snapshot = MarketSnapshot()
    from funding_arb.models import ExchangeData, FundingRate
    from datetime import datetime, timezone

    for i in range(num_exchanges):
        name = f"exchange_{i}"
        ed = ExchangeData()
        ed.rates["BTC/USDT:USDT"] = FundingRate(
            exchange=name, symbol="BTC/USDT:USDT", rate=0.0003,
            timestamp=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        )
        snapshot.update(name, ed)

    simple_opps = []
    for i in range(num_simple):
        simple_opps.append({
            "instrument": "BTC/USDT:USDT",
            "short_exchange": "binance",
            "short_rate": 0.000500,
            "long_exchange": "okx",
            "long_rate": -0.000100,
            "net_yield_per_period": 0.000600,
            "annualized_yield": 0.657,
        })

    graph_opps = [MagicMock() for _ in range(num_graph)]
    violations = [MagicMock() for _ in range(num_violations)]

    return {
        "snapshot": snapshot,
        "simple_opportunities": simple_opps,
        "graph_opportunities": graph_opps,
        "violations": violations,
    }


class TestScanCommand:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_basic(self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result()
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        mock_load_config.assert_called_once_with("config.toml")
        mock_orchestrator_cls.assert_called_once_with(mock_config)
        mock_orch.run_once.assert_called_once()
        output = capsys.readouterr().out
        assert "Exchanges scanned: 2" in output
        assert "Simple opportunities: 1" in output
        assert "Graph opportunities: 0" in output
        assert "Risk violations: 0" in output

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_prints_opportunity_details(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_simple=3)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Simple opportunities: 3" in output
        assert "BTC/USDT:USDT" in output
        assert "short binance" in output
        assert "long okx" in output
        assert "APR" in output

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_limits_to_5_opps(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_simple=7)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Simple opportunities: 7" in output
        detail_lines = [line for line in output.splitlines() if line.strip().startswith("BTC/USDT:USDT")]
        assert len(detail_lines) == 5

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_zero_opportunities(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_simple=0, num_graph=0, num_violations=0)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Simple opportunities: 0" in output
        assert "Graph opportunities: 0" in output

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_with_stale_exchanges(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result()
        result["snapshot"].mark_stale("kraken", "timeout")
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Stale:" in output
        assert "kraken" in output

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_with_violations(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_violations=3)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Risk violations: 3" in output


class TestRunCommand:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_default_runs_main_loop(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb"])
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        main()

        mock_load_config.assert_called_once_with("config.toml")
        mock_orchestrator_cls.assert_called_once_with(mock_config)
        mock_orch.start.assert_called_once()

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_explicit_run_command(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "run"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        main()

        mock_orch.start.assert_called_once()


class TestConfigFlag:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_custom_config_path_long(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--config", "custom.toml"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        main()

        mock_load_config.assert_called_once_with("custom.toml")

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_custom_config_path_short(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "-c", "/etc/arb/prod.toml"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        main()

        mock_load_config.assert_called_once_with("/etc/arb/prod.toml")

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_config_with_scan(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "-c", "prod.toml", "scan"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result()
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        mock_load_config.assert_called_once_with("prod.toml")
        mock_orch.run_once.assert_called_once()
        mock_orch.start.assert_not_called()


class TestLogLevel:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_log_level_debug(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--log-level", "DEBUG"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            mock_basic.assert_called_once()
            assert mock_basic.call_args[1]["level"] == logging.DEBUG

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_log_level_default_info(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            assert mock_basic.call_args[1]["level"] == logging.INFO


class TestKillSwitch:
    def test_kill_on(self, capsys):
        _handle_kill_switch("on")
        output = capsys.readouterr().out
        assert "ACTIVATED" in output
        KILL_SWITCH_PATH.unlink(missing_ok=True)

    def test_kill_off(self, capsys):
        KILL_SWITCH_PATH.touch()
        _handle_kill_switch("off")
        output = capsys.readouterr().out
        assert "DEACTIVATED" in output
        assert not KILL_SWITCH_PATH.exists()

    def test_kill_status_off(self, capsys):
        KILL_SWITCH_PATH.unlink(missing_ok=True)
        _handle_kill_switch("status")
        output = capsys.readouterr().out
        assert "OFF" in output

    def test_kill_status_on(self, capsys):
        KILL_SWITCH_PATH.touch()
        _handle_kill_switch("status")
        output = capsys.readouterr().out
        assert "ON" in output
        KILL_SWITCH_PATH.unlink(missing_ok=True)

    def test_kill_command_via_cli(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["funding-arb", "kill", "status"])
        KILL_SWITCH_PATH.unlink(missing_ok=True)
        main()
        output = capsys.readouterr().out
        assert "OFF" in output


class TestStatusCommand:
    @patch("funding_arb.__main__._show_status")
    @patch("funding_arb.__main__.load_config")
    def test_status_command(self, mock_load_config, mock_show_status, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "status"])
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        main()

        mock_show_status.assert_called_once_with(mock_config)


class TestBackupCommand:
    @patch("funding_arb.__main__.backup_all")
    @patch("funding_arb.__main__.load_config")
    def test_backup_command(self, mock_load_config, mock_backup_all, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["funding-arb", "backup"])
        mock_config = MagicMock()
        mock_config.database.state_db_path = "data/state.db"
        mock_config.database.trades_db_path = "data/trades.db"
        mock_config.database.funding_db_path = "data/funding.db"
        mock_load_config.return_value = mock_config

        from pathlib import Path
        mock_backup_all.return_value = [Path("data/backups/state_20260308.db")]

        main()

        mock_backup_all.assert_called_once_with(
            state_db="data/state.db",
            trades_db="data/trades.db",
            funding_db="data/funding.db",
            backup_dir="data/backups",
            max_backups=7,
        )
        output = capsys.readouterr().out
        assert "Backed up" in output

    @patch("funding_arb.__main__.backup_all")
    @patch("funding_arb.__main__.load_config")
    def test_backup_custom_dir_and_keep(
        self, mock_load_config, mock_backup_all, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "sys.argv", ["funding-arb", "backup", "--dir", "/mnt/bk", "--keep", "3"]
        )
        mock_config = MagicMock()
        mock_config.database.state_db_path = "data/state.db"
        mock_config.database.trades_db_path = "data/trades.db"
        mock_config.database.funding_db_path = "data/funding.db"
        mock_load_config.return_value = mock_config
        mock_backup_all.return_value = []

        main()

        mock_backup_all.assert_called_once_with(
            state_db="data/state.db",
            trades_db="data/trades.db",
            funding_db="data/funding.db",
            backup_dir="/mnt/bk",
            max_backups=3,
        )
        output = capsys.readouterr().out
        assert "No databases found" in output
