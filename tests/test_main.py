"""Tests for the __main__ CLI entry point."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from funding_arb.__main__ import main
from funding_arb.models import MarketSnapshot


def _make_scan_result(
    num_exchanges=2,
    num_simple=1,
    num_graph=0,
    num_violations=0,
):
    """Build a mock run_once() return dict."""
    snapshot = MarketSnapshot()
    # Populate exchange_data so .exchanges and .instruments work
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


class TestMainScanOnce:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_once_basic(self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
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
    def test_scan_once_prints_opportunity_details(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
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
    def test_scan_once_limits_to_5_opps(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        """The print loop slices [:5], so 7 opps should only print 5 detail lines."""
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_simple=7)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Simple opportunities: 7" in output
        # Count the detail lines (they start with "  BTC/USDT:USDT:")
        detail_lines = [line for line in output.splitlines() if line.strip().startswith("BTC/USDT:USDT")]
        assert len(detail_lines) == 5

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_scan_once_zero_opportunities(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
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
    def test_scan_once_with_stale_exchanges(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
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
    def test_scan_once_with_violations(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--scan-once"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result(num_violations=3)
        mock_orch.run_once = AsyncMock(return_value=result)

        main()

        output = capsys.readouterr().out
        assert "Risk violations: 3" in output


class TestMainContinuousMode:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_continuous_mode(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
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


class TestMainConfigFlag:
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
    def test_default_config(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        main()

        mock_load_config.assert_called_once_with("config.toml")


class TestMainLogLevel:
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
            call_kwargs = mock_basic.call_args
            assert call_kwargs[1]["level"] == logging.DEBUG

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_log_level_warning(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--log-level", "WARNING"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            assert mock_basic.call_args[1]["level"] == logging.WARNING

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_log_level_error(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb", "--log-level", "ERROR"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            assert mock_basic.call_args[1]["level"] == logging.ERROR

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

    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_log_format_string(self, mock_load_config, mock_orchestrator_cls, monkeypatch):
        monkeypatch.setattr("sys.argv", ["funding-arb"])
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.start = AsyncMock()

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            fmt = mock_basic.call_args[1]["format"]
            assert "%(levelname)s" in fmt
            assert "%(name)s" in fmt


class TestMainCombinedFlags:
    @patch("funding_arb.__main__.Orchestrator")
    @patch("funding_arb.__main__.load_config")
    def test_config_and_scan_once(
        self, mock_load_config, mock_orchestrator_cls, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "sys.argv", ["funding-arb", "-c", "prod.toml", "--scan-once", "--log-level", "DEBUG"]
        )
        mock_load_config.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        result = _make_scan_result()
        mock_orch.run_once = AsyncMock(return_value=result)

        with patch("funding_arb.__main__.logging.basicConfig") as mock_basic:
            main()
            mock_load_config.assert_called_once_with("prod.toml")
            mock_orch.run_once.assert_called_once()
            mock_orch.start.assert_not_called()
            assert mock_basic.call_args[1]["level"] == logging.DEBUG
