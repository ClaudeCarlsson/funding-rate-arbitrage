"""Tests for configuration management."""
from __future__ import annotations

import os
import textwrap

import pytest

from funding_arb.config import (
    Config,
    DatabaseConfig,
    ExchangeConfig,
    ExecutorConfig,
    OptimizerConfig,
    RiskConfig,
    ScannerConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# ExchangeConfig
# ---------------------------------------------------------------------------

class TestExchangeConfig:
    def test_defaults(self):
        ec = ExchangeConfig(name="binance")
        assert ec.name == "binance"
        assert ec.enabled is True
        assert ec.api_key_env == ""
        assert ec.api_secret_env == ""
        assert ec.sandbox is False
        assert ec.rate_limit_ms == 100
        assert ec.max_position_usd == 10_000.0

    def test_api_key_empty_env(self):
        """No env var name configured -> empty string."""
        ec = ExchangeConfig(name="binance")
        assert ec.api_key == ""

    def test_api_secret_empty_env(self):
        """No env var name configured -> empty string."""
        ec = ExchangeConfig(name="binance")
        assert ec.api_secret == ""

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my-api-key-123")
        ec = ExchangeConfig(name="binance", api_key_env="TEST_KEY")
        assert ec.api_key == "my-api-key-123"

    def test_api_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET", "my-secret-456")
        ec = ExchangeConfig(name="binance", api_secret_env="TEST_SECRET")
        assert ec.api_secret == "my-secret-456"

    def test_api_key_env_not_set(self, monkeypatch):
        """Env var name configured but the variable is not set -> empty string."""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        ec = ExchangeConfig(name="binance", api_key_env="NONEXISTENT_KEY")
        assert ec.api_key == ""

    def test_api_secret_env_not_set(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_SECRET", raising=False)
        ec = ExchangeConfig(name="binance", api_secret_env="NONEXISTENT_SECRET")
        assert ec.api_secret == ""

    def test_frozen(self):
        ec = ExchangeConfig(name="binance")
        with pytest.raises(AttributeError):
            ec.name = "bybit"  # type: ignore[misc]

    def test_custom_values(self):
        ec = ExchangeConfig(
            name="bybit",
            enabled=False,
            api_key_env="BK",
            api_secret_env="BS",
            sandbox=True,
            rate_limit_ms=200,
            max_position_usd=5000.0,
        )
        assert ec.name == "bybit"
        assert ec.enabled is False
        assert ec.sandbox is True
        assert ec.rate_limit_ms == 200
        assert ec.max_position_usd == 5000.0


# ---------------------------------------------------------------------------
# Other config dataclasses - defaults
# ---------------------------------------------------------------------------

class TestRiskConfig:
    def test_defaults(self):
        rc = RiskConfig()
        assert rc.max_delta_pct == 0.02
        assert rc.max_position_pct == 0.20
        assert rc.max_exchange_pct == 0.30
        assert rc.min_collateral_ratio == 2.0
        assert rc.max_drawdown == 0.05
        assert rc.max_gross_leverage == 3.0
        assert rc.correlation_floor == 0.95
        assert rc.kelly_fraction == 0.25

    def test_frozen(self):
        rc = RiskConfig()
        with pytest.raises(AttributeError):
            rc.max_delta_pct = 0.5  # type: ignore[misc]


class TestScannerConfig:
    def test_defaults(self):
        sc = ScannerConfig()
        assert sc.poll_interval_s == 60.0
        assert sc.order_book_depth == 5
        assert sc.staleness_threshold_s == 300.0
        assert sc.instruments == [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
        ]

    def test_custom_instruments(self):
        sc = ScannerConfig(instruments=["DOGE/USDT:USDT"])
        assert sc.instruments == ["DOGE/USDT:USDT"]


class TestOptimizerConfig:
    def test_defaults(self):
        oc = OptimizerConfig()
        assert oc.min_net_yield_bps == 5.0
        assert oc.recompute_threshold_pct == 0.10
        assert oc.max_cycles_to_evaluate == 50


class TestExecutorConfig:
    def test_defaults(self):
        ec = ExecutorConfig()
        assert ec.paper_trading is True
        assert ec.limit_order_timeout_s == 5.0
        assert ec.max_slippage_pct == 0.5
        assert ec.emergency_unwind_enabled is True


class TestDatabaseConfig:
    def test_defaults(self):
        dc = DatabaseConfig()
        assert dc.state_db_path == "data/state.db"
        assert dc.trades_db_path == "data/trades.db"
        assert dc.funding_db_path == "data/funding.db"
        assert dc.parquet_dir == "data/parquet"


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.exchanges == {}
        assert isinstance(cfg.risk, RiskConfig)
        assert isinstance(cfg.scanner, ScannerConfig)
        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert isinstance(cfg.executor, ExecutorConfig)
        assert isinstance(cfg.database, DatabaseConfig)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.exchanges == {}
        assert cfg.risk == RiskConfig()
        assert cfg.scanner == ScannerConfig()
        assert cfg.optimizer == OptimizerConfig()
        assert cfg.executor == ExecutorConfig()
        assert cfg.database == DatabaseConfig()

    def test_full_config(self, tmp_path):
        toml_content = textwrap.dedent("""\
            [scanner]
            poll_interval_s = 30.0
            order_book_depth = 10
            staleness_threshold_s = 120.0
            instruments = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

            [risk]
            max_delta_pct = 0.01
            max_position_pct = 0.15
            max_exchange_pct = 0.25
            min_collateral_ratio = 3.0
            max_drawdown = 0.03
            max_gross_leverage = 2.0
            correlation_floor = 0.90
            kelly_fraction = 0.50

            [optimizer]
            min_net_yield_bps = 10.0
            recompute_threshold_pct = 0.05
            max_cycles_to_evaluate = 100

            [executor]
            paper_trading = false
            limit_order_timeout_s = 10.0
            max_slippage_pct = 0.3
            emergency_unwind_enabled = false

            [database]
            state_db_path = "custom/state.db"
            trades_db_path = "custom/trades.db"
            funding_db_path = "custom/funding.db"
            parquet_dir = "custom/parquet"

            [exchanges.binance]
            enabled = true
            api_key_env = "BN_KEY"
            api_secret_env = "BN_SECRET"
            sandbox = false
            rate_limit_ms = 50
            max_position_usd = 20000.0

            [exchanges.bybit]
            enabled = false
            api_key_env = "BB_KEY"
            api_secret_env = "BB_SECRET"
            sandbox = true
            rate_limit_ms = 150
            max_position_usd = 8000.0
        """)
        config_path = tmp_path / "config.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)

        # Scanner
        assert cfg.scanner.poll_interval_s == 30.0
        assert cfg.scanner.order_book_depth == 10
        assert cfg.scanner.staleness_threshold_s == 120.0
        assert cfg.scanner.instruments == ["BTC/USDT:USDT", "ETH/USDT:USDT"]

        # Risk
        assert cfg.risk.max_delta_pct == 0.01
        assert cfg.risk.max_position_pct == 0.15
        assert cfg.risk.max_exchange_pct == 0.25
        assert cfg.risk.min_collateral_ratio == 3.0
        assert cfg.risk.max_drawdown == 0.03
        assert cfg.risk.max_gross_leverage == 2.0
        assert cfg.risk.correlation_floor == 0.90
        assert cfg.risk.kelly_fraction == 0.50

        # Optimizer
        assert cfg.optimizer.min_net_yield_bps == 10.0
        assert cfg.optimizer.recompute_threshold_pct == 0.05
        assert cfg.optimizer.max_cycles_to_evaluate == 100

        # Executor
        assert cfg.executor.paper_trading is False
        assert cfg.executor.limit_order_timeout_s == 10.0
        assert cfg.executor.max_slippage_pct == 0.3
        assert cfg.executor.emergency_unwind_enabled is False

        # Database
        assert cfg.database.state_db_path == "custom/state.db"
        assert cfg.database.trades_db_path == "custom/trades.db"
        assert cfg.database.funding_db_path == "custom/funding.db"
        assert cfg.database.parquet_dir == "custom/parquet"

        # Exchanges
        assert "binance" in cfg.exchanges
        assert "bybit" in cfg.exchanges
        bn = cfg.exchanges["binance"]
        assert bn.name == "binance"
        assert bn.enabled is True
        assert bn.api_key_env == "BN_KEY"
        assert bn.sandbox is False
        assert bn.rate_limit_ms == 50
        assert bn.max_position_usd == 20_000.0
        bb = cfg.exchanges["bybit"]
        assert bb.name == "bybit"
        assert bb.enabled is False
        assert bb.sandbox is True
        assert bb.rate_limit_ms == 150

    def test_partial_config_scanner_only(self, tmp_path):
        toml_content = textwrap.dedent("""\
            [scanner]
            poll_interval_s = 45.0
        """)
        config_path = tmp_path / "partial.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)

        # Scanner overridden
        assert cfg.scanner.poll_interval_s == 45.0
        # Scanner defaults preserved where not set
        assert cfg.scanner.order_book_depth == 5
        # Other sections use full defaults
        assert cfg.risk == RiskConfig()
        assert cfg.exchanges == {}

    def test_partial_config_exchanges_only(self, tmp_path):
        toml_content = textwrap.dedent("""\
            [exchanges.okx]
            enabled = true
            api_key_env = "OKX_KEY"
            api_secret_env = "OKX_SECRET"
            sandbox = false
            rate_limit_ms = 120
            max_position_usd = 7000.0
        """)
        config_path = tmp_path / "exonly.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert len(cfg.exchanges) == 1
        assert cfg.exchanges["okx"].name == "okx"
        assert cfg.exchanges["okx"].max_position_usd == 7000.0
        # Others are defaults
        assert cfg.scanner == ScannerConfig()

    def test_scanner_without_instruments(self, tmp_path):
        """Scanner section present but no instruments key -> default instruments."""
        toml_content = textwrap.dedent("""\
            [scanner]
            poll_interval_s = 90.0
            order_book_depth = 20
        """)
        config_path = tmp_path / "noinstr.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert cfg.scanner.poll_interval_s == 90.0
        assert cfg.scanner.order_book_depth == 20
        # Default instruments since none specified
        assert cfg.scanner.instruments == ScannerConfig().instruments

    def test_scanner_with_instruments(self, tmp_path):
        """Instruments explicitly provided in scanner section."""
        toml_content = textwrap.dedent("""\
            [scanner]
            instruments = ["DOGE/USDT:USDT", "XRP/USDT:USDT"]
        """)
        config_path = tmp_path / "instr.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert cfg.scanner.instruments == ["DOGE/USDT:USDT", "XRP/USDT:USDT"]

    def test_empty_toml_file(self, tmp_path):
        """An empty TOML file is valid and produces defaults."""
        config_path = tmp_path / "empty.toml"
        config_path.write_text("")

        cfg = load_config(config_path)
        assert cfg.exchanges == {}
        assert cfg.risk == RiskConfig()

    def test_multiple_exchanges(self, tmp_path):
        toml_content = textwrap.dedent("""\
            [exchanges.binance]
            enabled = true
            api_key_env = "BN_K"
            api_secret_env = "BN_S"
            sandbox = false
            rate_limit_ms = 100
            max_position_usd = 10000.0

            [exchanges.bybit]
            enabled = true
            api_key_env = "BB_K"
            api_secret_env = "BB_S"
            sandbox = true
            rate_limit_ms = 200
            max_position_usd = 5000.0

            [exchanges.okx]
            enabled = false
            api_key_env = "OKX_K"
            api_secret_env = "OKX_S"
            sandbox = false
            rate_limit_ms = 80
            max_position_usd = 15000.0
        """)
        config_path = tmp_path / "multi.toml"
        config_path.write_text(toml_content)

        cfg = load_config(config_path)
        assert len(cfg.exchanges) == 3
        assert cfg.exchanges["binance"].enabled is True
        assert cfg.exchanges["bybit"].sandbox is True
        assert cfg.exchanges["okx"].enabled is False

    def test_load_config_path_as_string(self, tmp_path):
        """load_config accepts a plain string path."""
        config_path = tmp_path / "strpath.toml"
        config_path.write_text("[optimizer]\nmin_net_yield_bps = 42.0\n")

        cfg = load_config(str(config_path))
        assert cfg.optimizer.min_net_yield_bps == 42.0
