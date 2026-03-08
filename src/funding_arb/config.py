"""Configuration management for the funding rate arbitrage system."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExchangeConfig:
    name: str
    enabled: bool = True
    api_key_env: str = ""
    api_secret_env: str = ""
    sandbox: bool = False
    rate_limit_ms: int = 100
    max_position_usd: float = 10_000.0

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env, "")

    @property
    def api_secret(self) -> str:
        if not self.api_secret_env:
            return ""
        return os.environ.get(self.api_secret_env, "")


@dataclass(frozen=True)
class RiskConfig:
    max_delta_pct: float = 0.02
    max_position_pct: float = 0.20
    max_exchange_pct: float = 0.30
    min_collateral_ratio: float = 2.0
    max_drawdown: float = 0.05
    max_gross_leverage: float = 3.0
    correlation_floor: float = 0.95
    kelly_fraction: float = 0.25


@dataclass(frozen=True)
class ScannerConfig:
    poll_interval_s: float = 60.0
    order_book_depth: int = 5
    staleness_threshold_s: float = 300.0
    instruments: list[str] = field(default_factory=lambda: [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    ])


@dataclass(frozen=True)
class OptimizerConfig:
    min_net_yield_bps: float = 5.0  # minimum 0.05% per period
    recompute_threshold_pct: float = 0.10  # 10% relative change triggers recompute
    max_cycles_to_evaluate: int = 50
    max_position_usd: float = 10_000.0


@dataclass(frozen=True)
class ExecutorConfig:
    paper_trading: bool = True
    limit_order_timeout_s: float = 5.0
    max_slippage_pct: float = 0.5
    emergency_unwind_enabled: bool = True


@dataclass(frozen=True)
class DatabaseConfig:
    state_db_path: str = "data/state.db"
    trades_db_path: str = "data/trades.db"
    funding_db_path: str = "data/funding.db"
    parquet_dir: str = "data/parquet"


@dataclass(frozen=True)
class Config:
    exchanges: dict[str, ExchangeConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


def load_config(path: str | Path = "config.toml") -> Config:
    """Load configuration from TOML file with env var overrides."""
    path = Path(path)
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    exchanges = {}
    for name, ex_data in raw.get("exchanges", {}).items():
        exchanges[name] = ExchangeConfig(name=name, **ex_data)

    return Config(
        exchanges=exchanges,
        risk=RiskConfig(**raw.get("risk", {})),
        scanner=ScannerConfig(**{
            k: v for k, v in raw.get("scanner", {}).items()
            if k != "instruments"
        } | ({"instruments": raw["scanner"]["instruments"]} if "instruments" in raw.get("scanner", {}) else {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        executor=ExecutorConfig(**raw.get("executor", {})),
        database=DatabaseConfig(**raw.get("database", {})),
    )
