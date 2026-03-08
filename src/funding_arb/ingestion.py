"""Historical funding rate data ingestion from exchanges and APIs."""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from .database import Database
from .models import FundingRate
from .scanner import FundingScanner

if TYPE_CHECKING:
    from pathlib import Path

    from .config import Config

logger = logging.getLogger(__name__)


class FundingRateIngester:
    """Ingests historical funding rate data from exchanges.

    Supports:
    - Direct exchange API via ccxt (fetch_funding_rate_history)
    - CSV file import (for Coinglass exports or manual data)
    - Incremental ingestion (only fetch new data since last ingestion)
    """

    def __init__(self, config: Config, database: Database | None = None):
        self.config = config
        self.scanner = FundingScanner(config)
        self.db = database or Database(
            state_db_path=config.database.state_db_path,
            trades_db_path=config.database.trades_db_path,
            funding_db_path=config.database.funding_db_path,
            parquet_dir=config.database.parquet_dir,
        )

    async def ingest_from_exchanges(
        self,
        exchanges: list[str] | None = None,
        symbols: list[str] | None = None,
        since: datetime | None = None,
        days_back: int = 180,
        batch_size: int = 500,
    ) -> int:
        """Ingest historical funding rates from exchange APIs.

        Args:
            exchanges: Exchange names to fetch from. Defaults to all configured.
            symbols: Symbols to fetch. Defaults to configured instruments.
            since: Start date. Defaults to days_back days ago.
            days_back: Days of history to fetch if since not specified.
            batch_size: Number of records per API call.

        Returns:
            Total number of funding rate records ingested.
        """
        await self.scanner.initialize()

        try:
            if exchanges is None:
                exchanges = [
                    name for name, cfg in self.config.exchanges.items()
                    if cfg.enabled
                ]
            if symbols is None:
                symbols = list(self.config.scanner.instruments)
            if since is None:
                since = datetime.now(UTC) - timedelta(days=days_back)

            total_ingested = 0

            for exchange in exchanges:
                for symbol in symbols:
                    count = await self._ingest_symbol(
                        exchange, symbol, since, batch_size
                    )
                    total_ingested += count
                    logger.info(
                        f"Ingested {count} rates for {symbol} on {exchange}"
                    )

            # Save to parquet for analytics
            all_rates = self.db.get_funding_rates(limit=100_000)
            if all_rates:
                self.db.save_funding_rates_parquet(all_rates)

            logger.info(f"Total ingested: {total_ingested} funding rate records")
            return total_ingested

        finally:
            await self.scanner.close()

    async def _ingest_symbol(
        self,
        exchange: str,
        symbol: str,
        since: datetime,
        batch_size: int,
    ) -> int:
        """Ingest funding rates for a single exchange/symbol pair."""
        # Check last ingested timestamp for incremental loading
        last_key = f"last_ingested_{exchange}_{symbol}"
        last_ts_str = self.db.get_state(last_key)
        if last_ts_str:
            last_ts = datetime.fromisoformat(last_ts_str)
            if last_ts > since:
                since = last_ts
                logger.info(f"Incremental load for {exchange}/{symbol} since {since}")

        count = 0
        current_since = since

        while True:
            rates = await self.scanner.fetch_funding_rate_history(
                exchange, symbol, since=current_since, limit=batch_size
            )
            if not rates:
                break

            self.db.save_funding_rates_batch(rates)
            count += len(rates)

            # Move window forward
            latest_ts = max(r.timestamp for r in rates)
            if latest_ts <= current_since:
                break  # No progress
            current_since = latest_ts + timedelta(seconds=1)

            # Rate limit
            await asyncio.sleep(0.5)

            if len(rates) < batch_size:
                break  # Last page

        # Record last ingestion point
        if count > 0:
            self.db.set_state(last_key, current_since.isoformat())

        return count

    def import_from_csv(
        self,
        filepath: str | Path,
        exchange: str | None = None,
        symbol: str | None = None,
        timestamp_col: str = "timestamp",
        rate_col: str = "rate",
        exchange_col: str | None = "exchange",
        symbol_col: str | None = "symbol",
    ) -> int:
        """Import funding rates from a CSV file.

        Flexible column mapping to support different CSV formats
        (Coinglass, manual exports, etc.)

        Args:
            filepath: Path to CSV file.
            exchange: Exchange name (required if no exchange column in CSV).
            symbol: Symbol (required if no symbol column in CSV).
            timestamp_col: Column name for timestamps.
            rate_col: Column name for funding rates.
            exchange_col: Column name for exchange (None if not in CSV).
            symbol_col: Column name for symbol (None if not in CSV).

        Returns:
            Number of records imported.
        """
        df = pd.read_csv(filepath)

        rates: list[FundingRate] = []
        for _, row in df.iterrows():
            ex = row[exchange_col] if exchange_col and exchange_col in df.columns else exchange
            sym = row[symbol_col] if symbol_col and symbol_col in df.columns else symbol

            if ex is None or sym is None:
                raise ValueError(
                    "Exchange and symbol must be specified either as columns or parameters"
                )

            ts = pd.to_datetime(row[timestamp_col])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            rates.append(FundingRate(
                exchange=str(ex),
                symbol=str(sym),
                rate=float(row[rate_col]),
                timestamp=ts.to_pydatetime(),
            ))

        if rates:
            self.db.save_funding_rates_batch(rates)
            logger.info(f"Imported {len(rates)} rates from {filepath}")

        return len(rates)

    def export_to_dataframe(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        """Export funding rates from database to a DataFrame for analysis/backtesting.

        Returns DataFrame with columns: timestamp, exchange, symbol, rate, annualized
        """
        rates = self.db.get_funding_rates(
            exchange=exchange, symbol=symbol, since=since, limit=limit
        )

        if not rates:
            return pd.DataFrame(columns=["timestamp", "exchange", "symbol", "rate", "annualized"])

        return pd.DataFrame([
            {
                "timestamp": r.timestamp,
                "exchange": r.exchange,
                "symbol": r.symbol,
                "rate": r.rate,
                "annualized": r.annualized,
            }
            for r in rates
        ])


async def run_ingestion(config_path: str = "config.toml") -> None:
    """CLI entry point for data ingestion."""
    from .config import load_config

    config = load_config(config_path)
    ingester = FundingRateIngester(config)
    count = await ingester.ingest_from_exchanges()
    print(f"Ingested {count} funding rate records")
