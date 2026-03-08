"""Tests for the data ingestion module."""
import os
import tempfile
import pytest
import pandas as pd
from datetime import datetime, timezone

from funding_arb.ingestion import FundingRateIngester
from funding_arb.database import Database
from funding_arb.config import Config, ExchangeConfig, ScannerConfig, DatabaseConfig


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database."""
    return Database(
        state_db_path=str(tmp_path / "state.db"),
        trades_db_path=str(tmp_path / "trades.db"),
        funding_db_path=str(tmp_path / "funding.db"),
        parquet_dir=str(tmp_path / "parquet"),
    )


@pytest.fixture
def config():
    return Config(
        database=DatabaseConfig(
            state_db_path="data/test_state.db",
            trades_db_path="data/test_trades.db",
            funding_db_path="data/test_funding.db",
            parquet_dir="data/test_parquet",
        ),
    )


class TestCSVImport:
    def test_import_csv(self, tmp_db, config):
        ingester = FundingRateIngester(config, database=tmp_db)

        # Create a test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,exchange,symbol,rate\n")
            f.write("2024-01-01T00:00:00+00:00,binance,BTC/USDT:USDT,0.0003\n")
            f.write("2024-01-01T08:00:00+00:00,binance,BTC/USDT:USDT,0.0005\n")
            f.write("2024-01-01T16:00:00+00:00,binance,BTC/USDT:USDT,0.0002\n")
            csv_path = f.name

        try:
            count = ingester.import_from_csv(csv_path)
            assert count == 3

            rates = tmp_db.get_funding_rates(exchange="binance", symbol="BTC/USDT:USDT")
            assert len(rates) == 3
        finally:
            os.unlink(csv_path)

    def test_import_csv_with_fixed_exchange(self, tmp_db, config):
        ingester = FundingRateIngester(config, database=tmp_db)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,rate\n")
            f.write("2024-01-01T00:00:00,0.0003\n")
            f.write("2024-01-01T08:00:00,0.0005\n")
            csv_path = f.name

        try:
            count = ingester.import_from_csv(
                csv_path,
                exchange="hyperliquid",
                symbol="ETH/USDT:USDT",
                exchange_col=None,
                symbol_col=None,
            )
            assert count == 2

            rates = tmp_db.get_funding_rates(exchange="hyperliquid")
            assert len(rates) == 2
            assert all(r.symbol == "ETH/USDT:USDT" for r in rates)
        finally:
            os.unlink(csv_path)

    def test_import_csv_missing_exchange_raises(self, tmp_db, config):
        ingester = FundingRateIngester(config, database=tmp_db)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,rate\n")
            f.write("2024-01-01T00:00:00,0.0003\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Exchange and symbol"):
                ingester.import_from_csv(
                    csv_path, exchange_col=None, symbol_col=None,
                )
        finally:
            os.unlink(csv_path)


class TestExportToDataFrame:
    def test_export_empty(self, tmp_db, config):
        ingester = FundingRateIngester(config, database=tmp_db)
        df = ingester.export_to_dataframe()
        assert df.empty
        assert set(df.columns) == {"timestamp", "exchange", "symbol", "rate", "annualized"}

    def test_export_after_import(self, tmp_db, config):
        ingester = FundingRateIngester(config, database=tmp_db)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,exchange,symbol,rate\n")
            f.write("2024-01-01T00:00:00+00:00,binance,BTC/USDT:USDT,0.0003\n")
            f.write("2024-01-01T08:00:00+00:00,binance,BTC/USDT:USDT,0.0005\n")
            csv_path = f.name

        try:
            ingester.import_from_csv(csv_path)
            df = ingester.export_to_dataframe(exchange="binance")
            assert len(df) == 2
            assert "annualized" in df.columns
        finally:
            os.unlink(csv_path)


# --- New tests for async ingestion methods (lines 58-92, 103-140, 230-235) ---

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from funding_arb.models import FundingRate as FR


@pytest.fixture
def ingestion_config():
    """Config with exchanges and instruments for async ingestion tests."""
    return Config(
        exchanges={
            "binance": ExchangeConfig(name="binance", enabled=True),
            "bybit": ExchangeConfig(name="bybit", enabled=False),
        },
        scanner=ScannerConfig(instruments=["BTC/USDT:USDT"]),
        database=DatabaseConfig(
            state_db_path="data/test_state.db",
            trades_db_path="data/test_trades.db",
            funding_db_path="data/test_funding.db",
            parquet_dir="data/test_parquet",
        ),
    )


class TestIngestFromExchanges:
    """Tests for ingest_from_exchanges (lines 58-92) and _ingest_symbol (lines 103-140)."""

    @pytest.mark.asyncio
    async def test_ingest_from_exchanges_defaults(self, tmp_db, ingestion_config):
        """Test ingest_from_exchanges with default params uses enabled exchanges and configured instruments."""
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        mock_rates = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003, timestamp=ts),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "initialize", new_callable=AsyncMock) as mock_init, \
             patch.object(ingester.scanner, "close", new_callable=AsyncMock) as mock_close, \
             patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            # Return rates on first call, empty on second (to end pagination)
            mock_fetch.side_effect = [mock_rates, []]

            count = await ingester.ingest_from_exchanges()

            mock_init.assert_called_once()
            mock_close.assert_called_once()
            assert count == 1
            # Only "binance" is enabled, "bybit" is disabled
            assert mock_fetch.call_count == 1  # 1 result < batch_size, so it breaks after 1 call

    @pytest.mark.asyncio
    async def test_ingest_from_exchanges_explicit_params(self, tmp_db, ingestion_config):
        """Test ingest_from_exchanges with explicit exchanges/symbols/since."""
        ts1 = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
        mock_rates = [
            FR(exchange="binance", symbol="ETH/USDT:USDT", rate=0.0002, timestamp=ts1),
            FR(exchange="binance", symbol="ETH/USDT:USDT", rate=0.0003, timestamp=ts2),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "initialize", new_callable=AsyncMock), \
             patch.object(ingester.scanner, "close", new_callable=AsyncMock), \
             patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [mock_rates, []]

            count = await ingester.ingest_from_exchanges(
                exchanges=["binance"],
                symbols=["ETH/USDT:USDT"],
                since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

            assert count == 2

    @pytest.mark.asyncio
    async def test_ingest_saves_parquet_when_rates_exist(self, tmp_db, ingestion_config):
        """Test that parquet is saved after ingestion when rates exist."""
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        mock_rates = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003, timestamp=ts),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "initialize", new_callable=AsyncMock), \
             patch.object(ingester.scanner, "close", new_callable=AsyncMock), \
             patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch, \
             patch.object(tmp_db, "save_funding_rates_parquet") as mock_parquet:
            mock_fetch.side_effect = [mock_rates, []]

            await ingester.ingest_from_exchanges(exchanges=["binance"], symbols=["BTC/USDT:USDT"])

            # Parquet should be called since rates were saved
            mock_parquet.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_closes_scanner_on_exception(self, tmp_db, ingestion_config):
        """Test scanner.close() is called even when an exception occurs (finally block)."""
        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "initialize", new_callable=AsyncMock), \
             patch.object(ingester.scanner, "close", new_callable=AsyncMock) as mock_close, \
             patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = RuntimeError("API error")

            with pytest.raises(RuntimeError, match="API error"):
                await ingester.ingest_from_exchanges(exchanges=["binance"], symbols=["BTC/USDT:USDT"])

            mock_close.assert_called_once()


class TestIngestSymbol:
    """Tests for _ingest_symbol (lines 103-140) directly."""

    @pytest.mark.asyncio
    async def test_ingest_symbol_incremental(self, tmp_db, ingestion_config):
        """Test incremental ingestion uses last_ingested timestamp."""
        # Set a prior ingestion timestamp
        since_initial = datetime(2024, 1, 1, tzinfo=timezone.utc)
        last_ingested = datetime(2024, 6, 1, tzinfo=timezone.utc)
        tmp_db.set_state("last_ingested_binance_BTC/USDT:USDT", last_ingested.isoformat())

        ts = datetime(2024, 6, 2, tzinfo=timezone.utc)
        mock_rates = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0004, timestamp=ts),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            # Return one batch then empty
            mock_fetch.side_effect = [mock_rates, []]

            count = await ingester._ingest_symbol("binance", "BTC/USDT:USDT", since_initial, 500)

            assert count == 1
            # The fetch should have used last_ingested as since (because it's > since_initial)
            first_call_kwargs = mock_fetch.call_args_list[0]
            assert first_call_kwargs[1]["since"] >= last_ingested or first_call_kwargs[0][2] >= last_ingested

    @pytest.mark.asyncio
    async def test_ingest_symbol_pagination(self, tmp_db, ingestion_config):
        """Test multi-page ingestion with batch_size boundary."""
        ts1 = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 6, 1, 16, 0, 0, tzinfo=timezone.utc)

        batch1 = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003, timestamp=ts1),
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0004, timestamp=ts2),
        ]
        batch2 = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0005, timestamp=ts3),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            # batch_size=2: first call returns full batch (2 items), second returns partial (1 item)
            mock_fetch.side_effect = [batch1, batch2]

            count = await ingester._ingest_symbol(
                "binance", "BTC/USDT:USDT",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                batch_size=2,
            )

            assert count == 3
            assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_ingest_symbol_no_progress_breaks(self, tmp_db, ingestion_config):
        """Test that ingestion stops when latest_ts <= current_since (no progress)."""
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        same_rates = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003, timestamp=ts),
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0004, timestamp=ts),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            # Since is already at or past max(timestamps), loop should break on no-progress
            mock_fetch.return_value = same_rates

            count = await ingester._ingest_symbol(
                "binance", "BTC/USDT:USDT",
                ts,  # since == latest_ts, so no progress
                batch_size=500,
            )

            assert count == 2
            assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_symbol_empty_result(self, tmp_db, ingestion_config):
        """Test that zero rates returned immediately returns 0."""
        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            count = await ingester._ingest_symbol(
                "binance", "BTC/USDT:USDT",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                batch_size=500,
            )

            assert count == 0

    @pytest.mark.asyncio
    async def test_ingest_symbol_records_last_state(self, tmp_db, ingestion_config):
        """Test that last ingestion state is saved when count > 0."""
        ts = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
        mock_rates = [
            FR(exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003, timestamp=ts),
        ]

        ingester = FundingRateIngester(ingestion_config, database=tmp_db)

        with patch.object(ingester.scanner, "fetch_funding_rate_history", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [mock_rates, []]

            await ingester._ingest_symbol(
                "binance", "BTC/USDT:USDT",
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                batch_size=500,
            )

            saved = tmp_db.get_state("last_ingested_binance_BTC/USDT:USDT")
            assert saved is not None


class TestRunIngestion:
    """Tests for the run_ingestion() CLI entry point (lines 230-235)."""

    @pytest.mark.asyncio
    async def test_run_ingestion(self):
        """Test run_ingestion loads config and calls ingest_from_exchanges."""
        from funding_arb.ingestion import run_ingestion

        mock_config = Config(
            exchanges={"binance": ExchangeConfig(name="binance", enabled=True)},
            scanner=ScannerConfig(instruments=["BTC/USDT:USDT"]),
        )

        with patch("funding_arb.config.load_config", return_value=mock_config) as mock_load, \
             patch.object(FundingRateIngester, "ingest_from_exchanges", new_callable=AsyncMock, return_value=42) as mock_ingest:
            await run_ingestion("config.toml")

            mock_load.assert_called_once_with("config.toml")
            mock_ingest.assert_called_once()
