"""Tests for the database module - SQLite and Parquet storage."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from funding_arb.database import Database
from funding_arb.models import ArbitragePosition, FundingRate, OrderResult, OrderSide


@pytest.fixture
def db(tmp_path):
    """Create a Database instance using temporary paths."""
    return Database(
        state_db_path=str(tmp_path / "state.db"),
        trades_db_path=str(tmp_path / "trades.db"),
        funding_db_path=str(tmp_path / "funding.db"),
        parquet_dir=str(tmp_path / "parquet"),
    )


@pytest.fixture
def sample_rate():
    """A single funding rate for reuse."""
    return FundingRate(
        exchange="binance",
        symbol="BTC/USDT:USDT",
        rate=0.0003,
        timestamp=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        next_funding_time=datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_rates():
    """A batch of diverse funding rates."""
    base = datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    return [
        FundingRate(
            exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003,
            timestamp=base,
            next_funding_time=base + timedelta(hours=8),
        ),
        FundingRate(
            exchange="binance", symbol="ETH/USDT:USDT", rate=0.0005,
            timestamp=base,
            next_funding_time=base + timedelta(hours=8),
        ),
        FundingRate(
            exchange="okx", symbol="BTC/USDT:USDT", rate=-0.0001,
            timestamp=base + timedelta(hours=1),
            next_funding_time=None,
        ),
        FundingRate(
            exchange="okx", symbol="ETH/USDT:USDT", rate=0.0002,
            timestamp=base + timedelta(hours=2),
        ),
        FundingRate(
            exchange="bybit", symbol="BTC/USDT:USDT", rate=0.0004,
            timestamp=base + timedelta(hours=3),
        ),
    ]


def _make_position(
    position_id="pos-001",
    leg_a_exchange="binance",
    leg_b_exchange="okx",
    closed_at=None,
    pnl=0.0,
    funding=0.0,
):
    """Helper to build an ArbitragePosition with both legs."""
    leg_a = OrderResult(
        order_id="ord-a1", exchange=leg_a_exchange, symbol="BTC/USDT:USDT",
        side=OrderSide.BUY, amount=0.1, avg_price=50000.0, fee=3.0, is_filled=True,
    )
    leg_b = OrderResult(
        order_id="ord-b1", exchange=leg_b_exchange, symbol="BTC/USDT:USDT",
        side=OrderSide.SELL, amount=0.1, avg_price=50050.0, fee=3.0, is_filled=True,
    )
    return ArbitragePosition(
        id=position_id, leg_a=leg_a, leg_b=leg_b,
        entry_funding_rate=0.0003, entry_spread=0.001,
        opened_at=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        closed_at=closed_at,
        realized_pnl=pnl, funding_collected=funding,
    )


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

class TestDatabaseInit:
    def test_creates_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        db = Database(
            state_db_path=str(nested / "state.db"),
            trades_db_path=str(nested / "trades.db"),
            funding_db_path=str(nested / "funding.db"),
            parquet_dir=str(nested / "parquet"),
        )
        assert nested.exists()
        assert (nested / "parquet").exists()

    def test_tables_created(self, db):
        with sqlite3.connect(db.funding_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
        assert "funding_rates" in table_names

        with sqlite3.connect(db.trades_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
        assert "positions" in table_names
        assert "orders" in table_names

        with sqlite3.connect(db.state_db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
        assert "system_state" in table_names

    def test_idempotent_init(self, db):
        """Calling _init_databases twice should not raise."""
        db._init_databases()


# ---------------------------------------------------------------------------
# Funding rates (SQLite)
# ---------------------------------------------------------------------------

class TestSaveFundingRate:
    def test_single_rate(self, db, sample_rate):
        db.save_funding_rate(sample_rate)
        results = db.get_funding_rates()
        assert len(results) == 1
        r = results[0]
        assert r.exchange == "binance"
        assert r.symbol == "BTC/USDT:USDT"
        assert r.rate == pytest.approx(0.0003)
        assert r.timestamp == sample_rate.timestamp
        assert r.next_funding_time == sample_rate.next_funding_time

    def test_single_rate_no_next_funding_time(self, db):
        rate = FundingRate(
            exchange="bybit", symbol="SOL/USDT:USDT", rate=0.0001,
            timestamp=datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
            next_funding_time=None,
        )
        db.save_funding_rate(rate)
        results = db.get_funding_rates()
        assert len(results) == 1
        assert results[0].next_funding_time is None


class TestSaveFundingRatesBatch:
    def test_batch_insert(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(limit=100)
        assert len(results) == 5

    def test_batch_empty_list(self, db):
        db.save_funding_rates_batch([])
        results = db.get_funding_rates()
        assert len(results) == 0

    def test_batch_rates_with_and_without_next_funding_time(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(limit=100)
        has_nft = [r for r in results if r.next_funding_time is not None]
        no_nft = [r for r in results if r.next_funding_time is None]
        assert len(has_nft) >= 2
        assert len(no_nft) >= 1


class TestGetFundingRates:
    def test_no_filters(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates()
        assert len(results) == 5

    def test_filter_by_exchange(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(exchange="binance")
        assert len(results) == 2
        assert all(r.exchange == "binance" for r in results)

    def test_filter_by_symbol(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(symbol="BTC/USDT:USDT")
        assert len(results) == 3
        assert all(r.symbol == "BTC/USDT:USDT" for r in results)

    def test_filter_by_exchange_and_symbol(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(exchange="binance", symbol="ETH/USDT:USDT")
        assert len(results) == 1
        assert results[0].exchange == "binance"
        assert results[0].symbol == "ETH/USDT:USDT"

    def test_filter_by_since(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        cutoff = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        results = db.get_funding_rates(since=cutoff)
        assert all(r.timestamp >= cutoff for r in results)
        assert len(results) == 3  # okx BTC (+1h), okx ETH (+2h), bybit BTC (+3h)

    def test_filter_exchange_and_since(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        cutoff = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        results = db.get_funding_rates(exchange="okx", since=cutoff)
        assert len(results) == 2
        assert all(r.exchange == "okx" for r in results)

    def test_filter_all_three(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        cutoff = datetime(2025, 1, 15, 8, 30, 0, tzinfo=timezone.utc)
        results = db.get_funding_rates(
            exchange="okx", symbol="BTC/USDT:USDT", since=cutoff,
        )
        assert len(results) == 1
        assert results[0].exchange == "okx"
        assert results[0].symbol == "BTC/USDT:USDT"

    def test_limit(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(limit=2)
        assert len(results) == 2

    def test_order_desc_by_timestamp(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(limit=100)
        timestamps = [r.timestamp for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_no_results(self, db, sample_rates):
        db.save_funding_rates_batch(sample_rates)
        results = db.get_funding_rates(exchange="nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestSavePosition:
    def test_save_and_retrieve_open_position(self, db):
        pos = _make_position()
        db.save_position(pos)
        open_positions = db.get_open_positions()
        assert len(open_positions) == 1
        p = open_positions[0]
        assert p.id == "pos-001"
        assert p.entry_funding_rate == pytest.approx(0.0003)
        assert p.entry_spread == pytest.approx(0.001)
        assert p.realized_pnl == pytest.approx(0.0)
        assert p.funding_collected == pytest.approx(0.0)
        assert p.closed_at is None

    def test_position_leg_a_fields(self, db):
        pos = _make_position()
        db.save_position(pos)
        p = db.get_open_positions()[0]
        assert p.leg_a is not None
        assert p.leg_a.exchange == "binance"
        assert p.leg_a.symbol == "BTC/USDT:USDT"
        assert p.leg_a.side == OrderSide.BUY
        assert p.leg_a.amount == pytest.approx(0.1)
        assert p.leg_a.avg_price == pytest.approx(50000.0)
        assert p.leg_a.fee == pytest.approx(3.0)
        assert p.leg_a.is_filled is True

    def test_position_leg_b_fields(self, db):
        pos = _make_position()
        db.save_position(pos)
        p = db.get_open_positions()[0]
        assert p.leg_b is not None
        assert p.leg_b.exchange == "okx"
        assert p.leg_b.side == OrderSide.SELL
        assert p.leg_b.amount == pytest.approx(0.1)
        assert p.leg_b.avg_price == pytest.approx(50050.0)
        assert p.leg_b.fee == pytest.approx(3.0)

    def test_position_no_leg_a(self, db):
        pos = ArbitragePosition(
            id="pos-no-a", leg_a=None,
            leg_b=OrderResult(
                order_id="ord-b", exchange="okx", symbol="ETH/USDT:USDT",
                side=OrderSide.SELL, amount=1.0, avg_price=3000.0, fee=1.8,
                is_filled=True,
            ),
            entry_funding_rate=0.0001, entry_spread=0.002,
            opened_at=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        )
        db.save_position(pos)
        p = db.get_open_positions()[0]
        assert p.leg_a is None
        assert p.leg_b is not None
        assert p.leg_b.exchange == "okx"

    def test_position_no_leg_b(self, db):
        pos = ArbitragePosition(
            id="pos-no-b",
            leg_a=OrderResult(
                order_id="ord-a", exchange="binance", symbol="ETH/USDT:USDT",
                side=OrderSide.BUY, amount=1.0, avg_price=3000.0, fee=1.8,
                is_filled=True,
            ),
            leg_b=None,
            entry_funding_rate=0.0002, entry_spread=0.003,
            opened_at=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        )
        db.save_position(pos)
        p = db.get_open_positions()[0]
        assert p.leg_a is not None
        assert p.leg_a.exchange == "binance"
        assert p.leg_b is None

    def test_position_both_legs_none(self, db):
        pos = ArbitragePosition(
            id="pos-empty", leg_a=None, leg_b=None,
            opened_at=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        )
        db.save_position(pos)
        p = db.get_open_positions()[0]
        assert p.leg_a is None
        assert p.leg_b is None
        assert p.id == "pos-empty"

    def test_update_position_replaces(self, db):
        pos = _make_position(pnl=0.0)
        db.save_position(pos)
        pos.realized_pnl = 42.5
        pos.funding_collected = 10.0
        db.save_position(pos)
        open_positions = db.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].realized_pnl == pytest.approx(42.5)
        assert open_positions[0].funding_collected == pytest.approx(10.0)

    def test_multiple_positions(self, db):
        for i in range(5):
            db.save_position(_make_position(position_id=f"pos-{i:03d}"))
        assert len(db.get_open_positions()) == 5


class TestClosePosition:
    def test_closed_position_not_in_open(self, db):
        close_time = datetime(2025, 1, 16, 8, 0, 0, tzinfo=timezone.utc)
        pos = _make_position(closed_at=close_time, pnl=150.0, funding=25.0)
        db.save_position(pos)
        assert db.get_open_positions() == []

    def test_close_existing_open_position(self, db):
        pos = _make_position()
        db.save_position(pos)
        assert len(db.get_open_positions()) == 1

        close_time = datetime(2025, 1, 16, 8, 0, 0, tzinfo=timezone.utc)
        pos.closed_at = close_time
        pos.realized_pnl = 200.0
        pos.funding_collected = 30.0
        db.save_position(pos)
        assert db.get_open_positions() == []

    def test_mix_open_and_closed(self, db):
        open_pos = _make_position(position_id="open-1")
        closed_pos = _make_position(
            position_id="closed-1",
            closed_at=datetime(2025, 1, 16, tzinfo=timezone.utc),
        )
        db.save_position(open_pos)
        db.save_position(closed_pos)
        open_positions = db.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].id == "open-1"


# ---------------------------------------------------------------------------
# _row_to_position
# ---------------------------------------------------------------------------

class TestRowToPosition:
    def test_full_row(self, db):
        """Test _row_to_position with a complete row from the database."""
        pos = _make_position()
        db.save_position(pos)
        with sqlite3.connect(db.trades_db_path) as conn:
            row = conn.execute("SELECT * FROM positions").fetchone()
        result = db._row_to_position(row)
        assert result.id == "pos-001"
        assert result.leg_a is not None
        assert result.leg_b is not None
        assert result.leg_a.exchange == "binance"
        assert result.leg_b.exchange == "okx"

    def test_row_with_null_legs(self, db):
        """Test _row_to_position when both legs are NULL in the row."""
        pos = ArbitragePosition(
            id="null-legs", leg_a=None, leg_b=None,
            opened_at=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        )
        db.save_position(pos)
        with sqlite3.connect(db.trades_db_path) as conn:
            row = conn.execute("SELECT * FROM positions").fetchone()
        result = db._row_to_position(row)
        assert result.leg_a is None
        assert result.leg_b is None

    def test_row_with_null_side_defaults_to_buy(self, db):
        """When side is NULL in the row, _row_to_position defaults to BUY."""
        with sqlite3.connect(db.trades_db_path) as conn:
            conn.execute(
                """INSERT INTO positions
                   (id, leg_a_exchange, leg_a_symbol, leg_a_side,
                    leg_a_amount, leg_a_price, leg_a_fee,
                    leg_b_exchange, leg_b_symbol, leg_b_side,
                    leg_b_amount, leg_b_price, leg_b_fee,
                    entry_funding_rate, entry_spread, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("side-null", "binance", "BTC/USDT:USDT", None, 0.1, 50000.0, 3.0,
                 "okx", "BTC/USDT:USDT", None, 0.1, 50050.0, 3.0,
                 0.0003, 0.001, "2025-01-15T08:00:00+00:00"),
            )
            row = conn.execute("SELECT * FROM positions WHERE id = 'side-null'").fetchone()
        result = db._row_to_position(row)
        assert result.leg_a.side == OrderSide.BUY
        assert result.leg_b.side == OrderSide.BUY

    def test_row_with_null_numeric_fields(self, db):
        """When numeric fields (amount, price, fee) are NULL, they default to 0.0."""
        with sqlite3.connect(db.trades_db_path) as conn:
            conn.execute(
                """INSERT INTO positions
                   (id, leg_a_exchange, leg_a_symbol, leg_a_side,
                    leg_a_amount, leg_a_price, leg_a_fee,
                    leg_b_exchange, leg_b_symbol, leg_b_side,
                    leg_b_amount, leg_b_price, leg_b_fee,
                    entry_funding_rate, entry_spread, opened_at,
                    realized_pnl, funding_collected)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("null-nums", "binance", None, "buy", None, None, None,
                 "okx", None, "sell", None, None, None,
                 None, None, "2025-01-15T08:00:00+00:00", None, None),
            )
            row = conn.execute("SELECT * FROM positions WHERE id = 'null-nums'").fetchone()
        result = db._row_to_position(row)
        assert result.leg_a.symbol == ""
        assert result.leg_a.amount == 0.0
        assert result.leg_a.avg_price == 0.0
        assert result.leg_a.fee == 0.0
        assert result.leg_b.symbol == ""
        assert result.leg_b.amount == 0.0
        assert result.entry_funding_rate == 0.0
        assert result.entry_spread == 0.0
        assert result.realized_pnl == 0.0
        assert result.funding_collected == 0.0

    def test_row_with_closed_at(self, db):
        close_time = datetime(2025, 1, 16, 8, 0, 0, tzinfo=timezone.utc)
        pos = _make_position(closed_at=close_time)
        db.save_position(pos)
        with sqlite3.connect(db.trades_db_path) as conn:
            row = conn.execute("SELECT * FROM positions").fetchone()
        result = db._row_to_position(row)
        assert result.closed_at == close_time


# ---------------------------------------------------------------------------
# Parquet storage
# ---------------------------------------------------------------------------

class TestParquet:
    def test_save_parquet(self, db, sample_rates):
        path = db.save_funding_rates_parquet(sample_rates, filename="test.parquet")
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 5
        assert set(df.columns) == {"exchange", "symbol", "rate", "annualized", "timestamp"}

    def test_save_parquet_default_filename(self, db, sample_rates):
        path = db.save_funding_rates_parquet(sample_rates)
        assert path.exists()
        assert path.suffix == ".parquet"
        assert "funding_rates_" in path.name

    def test_save_parquet_data_integrity(self, db):
        rates = [
            FundingRate(
                exchange="binance", symbol="BTC/USDT:USDT", rate=0.0003,
                timestamp=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
            ),
        ]
        path = db.save_funding_rates_parquet(rates, filename="integrity.parquet")
        df = pd.read_parquet(path)
        assert df.iloc[0]["exchange"] == "binance"
        assert df.iloc[0]["symbol"] == "BTC/USDT:USDT"
        assert df.iloc[0]["rate"] == pytest.approx(0.0003)
        assert df.iloc[0]["annualized"] == pytest.approx(0.0003 * 3 * 365)

    def test_save_parquet_empty_raises(self, db):
        with pytest.raises(ValueError, match="No rates to save"):
            db.save_funding_rates_parquet([])


# ---------------------------------------------------------------------------
# System state (key-value)
# ---------------------------------------------------------------------------

class TestSystemState:
    def test_set_and_get(self, db):
        db.set_state("last_scan_time", "2025-01-15T08:00:00Z")
        assert db.get_state("last_scan_time") == "2025-01-15T08:00:00Z"

    def test_get_missing_key_returns_none(self, db):
        assert db.get_state("nonexistent_key") is None

    def test_overwrite_state(self, db):
        db.set_state("mode", "scanning")
        db.set_state("mode", "trading")
        assert db.get_state("mode") == "trading"

    def test_multiple_keys(self, db):
        db.set_state("key_a", "value_a")
        db.set_state("key_b", "value_b")
        db.set_state("key_c", "value_c")
        assert db.get_state("key_a") == "value_a"
        assert db.get_state("key_b") == "value_b"
        assert db.get_state("key_c") == "value_c"

    def test_state_stored_with_timestamp(self, db):
        db.set_state("check", "yes")
        with sqlite3.connect(db.state_db_path) as conn:
            row = conn.execute(
                "SELECT updated_at FROM system_state WHERE key = ?", ("check",)
            ).fetchone()
        assert row is not None
        # Verify it parses as a valid ISO datetime
        dt = datetime.fromisoformat(row[0])
        assert dt.year >= 2025
