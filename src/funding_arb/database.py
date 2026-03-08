"""Database layer: SQLite for operational state, Parquet for time series."""
from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .models import ArbitragePosition, FundingRate, OrderResult, OrderSide

logger = logging.getLogger(__name__)


class Database:
    """Manages SQLite databases for state and trades, Parquet for time series."""

    def __init__(
        self,
        state_db_path: str = "data/state.db",
        trades_db_path: str = "data/trades.db",
        funding_db_path: str = "data/funding.db",
        parquet_dir: str = "data/parquet",
    ):
        self.state_db_path = Path(state_db_path)
        self.trades_db_path = Path(trades_db_path)
        self.funding_db_path = Path(funding_db_path)
        self.parquet_dir = Path(parquet_dir)

        # Ensure directories exist
        for path in [self.state_db_path, self.trades_db_path, self.funding_db_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self._init_databases()

    def _init_databases(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.funding_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    rate REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    next_funding_time TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_funding_exchange_symbol
                ON funding_rates (exchange, symbol, timestamp)
            """)

        with sqlite3.connect(self.trades_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    leg_a_exchange TEXT,
                    leg_a_symbol TEXT,
                    leg_a_side TEXT,
                    leg_a_amount REAL,
                    leg_a_price REAL,
                    leg_a_fee REAL,
                    leg_b_exchange TEXT,
                    leg_b_symbol TEXT,
                    leg_b_side TEXT,
                    leg_b_amount REAL,
                    leg_b_price REAL,
                    leg_b_fee REAL,
                    entry_funding_rate REAL,
                    entry_spread REAL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    realized_pnl REAL DEFAULT 0,
                    funding_collected REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    is_filled INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    raw_json TEXT
                )
            """)

        with sqlite3.connect(self.state_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def save_funding_rate(self, rate: FundingRate) -> None:
        """Save a single funding rate observation."""
        with sqlite3.connect(self.funding_db_path) as conn:
            conn.execute(
                "INSERT INTO funding_rates (exchange, symbol, rate, timestamp, next_funding_time) VALUES (?, ?, ?, ?, ?)",
                (
                    rate.exchange,
                    rate.symbol,
                    rate.rate,
                    rate.timestamp.isoformat(),
                    rate.next_funding_time.isoformat() if rate.next_funding_time else None,
                ),
            )

    def save_funding_rates_batch(self, rates: list[FundingRate]) -> None:
        """Save multiple funding rate observations."""
        with sqlite3.connect(self.funding_db_path) as conn:
            conn.executemany(
                "INSERT INTO funding_rates (exchange, symbol, rate, timestamp, next_funding_time) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r.exchange, r.symbol, r.rate,
                        r.timestamp.isoformat(),
                        r.next_funding_time.isoformat() if r.next_funding_time else None,
                    )
                    for r in rates
                ],
            )

    def get_funding_rates(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[FundingRate]:
        """Query funding rates with optional filters."""
        query = "SELECT exchange, symbol, rate, timestamp, next_funding_time FROM funding_rates WHERE 1=1"
        params: list = []

        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.funding_db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            FundingRate(
                exchange=row[0],
                symbol=row[1],
                rate=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                next_funding_time=datetime.fromisoformat(row[4]) if row[4] else None,
            )
            for row in rows
        ]

    def save_position(self, position: ArbitragePosition) -> None:
        """Save or update an arbitrage position."""
        with sqlite3.connect(self.trades_db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO positions
                   (id, leg_a_exchange, leg_a_symbol, leg_a_side, leg_a_amount, leg_a_price, leg_a_fee,
                    leg_b_exchange, leg_b_symbol, leg_b_side, leg_b_amount, leg_b_price, leg_b_fee,
                    entry_funding_rate, entry_spread, opened_at, closed_at, realized_pnl, funding_collected)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    position.id,
                    position.leg_a.exchange if position.leg_a else None,
                    position.leg_a.symbol if position.leg_a else None,
                    position.leg_a.side.value if position.leg_a else None,
                    position.leg_a.amount if position.leg_a else None,
                    position.leg_a.avg_price if position.leg_a else None,
                    position.leg_a.fee if position.leg_a else None,
                    position.leg_b.exchange if position.leg_b else None,
                    position.leg_b.symbol if position.leg_b else None,
                    position.leg_b.side.value if position.leg_b else None,
                    position.leg_b.amount if position.leg_b else None,
                    position.leg_b.avg_price if position.leg_b else None,
                    position.leg_b.fee if position.leg_b else None,
                    position.entry_funding_rate,
                    position.entry_spread,
                    position.opened_at.isoformat(),
                    position.closed_at.isoformat() if position.closed_at else None,
                    position.realized_pnl,
                    position.funding_collected,
                ),
            )

    def get_open_positions(self) -> list[ArbitragePosition]:
        """Get all open positions."""
        with sqlite3.connect(self.trades_db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL"
            ).fetchall()

        return [self._row_to_position(row) for row in rows]

    def _row_to_position(self, row: tuple) -> ArbitragePosition:
        """Convert a database row to ArbitragePosition."""
        leg_a = None
        if row[1]:  # leg_a_exchange
            leg_a = OrderResult(
                order_id="",
                exchange=row[1],
                symbol=row[2] or "",
                side=OrderSide(row[3]) if row[3] else OrderSide.BUY,
                amount=row[4] or 0.0,
                avg_price=row[5] or 0.0,
                fee=row[6] or 0.0,
                is_filled=True,
            )

        leg_b = None
        if row[7]:  # leg_b_exchange
            leg_b = OrderResult(
                order_id="",
                exchange=row[7],
                symbol=row[8] or "",
                side=OrderSide(row[9]) if row[9] else OrderSide.BUY,
                amount=row[10] or 0.0,
                avg_price=row[11] or 0.0,
                fee=row[12] or 0.0,
                is_filled=True,
            )

        return ArbitragePosition(
            id=row[0],
            leg_a=leg_a,
            leg_b=leg_b,
            entry_funding_rate=row[13] or 0.0,
            entry_spread=row[14] or 0.0,
            opened_at=datetime.fromisoformat(row[15]),
            closed_at=datetime.fromisoformat(row[16]) if row[16] else None,
            realized_pnl=row[17] or 0.0,
            funding_collected=row[18] or 0.0,
        )

    def save_funding_rates_parquet(
        self, rates: list[FundingRate], filename: str | None = None
    ) -> Path:
        """Save funding rates to Parquet file for analytics."""
        if not rates:
            raise ValueError("No rates to save")

        df = pd.DataFrame([
            {
                "exchange": r.exchange,
                "symbol": r.symbol,
                "rate": r.rate,
                "annualized": r.annualized,
                "timestamp": r.timestamp,
            }
            for r in rates
        ])

        if filename is None:
            filename = f"funding_rates_{datetime.now(UTC).strftime('%Y%m%d')}.parquet"

        path = self.parquet_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(rates)} funding rates to {path}")
        return path

    def set_state(self, key: str, value: str) -> None:
        """Set a system state value."""
        with sqlite3.connect(self.state_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now(UTC).isoformat()),
            )

    def get_state(self, key: str) -> str | None:
        """Get a system state value."""
        with sqlite3.connect(self.state_db_path) as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None
