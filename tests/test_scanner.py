"""Tests for the funding rate scanner module."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from funding_arb.config import Config, ExchangeConfig, ScannerConfig
from funding_arb.models import (
    Balance,
    ExchangeData,
    FundingRate,
    MarketSnapshot,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
)
from funding_arb.scanner import FundingScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    exchanges: dict[str, ExchangeConfig] | None = None,
    instruments: list[str] | None = None,
    order_book_depth: int = 5,
) -> Config:
    if instruments is None:
        instruments = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    scanner = ScannerConfig(instruments=instruments, order_book_depth=order_book_depth)
    if exchanges is None:
        exchanges = {
            "binance": ExchangeConfig(name="binance", enabled=True),
        }
    return Config(exchanges=exchanges, scanner=scanner)


def _mock_exchange_class(name: str = "binance") -> MagicMock:
    """Return a mock ccxt exchange instance with default async stubs."""
    ex = MagicMock()
    ex.close = AsyncMock()
    ex.fetch_funding_rates = AsyncMock(return_value={})
    ex.fetch_funding_rate = AsyncMock(return_value={})
    ex.fetch_order_book = AsyncMock(return_value={"bids": [], "asks": []})
    ex.fetch_open_interest = AsyncMock(return_value={})
    ex.fetch_balance = AsyncMock(return_value={"total": {}, "free": {}, "used": {}})
    ex.fetch_funding_rate_history = AsyncMock(return_value=[])
    return ex


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    async def test_enabled_exchange(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)

        mock_cls = MagicMock(return_value=_mock_exchange_class())
        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = mock_cls
            await scanner.initialize()

        assert scanner._initialized is True
        assert "binance" in scanner._exchanges
        mock_cls.assert_called_once()

    async def test_disabled_exchange_skipped(self):
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance", enabled=False),
        })
        scanner = FundingScanner(cfg)

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = MagicMock()
            await scanner.initialize()

        assert scanner._initialized is True
        assert "binance" not in scanner._exchanges

    async def test_unknown_exchange_skipped(self):
        cfg = _make_config(exchanges={
            "unknown_exchange_xyz": ExchangeConfig(name="unknown_exchange_xyz"),
        })
        scanner = FundingScanner(cfg)

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.unknown_exchange_xyz = None  # getattr returns None
            await scanner.initialize()

        assert scanner._initialized is True
        assert "unknown_exchange_xyz" not in scanner._exchanges

    async def test_exchange_with_api_keys(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "key123")
        monkeypatch.setenv("MY_SECRET", "secret456")
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(
                name="binance",
                api_key_env="MY_KEY",
                api_secret_env="MY_SECRET",
            ),
        })
        scanner = FundingScanner(cfg)

        captured_params = {}
        def capture_init(params):
            captured_params.update(params)
            return _mock_exchange_class()

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = capture_init
            await scanner.initialize()

        assert captured_params["apiKey"] == "key123"
        assert captured_params["secret"] == "secret456"

    async def test_sandbox_mode(self):
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance", sandbox=True),
        })
        scanner = FundingScanner(cfg)

        captured_params = {}
        def capture_init(params):
            captured_params.update(params)
            return _mock_exchange_class()

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = capture_init
            await scanner.initialize()

        assert captured_params["sandbox"] is True

    async def test_rate_limit_forwarded(self):
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance", rate_limit_ms=250),
        })
        scanner = FundingScanner(cfg)

        captured_params = {}
        def capture_init(params):
            captured_params.update(params)
            return _mock_exchange_class()

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = capture_init
            await scanner.initialize()

        assert captured_params["enableRateLimit"] is True
        assert captured_params["rateLimit"] == 250

    async def test_init_exception_caught(self):
        """Exchange constructor raising should not crash; exchange is skipped."""
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance"),
        })
        scanner = FundingScanner(cfg)

        def raise_on_init(params):
            raise RuntimeError("connection failed")

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = raise_on_init
            await scanner.initialize()

        assert scanner._initialized is True
        assert "binance" not in scanner._exchanges

    async def test_multiple_exchanges(self):
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance"),
            "bybit": ExchangeConfig(name="bybit"),
        })
        scanner = FundingScanner(cfg)

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = MagicMock(return_value=_mock_exchange_class())
            mock_ccxt.bybit = MagicMock(return_value=_mock_exchange_class())
            await scanner.initialize()

        assert "binance" in scanner._exchanges
        assert "bybit" in scanner._exchanges

    async def test_no_api_key_omits_param(self):
        """When api_key_env is empty, apiKey should not appear in params."""
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance"),
        })
        scanner = FundingScanner(cfg)

        captured_params = {}
        def capture_init(params):
            captured_params.update(params)
            return _mock_exchange_class()

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = capture_init
            await scanner.initialize()

        assert "apiKey" not in captured_params
        assert "secret" not in captured_params


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

class TestClose:
    async def test_close_exchanges(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        scanner._exchanges = {"binance": ex_mock}
        scanner._initialized = True

        await scanner.close()

        ex_mock.close.assert_awaited_once()
        assert scanner._exchanges == {}
        assert scanner._initialized is False

    async def test_close_handles_exception(self):
        """close() should not raise even if an exchange close fails."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.close = AsyncMock(side_effect=RuntimeError("socket error"))
        scanner._exchanges = {"binance": ex_mock}
        scanner._initialized = True

        await scanner.close()  # should not raise

        assert scanner._exchanges == {}
        assert scanner._initialized is False

    async def test_close_multiple(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex1 = _mock_exchange_class()
        ex2 = _mock_exchange_class()
        scanner._exchanges = {"binance": ex1, "bybit": ex2}
        scanner._initialized = True

        await scanner.close()

        ex1.close.assert_awaited_once()
        ex2.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class TestScan:
    async def test_scan_calls_initialize_if_needed(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)

        with patch("funding_arb.scanner.ccxt") as mock_ccxt:
            mock_ccxt.binance = MagicMock(return_value=_mock_exchange_class())
            snapshot = await scanner.scan()

        assert scanner._initialized is True
        assert isinstance(snapshot, MarketSnapshot)

    async def test_scan_does_not_reinitialize(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        scanner._exchanges = {"binance": ex_mock}
        scanner._initialized = True

        snapshot = await scanner.scan()
        assert "binance" in snapshot.exchange_data

    async def test_scan_handles_exchange_failure(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        # Make all fetch methods raise
        ex_mock.fetch_funding_rates = AsyncMock(side_effect=RuntimeError("API down"))
        ex_mock.fetch_order_book = AsyncMock(side_effect=RuntimeError("API down"))
        ex_mock.fetch_open_interest = AsyncMock(side_effect=RuntimeError("API down"))
        ex_mock.fetch_balance = AsyncMock(side_effect=RuntimeError("API down"))
        scanner._exchanges = {"binance": ex_mock}
        scanner._initialized = True

        # Patch _fetch_exchange to raise directly for the gather path
        with patch.object(scanner, "_fetch_exchange", side_effect=RuntimeError("total failure")):
            snapshot = await scanner.scan()

        assert "binance" in snapshot.stale_exchanges
        assert "total failure" in snapshot.stale_exchanges["binance"]

    async def test_scan_mixed_success_and_failure(self):
        cfg = _make_config(exchanges={
            "binance": ExchangeConfig(name="binance"),
            "bybit": ExchangeConfig(name="bybit"),
        })
        scanner = FundingScanner(cfg)
        good_ex = _mock_exchange_class()
        bad_ex = _mock_exchange_class()
        scanner._exchanges = {"binance": good_ex, "bybit": bad_ex}
        scanner._initialized = True

        async def good_fetch(name, ex):
            return ExchangeData(fetched_at=datetime.now(timezone.utc))

        async def bad_fetch(name, ex):
            raise ConnectionError("timeout")

        with patch.object(scanner, "_fetch_exchange", side_effect=[good_fetch, bad_fetch]) as mock_fetch:
            # Need to handle the gather properly - let's use a different approach
            pass

        # Direct approach: patch at a lower level
        good_ex.fetch_funding_rates = AsyncMock(return_value={})
        good_ex.fetch_order_book = AsyncMock(return_value={"bids": [], "asks": []})
        good_ex.fetch_open_interest = AsyncMock(return_value={})
        good_ex.fetch_balance = AsyncMock(return_value={"total": {}, "free": {}, "used": {}})

        bad_ex.fetch_funding_rates = AsyncMock(side_effect=Exception("dead"))
        bad_ex.fetch_order_book = AsyncMock(side_effect=Exception("dead"))
        bad_ex.fetch_open_interest = AsyncMock(side_effect=Exception("dead"))
        bad_ex.fetch_balance = AsyncMock(side_effect=Exception("dead"))

        snapshot = await scanner.scan()

        # Both appear in exchange_data because _fetch_exchange catches exceptions internally
        assert "binance" in snapshot.exchange_data
        assert "bybit" in snapshot.exchange_data


# ---------------------------------------------------------------------------
# _fetch_exchange
# ---------------------------------------------------------------------------

class TestFetchExchange:
    async def test_successful_fetch(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rates = AsyncMock(return_value={
            "BTC/USDT:USDT": {"fundingRate": 0.0003, "fundingDatetime": None},
        })
        ex_mock.fetch_order_book = AsyncMock(return_value={
            "bids": [[50000, 1.0]], "asks": [[50010, 0.5]],
        })
        ex_mock.fetch_open_interest = AsyncMock(return_value={
            "openInterestAmount": 1000.0, "openInterestValue": 50_000_000.0,
        })
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 10000.0}, "free": {"USDT": 8000.0}, "used": {"USDT": 2000.0},
        })

        result = await scanner._fetch_exchange("binance", ex_mock)

        assert isinstance(result, ExchangeData)
        assert result.fetched_at is not None

    async def test_subtask_exception_returns_empty(self):
        """If a subtask raises, its field defaults to empty dict."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rates = AsyncMock(side_effect=RuntimeError("oops"))
        ex_mock.fetch_order_book = AsyncMock(side_effect=RuntimeError("oops"))
        # fetch_open_interest missing to simulate no method
        del ex_mock.fetch_open_interest
        ex_mock.fetch_balance = AsyncMock(side_effect=RuntimeError("oops"))

        result = await scanner._fetch_exchange("binance", ex_mock)

        assert isinstance(result, ExchangeData)
        assert result.rates == {}
        assert result.books == {}
        assert result.open_interest == {}
        assert result.balances == {}


# ---------------------------------------------------------------------------
# _fetch_funding_rates
# ---------------------------------------------------------------------------

class TestFetchFundingRates:
    async def test_batch_fetch_funding_rates(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ts_ms = 1700000000000
        ex_mock.fetch_funding_rates = AsyncMock(return_value={
            "BTC/USDT:USDT": {"fundingRate": 0.0005, "fundingDatetime": ts_ms},
            "ETH/USDT:USDT": {"fundingRate": -0.0001, "fundingDatetime": None},
            "DOGE/USDT:USDT": {"fundingRate": 0.001, "fundingDatetime": None},  # not in instruments
        })

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in rates
        assert "ETH/USDT:USDT" in rates
        assert "DOGE/USDT:USDT" not in rates  # filtered out
        assert rates["BTC/USDT:USDT"].rate == 0.0005
        assert rates["BTC/USDT:USDT"].exchange == "binance"
        assert rates["BTC/USDT:USDT"].next_funding_time is not None
        assert rates["ETH/USDT:USDT"].rate == -0.0001
        assert rates["ETH/USDT:USDT"].next_funding_time is None

    async def test_per_symbol_fallback(self):
        """When exchange lacks fetch_funding_rates, falls back to per-symbol."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_funding_rates  # remove batch method
        ex_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0002, "fundingDatetime": None,
        })

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in rates
        assert rates["BTC/USDT:USDT"].rate == 0.0002
        ex_mock.fetch_funding_rate.assert_awaited_once_with("BTC/USDT:USDT")

    async def test_per_symbol_partial_failure(self):
        """One symbol fails, others succeed."""
        cfg = _make_config(instruments=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_funding_rates

        async def side_effect(symbol):
            if symbol == "ETH/USDT:USDT":
                raise RuntimeError("not found")
            return {"fundingRate": 0.0001, "fundingDatetime": None}

        ex_mock.fetch_funding_rate = AsyncMock(side_effect=side_effect)

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in rates
        assert "ETH/USDT:USDT" not in rates

    async def test_batch_fetch_exception(self):
        """Total failure of batch fetch returns empty rates."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rates = AsyncMock(side_effect=RuntimeError("rate limit"))

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert rates == {}

    async def test_null_funding_rate_defaults_to_zero(self):
        """fundingRate of None should become 0.0."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rates = AsyncMock(return_value={
            "BTC/USDT:USDT": {"fundingRate": None, "fundingDatetime": None},
        })

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert rates["BTC/USDT:USDT"].rate == 0.0

    async def test_per_symbol_null_rate_defaults_to_zero(self):
        """Per-symbol path: fundingRate of None -> 0.0."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_funding_rates
        ex_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": None, "fundingDatetime": None,
        })

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        assert rates["BTC/USDT:USDT"].rate == 0.0

    async def test_per_symbol_with_funding_datetime(self):
        """Per-symbol path with a valid fundingDatetime timestamp."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_funding_rates
        ts_ms = 1700000000000
        ex_mock.fetch_funding_rate = AsyncMock(return_value={
            "fundingRate": 0.0003, "fundingDatetime": ts_ms,
        })

        rates = await scanner._fetch_funding_rates("binance", ex_mock, cfg.scanner.instruments)

        fr = rates["BTC/USDT:USDT"]
        assert fr.next_funding_time is not None
        assert fr.next_funding_time.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _fetch_order_books
# ---------------------------------------------------------------------------

class TestFetchOrderBooks:
    async def test_successful_fetch(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"], order_book_depth=10)
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_order_book = AsyncMock(return_value={
            "bids": [[50000.0, 1.0], [49990.0, 2.0]],
            "asks": [[50010.0, 0.5], [50020.0, 1.5]],
        })

        books = await scanner._fetch_order_books("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in books
        book = books["BTC/USDT:USDT"]
        assert book.exchange == "binance"
        assert book.symbol == "BTC/USDT:USDT"
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids[0].price == 50000.0
        assert book.asks[0].amount == 0.5
        ex_mock.fetch_order_book.assert_awaited_once_with("BTC/USDT:USDT", limit=10)

    async def test_fetch_order_book_failure(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()

        call_count = 0
        async def side_effect(symbol, limit=5):
            nonlocal call_count
            call_count += 1
            if symbol == "BTC/USDT:USDT":
                raise RuntimeError("timeout")
            return {"bids": [[3000.0, 5.0]], "asks": [[3001.0, 4.0]]}

        ex_mock.fetch_order_book = AsyncMock(side_effect=side_effect)

        books = await scanner._fetch_order_books("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" not in books
        assert "ETH/USDT:USDT" in books

    async def test_empty_bids_asks(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_order_book = AsyncMock(return_value={})

        books = await scanner._fetch_order_books("binance", ex_mock, cfg.scanner.instruments)

        book = books["BTC/USDT:USDT"]
        assert book.bids == []
        assert book.asks == []


# ---------------------------------------------------------------------------
# _fetch_open_interest
# ---------------------------------------------------------------------------

class TestFetchOpenInterest:
    async def test_successful_fetch(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_open_interest = AsyncMock(return_value={
            "openInterestAmount": 5000.0,
            "openInterestValue": 250_000_000.0,
        })

        oi = await scanner._fetch_open_interest("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in oi
        assert oi["BTC/USDT:USDT"].oi_contracts == 5000.0
        assert oi["BTC/USDT:USDT"].oi_usd == 250_000_000.0
        assert oi["BTC/USDT:USDT"].exchange == "binance"

    async def test_no_method_returns_empty(self):
        """Exchange without fetch_open_interest returns empty dict."""
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_open_interest  # remove the method

        oi = await scanner._fetch_open_interest("binance", ex_mock, cfg.scanner.instruments)

        assert oi == {}

    async def test_partial_failure(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()

        async def side_effect(symbol):
            if symbol == "ETH/USDT:USDT":
                raise RuntimeError("not available")
            return {"openInterestAmount": 100.0, "openInterestValue": 5_000_000.0}

        ex_mock.fetch_open_interest = AsyncMock(side_effect=side_effect)

        oi = await scanner._fetch_open_interest("binance", ex_mock, cfg.scanner.instruments)

        assert "BTC/USDT:USDT" in oi
        assert "ETH/USDT:USDT" not in oi

    async def test_null_values_default_to_zero(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_open_interest = AsyncMock(return_value={
            "openInterestAmount": None,
            "openInterestValue": None,
        })

        oi = await scanner._fetch_open_interest("binance", ex_mock, cfg.scanner.instruments)

        assert oi["BTC/USDT:USDT"].oi_contracts == 0.0
        assert oi["BTC/USDT:USDT"].oi_usd == 0.0


# ---------------------------------------------------------------------------
# _fetch_balances
# ---------------------------------------------------------------------------

class TestFetchBalances:
    async def test_successful_fetch(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 10000.0, "BTC": 0.5},
            "free": {"USDT": 8000.0, "BTC": 0.3},
            "used": {"USDT": 2000.0, "BTC": 0.2},
        })

        balances = await scanner._fetch_balances("binance", ex_mock)

        assert "USDT" in balances
        assert "BTC" in balances
        assert balances["USDT"].total == 10000.0
        assert balances["USDT"].free == 8000.0
        assert balances["USDT"].used == 2000.0
        assert balances["BTC"].total == 0.5

    async def test_fetch_balance_failure(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_balance = AsyncMock(side_effect=RuntimeError("auth failed"))

        balances = await scanner._fetch_balances("binance", ex_mock)

        assert balances == {}

    async def test_zero_balance_skipped(self):
        """Currencies with zero total balance are excluded."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 5000.0, "BTC": 0.0, "ETH": 0},
            "free": {"USDT": 5000.0, "BTC": 0.0, "ETH": 0},
            "used": {"USDT": 0.0, "BTC": 0.0, "ETH": 0},
        })

        balances = await scanner._fetch_balances("binance", ex_mock)

        assert "USDT" in balances
        assert "BTC" not in balances
        assert "ETH" not in balances

    async def test_none_balance_skipped(self):
        """Currencies with None total balance are excluded."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 1000.0, "BTC": None},
            "free": {"USDT": 1000.0, "BTC": None},
            "used": {"USDT": 0.0, "BTC": None},
        })

        balances = await scanner._fetch_balances("binance", ex_mock)

        assert "USDT" in balances
        assert "BTC" not in balances

    async def test_null_free_used_defaults_to_zero(self):
        """free/used fields that are None should default to 0.0."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 500.0},
            "free": {"USDT": None},
            "used": {},
        })

        balances = await scanner._fetch_balances("binance", ex_mock)

        assert balances["USDT"].free == 0.0
        assert balances["USDT"].used == 0.0


# ---------------------------------------------------------------------------
# fetch_funding_rate_history
# ---------------------------------------------------------------------------

class TestFetchFundingRateHistory:
    async def test_successful_fetch(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ts_ms = 1700000000000
        ex_mock.fetch_funding_rate_history = AsyncMock(return_value=[
            {"fundingRate": 0.0001, "timestamp": ts_ms},
            {"fundingRate": 0.0002, "timestamp": ts_ms + 28800000},
        ])
        scanner._exchanges = {"binance": ex_mock}

        since = datetime(2023, 11, 14, tzinfo=timezone.utc)
        result = await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT", since=since, limit=50)

        assert len(result) == 2
        assert result[0].rate == 0.0001
        assert result[1].rate == 0.0002
        assert result[0].exchange == "binance"
        assert result[0].symbol == "BTC/USDT:USDT"
        assert result[0].timestamp.tzinfo == timezone.utc

        call_args = ex_mock.fetch_funding_rate_history.call_args
        assert call_args.kwargs["since"] == int(since.timestamp() * 1000)
        assert call_args.kwargs["limit"] == 50

    async def test_missing_exchange(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        scanner._exchanges = {}

        result = await scanner.fetch_funding_rate_history("nonexistent", "BTC/USDT:USDT")

        assert result == []

    async def test_no_method_on_exchange(self):
        """Exchange object exists but lacks fetch_funding_rate_history."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        del ex_mock.fetch_funding_rate_history
        scanner._exchanges = {"binance": ex_mock}

        result = await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT")

        assert result == []

    async def test_api_error_returns_empty(self):
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rate_history = AsyncMock(side_effect=RuntimeError("forbidden"))
        scanner._exchanges = {"binance": ex_mock}

        result = await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT")

        assert result == []

    async def test_since_none(self):
        """When since is None, since_ms should be None."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rate_history = AsyncMock(return_value=[])
        scanner._exchanges = {"binance": ex_mock}

        await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT", since=None)

        call_args = ex_mock.fetch_funding_rate_history.call_args
        assert call_args.kwargs["since"] is None

    async def test_entry_without_timestamp(self):
        """Entry missing timestamp field should use current time."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rate_history = AsyncMock(return_value=[
            {"fundingRate": 0.0003},
        ])
        scanner._exchanges = {"binance": ex_mock}

        result = await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT")

        assert len(result) == 1
        assert result[0].rate == 0.0003
        assert result[0].timestamp is not None

    async def test_null_funding_rate_in_history(self):
        """fundingRate of None in history entry defaults to 0.0."""
        cfg = _make_config()
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ex_mock.fetch_funding_rate_history = AsyncMock(return_value=[
            {"fundingRate": None, "timestamp": 1700000000000},
        ])
        scanner._exchanges = {"binance": ex_mock}

        result = await scanner.fetch_funding_rate_history("binance", "BTC/USDT:USDT")

        assert result[0].rate == 0.0


# ---------------------------------------------------------------------------
# Integration-style: scan end-to-end with realistic mock data
# ---------------------------------------------------------------------------

class TestScanEndToEnd:
    async def test_full_scan_produces_populated_snapshot(self):
        cfg = _make_config(instruments=["BTC/USDT:USDT"])
        scanner = FundingScanner(cfg)
        ex_mock = _mock_exchange_class()
        ts_ms = 1700000000000

        ex_mock.fetch_funding_rates = AsyncMock(return_value={
            "BTC/USDT:USDT": {"fundingRate": 0.0005, "fundingDatetime": ts_ms},
        })
        ex_mock.fetch_order_book = AsyncMock(return_value={
            "bids": [[50000.0, 1.0]], "asks": [[50010.0, 0.5]],
        })
        ex_mock.fetch_open_interest = AsyncMock(return_value={
            "openInterestAmount": 1000.0, "openInterestValue": 50_000_000.0,
        })
        ex_mock.fetch_balance = AsyncMock(return_value={
            "total": {"USDT": 10000.0},
            "free": {"USDT": 8000.0},
            "used": {"USDT": 2000.0},
        })

        scanner._exchanges = {"binance": ex_mock}
        scanner._initialized = True

        snapshot = await scanner.scan()

        assert isinstance(snapshot, MarketSnapshot)
        assert "binance" in snapshot.exchange_data
        ed = snapshot.exchange_data["binance"]

        # Funding rates
        assert "BTC/USDT:USDT" in ed.rates
        assert ed.rates["BTC/USDT:USDT"].rate == 0.0005

        # Order books
        assert "BTC/USDT:USDT" in ed.books
        assert len(ed.books["BTC/USDT:USDT"].bids) == 1
        assert ed.books["BTC/USDT:USDT"].bids[0].price == 50000.0

        # Open interest
        assert "BTC/USDT:USDT" in ed.open_interest
        assert ed.open_interest["BTC/USDT:USDT"].oi_contracts == 1000.0

        # Balances
        assert "USDT" in ed.balances
        assert ed.balances["USDT"].total == 10000.0

        # No stale exchanges
        assert snapshot.stale_exchanges == {}
