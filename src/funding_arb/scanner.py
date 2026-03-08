"""Funding rate scanner - polls all configured exchanges."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import ccxt.async_support as ccxt

from .models import (
    Balance,
    ExchangeData,
    FundingRate,
    MarketSnapshot,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
)

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


class FundingScanner:
    """Polls all configured exchanges for funding rates, order books, OI, and balances."""

    def __init__(self, config: Config):
        self.config = config
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Create ccxt exchange instances."""
        for name, ex_config in self.config.exchanges.items():
            if not ex_config.enabled:
                continue
            try:
                exchange_class = getattr(ccxt, name, None)
                if exchange_class is None:
                    logger.warning(f"Exchange {name} not found in ccxt")
                    continue

                params: dict = {
                    "enableRateLimit": True,
                    "rateLimit": ex_config.rate_limit_ms,
                }
                if ex_config.api_key:
                    params["apiKey"] = ex_config.api_key
                if ex_config.api_secret:
                    params["secret"] = ex_config.api_secret
                if ex_config.sandbox:
                    params["sandbox"] = True

                self._exchanges[name] = exchange_class(params)
                logger.info(f"Initialized exchange: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

        self._initialized = True

    async def close(self) -> None:
        """Close all exchange connections."""
        for ex in self._exchanges.values():
            with contextlib.suppress(Exception):
                await ex.close()
        self._exchanges.clear()
        self._initialized = False

    async def scan(self) -> MarketSnapshot:
        """Concurrent fetch across all exchanges. Returns unified snapshot."""
        if not self._initialized:
            await self.initialize()

        snapshot = MarketSnapshot(timestamp=datetime.now(UTC))
        tasks = {
            name: self._fetch_exchange(name, ex)
            for name, ex in self._exchanges.items()
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        for name, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception):
                snapshot.mark_stale(name, str(result))
                logger.warning(f"Scan failed for {name}: {result}")
            else:
                snapshot.update(name, result)

        return snapshot

    async def _fetch_exchange(self, name: str, ex: ccxt.Exchange) -> ExchangeData:
        """Fetch rates, book, OI, and balances for one exchange."""
        instruments = self.config.scanner.instruments

        rates_task = self._fetch_funding_rates(name, ex, instruments)
        books_task = self._fetch_order_books(name, ex, instruments)
        oi_task = self._fetch_open_interest(name, ex, instruments)
        balances_task = self._fetch_balances(name, ex)

        rates, books, oi, balances = await asyncio.gather(
            rates_task, books_task, oi_task, balances_task,
            return_exceptions=True,
        )

        return ExchangeData(
            rates=rates if not isinstance(rates, Exception) else {},
            books=books if not isinstance(books, Exception) else {},
            open_interest=oi if not isinstance(oi, Exception) else {},
            balances=balances if not isinstance(balances, Exception) else {},
            fetched_at=datetime.now(UTC),
        )

    async def _fetch_funding_rates(
        self, name: str, ex: ccxt.Exchange, instruments: list[str]
    ) -> dict[str, FundingRate]:
        """Fetch current funding rates for all instruments."""
        rates: dict[str, FundingRate] = {}
        try:
            if hasattr(ex, "fetch_funding_rates"):
                raw_rates = await ex.fetch_funding_rates(instruments)
                for symbol, data in raw_rates.items():
                    if symbol in instruments:
                        rates[symbol] = FundingRate(
                            exchange=name,
                            symbol=symbol,
                            rate=data.get("fundingRate", 0.0) or 0.0,
                            next_funding_time=datetime.fromtimestamp(
                                data["fundingDatetime"] / 1000, tz=UTC
                            ) if data.get("fundingDatetime") else None,
                            timestamp=datetime.now(UTC),
                        )
            else:
                for symbol in instruments:
                    try:
                        data = await ex.fetch_funding_rate(symbol)
                        rates[symbol] = FundingRate(
                            exchange=name,
                            symbol=symbol,
                            rate=data.get("fundingRate", 0.0) or 0.0,
                            next_funding_time=datetime.fromtimestamp(
                                data["fundingDatetime"] / 1000, tz=UTC
                            ) if data.get("fundingDatetime") else None,
                            timestamp=datetime.now(UTC),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to fetch funding rate for {symbol} on {name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch funding rates from {name}: {e}")
        return rates

    async def _fetch_order_books(
        self, name: str, ex: ccxt.Exchange, instruments: list[str]
    ) -> dict[str, OrderBook]:
        """Fetch order books for all instruments."""
        books: dict[str, OrderBook] = {}
        depth = self.config.scanner.order_book_depth
        for symbol in instruments:
            try:
                raw = await ex.fetch_order_book(symbol, limit=depth)
                books[symbol] = OrderBook(
                    exchange=name,
                    symbol=symbol,
                    bids=[OrderBookLevel(price=b[0], amount=b[1]) for b in raw.get("bids", [])],
                    asks=[OrderBookLevel(price=a[0], amount=a[1]) for a in raw.get("asks", [])],
                    timestamp=datetime.now(UTC),
                )
            except Exception as e:
                logger.debug(f"Failed to fetch order book for {symbol} on {name}: {e}")
        return books

    async def _fetch_open_interest(
        self, name: str, ex: ccxt.Exchange, instruments: list[str]
    ) -> dict[str, OpenInterest]:
        """Fetch open interest for all instruments."""
        oi_data: dict[str, OpenInterest] = {}
        if not hasattr(ex, "fetch_open_interest"):
            return oi_data
        for symbol in instruments:
            try:
                raw = await ex.fetch_open_interest(symbol)
                oi_data[symbol] = OpenInterest(
                    exchange=name,
                    symbol=symbol,
                    oi_contracts=raw.get("openInterestAmount", 0.0) or 0.0,
                    oi_usd=raw.get("openInterestValue", 0.0) or 0.0,
                    timestamp=datetime.now(UTC),
                )
            except Exception as e:
                logger.debug(f"Failed to fetch OI for {symbol} on {name}: {e}")
        return oi_data

    async def _fetch_balances(
        self, name: str, ex: ccxt.Exchange
    ) -> dict[str, Balance]:
        """Fetch account balances."""
        balances: dict[str, Balance] = {}
        try:
            raw = await ex.fetch_balance()
            for currency, data in raw.get("total", {}).items():
                if data and data > 0:
                    balances[currency] = Balance(
                        currency=currency,
                        free=raw.get("free", {}).get(currency, 0.0) or 0.0,
                        used=raw.get("used", {}).get(currency, 0.0) or 0.0,
                        total=data,
                    )
        except Exception as e:
            logger.debug(f"Failed to fetch balances from {name}: {e}")
        return balances

    async def fetch_funding_rate_history(
        self, exchange: str, symbol: str, since: datetime | None = None, limit: int = 100
    ) -> list[FundingRate]:
        """Fetch historical funding rates for backtesting."""
        ex = self._exchanges.get(exchange)
        if ex is None or not hasattr(ex, "fetch_funding_rate_history"):
            return []

        since_ms = int(since.timestamp() * 1000) if since else None
        try:
            raw = await ex.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
            return [
                FundingRate(
                    exchange=exchange,
                    symbol=symbol,
                    rate=entry.get("fundingRate", 0.0) or 0.0,
                    timestamp=datetime.fromtimestamp(
                        entry["timestamp"] / 1000, tz=UTC
                    ) if entry.get("timestamp") else datetime.now(UTC),
                )
                for entry in raw
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate history for {symbol} on {exchange}: {e}")
            return []
