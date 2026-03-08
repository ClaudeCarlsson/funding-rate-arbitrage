"""Trade executor — places real or paper orders via ccxt with verification and kill switch."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import ccxt.async_support as ccxt

from .config import ExecutorConfig
from .database import Database
from .models import (
    ArbitragePosition,
    Opportunity,
    OrderResult,
    OrderSide,
    TradeLeg,
)

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

KILL_SWITCH_PATH = Path("/tmp/kill_trading")
MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0


class TradeExecutor:
    """Executes arbitrage trades via ccxt with position verification and kill switch.

    Three modes:
    - paper (default): fully simulated fills, no exchange calls for orders
    - dry_run: real scanning/prices, orders constructed but NOT submitted
    - live: real market orders with verification
    """

    def __init__(self, config: Config, database: Database | None = None):
        self.executor_config = config.executor
        self.config = config
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self._initialized = False
        self.db = database or Database(
            state_db_path=config.database.state_db_path,
            trades_db_path=config.database.trades_db_path,
            funding_db_path=config.database.funding_db_path,
            parquet_dir=config.database.parquet_dir,
        )

    async def initialize(self) -> None:
        """Create ccxt exchange instances for enabled exchanges."""
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
                logger.info(f"Executor initialized exchange: {name}")
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

    def is_killed(self) -> bool:
        """Check if the kill switch file exists."""
        return KILL_SWITCH_PATH.exists()

    async def open_position(
        self, opp: Opportunity, size_usd: float
    ) -> ArbitragePosition | None:
        """Execute a two-leg arbitrage position.

        Args:
            opp: Opportunity with leg_a (short) and leg_b (long) defined.
            size_usd: Total notional in USD for each leg.

        Returns:
            ArbitragePosition if both legs filled, None if failed or killed.
        """
        if self.is_killed():
            logger.warning("Kill switch active — refusing to open position")
            return None

        if opp.leg_a is None or opp.leg_b is None:
            logger.error("Opportunity missing trade legs")
            return None

        # Enforce hard position size cap
        max_pos = self.executor_config.max_position_usd
        if size_usd > max_pos:
            logger.info(f"Capping position size from ${size_usd:.2f} to ${max_pos:.2f}")
            size_usd = max_pos

        position_id = str(uuid.uuid4())[:12]

        # Execute both legs
        leg_a_result = await self._execute_leg(opp.leg_a, size_usd, position_id, "A")
        if leg_a_result is None:
            logger.error(f"Position {position_id}: leg A failed, aborting")
            return None

        leg_b_result = await self._execute_leg(opp.leg_b, size_usd, position_id, "B")
        if leg_b_result is None:
            logger.error(
                f"Position {position_id}: leg B failed after leg A filled — "
                f"MANUAL INTERVENTION NEEDED to close leg A on {opp.leg_a.exchange}"
            )
            # Still record the partial position so we can track and unwind it
            position = ArbitragePosition(
                id=position_id,
                leg_a=leg_a_result,
                leg_b=None,
                entry_funding_rate=opp.net_rate,
                entry_spread=opp.expected_spread,
            )
            self.db.save_position(position)
            return None

        position = ArbitragePosition(
            id=position_id,
            leg_a=leg_a_result,
            leg_b=leg_b_result,
            entry_funding_rate=opp.net_rate,
            entry_spread=opp.expected_spread,
        )
        self.db.save_position(position)
        logger.info(
            f"Position {position_id} opened: "
            f"leg_a={leg_a_result.exchange} {leg_a_result.side.value} {leg_a_result.amount:.6f} @ {leg_a_result.avg_price:.2f}, "
            f"leg_b={leg_b_result.exchange} {leg_b_result.side.value} {leg_b_result.amount:.6f} @ {leg_b_result.avg_price:.2f}"
        )
        return position

    async def close_position(self, position: ArbitragePosition) -> bool:
        """Close an open arbitrage position by placing opposing orders.

        Returns True if both legs closed successfully.
        """
        if not position.is_open:
            logger.warning(f"Position {position.id} already closed")
            return False

        success = True

        if position.leg_a:
            close_side = OrderSide.BUY if position.leg_a.side == OrderSide.SELL else OrderSide.SELL
            close_leg_a = TradeLeg(
                exchange=position.leg_a.exchange,
                symbol=position.leg_a.symbol,
                side=close_side,
            )
            result = await self._execute_leg(
                close_leg_a, position.leg_a.amount * position.leg_a.avg_price,
                position.id, "close_A"
            )
            if result is None:
                logger.error(f"Failed to close leg A of position {position.id}")
                success = False

        if position.leg_b:
            close_side = OrderSide.BUY if position.leg_b.side == OrderSide.SELL else OrderSide.SELL
            close_leg_b = TradeLeg(
                exchange=position.leg_b.exchange,
                symbol=position.leg_b.symbol,
                side=close_side,
            )
            result = await self._execute_leg(
                close_leg_b, position.leg_b.amount * position.leg_b.avg_price,
                position.id, "close_B"
            )
            if result is None:
                logger.error(f"Failed to close leg B of position {position.id}")
                success = False

        if success:
            position.closed_at = datetime.now(UTC)
            self.db.save_position(position)
            logger.info(f"Position {position.id} closed")

        return success

    async def _execute_leg(
        self, leg: TradeLeg, size_usd: float, position_id: str, label: str
    ) -> OrderResult | None:
        """Execute a single order leg with retry and verification.

        Returns OrderResult on success, None on failure after retries.
        """
        mode = self.executor_config.mode

        if mode == "paper":
            return self._paper_fill(leg, size_usd, position_id, label)

        if mode == "dry_run":
            return await self._dry_run_fill(leg, size_usd, position_id, label)

        ex = self._exchanges.get(leg.exchange)
        if ex is None:
            logger.error(f"Exchange {leg.exchange} not available")
            return None

        # Estimate amount from size_usd using the order book or last price
        try:
            ticker = await ex.fetch_ticker(leg.symbol)
            last_price = ticker.get("last", 0.0)
            if not last_price or last_price <= 0:
                logger.error(f"No valid price for {leg.symbol} on {leg.exchange}")
                return None
            amount = size_usd / last_price
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {leg.symbol} on {leg.exchange}: {e}")
            return None

        # Retry loop with exponential backoff
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[{position_id}:{label}] Attempt {attempt}: "
                    f"{leg.side.value} {amount:.6f} {leg.symbol} on {leg.exchange}"
                )
                raw_order = await ex.create_order(
                    symbol=leg.symbol,
                    type="market",
                    side=leg.side.value,
                    amount=amount,
                )

                order_id = raw_order.get("id", "")

                # Verify the fill by fetching the order
                filled_order = await self._verify_fill(ex, order_id, leg.symbol)
                if filled_order is None:
                    logger.warning(f"[{position_id}:{label}] Fill verification failed, using order response")
                    filled_order = raw_order

                avg_price = filled_order.get("average", filled_order.get("price", last_price)) or last_price
                filled_amount = filled_order.get("filled", amount) or amount
                fee_cost = 0.0
                fee_info = filled_order.get("fee")
                if fee_info and isinstance(fee_info, dict):
                    fee_cost = fee_info.get("cost", 0.0) or 0.0

                result = OrderResult(
                    order_id=str(order_id),
                    exchange=leg.exchange,
                    symbol=leg.symbol,
                    side=leg.side,
                    amount=float(filled_amount),
                    avg_price=float(avg_price),
                    fee=float(fee_cost),
                    is_filled=True,
                    raw=filled_order,
                )

                # Check slippage
                slippage_pct = abs(avg_price - last_price) / last_price * 100
                if slippage_pct > self.executor_config.max_slippage_pct:
                    logger.warning(
                        f"[{position_id}:{label}] Slippage {slippage_pct:.3f}% "
                        f"exceeds max {self.executor_config.max_slippage_pct}%"
                    )

                logger.info(
                    f"[{position_id}:{label}] Filled: {filled_amount:.6f} @ {avg_price:.2f} "
                    f"(fee={fee_cost:.4f}, slippage={slippage_pct:.3f}%)"
                )
                return result

            except Exception as e:
                logger.warning(f"[{position_id}:{label}] Attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    backoff = BASE_BACKOFF_S * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

        logger.error(f"[{position_id}:{label}] All {MAX_RETRIES} attempts failed")
        return None

    async def _verify_fill(
        self, ex: ccxt.Exchange, order_id: str, symbol: str
    ) -> dict | None:
        """Fetch the order from the exchange to verify fill status."""
        for _ in range(3):
            try:
                order = await ex.fetch_order(order_id, symbol)
                status = order.get("status", "")
                if status in ("closed", "filled"):
                    return order
                if status in ("canceled", "cancelled", "rejected", "expired"):
                    logger.error(f"Order {order_id} was {status}")
                    return None
                # Still open — wait and retry
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Verify fill error for {order_id}: {e}")
                await asyncio.sleep(0.5)
        return None

    async def _dry_run_fill(
        self, leg: TradeLeg, size_usd: float, position_id: str, label: str
    ) -> OrderResult | None:
        """Dry run: fetch real price, construct order, log it, but don't submit."""
        ex = self._exchanges.get(leg.exchange)
        if ex is None:
            logger.error(f"[DRY RUN] Exchange {leg.exchange} not available")
            return None

        try:
            ticker = await ex.fetch_ticker(leg.symbol)
            last_price = ticker.get("last", 0.0)
            if not last_price or last_price <= 0:
                logger.error(f"[DRY RUN] No valid price for {leg.symbol} on {leg.exchange}")
                return None

            bid = ticker.get("bid", last_price)
            ask = ticker.get("ask", last_price)
            amount = size_usd / last_price
            estimated_fee = size_usd * 0.0004  # maker fee estimate

            order_payload = {
                "position_id": position_id,
                "label": label,
                "exchange": leg.exchange,
                "symbol": leg.symbol,
                "side": leg.side.value,
                "amount": amount,
                "would_fill_at": last_price,
                "bid": bid,
                "ask": ask,
                "spread_bps": ((ask - bid) / last_price * 10000) if last_price > 0 else 0,
                "estimated_fee": estimated_fee,
                "size_usd": size_usd,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            logger.info(
                f"[DRY RUN] [{position_id}:{label}] Would {leg.side.value} "
                f"{amount:.6f} {leg.symbol} on {leg.exchange} @ {last_price:.2f} "
                f"(bid={bid:.2f}, ask={ask:.2f}, spread={order_payload['spread_bps']:.1f}bps, "
                f"est_fee=${estimated_fee:.4f})"
            )

            self.db.save_dry_trade(order_payload)

            return OrderResult(
                order_id=f"dry-{position_id}-{label}",
                exchange=leg.exchange,
                symbol=leg.symbol,
                side=leg.side,
                amount=amount,
                avg_price=last_price,
                fee=estimated_fee,
                is_filled=True,
                raw=order_payload,
            )

        except Exception as e:
            logger.error(f"[DRY RUN] Failed to fetch data for {leg.symbol} on {leg.exchange}: {e}")
            return None

    def _paper_fill(
        self, leg: TradeLeg, size_usd: float, position_id: str, label: str
    ) -> OrderResult:
        """Simulate a fill for paper trading."""
        # Use aggressive_price if set, otherwise assume ~$50k BTC as rough default
        price = leg.aggressive_price if leg.aggressive_price > 0 else 50000.0
        amount = size_usd / price
        order_id = f"paper-{position_id}-{label}"

        logger.info(
            f"[PAPER] [{position_id}:{label}] {leg.side.value} {amount:.6f} "
            f"{leg.symbol} @ {price:.2f} on {leg.exchange}"
        )

        return OrderResult(
            order_id=order_id,
            exchange=leg.exchange,
            symbol=leg.symbol,
            side=leg.side,
            amount=amount,
            avg_price=price,
            fee=size_usd * 0.0006,  # simulated taker fee
            is_filled=True,
        )
