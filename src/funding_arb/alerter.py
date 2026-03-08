"""Telegram alerter for system notifications."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ArbitragePosition, Portfolio, Violation

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends alerts via Telegram bot API.

    Requires environment variables:
        TELEGRAM_BOT_TOKEN: Bot API token from @BotFather
        TELEGRAM_CHAT_ID: Chat ID to send messages to
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self._rate_limit_s = 1.0  # min seconds between messages
        self._last_send_time: float = 0

        if not self.enabled:
            logger.info("Telegram alerter disabled (missing token or chat_id)")

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            logger.debug(f"Alert (disabled): {text[:100]}...")
            return False

        # Rate limiting
        now = asyncio.get_event_loop().time()
        if now - self._last_send_time < self._rate_limit_s:
            await asyncio.sleep(self._rate_limit_s - (now - self._last_send_time))

        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            async with aiohttp.ClientSession() as session, session.post(url, json=payload) as resp:
                if resp.status == 200:
                    self._last_send_time = asyncio.get_event_loop().time()
                    return True
                else:
                    body = await resp.text()
                    logger.error(f"Telegram API error {resp.status}: {body}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def notify_new_position(self, position: ArbitragePosition) -> None:
        """Alert when a new arbitrage position is opened."""
        legs = []
        if position.leg_a:
            legs.append(
                f"  Leg A: {position.leg_a.side.value.upper()} {position.leg_a.amount:.4f} "
                f"{position.leg_a.symbol} @ {position.leg_a.avg_price:.2f} on {position.leg_a.exchange}"
            )
        if position.leg_b:
            legs.append(
                f"  Leg B: {position.leg_b.side.value.upper()} {position.leg_b.amount:.4f} "
                f"{position.leg_b.symbol} @ {position.leg_b.avg_price:.2f} on {position.leg_b.exchange}"
            )

        text = (
            f"<b>NEW POSITION OPENED</b>\n"
            f"ID: <code>{position.id}</code>\n"
            f"Funding rate: {position.entry_funding_rate:.6f}\n"
            f"Notional: ${position.notional_usd:,.2f}\n"
            + "\n".join(legs)
        )
        await self.send_message(text)

    async def notify_position_closed(
        self, position: ArbitragePosition, reason: str = ""
    ) -> None:
        """Alert when a position is closed."""
        text = (
            f"<b>POSITION CLOSED</b>\n"
            f"ID: <code>{position.id}</code>\n"
            f"Reason: {reason or 'manual'}\n"
            f"P&L: ${position.realized_pnl:+,.2f}\n"
            f"Funding collected: ${position.funding_collected:,.2f}\n"
            f"Duration: {self._format_duration(position)}"
        )
        await self.send_message(text)

    async def notify_risk_violation(self, violation: Violation) -> None:
        """Alert on risk violations."""
        emoji = "\U0001f534" if violation.severity == "critical" else "\U0001f7e1"
        text = (
            f"{emoji} <b>RISK ALERT ({violation.severity.upper()})</b>\n"
            f"Type: {violation.type.name}\n"
            f"{violation.message}"
        )
        await self.send_message(text)

    async def notify_emergency_unwind(self, details: str = "") -> None:
        """Alert on emergency position unwind."""
        text = (
            f"\U0001f6a8 <b>EMERGENCY UNWIND</b>\n"
            f"An emergency position unwind was triggered.\n"
            f"{details}"
        )
        await self.send_message(text)

    async def notify_daily_summary(self, portfolio: Portfolio) -> None:
        """Send daily P&L and portfolio summary."""
        open_positions = [p for p in portfolio.positions if p.is_open]
        total_funding = sum(p.funding_collected for p in portfolio.positions)
        total_pnl = sum(p.realized_pnl for p in portfolio.positions if not p.is_open)

        text = (
            f"<b>DAILY SUMMARY</b>\n"
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            f"Total equity: ${portfolio.total_equity:,.2f}\n"
            f"Peak equity: ${portfolio.peak_equity:,.2f}\n"
            f"Drawdown: {portfolio.drawdown_from_peak*100:.2f}%\n"
            f"\n"
            f"Open positions: {len(open_positions)}\n"
            f"Total funding earned: ${total_funding:,.2f}\n"
            f"Realized P&L: ${total_pnl:+,.2f}\n"
            f"\n"
            f"<b>Equity by exchange:</b>\n"
        )
        for ex, eq in sorted(portfolio.equity_by_exchange.items()):
            pct = eq / max(portfolio.total_equity, 1) * 100
            text += f"  {ex}: ${eq:,.2f} ({pct:.1f}%)\n"

        await self.send_message(text)

    async def notify_opportunity(self, opp: dict[str, Any]) -> None:
        """Alert on a new funding rate opportunity."""
        text = (
            f"<b>OPPORTUNITY DETECTED</b>\n"
            f"Instrument: {opp.get('instrument', 'N/A')}\n"
            f"Short: {opp.get('short_exchange', '')} ({opp.get('short_rate', 0):.6f})\n"
            f"Long: {opp.get('long_exchange', '')} ({opp.get('long_rate', 0):.6f})\n"
            f"Net yield: {opp.get('net_yield_per_period', 0):.6f}/period\n"
            f"Annualized: {opp.get('annualized_yield', 0)*100:.1f}% APR"
        )
        await self.send_message(text)

    async def notify_system_start(self) -> None:
        """Alert when the system starts."""
        text = (
            f"<b>SYSTEM STARTED</b>\n"
            f"Funding Rate Arbitrage System is online.\n"
            f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        await self.send_message(text)

    async def notify_system_stop(self, reason: str = "") -> None:
        """Alert when the system stops."""
        text = (
            f"<b>SYSTEM STOPPED</b>\n"
            f"Reason: {reason or 'normal shutdown'}\n"
            f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        await self.send_message(text)

    async def notify_scan_failure(self, exchange: str, error: str) -> None:
        """Alert when an exchange scan fails."""
        text = (
            f"<b>SCAN FAILURE</b>\n"
            f"Exchange: {exchange}\n"
            f"Error: {error[:200]}"
        )
        await self.send_message(text)

    def _format_duration(self, position: ArbitragePosition) -> str:
        """Format position duration as human-readable string."""
        if position.closed_at is None:
            return "still open"
        delta = position.closed_at - position.opened_at
        hours = delta.total_seconds() / 3600
        if hours < 24:
            return f"{hours:.1f}h"
        days = hours / 24
        return f"{days:.1f}d"


class LogAlerter:
    """Fallback alerter that logs instead of sending Telegram messages.

    Same interface as TelegramAlerter for use in testing/development.
    """

    async def send_message(self, text: str, **kwargs: Any) -> bool:
        logger.info(f"[ALERT] {text}")
        return True

    async def notify_new_position(self, position: ArbitragePosition) -> None:
        logger.info(f"[ALERT] New position: {position.id}, notional=${position.notional_usd:,.2f}")

    async def notify_position_closed(self, position: ArbitragePosition, reason: str = "") -> None:
        logger.info(f"[ALERT] Position closed: {position.id}, reason={reason}")

    async def notify_risk_violation(self, violation: Violation) -> None:
        logger.warning(f"[ALERT] Risk violation: {violation.message}")

    async def notify_emergency_unwind(self, details: str = "") -> None:
        logger.critical(f"[ALERT] Emergency unwind: {details}")

    async def notify_daily_summary(self, portfolio: Portfolio) -> None:
        logger.info(f"[ALERT] Daily summary: equity=${portfolio.total_equity:,.2f}")

    async def notify_opportunity(self, opp: dict[str, Any]) -> None:
        logger.info(f"[ALERT] Opportunity: {opp.get('instrument', 'N/A')}")

    async def notify_system_start(self) -> None:
        logger.info("[ALERT] System started")

    async def notify_system_stop(self, reason: str = "") -> None:
        logger.info(f"[ALERT] System stopped: {reason}")

    async def notify_scan_failure(self, exchange: str, error: str) -> None:
        logger.warning(f"[ALERT] Scan failure: {exchange}: {error}")
