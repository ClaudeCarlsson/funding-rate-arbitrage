"""Tests for the alerter module."""
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from funding_arb.alerter import LogAlerter, TelegramAlerter
from funding_arb.models import (
    ArbitragePosition,
    OrderResult,
    OrderSide,
    Portfolio,
    Violation,
    ViolationType,
)


class TestTelegramAlerterInit:
    def test_disabled_without_credentials(self):
        alerter = TelegramAlerter()
        assert not alerter.enabled

    def test_disabled_explicitly(self):
        alerter = TelegramAlerter(bot_token="test", chat_id="123", enabled=False)
        assert not alerter.enabled

    def test_enabled_with_credentials(self):
        alerter = TelegramAlerter(bot_token="test", chat_id="123")
        assert alerter.enabled


class TestLogAlerter:
    @pytest.fixture
    def alerter(self):
        return LogAlerter()

    @pytest.mark.asyncio
    async def test_send_message(self, alerter):
        result = await alerter.send_message("test message")
        assert result is True

    @pytest.mark.asyncio
    async def test_notify_new_position(self, alerter):
        pos = ArbitragePosition(
            id="test-123",
            leg_a=OrderResult(
                order_id="a", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.SELL, amount=0.1, avg_price=50000, fee=3.0,
                is_filled=True,
            ),
        )
        await alerter.notify_new_position(pos)

    @pytest.mark.asyncio
    async def test_notify_risk_violation(self, alerter):
        v = Violation(
            type=ViolationType.DELTA_DRIFT,
            message="Delta too high",
            severity="critical",
        )
        await alerter.notify_risk_violation(v)

    @pytest.mark.asyncio
    async def test_notify_daily_summary(self, alerter):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 5000, "bybit": 5000},
            peak_equity=10000,
        )
        await alerter.notify_daily_summary(portfolio)

    @pytest.mark.asyncio
    async def test_notify_opportunity(self, alerter):
        opp = {
            "instrument": "BTC/USDT:USDT",
            "short_exchange": "hyperliquid",
            "long_exchange": "binance",
            "short_rate": 0.003,
            "long_rate": 0.0005,
            "net_yield_per_period": 0.0025,
            "annualized_yield": 0.27,
        }
        await alerter.notify_opportunity(opp)

    @pytest.mark.asyncio
    async def test_notify_system_lifecycle(self, alerter):
        await alerter.notify_system_start()
        await alerter.notify_system_stop("test shutdown")

    @pytest.mark.asyncio
    async def test_notify_scan_failure(self, alerter):
        await alerter.notify_scan_failure("binance", "Connection timeout")

    @pytest.mark.asyncio
    async def test_notify_emergency_unwind(self, alerter):
        await alerter.notify_emergency_unwind("Leg A failed")

    @pytest.mark.asyncio
    async def test_notify_position_closed(self, alerter):
        pos = ArbitragePosition(
            id="close-1",
            leg_a=OrderResult(
                order_id="a", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.SELL, amount=0.1, avg_price=50000, fee=3.0,
                is_filled=True,
            ),
            realized_pnl=42.0,
            funding_collected=15.0,
        )
        await alerter.notify_position_closed(pos, reason="target hit")


# ---------------------------------------------------------------------------
# Helpers for TelegramAlerter tests
# ---------------------------------------------------------------------------

def _make_mock_session(status: int = 200, body_text: str = "ok"):
    """Create a fully mocked aiohttp.ClientSession context manager."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=body_text)

    # response context manager (async with session.post(...) as resp)
    resp_cm = AsyncMock()
    resp_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    resp_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=resp_cm)

    return mock_session, mock_resp


def _make_enabled_alerter() -> TelegramAlerter:
    alerter = TelegramAlerter(bot_token="test-token", chat_id="12345")
    # Reset rate-limit so we never sleep in tests
    alerter._last_send_time = 0
    return alerter


def _sample_position(
    closed: bool = False,
    hours_open: float = 5.0,
) -> ArbitragePosition:
    opened = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    closed_at = opened + timedelta(hours=hours_open) if closed else None
    return ArbitragePosition(
        id="pos-abc",
        leg_a=OrderResult(
            order_id="a1", exchange="binance", symbol="BTC/USDT:USDT",
            side=OrderSide.SELL, amount=0.5, avg_price=40000, fee=5.0,
            is_filled=True,
        ),
        leg_b=OrderResult(
            order_id="b1", exchange="bybit", symbol="BTC/USDT:USDT",
            side=OrderSide.BUY, amount=0.5, avg_price=40050, fee=5.0,
            is_filled=True,
        ),
        entry_funding_rate=0.0003,
        opened_at=opened,
        closed_at=closed_at,
        realized_pnl=120.50 if closed else 0.0,
        funding_collected=35.25,
    )


def _sample_portfolio() -> Portfolio:
    closed_pos = _sample_position(closed=True)
    open_pos = _sample_position(closed=False)
    return Portfolio(
        positions=[open_pos, closed_pos],
        equity_by_exchange={"binance": 5000.0, "bybit": 3000.0},
        peak_equity=9000.0,
    )


# ---------------------------------------------------------------------------
# TelegramAlerter.send_message
# ---------------------------------------------------------------------------

class TestTelegramAlerterSendMessage:

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """send_message returns True on HTTP 200."""
        alerter = _make_enabled_alerter()
        mock_session, _ = _make_mock_session(status=200)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await alerter.send_message("hello world")

        assert result is True
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert "sendMessage" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["text"] == "hello world"
        assert call_kwargs[1]["json"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_send_message_non_200(self):
        """send_message returns False on non-200 response."""
        alerter = _make_enabled_alerter()
        mock_session, _ = _make_mock_session(status=403, body_text="Forbidden")

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await alerter.send_message("test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """send_message returns False when aiohttp raises."""
        alerter = _make_enabled_alerter()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=ConnectionError("no network"))
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await alerter.send_message("test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_disabled(self):
        """send_message returns False when alerter is disabled."""
        alerter = TelegramAlerter()  # no credentials => disabled
        result = await alerter.send_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_rate_limiting(self):
        """send_message sleeps when calls are too close together."""
        alerter = _make_enabled_alerter()
        mock_session, _ = _make_mock_session(status=200)

        loop = asyncio.get_event_loop()
        # Pretend the last send was just now so rate limiter kicks in
        alerter._last_send_time = loop.time()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await alerter.send_message("rate limited")

        assert result is True
        mock_sleep.assert_awaited_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg > 0


# ---------------------------------------------------------------------------
# TelegramAlerter notification methods
# ---------------------------------------------------------------------------

class TestTelegramAlerterNotifications:
    """Tests that each notify_* method composes the correct text and calls send_message."""

    @pytest.fixture
    def alerter(self):
        a = _make_enabled_alerter()
        a.send_message = AsyncMock(return_value=True)
        return a

    # -- notify_new_position --

    @pytest.mark.asyncio
    async def test_notify_new_position_both_legs(self, alerter):
        pos = _sample_position(closed=False)
        await alerter.notify_new_position(pos)

        alerter.send_message.assert_awaited_once()
        text = alerter.send_message.call_args[0][0]
        assert "NEW POSITION OPENED" in text
        assert pos.id in text
        assert "Leg A" in text
        assert "Leg B" in text
        assert "binance" in text
        assert "bybit" in text
        assert "0.0003" in text  # entry_funding_rate

    @pytest.mark.asyncio
    async def test_notify_new_position_single_leg(self, alerter):
        pos = ArbitragePosition(
            id="single-leg",
            leg_a=OrderResult(
                order_id="a1", exchange="binance", symbol="ETH/USDT:USDT",
                side=OrderSide.BUY, amount=1.0, avg_price=3000, fee=2.0,
                is_filled=True,
            ),
        )
        await alerter.notify_new_position(pos)

        text = alerter.send_message.call_args[0][0]
        assert "Leg A" in text
        assert "Leg B" not in text

    # -- notify_position_closed --

    @pytest.mark.asyncio
    async def test_notify_position_closed(self, alerter):
        pos = _sample_position(closed=True, hours_open=12.5)
        await alerter.notify_position_closed(pos, reason="funding reversal")

        alerter.send_message.assert_awaited_once()
        text = alerter.send_message.call_args[0][0]
        assert "POSITION CLOSED" in text
        assert pos.id in text
        assert "funding reversal" in text
        assert "+120.50" in text  # realized_pnl
        assert "35.25" in text  # funding_collected
        assert "12.5h" in text  # duration

    # -- notify_risk_violation --

    @pytest.mark.asyncio
    async def test_notify_risk_violation_critical(self, alerter):
        v = Violation(
            type=ViolationType.DELTA_DRIFT,
            message="Delta drift exceeded 5%",
            severity="critical",
        )
        await alerter.notify_risk_violation(v)

        text = alerter.send_message.call_args[0][0]
        assert "RISK ALERT (CRITICAL)" in text
        assert "DELTA_DRIFT" in text
        assert "Delta drift exceeded 5%" in text
        assert "\U0001f534" in text  # red circle

    @pytest.mark.asyncio
    async def test_notify_risk_violation_warning(self, alerter):
        v = Violation(
            type=ViolationType.LOW_COLLATERAL,
            message="Collateral below threshold",
            severity="warning",
        )
        await alerter.notify_risk_violation(v)

        text = alerter.send_message.call_args[0][0]
        assert "RISK ALERT (WARNING)" in text
        assert "LOW_COLLATERAL" in text
        assert "\U0001f7e1" in text  # yellow circle

    # -- notify_emergency_unwind --

    @pytest.mark.asyncio
    async def test_notify_emergency_unwind(self, alerter):
        await alerter.notify_emergency_unwind("Margin call on bybit")

        text = alerter.send_message.call_args[0][0]
        assert "EMERGENCY UNWIND" in text
        assert "Margin call on bybit" in text
        assert "\U0001f6a8" in text  # siren

    # -- notify_daily_summary --

    @pytest.mark.asyncio
    async def test_notify_daily_summary(self, alerter):
        portfolio = _sample_portfolio()
        await alerter.notify_daily_summary(portfolio)

        text = alerter.send_message.call_args[0][0]
        assert "DAILY SUMMARY" in text
        assert "8,000.00" in text  # total_equity = 5000 + 3000
        assert "9,000.00" in text  # peak_equity
        assert "Open positions: 1" in text  # one open, one closed
        assert "70.50" in text  # total funding from both positions (35.25 * 2)
        assert "+120.50" in text  # realized pnl from closed position
        assert "binance" in text
        assert "bybit" in text

    # -- notify_opportunity --

    @pytest.mark.asyncio
    async def test_notify_opportunity(self, alerter):
        opp = {
            "instrument": "ETH/USDT:USDT",
            "short_exchange": "hyperliquid",
            "long_exchange": "binance",
            "short_rate": 0.003,
            "long_rate": 0.0005,
            "net_yield_per_period": 0.0025,
            "annualized_yield": 0.27,
        }
        await alerter.notify_opportunity(opp)

        text = alerter.send_message.call_args[0][0]
        assert "OPPORTUNITY DETECTED" in text
        assert "ETH/USDT:USDT" in text
        assert "hyperliquid" in text
        assert "binance" in text
        assert "0.003000" in text
        assert "0.000500" in text
        assert "0.002500" in text
        assert "27.0% APR" in text

    # -- notify_system_start --

    @pytest.mark.asyncio
    async def test_notify_system_start(self, alerter):
        await alerter.notify_system_start()

        text = alerter.send_message.call_args[0][0]
        assert "SYSTEM STARTED" in text
        assert "online" in text
        assert "UTC" in text

    # -- notify_system_stop --

    @pytest.mark.asyncio
    async def test_notify_system_stop(self, alerter):
        await alerter.notify_system_stop("maintenance window")

        text = alerter.send_message.call_args[0][0]
        assert "SYSTEM STOPPED" in text
        assert "maintenance window" in text
        assert "UTC" in text

    # -- notify_scan_failure --

    @pytest.mark.asyncio
    async def test_notify_scan_failure(self, alerter):
        await alerter.notify_scan_failure("okx", "Rate limit exceeded")

        text = alerter.send_message.call_args[0][0]
        assert "SCAN FAILURE" in text
        assert "okx" in text
        assert "Rate limit exceeded" in text


# ---------------------------------------------------------------------------
# TelegramAlerter._format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:

    def test_format_duration_hours(self):
        alerter = _make_enabled_alerter()
        pos = _sample_position(closed=True, hours_open=5.0)
        assert alerter._format_duration(pos) == "5.0h"

    def test_format_duration_days(self):
        alerter = _make_enabled_alerter()
        pos = _sample_position(closed=True, hours_open=72.0)
        assert alerter._format_duration(pos) == "3.0d"

    def test_format_duration_still_open(self):
        alerter = _make_enabled_alerter()
        pos = _sample_position(closed=False)
        assert alerter._format_duration(pos) == "still open"

    def test_format_duration_boundary_24h(self):
        """Exactly 24 hours should render as days."""
        alerter = _make_enabled_alerter()
        pos = _sample_position(closed=True, hours_open=24.0)
        assert alerter._format_duration(pos) == "1.0d"
