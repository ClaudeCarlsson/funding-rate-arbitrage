"""Tests for the balance rebalancer module."""
from __future__ import annotations

import pytest

from funding_arb.models import Portfolio
from funding_arb.rebalancer import Rebalancer, SkewAlert


def _make_portfolio(equity_by_exchange: dict[str, float]) -> Portfolio:
    return Portfolio(equity_by_exchange=equity_by_exchange)


class TestCheckSkew:
    def test_no_skew(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 500, "bybit": 500})
        alerts = r.check_skew(portfolio)
        assert alerts == []

    def test_warning_skew(self):
        r = Rebalancer(skew_alert_pct=0.65, skew_critical_pct=0.80)
        portfolio = _make_portfolio({"binance": 700, "bybit": 300})
        alerts = r.check_skew(portfolio)
        assert len(alerts) == 1
        assert alerts[0].exchange == "binance"
        assert alerts[0].severity == "warning"

    def test_critical_skew(self):
        r = Rebalancer(skew_alert_pct=0.65, skew_critical_pct=0.80)
        portfolio = _make_portfolio({"binance": 900, "bybit": 100})
        alerts = r.check_skew(portfolio)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_empty_portfolio(self):
        r = Rebalancer()
        portfolio = _make_portfolio({})
        alerts = r.check_skew(portfolio)
        assert alerts == []

    def test_single_exchange_always_at_100pct(self):
        r = Rebalancer(skew_alert_pct=0.65)
        portfolio = _make_portfolio({"binance": 1000})
        alerts = r.check_skew(portfolio)
        assert len(alerts) == 1
        assert alerts[0].pct == 1.0

    def test_three_exchanges_balanced(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 333, "bybit": 333, "okx": 334})
        alerts = r.check_skew(portfolio)
        assert alerts == []

    def test_alert_message_format(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 800, "bybit": 200})
        alerts = r.check_skew(portfolio)
        assert len(alerts) == 1
        assert "binance" in alerts[0].message
        assert "%" in alerts[0].message


class TestSuggestTransfers:
    def test_balanced_no_suggestions(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 500, "bybit": 500})
        suggestions = r.suggest_transfers(portfolio)
        assert suggestions == []

    def test_suggests_transfer_from_surplus_to_deficit(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 800, "bybit": 200})
        suggestions = r.suggest_transfers(portfolio)
        assert len(suggestions) == 1
        assert suggestions[0]["from"] == "binance"
        assert suggestions[0]["to"] == "bybit"
        assert suggestions[0]["amount_usd"] == 300.0

    def test_three_way_rebalance(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 600, "bybit": 300, "okx": 100})
        suggestions = r.suggest_transfers(portfolio)
        total_transferred = sum(s["amount_usd"] for s in suggestions)
        # Should move ~266.67 to each exchange; total transfers match surplus
        assert total_transferred > 0
        assert all(s["from"] == "binance" for s in suggestions)

    def test_single_exchange_no_suggestions(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 1000})
        suggestions = r.suggest_transfers(portfolio)
        assert suggestions == []

    def test_small_amounts_ignored(self):
        r = Rebalancer()
        portfolio = _make_portfolio({"binance": 505, "bybit": 495})
        suggestions = r.suggest_transfers(portfolio)
        # $5 difference is below $10 threshold
        assert suggestions == []
