"""Tests for the Prometheus metrics module."""
from __future__ import annotations

from funding_arb.metrics import (
    BEST_YIELD_APR,
    EXCHANGES_SCANNED,
    FUNDING_RATES_SAVED,
    KILL_SWITCH_ACTIVE,
    OPEN_POSITIONS,
    POSITIONS_OPENED,
    REGISTRY,
    RISK_VIOLATIONS,
    SIMPLE_OPPORTUNITIES,
    TICK_COUNT,
    TICK_DURATION,
    TICK_ERRORS,
    TOTAL_EQUITY_USD,
    update_portfolio_metrics,
)
from funding_arb.models import ArbitragePosition, Portfolio


class TestMetricsExist:
    def test_tick_count_increments(self):
        before = REGISTRY.get_sample_value("funding_arb_ticks_total") or 0
        TICK_COUNT.inc()
        after = REGISTRY.get_sample_value("funding_arb_ticks_total")
        assert after == before + 1

    def test_tick_errors_increments(self):
        before = REGISTRY.get_sample_value("funding_arb_tick_errors_total") or 0
        TICK_ERRORS.inc()
        after = REGISTRY.get_sample_value("funding_arb_tick_errors_total")
        assert after == before + 1

    def test_tick_duration_observes(self):
        TICK_DURATION.observe(1.5)
        # Just verify no exception is raised

    def test_gauge_set(self):
        EXCHANGES_SCANNED.set(3)
        val = REGISTRY.get_sample_value("funding_arb_exchanges_scanned")
        assert val == 3

    def test_simple_opportunities_gauge(self):
        SIMPLE_OPPORTUNITIES.set(5)
        val = REGISTRY.get_sample_value("funding_arb_simple_opportunities")
        assert val == 5

    def test_best_yield_apr(self):
        BEST_YIELD_APR.set(42.5)
        val = REGISTRY.get_sample_value("funding_arb_best_yield_apr_pct")
        assert val == 42.5

    def test_kill_switch_gauge(self):
        KILL_SWITCH_ACTIVE.set(1)
        val = REGISTRY.get_sample_value("funding_arb_kill_switch_active")
        assert val == 1
        KILL_SWITCH_ACTIVE.set(0)
        val = REGISTRY.get_sample_value("funding_arb_kill_switch_active")
        assert val == 0

    def test_risk_violations_labelled(self):
        before_w = REGISTRY.get_sample_value(
            "funding_arb_risk_violations_total", {"severity": "warning"}
        ) or 0
        before_c = REGISTRY.get_sample_value(
            "funding_arb_risk_violations_total", {"severity": "critical"}
        ) or 0
        RISK_VIOLATIONS.labels(severity="warning").inc()
        RISK_VIOLATIONS.labels(severity="critical").inc(2)
        assert REGISTRY.get_sample_value(
            "funding_arb_risk_violations_total", {"severity": "warning"}
        ) == before_w + 1
        assert REGISTRY.get_sample_value(
            "funding_arb_risk_violations_total", {"severity": "critical"}
        ) == before_c + 2

    def test_positions_opened_counter(self):
        before = REGISTRY.get_sample_value("funding_arb_positions_opened_total") or 0
        POSITIONS_OPENED.inc()
        after = REGISTRY.get_sample_value("funding_arb_positions_opened_total")
        assert after == before + 1

    def test_funding_rates_saved(self):
        before = REGISTRY.get_sample_value("funding_arb_funding_rates_saved_total") or 0
        FUNDING_RATES_SAVED.inc(10)
        after = REGISTRY.get_sample_value("funding_arb_funding_rates_saved_total")
        assert after == before + 10


class TestUpdatePortfolioMetrics:
    def test_update_with_empty_portfolio(self):
        portfolio = Portfolio()
        update_portfolio_metrics(portfolio)
        assert REGISTRY.get_sample_value("funding_arb_total_equity_usd") == 0
        assert REGISTRY.get_sample_value("funding_arb_open_positions") == 0

    def test_update_with_positions(self):
        portfolio = Portfolio(
            equity_by_exchange={"binance": 500.0, "bybit": 300.0},
            peak_equity=1000.0,
        )
        pos_open = ArbitragePosition(id="a")
        pos_closed = ArbitragePosition(id="b", closed_at=pos_open.opened_at)
        pos_closed.realized_pnl = 15.0
        pos_open.funding_collected = 5.0
        portfolio.positions = [pos_open, pos_closed]

        update_portfolio_metrics(portfolio)

        assert REGISTRY.get_sample_value("funding_arb_total_equity_usd") == 800.0
        assert REGISTRY.get_sample_value("funding_arb_open_positions") == 1
        assert REGISTRY.get_sample_value("funding_arb_funding_collected_usd") == 5.0
        assert REGISTRY.get_sample_value("funding_arb_realized_pnl_usd") == 15.0
