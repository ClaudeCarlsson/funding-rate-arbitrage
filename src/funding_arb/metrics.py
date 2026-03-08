"""Prometheus metrics for the funding rate arbitrage system."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

if TYPE_CHECKING:
    from .models import Portfolio

logger = logging.getLogger(__name__)

# Use a custom registry so tests don't conflict with global state
REGISTRY = CollectorRegistry()

# --- Tick / Loop metrics ---
TICK_COUNT = Counter(
    "funding_arb_ticks_total",
    "Total number of orchestrator ticks",
    registry=REGISTRY,
)
TICK_DURATION = Histogram(
    "funding_arb_tick_duration_seconds",
    "Duration of each orchestrator tick",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
    registry=REGISTRY,
)
TICK_ERRORS = Counter(
    "funding_arb_tick_errors_total",
    "Total number of failed ticks",
    registry=REGISTRY,
)

# --- Exchange / Scanning ---
EXCHANGES_SCANNED = Gauge(
    "funding_arb_exchanges_scanned",
    "Number of exchanges in last scan",
    registry=REGISTRY,
)
STALE_EXCHANGES = Gauge(
    "funding_arb_stale_exchanges",
    "Number of stale exchanges in last scan",
    registry=REGISTRY,
)
FUNDING_RATES_SAVED = Counter(
    "funding_arb_funding_rates_saved_total",
    "Total funding rate observations saved",
    registry=REGISTRY,
)

# --- Opportunities ---
SIMPLE_OPPORTUNITIES = Gauge(
    "funding_arb_simple_opportunities",
    "Number of simple opportunities found in last scan",
    registry=REGISTRY,
)
GRAPH_OPPORTUNITIES = Gauge(
    "funding_arb_graph_opportunities",
    "Number of graph opportunities found in last scan",
    registry=REGISTRY,
)
BEST_YIELD_APR = Gauge(
    "funding_arb_best_yield_apr_pct",
    "Best annualized yield percentage from last scan",
    registry=REGISTRY,
)

# --- Positions / Trading ---
OPEN_POSITIONS = Gauge(
    "funding_arb_open_positions",
    "Number of currently open positions",
    registry=REGISTRY,
)
POSITIONS_OPENED = Counter(
    "funding_arb_positions_opened_total",
    "Total number of positions opened",
    registry=REGISTRY,
)
POSITIONS_CLOSED = Counter(
    "funding_arb_positions_closed_total",
    "Total number of positions closed",
    registry=REGISTRY,
)
FUNDING_COLLECTED_USD = Gauge(
    "funding_arb_funding_collected_usd",
    "Total funding collected across all positions (USD)",
    registry=REGISTRY,
)
REALIZED_PNL_USD = Gauge(
    "funding_arb_realized_pnl_usd",
    "Total realized P&L across closed positions (USD)",
    registry=REGISTRY,
)

# --- Risk ---
RISK_VIOLATIONS = Counter(
    "funding_arb_risk_violations_total",
    "Total risk violations detected",
    ["severity"],
    registry=REGISTRY,
)
KILL_SWITCH_ACTIVE = Gauge(
    "funding_arb_kill_switch_active",
    "Whether the kill switch is currently active (1=on, 0=off)",
    registry=REGISTRY,
)

# --- Portfolio ---
TOTAL_EQUITY_USD = Gauge(
    "funding_arb_total_equity_usd",
    "Total portfolio equity in USD",
    registry=REGISTRY,
)
DRAWDOWN_PCT = Gauge(
    "funding_arb_drawdown_pct",
    "Current drawdown from peak as percentage",
    registry=REGISTRY,
)


def update_portfolio_metrics(portfolio: Portfolio) -> None:
    """Update all portfolio-related gauges from a Portfolio snapshot."""
    TOTAL_EQUITY_USD.set(portfolio.total_equity)
    DRAWDOWN_PCT.set(portfolio.drawdown_from_peak * 100)

    open_count = sum(1 for p in portfolio.positions if p.is_open)
    OPEN_POSITIONS.set(open_count)

    total_funding = sum(p.funding_collected for p in portfolio.positions)
    FUNDING_COLLECTED_USD.set(total_funding)

    total_pnl = sum(p.realized_pnl for p in portfolio.positions if not p.is_open)
    REALIZED_PNL_USD.set(total_pnl)


_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics server (idempotent)."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        try:
            start_http_server(port, registry=REGISTRY)
            _server_started = True
            logger.info(f"Prometheus metrics server started on :{port}")
        except OSError as e:
            logger.warning(f"Could not start metrics server on :{port}: {e}")
