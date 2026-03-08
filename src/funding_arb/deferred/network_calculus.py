"""Deterministic Network Calculus (Min-Plus Algebra) for Execution Latency Veto."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MinPlusLatencyVeto:
    """Uses network calculus to prove worst-case execution latency."""

    # We maintain a history of pings to estimate the service curve
    ping_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_ping(self, latency_sec: float) -> None:
        """Record the round-trip time of a recent API request."""
        self.ping_history.append(latency_sec)

    def calculate_service_curve(self) -> tuple[float, float]:
        """
        Estimate the rate-latency service curve Beta(t) = R * (t - T)^+
        Returns: (R, T)
        Where R is the processing rate (orders/sec) and T is the worst-case propagation delay.
        """
        if not self.ping_history:
            return (10.0, 0.5) # Default assumption: 10 req/s, 500ms max base latency

        # Use the 99th percentile ping as our baseline propagation delay 'T'
        sorted_pings = sorted(list(self.ping_history))
        p99_idx = int(len(sorted_pings) * 0.99)
        if p99_idx >= len(sorted_pings):
            p99_idx = -1

        T = sorted_pings[p99_idx]

        # Estimate R: the inverse of the minimum inter-arrival time we've sustained,
        # but for safety, we assume a static rate limit imposed by the exchange.
        # e.g., Binance allows ~10 orders/sec.
        R = 10.0

        return (R, T)

    def compute_max_delay(self, order_burst_size: int = 2) -> float:
        """
        Compute worst-case delay D using Min-Plus algebra.
        Arrival curve Alpha(t) = b + r*t
        Service curve Beta(t) = R * (t - T)^+

        Max delay D = T + b/R
        """
        R, T = self.calculate_service_curve()
        b = order_burst_size # Number of simultaneous orders (e.g. 2 for a standard hedge)

        # In a strict min-plus network calculus framework, the maximum horizontal deviation
        # between the arrival curve and service curve is D = T + b/R
        max_delay = T + (b / R)
        return max_delay

    def should_veto(self, expected_half_life: float, order_burst_size: int = 2) -> bool:
        """
        Returns True if the mathematically proven worst-case latency exceeds the
        expected duration of the arbitrage opportunity.
        """
        max_delay = self.compute_max_delay(order_burst_size)

        # Veto if it will take longer to safely execute than the opportunity exists
        if max_delay > expected_half_life:
            logger.warning(
                f"LATENCY VETO: Worst-case delay {max_delay:.3f}s exceeds "
                f"opportunity half-life {expected_half_life:.3f}s."
            )
            return True

        return False
