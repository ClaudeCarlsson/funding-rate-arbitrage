"""Avellaneda-Stoikov Market Making Engine for Passive Execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OptimalQuotes:
    bid_price: float
    ask_price: float
    reservation_price: float
    spread: float


class AvellanedaStoikovEngine:
    """
    Stochastic control engine for optimal limit order placement.
    Minimizes inventory risk while maximizing maker rebate capture.
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,    # gamma
        order_intensity: float = 1.5,  # kappa: likelihood of fill at distance
        horizon_sec: float = 60.0      # T: execution time window
    ):
        self.gamma = risk_aversion
        self.kappa = order_intensity
        self.T = horizon_sec

    def calculate_quotes(
        self,
        mid_price: float,
        volatility: float,
        current_q: float,
        target_q: float,
        time_elapsed: float = 0.0
    ) -> OptimalQuotes:
        """
        Compute optimal bid/ask offsets from mid-price.

        Args:
            mid_price: Current fair value from Particle Filter.
            volatility: Price variance (sigma^2) from Particle Filter.
            current_q: Our current position in base asset units.
            target_q: The target position from the Brain Node.
            time_elapsed: Seconds since the start of the execution window.
        """
        # Time remaining in the horizon
        t_remaining = max(0.01, self.T - time_elapsed)

        # Inventory relative to target
        q_error = current_q - target_q

        # 1. Reservation Price: Skew the mid-price based on inventory risk
        # r = s - q * gamma * sigma^2 * (T - t)
        reservation_price = mid_price - (q_error * self.gamma * volatility * t_remaining)

        # 2. Optimal Spread: Based on risk aversion and market depth (kappa)
        # Simplified AS spread formula: delta = (2/gamma) * ln(1 + gamma/kappa)
        # Adjusted for dynamic volatility
        spread = (self.gamma * volatility * t_remaining) + (2 / self.gamma) * np.log(1 + (self.gamma / self.kappa))

        # 3. Final Quotes centered around reservation price
        half_spread = spread / 2.0
        bid = reservation_price - half_spread
        ask = reservation_price + half_spread

        # Edge case: Ensure we don't cross our own spread (bid < fair < ask logic)
        # but in AS, the entire pocket can move above/below mid-price to force fills.

        return OptimalQuotes(
            bid_price=bid,
            ask_price=ask,
            reservation_price=reservation_price,
            spread=spread
        )

    def get_order_sizes(self, current_q: float, target_q: float, max_batch: float = 1.0) -> tuple[float, float]:
        """Determine sizes for the resting bid and ask based on distance to target."""
        q_error = target_q - current_q

        if q_error > 0:
            # We need to buy
            return (min(q_error, max_batch), 0.0)
        elif q_error < 0:
            # We need to sell
            return (0.0, min(abs(q_error), max_batch))

        return (0.0, 0.0)
