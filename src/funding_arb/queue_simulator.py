"""Limit Order Book (LOB) FIFO Queue Simulator for realistic paper trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimulatedOrder:
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    queue_position: float  # Q(t): volume ahead of us in the queue
    filled_amount: float = 0.0
    is_active: bool = True


class FIFOQueueSimulator:
    """
    Simulates realistic queue position decay using stochastic differential equations.
    Prevents the 'Queue Illusion' in paper trading environments.
    """

    def __init__(self, cancellation_hazard_rate: float = 0.1):
        # theta: the rate at which resting orders ahead of us are cancelled
        self.theta = cancellation_hazard_rate
        self.active_orders: dict[str, SimulatedOrder] = {}

    def place_order(self, order_id: str, side: str, price: float, amount: float, current_book_depth_at_price: float) -> None:
        """
        Place an order at the back of the FIFO queue.
        current_book_depth_at_price becomes our initial Q(0).
        """
        self.active_orders[order_id] = SimulatedOrder(
            order_id=order_id,
            side=side,
            price=price,
            amount=amount,
            queue_position=current_book_depth_at_price
        )
        logger.debug(f"Order {order_id} placed. Queue ahead: {current_book_depth_at_price:.2f}")

    def cancel_order(self, order_id: str) -> None:
        if order_id in self.active_orders:
            self.active_orders[order_id].is_active = False

    def process_tick(self, dt: float, aggressive_buy_volume: float, aggressive_sell_volume: float) -> list[tuple[str, float]]:
        """
        Decay queue positions based on time elapsed (dt) and aggressive taker volume hitting our levels.
        Returns a list of tuples: (order_id, fill_amount)
        """
        fills = []

        for order_id, order in self.active_orders.items():
            if not order.is_active or order.filled_amount >= order.amount:
                continue

            # mu(t): aggressive volume hitting our side of the book
            # If we are a 'buy' (bid), we get hit by aggressive 'sell' volume
            mu = aggressive_sell_volume if order.side == 'buy' else aggressive_buy_volume

            # Decay equation: dQ = -(mu + theta * Q) * dt
            # If Q is large, cancellations (theta * Q) dominate.
            # If Q is small, aggressive takers (mu) dominate.
            dQ = (mu + (self.theta * order.queue_position)) * dt

            order.queue_position -= dQ

            # If we reached the front of the queue, we start getting filled
            if order.queue_position <= 0:
                # The "overshoot" of the queue position is volume that hits our order
                available_to_fill = abs(order.queue_position)

                # Cap fill by our remaining order size
                fill_qty = min(available_to_fill, order.amount - order.filled_amount)

                if fill_qty > 0:
                    order.filled_amount += fill_qty
                    fills.append((order_id, fill_qty))

                # Reset queue position to 0 since we are at the absolute front
                order.queue_position = 0.0

            if order.filled_amount >= order.amount:
                order.is_active = False

        return fills
