"""VPIN (Volume-Synchronized Probability of Informed Trading) for Toxic Flow Detection."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    timestamp: float
    price: float
    amount: float
    side: str # 'buy' or 'sell' (taker side)


class VPINDetector:
    """
    Detects toxic order flow by analyzing the entropy of trade volume buckets.
    """

    def __init__(self, bucket_volume: float = 1.0, num_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets

        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0

        # We store the (buy_vol - sell_vol) imbalance for each bucket
        self.imbalances = deque(maxlen=num_buckets)

    def process_trade(self, trade: Trade) -> float | None:
        """
        Ingest a new trade from the tape.
        Returns current VPIN value if a bucket was completed, else None.
        """
        if trade.side == 'buy':
            self.current_bucket_buy += trade.amount
        else:
            self.current_bucket_sell += trade.amount

        current_total = self.current_bucket_buy + self.current_bucket_sell

        if current_total >= self.bucket_volume:
            # Bucket completed. Calculate imbalance
            imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
            self.imbalances.append(imbalance)

            # Reset for next bucket
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0

            return self.calculate_vpin()

        return None

    def calculate_vpin(self) -> float:
        """
        VPIN = sum(|Buy - Sell|) / (n * V_bucket)
        Ranges from 0 (perfectly balanced) to 1 (purely directional/toxic).
        """
        if not self.imbalances:
            return 0.0

        total_imbalance = sum(self.imbalances)
        vpin = total_imbalance / (len(self.imbalances) * self.bucket_volume)
        return vpin

    def calculate_shannon_entropy(self) -> float:
        """
        Calculates the predictability of the flow.
        High entropy = random/noise. Low entropy = directional/toxic.
        """
        if len(self.imbalances) < 2:
            return 1.0

        # Convert imbalances to a distribution
        data = np.array(self.imbalances)
        # Simple discretization into 10 bins
        hist, _ = np.histogram(data, bins=10, density=True)
        # Filter zero-probabilities to avoid log(0)
        probs = hist[hist > 0]

        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def is_toxic(self, threshold: float = 0.8) -> bool:
        """Determines if the current order flow suggests an imminent adverse price move."""
        return self.calculate_vpin() > threshold
