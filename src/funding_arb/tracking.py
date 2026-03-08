"""Sequential Monte Carlo (Particle Filter) for Order Book Tracking."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StateEstimate:
    expected_mid_price: float
    expected_bid_depth: float
    expected_ask_depth: float
    price_variance: float
    depth_variance: float


class OrderBookParticleFilter:
    """Maintains a probability cloud of the order book state during data blackouts."""

    def __init__(self, num_particles: int = 1000):
        self.num_particles = num_particles

        # State: [mid_price, bid_depth, ask_depth]
        self.particles = np.zeros((num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles

        self.is_initialized = False
        self.last_update_time = 0.0

        # Model parameters
        self.base_volatility = 0.001   # baseline price volatility
        self.depth_decay = 0.1         # kappa: mean reversion / decay rate for depth
        self.depth_volatility = 0.05   # eta: noise in depth
        self.observation_noise = np.array([0.0005, 10.0, 10.0]) # [price_std, bid_depth_std, ask_depth_std]

    def initialize(self, mid_price: float, bid_depth: float, ask_depth: float, timestamp: float) -> None:
        """Initialize all particles tightly around the first exact observation."""
        self.particles[:, 0] = mid_price
        self.particles[:, 1] = bid_depth
        self.particles[:, 2] = ask_depth
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_update_time = timestamp
        self.is_initialized = True

    def predict(self, current_time: float, hawkes_intensity: float, baseline_intensity: float = 0.01) -> None:
        """
        Propagate particles forward in time.
        hawkes_intensity scales the variance to simulate uncertainty during liquidations.
        """
        if not self.is_initialized:
            return

        dt = current_time - self.last_update_time
        if dt <= 0:
            return

        # Scale volatility by the Hawkes intensity ratio (min 1.0 to avoid 0 div/crash)
        intensity_ratio = max(1.0, hawkes_intensity / baseline_intensity)
        dynamic_vol = self.base_volatility * np.sqrt(intensity_ratio)

        # 1. Price Diffusion (Geometric Brownian Motion)
        # S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        z_price = np.random.standard_normal(self.num_particles)
        drift = -0.5 * (dynamic_vol ** 2) * dt
        diffusion = dynamic_vol * np.sqrt(dt) * z_price
        self.particles[:, 0] *= np.exp(drift + diffusion)

        # 2. Depth Decay / Reconstitution
        # Depth drains during high intensity (liquidity pulled)
        decay_factor = np.exp(-self.depth_decay * intensity_ratio * dt)
        z_bid = np.random.standard_normal(self.num_particles)
        z_ask = np.random.standard_normal(self.num_particles)

        self.particles[:, 1] = self.particles[:, 1] * decay_factor + self.depth_volatility * np.sqrt(dt) * self.particles[:, 1] * z_bid
        self.particles[:, 2] = self.particles[:, 2] * decay_factor + self.depth_volatility * np.sqrt(dt) * self.particles[:, 2] * z_ask

        # Ensure depth doesn't go negative
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 0.1)
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 0.1)

        self.last_update_time = current_time

    def update(self, obs_mid: float, obs_bid_depth: float, obs_ask_depth: float, timestamp: float) -> None:
        """Bayesian update when a new websocket packet arrives."""
        if not self.is_initialized:
            self.initialize(obs_mid, obs_bid_depth, obs_ask_depth, timestamp)
            return

        # Calculate likelihood p(y_t | x_t)
        # Assume Gaussian observation noise
        diff_price = self.particles[:, 0] - obs_mid
        diff_bid = self.particles[:, 1] - obs_bid_depth
        diff_ask = self.particles[:, 2] - obs_ask_depth

        # log likelihood to avoid underflow
        log_likelihood = (
            -0.5 * (diff_price / self.observation_noise[0])**2
            -0.5 * (diff_bid / self.observation_noise[1])**2
            -0.5 * (diff_ask / self.observation_noise[2])**2
        )

        # Update weights
        # w_t = w_{t-1} * p(y_t | x_t)
        # Using max subtraction trick for numerical stability
        max_ll = np.max(log_likelihood)
        likelihood = np.exp(log_likelihood - max_ll)

        self.weights *= likelihood
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample if Effective Sample Size (ESS) is too low
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.num_particles / 2.0:
            self._resample()

        self.last_update_time = timestamp

    def _resample(self) -> None:
        """Systematic resampling."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Guarantee sum to 1

        # Draw deterministic evenly spaced points with a single random offset
        step = 1.0 / self.num_particles
        u = np.random.uniform(0, step)
        pointers = u + np.arange(self.num_particles) * step

        # Find indices
        indices = np.searchsorted(cumulative_sum, pointers)

        # Resample particles
        self.particles = self.particles[indices]

        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_state(self) -> StateEstimate:
        """Return the expected state and uncertainty."""
        if not self.is_initialized:
            return StateEstimate(0.0, 0.0, 0.0, 0.0, 0.0)

        expected_state = np.average(self.particles, weights=self.weights, axis=0)

        # Variance calculation
        variance_state = np.average((self.particles - expected_state)**2, weights=self.weights, axis=0)

        return StateEstimate(
            expected_mid_price=expected_state[0],
            expected_bid_depth=expected_state[1],
            expected_ask_depth=expected_state[2],
            price_variance=variance_state[0],
            depth_variance=variance_state[1] + variance_state[2]  # Combined depth variance
        )
