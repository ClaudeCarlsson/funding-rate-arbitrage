"""Funding rate prediction models: Hawkes Processes, ARIMA, GARCH, and regime detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class HawkesPredictor:
    """Microstructure-based predictor using self-exciting Hawkes Processes."""

    def __init__(self, baseline: float = 0.01, alpha: float = 0.1, beta: float = 0.5):
        self.mu = baseline
        self.alpha = alpha
        self.beta = beta
        self.events: list[float] = []
        self.internal_events: list[float] = [] # Tracks our own trades to prevent toxic feedback

    def compute_intensity(self, t: float) -> float:
        """Calculate the current conditional intensity λ(t), blanking internal trades."""
        intensity = self.mu

        # Add excitation from all public events
        for event_t in self.events:
            if event_t < t:
                intensity += self.alpha * np.exp(-self.beta * (t - event_t))

        # Subtract excitation that we caused ourselves (Internal Signal Blanking)
        for internal_t in self.internal_events:
            if internal_t < t:
                # We assume our own trade caused a public event tape print
                intensity -= self.alpha * np.exp(-self.beta * (t - internal_t))

        return max(self.mu, intensity) # Can't go below baseline

    def register_internal_trade(self, t: float) -> None:
        """Record a time when the Janitor or Brawn executed an aggressive trade."""
        self.internal_events.append(t)
        self.internal_events = self.internal_events[-1000:]

    def update(self, new_events: list[float]) -> float:
        """Update event history and return current intensity."""
        if not new_events:
            return self.mu

        self.events.extend(new_events)
        self.events = self.events[-1000:]  # Memory management
        return self.compute_intensity(new_events[-1])

    def detect_regime_shift(self, t: float) -> bool:
        """Signals high-intensity spikes suggesting liquidity cascades."""
        intensity = self.compute_intensity(t)
        return intensity > self.mu * 10


class FundingRegime(Enum):
    """Market regime classifications based on funding rate levels."""
    HIGH_POSITIVE = auto()   # >0.03%, aggressive yield harvesting
    LOW_POSITIVE = auto()    # 0.005-0.03%, standard delta-neutral
    NEAR_ZERO = auto()       # ±0.005%, reduce positions
    NEGATIVE = auto()        # <-0.005%, reverse or exit


@dataclass
class RegimeState:
    """Wrapper for regime for backward compatibility."""
    current_regime: FundingRegime
    probabilities: dict[FundingRegime, float] = field(default_factory=dict)
    duration_in_regime: int = 0


@dataclass
class PredictionResult:
    """Result from a funding rate prediction."""
    predicted_rate: float
    confidence_lower: float
    confidence_upper: float
    predicted_volatility: float
    regime: FundingRegime
    hawkes_intensity: float
    periods_above_breakeven: int
    model_used: str = ""


class FundingPredictor:
    """Predicts funding rates using a multi-tier approach.

    Tier 1: Hawkes Process - microstructure intensity (High frequency)
    Tier 2: ARIMA/GARCH - baseline statistical mean-reversion and volatility
    Tier 3: Regime detection - market state classification
    """

    def __init__(self, breakeven_rate: float = 0.0006):
        self.breakeven_rate = breakeven_rate
        self.hawkes = HawkesPredictor()
        self._regime_thresholds = {
            FundingRegime.HIGH_POSITIVE: 0.0003,
            FundingRegime.LOW_POSITIVE: 0.00005,
            FundingRegime.NEAR_ZERO: -0.00005,
        }

    def predict(
        self,
        rates: pd.Series,
        event_timestamps: list[float] | None = None,
        periods_ahead: int = 1,
    ) -> PredictionResult:
        """Generate a prediction using Hawkes intensity and statistical baselines."""
        current_t = float(rates.index[-1].timestamp()) if hasattr(rates.index[-1], 'timestamp') else len(rates)

        # 1. Hawkes Intensity
        intensity = self.hawkes.update(event_timestamps or [])

        if len(rates) < 10:
            current = float(rates.iloc[-1]) if len(rates) > 0 else 0.0
            return PredictionResult(
                predicted_rate=current,
                confidence_lower=current - 0.001,
                confidence_upper=current + 0.001,
                predicted_volatility=0.001,
                regime=self._classify_regime(current),
                hawkes_intensity=intensity,
                periods_above_breakeven=0,
                model_used="naive",
            )

        # 2. Statistical Baselines (ARIMA/Mean Reversion)
        try:
            pred_rate, ci_lower, ci_upper = self._arima_predict(rates, periods_ahead)
            model_used = "arima"
        except Exception:
            pred_rate, ci_lower, ci_upper = self._mean_reversion_predict(rates, periods_ahead)
            model_used = "mean_reversion"

        # 3. Volatility (GARCH/EWMA)
        try:
            vol = self._garch_volatility(rates)
        except Exception:
            vol = float(rates.std())

        # 4. Hawkes-Adjusted Prediction
        # If intensity is high, we expect more variance and possible regime shift
        if self.hawkes.detect_regime_shift(current_t):
            model_used += "+hawkes_spike"
            vol *= 1.5

        return PredictionResult(
            predicted_rate=pred_rate,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            predicted_volatility=vol,
            regime=self._classify_regime(pred_rate),
            hawkes_intensity=intensity,
            periods_above_breakeven=self._estimate_survival(rates, self.breakeven_rate),
            model_used=model_used,
        )

    def classify_regime(self, rates: pd.Series) -> RegimeState:
        """Classify the current market regime."""
        if len(rates) == 0:
            return RegimeState(current_regime=FundingRegime.NEAR_ZERO, probabilities={r: 0.25 for r in FundingRegime})

        ewm_rate = float(rates.ewm(span=min(6, len(rates))).mean().iloc[-1])
        regime = self._classify_regime(ewm_rate)

        # Calculate duration: how many consecutive recent rates are in the same regime
        duration = 0
        for r in reversed(rates):
            if self._classify_regime(float(r)) == regime:
                duration += 1
            else:
                break

        # Mock probabilities for compatibility
        probs = {r: 0.0 for r in FundingRegime}
        probs[regime] = 1.0
        return RegimeState(current_regime=regime, probabilities=probs, duration_in_regime=duration)

    def _classify_regime(self, rate: float) -> FundingRegime:
        if rate >= self._regime_thresholds[FundingRegime.HIGH_POSITIVE]:
            return FundingRegime.HIGH_POSITIVE
        elif rate >= self._regime_thresholds[FundingRegime.LOW_POSITIVE]:
            return FundingRegime.LOW_POSITIVE
        elif rate >= self._regime_thresholds[FundingRegime.NEAR_ZERO]:
            return FundingRegime.NEAR_ZERO
        else:
            return FundingRegime.NEGATIVE

    def _arima_predict(
        self, rates: pd.Series, periods_ahead: int
    ) -> tuple[float, float, float]:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(rates.values, order=(1, 0, 1))
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=periods_ahead)
        pred = float(forecast.predicted_mean[-1])
        ci = forecast.conf_int(alpha=0.05)
        return pred, float(ci[-1, 0]), float(ci[-1, 1])

    def _mean_reversion_predict(
        self, rates: pd.Series, periods_ahead: int
    ) -> tuple[float, float, float]:
        y, x = rates.values[1:], rates.values[:-1]
        x_mean, y_mean = np.mean(x), np.mean(y)
        beta = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12)
        alpha = y_mean - beta * x_mean
        pred = float(rates.iloc[-1])
        for _ in range(periods_ahead):
            pred = alpha + beta * pred
        std_resid = float(np.std(y - (alpha + beta * x)))
        ci_width = 1.96 * std_resid * (periods_ahead ** 0.5)
        return pred, pred - ci_width, pred + ci_width

    def _garch_volatility(self, rates: pd.Series) -> float:
        from arch import arch_model
        scaled = rates * 10000
        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="AR", lags=1)
        result = model.fit(disp="off", show_warning=False)
        return float(result.conditional_volatility.iloc[-1]) / 10000

    def _estimate_survival(
        self, rates: pd.Series, breakeven: float
    ) -> int:
        recent = rates.tail(90)
        runs, current_run = [], 0
        for rate in recent:
            if rate > breakeven:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        return int(np.median(runs)) if runs else 0
