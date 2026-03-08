"""Funding rate prediction: EWM trend, mean-reversion, and regime detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class FundingRegime(Enum):
    """Market regime classifications based on funding rate levels."""
    HIGH_POSITIVE = auto()   # >0.03%, aggressive yield harvesting
    LOW_POSITIVE = auto()    # 0.005-0.03%, standard delta-neutral
    NEAR_ZERO = auto()       # +/-0.005%, reduce positions
    NEGATIVE = auto()        # <-0.005%, reverse or exit


@dataclass
class RegimeState:
    """Current regime with duration tracking."""
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
    periods_above_breakeven: int
    model_used: str = ""


class FundingPredictor:
    """Predicts funding rates using EWM trend + mean-reversion + regime detection.

    Optimized for 8-hour funding cycles where simple statistical methods
    capture the signal that matters.
    """

    def __init__(self, breakeven_rate: float = 0.0006):
        self.breakeven_rate = breakeven_rate
        self._regime_thresholds = {
            FundingRegime.HIGH_POSITIVE: 0.0003,
            FundingRegime.LOW_POSITIVE: 0.00005,
            FundingRegime.NEAR_ZERO: -0.00005,
        }

    def predict(
        self,
        rates: pd.Series,
        periods_ahead: int = 1,
    ) -> PredictionResult:
        """Generate a prediction using EWM trend and mean-reversion."""
        if len(rates) < 10:
            current = float(rates.iloc[-1]) if len(rates) > 0 else 0.0
            return PredictionResult(
                predicted_rate=current,
                confidence_lower=current - 0.001,
                confidence_upper=current + 0.001,
                predicted_volatility=0.001,
                regime=self._classify_regime(current),
                periods_above_breakeven=0,
                model_used="naive",
            )

        # EWM trend as primary signal
        ewm = rates.ewm(span=6).mean()
        ewm_rate = float(ewm.iloc[-1])

        # Mean-reversion prediction for confidence interval
        pred_rate, ci_lower, ci_upper = self._mean_reversion_predict(rates, periods_ahead)

        # Blend: 60% mean-reversion, 40% EWM trend
        pred_rate = 0.6 * pred_rate + 0.4 * ewm_rate

        # Simple EWMA volatility
        vol = float(rates.ewm(span=12).std().iloc[-1])

        return PredictionResult(
            predicted_rate=pred_rate,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            predicted_volatility=vol,
            regime=self._classify_regime(pred_rate),
            periods_above_breakeven=self._estimate_survival(rates, self.breakeven_rate),
            model_used="ewm_mean_reversion",
        )

    def classify_regime(self, rates: pd.Series) -> RegimeState:
        """Classify the current market regime from recent rates."""
        if len(rates) == 0:
            return RegimeState(
                current_regime=FundingRegime.NEAR_ZERO,
                probabilities={r: 0.25 for r in FundingRegime},
            )

        ewm_rate = float(rates.ewm(span=min(6, len(rates))).mean().iloc[-1])
        regime = self._classify_regime(ewm_rate)

        # Duration: consecutive recent rates in the same regime
        duration = 0
        for r in reversed(rates):
            if self._classify_regime(float(r)) == regime:
                duration += 1
            else:
                break

        probs = {r: 0.0 for r in FundingRegime}
        probs[regime] = 1.0
        return RegimeState(
            current_regime=regime, probabilities=probs, duration_in_regime=duration
        )

    def _classify_regime(self, rate: float) -> FundingRegime:
        if rate >= self._regime_thresholds[FundingRegime.HIGH_POSITIVE]:
            return FundingRegime.HIGH_POSITIVE
        elif rate >= self._regime_thresholds[FundingRegime.LOW_POSITIVE]:
            return FundingRegime.LOW_POSITIVE
        elif rate >= self._regime_thresholds[FundingRegime.NEAR_ZERO]:
            return FundingRegime.NEAR_ZERO
        else:
            return FundingRegime.NEGATIVE

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
