"""Tests for the funding rate prediction module."""
import numpy as np
import pandas as pd
import pytest

from funding_arb.prediction import (
    FundingPredictor,
    FundingRegime,
    PredictionResult,
    RegimeState,
)


@pytest.fixture
def predictor():
    return FundingPredictor(breakeven_rate=0.0006)


@pytest.fixture
def positive_rates():
    """Simulated positive funding rate series (mean ~0.05%)."""
    rng = np.random.default_rng(42)
    base = 0.0005
    rates = []
    r = base
    for _ in range(100):
        r = r + 0.1 * (base - r) + rng.normal(0, 0.00001)
        rates.append(r)
    return pd.Series(rates)


@pytest.fixture
def negative_rates():
    """Simulated negative funding rate series."""
    rng = np.random.default_rng(42)
    base = -0.0002
    rates = []
    r = base
    for _ in range(100):
        r = r + 0.1 * (base - r) + rng.normal(0, 0.0001)
        rates.append(r)
    return pd.Series(rates)


class TestRegimeClassification:
    def test_high_positive(self, predictor):
        rates = pd.Series([0.0005, 0.0004, 0.0003, 0.0004, 0.0005])
        state = predictor.classify_regime(rates)
        assert state.current_regime == FundingRegime.HIGH_POSITIVE

    def test_near_zero(self, predictor):
        rates = pd.Series([0.0, 0.00001, -0.00001, 0.0])
        state = predictor.classify_regime(rates)
        assert state.current_regime == FundingRegime.NEAR_ZERO

    def test_regime_probabilities_sum_to_one(self, predictor, positive_rates):
        state = predictor.classify_regime(positive_rates)
        total = sum(state.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_duration_tracking(self, predictor):
        rates = pd.Series([0.0005] * 10)
        state = predictor.classify_regime(rates)
        assert state.duration_in_regime == 10

    def test_empty_rates(self, predictor):
        state = predictor.classify_regime(pd.Series([], dtype=float))
        assert state.current_regime == FundingRegime.NEAR_ZERO


class TestPrediction:
    def test_predict_positive(self, predictor, positive_rates):
        result = predictor.predict(positive_rates)
        assert result.predicted_rate > 0
        assert result.regime in [FundingRegime.HIGH_POSITIVE, FundingRegime.LOW_POSITIVE]

    def test_predict_regime(self, predictor, negative_rates):
        result = predictor.predict(negative_rates)
        assert result.regime == FundingRegime.NEGATIVE

    def test_survival_estimate(self, predictor, positive_rates):
        result = predictor.predict(positive_rates)
        assert result.periods_above_breakeven >= 0

    def test_predict_short_series(self, predictor):
        rates = pd.Series([0.0003, 0.0004, 0.0005])
        result = predictor.predict(rates)
        assert result.model_used == "naive"

    def test_predict_has_confidence_interval(self, predictor, positive_rates):
        result = predictor.predict(positive_rates)
        assert result.confidence_lower < result.predicted_rate
        assert result.confidence_upper > result.predicted_rate

    def test_predict_volatility_positive(self, predictor, positive_rates):
        result = predictor.predict(positive_rates)
        assert result.predicted_volatility > 0
