"""Tests for Z3-based formal verification of risk invariants."""
import pytest

try:
    from z3 import Real, Solver, unsat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from funding_arb.verification import (
    FormalRobustnessVerifier,
    RiskVerifier,
    Z3_AVAILABLE as MODULE_Z3,
    run_verification,
)


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver not installed")
class TestRiskVerification:
    @pytest.fixture
    def verifier(self):
        return RiskVerifier()

    def test_position_size_bounded(self, verifier):
        assert verifier.verify_position_size_bounded() is True

    def test_kelly_never_negative(self, verifier):
        assert verifier.verify_kelly_never_negative() is True

    def test_delta_check_correct(self, verifier):
        assert verifier.verify_delta_check_correct() is True

    def test_leverage_check_correct(self, verifier):
        assert verifier.verify_leverage_check_correct() is True

    def test_drawdown_check_correct(self, verifier):
        assert verifier.verify_drawdown_check_correct() is True

    def test_collateral_ratio_correct(self, verifier):
        assert verifier.verify_collateral_ratio_correct() is True

    def test_size_respects_exchange_limit(self, verifier):
        assert verifier.verify_size_respects_exchange_limit() is True

    def test_verify_all(self, verifier):
        results = verifier.verify_all()
        assert all(results.values())
        assert len(results) == 7

    def test_run_verification(self):
        assert run_verification() is True

    def test_formal_robustness_verifier(self):
        verifier = FormalRobustnessVerifier()

        # Test 1: Given a reasonable max gross leverage (e.g., 3.0),
        # even if nominal_size requests up to 100x equity, 
        # if uncertainty is high enough (gamma * uncertainty_bound >= 33.3), 
        # the invariant holds.
        # gain = 1 / (1 + 1 * 35) = 1/36.  100 / 36 = 2.77 < 3.0. Should be True.
        assert verifier.verify_margin_invariant(max_gross_leverage=3.0, gamma=1.0, uncertainty_bound=35.0) is True

        # Test 2: If uncertainty is 0, the controller allows higher leverage.
        # We use a high max_position_pct to trigger a violation.
        assert verifier.verify_margin_invariant(max_gross_leverage=3.0, gamma=1.0, uncertainty_bound=0.0, max_position_pct=5.0) is False
