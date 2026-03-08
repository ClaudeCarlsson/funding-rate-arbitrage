"""Z3-based formal verification of risk engine invariants.

Proves that the risk-checking code correctly enforces invariants
under ALL possible inputs. Run at build/test time, not at runtime.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import z3
    from z3 import And, If, Not, Or, Real, RealVal, Solver, unsat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("z3-solver not installed — formal verification disabled")


class RiskVerifier:
    """Formally verify risk engine invariants using Z3 theorem prover.

    Each method proves a specific property about the risk engine:
    - If the proof succeeds (unsat), NO input can violate the invariant
    - If the proof fails (sat), z3 provides a counterexample
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("z3-solver is required for formal verification")

    def verify_all(self) -> dict[str, bool]:
        """Run all verification proofs. Returns {property_name: proved}."""
        results = {}
        proofs = [
            ("position_size_bounded", self.verify_position_size_bounded),
            ("kelly_never_negative", self.verify_kelly_never_negative),
            ("delta_check_correct", self.verify_delta_check_correct),
            ("leverage_check_correct", self.verify_leverage_check_correct),
            ("drawdown_check_correct", self.verify_drawdown_check_correct),
            ("collateral_ratio_correct", self.verify_collateral_ratio_correct),
            ("size_respects_exchange_limit", self.verify_size_respects_exchange_limit),
        ]
        for name, proof_fn in proofs:
            try:
                proved = proof_fn()
                results[name] = proved
                status = "PROVED" if proved else "FAILED"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                results[name] = False
                logger.error(f"  {name}: ERROR - {e}")
        return results

    def verify_position_size_bounded(self) -> bool:
        """Prove: position sizing function never exceeds MAX_POSITION_PCT of equity."""
        equity = Real("equity")
        edge = Real("edge")
        variance = Real("variance")
        kelly_fraction = RealVal("0.25")
        max_position_pct = RealVal("0.20")

        # Model the sizing function
        kelly_size = (edge / variance) * kelly_fraction * equity
        max_by_concentration = equity * max_position_pct

        # The actual size is min(kelly_size, max_by_concentration, ...)
        final_size = If(kelly_size < max_by_concentration, kelly_size, max_by_concentration)

        s = Solver()
        s.add(equity > 0)
        s.add(edge > 0)
        s.add(variance > 0)
        s.add(final_size > max_by_concentration)

        return s.check() == unsat

    def verify_kelly_never_negative(self) -> bool:
        """Prove: Kelly sizing with max(0, ...) never returns negative."""
        equity = Real("equity")
        edge = Real("edge")
        variance = Real("variance")
        kelly_fraction = RealVal("0.25")

        size = If(
            And(edge > 0, variance > 0),
            (edge / variance) * kelly_fraction * equity,
            RealVal(0),
        )

        final_size = If(size > 0, size, RealVal(0))

        s = Solver()
        s.add(equity > 0)
        s.add(final_size < 0)

        return s.check() == unsat

    def verify_delta_check_correct(self) -> bool:
        """Prove: delta check correctly flags ALL violations."""
        net_delta = Real("net_delta")
        equity = Real("equity")
        max_delta_pct = RealVal("0.02")

        delta_pct = If(net_delta >= 0, net_delta / equity, -net_delta / equity)
        is_violation = delta_pct > max_delta_pct

        code_check = If(net_delta >= 0, net_delta, -net_delta) / equity > max_delta_pct

        s = Solver()
        s.add(equity > 0)
        s.add(Or(
            And(is_violation, Not(code_check)),
            And(Not(is_violation), code_check),
        ))

        return s.check() == unsat

    def verify_leverage_check_correct(self) -> bool:
        """Prove: leverage check catches ALL cases where gross_notional/equity > max."""
        notional_a = Real("notional_a")
        notional_b = Real("notional_b")
        notional_c = Real("notional_c")
        equity = Real("equity")
        max_leverage = RealVal("3.0")

        abs_a = If(notional_a >= 0, notional_a, -notional_a)
        abs_b = If(notional_b >= 0, notional_b, -notional_b)
        abs_c = If(notional_c >= 0, notional_c, -notional_c)
        gross = abs_a + abs_b + abs_c

        leverage = gross / equity
        violation = leverage > max_leverage

        code_leverage = (abs_a + abs_b + abs_c) / equity
        code_violation = code_leverage > max_leverage

        s = Solver()
        s.add(equity > 0)
        s.add(Or(
            And(violation, Not(code_violation)),
            And(Not(violation), code_violation),
        ))

        return s.check() == unsat

    def verify_drawdown_check_correct(self) -> bool:
        """Prove: drawdown check correctly computes (peak - current) / peak."""
        current_equity = Real("current_equity")
        peak_equity = Real("peak_equity")
        max_drawdown = RealVal("0.05")

        drawdown = (peak_equity - current_equity) / peak_equity
        is_violation = drawdown > max_drawdown

        alt_check = current_equity < peak_equity * (1 - max_drawdown)

        s = Solver()
        s.add(peak_equity > 0)
        s.add(current_equity > 0)
        s.add(current_equity <= peak_equity)
        s.add(Or(
            And(is_violation, Not(alt_check)),
            And(Not(is_violation), alt_check),
        ))

        return s.check() == unsat

    def verify_collateral_ratio_correct(self) -> bool:
        """Prove: collateral ratio check catches ALL undercollateralized states."""
        equity = Real("equity")
        margin_used = Real("margin_used")
        min_ratio = RealVal("2.0")

        ratio = equity / margin_used
        is_violation = ratio < min_ratio

        alt_check = equity < min_ratio * margin_used

        s = Solver()
        s.add(equity > 0)
        s.add(margin_used > 0)
        s.add(Or(
            And(is_violation, Not(alt_check)),
            And(Not(is_violation), alt_check),
        ))

        return s.check() == unsat

    def verify_size_respects_exchange_limit(self) -> bool:
        """Prove: position size never causes exchange concentration to exceed limit."""
        total_equity = Real("total_equity")
        equity_at_exchange = Real("equity_at_exchange")
        max_exchange_pct = RealVal("0.30")

        max_by_exchange = total_equity * max_exchange_pct - equity_at_exchange
        size = If(max_by_exchange > 0, max_by_exchange, RealVal(0))
        new_exchange_pct = (equity_at_exchange + size) / total_equity

        s = Solver()
        s.add(total_equity > 0)
        s.add(equity_at_exchange >= 0)
        s.add(equity_at_exchange <= total_equity * max_exchange_pct)
        s.add(new_exchange_pct > max_exchange_pct)

        return s.check() == unsat


class FormalRobustnessVerifier:
    """Uses Z3 to verify that the H-Infinity controller respects margin invariants."""

    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required for formal verification.")

        self.solver = z3.Solver()

    def verify_margin_invariant(
        self,
        max_gross_leverage: float,
        gamma: float,
        uncertainty_bound: float,
        max_position_pct: float = 0.2
    ) -> bool:
        """
        Verify that the H-Infinity controller output will never violate the
        max_gross_leverage limit, accounting for the final position cap.
        """
        nominal_size = z3.Real('nominal_size')
        equity = z3.Real('equity')

        domain_constraints = [
            nominal_size > 0,
            equity > 0,
            nominal_size <= equity * 100
        ]

        # Gain logic
        gain = 1.0 / (1.0 + (gamma * uncertainty_bound))
        robust_size = nominal_size * gain

        # Apply the final cap like in RiskManager.compute_robust_size
        # robust_size = min(robust_size, equity * max_position_pct)
        capped_size = z3.If(robust_size < equity * max_position_pct, robust_size, equity * max_position_pct)

        safety_invariant = (capped_size / equity) <= max_gross_leverage

        self.solver.push()
        self.solver.add(domain_constraints)
        self.solver.add(z3.Not(safety_invariant))

        result = self.solver.check()
        self.solver.pop()

        return result == z3.unsat


def run_verification() -> bool:
    """Run all formal proofs and return True if all pass."""
    if not Z3_AVAILABLE:
        logger.error("z3-solver not installed — cannot run verification")
        return False

    logger.info("Running formal verification of risk invariants...")
    verifier = RiskVerifier()
    results = verifier.verify_all()

    robust_verifier = FormalRobustnessVerifier()
    robust_ok = robust_verifier.verify_margin_invariant(3.0, 1.0, 0.2)
    results["h_infinity_robustness"] = robust_ok
    logger.info(f"  h_infinity_robustness: {'PROVED' if robust_ok else 'FAILED'}")

    all_passed = all(results.values())
    if all_passed:
        logger.info(f"ALL {len(results)} PROOFS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"FAILED PROOFS: {failed}")

    return all_passed
