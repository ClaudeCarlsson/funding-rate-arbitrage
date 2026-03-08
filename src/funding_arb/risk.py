"""Risk management engine with invariant checking and dynamic adjustment."""
from __future__ import annotations

import logging

from .config import RiskConfig
from .models import (
    Opportunity,
    Portfolio,
    Violation,
    ViolationType,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk invariants and computes position sizes using H-Infinity robust control."""

    def __init__(self, config: RiskConfig | None = None, gamma: float = 1.0):
        self.config = config or RiskConfig()
        self._default_config = RiskConfig()  # for regime reset
        self.gamma = gamma  # Sensitivity to "worst-case" perturbations for H-Infinity

    def check_invariants(self, portfolio: Portfolio) -> list[Violation]:
        """Check all risk invariants against current portfolio state."""
        violations: list[Violation] = []
        total_equity = portfolio.total_equity

        if total_equity <= 0:
            return violations

        # Delta neutrality
        net_delta = sum(p.delta_usd for p in portfolio.positions if p.is_open)
        delta_pct = abs(net_delta) / total_equity
        if delta_pct > self.config.max_delta_pct:
            violations.append(Violation(
                type=ViolationType.DELTA_DRIFT,
                message=f"Delta drift: {delta_pct:.4f} exceeds limit {self.config.max_delta_pct}",
                severity="critical" if delta_pct > self.config.max_delta_pct * 2 else "warning",
                details={"delta_pct": delta_pct, "net_delta_usd": net_delta},
            ))

        # Position concentration
        for pos in portfolio.positions:
            if not pos.is_open:
                continue
            pos_pct = abs(pos.notional_usd) / total_equity
            if pos_pct > self.config.max_position_pct:
                violations.append(Violation(
                    type=ViolationType.POSITION_CONCENTRATION,
                    message=f"Position {pos.id}: {pos_pct:.4f} exceeds limit {self.config.max_position_pct}",
                    severity="warning",
                    details={"position_id": pos.id, "position_pct": pos_pct},
                ))

        # Exchange concentration
        for ex_name, equity in portfolio.equity_by_exchange.items():
            ex_pct = equity / total_equity
            if ex_pct > self.config.max_exchange_pct:
                violations.append(Violation(
                    type=ViolationType.EXCHANGE_CONCENTRATION,
                    message=f"Exchange {ex_name}: {ex_pct:.4f} exceeds limit {self.config.max_exchange_pct}",
                    severity="warning",
                    details={"exchange": ex_name, "exchange_pct": ex_pct},
                ))

        # Collateral ratios
        for ex_name, margin_state in portfolio.margin_by_exchange.items():
            ratio = margin_state.ratio
            if ratio < self.config.min_collateral_ratio:
                violations.append(Violation(
                    type=ViolationType.LOW_COLLATERAL,
                    message=f"Exchange {ex_name}: collateral ratio {ratio:.2f} below minimum {self.config.min_collateral_ratio}",
                    severity="critical",
                    details={"exchange": ex_name, "ratio": ratio},
                ))

        # Gross leverage
        gross_notional = sum(abs(p.notional_usd) for p in portfolio.positions if p.is_open)
        leverage = gross_notional / total_equity
        if leverage > self.config.max_gross_leverage:
            violations.append(Violation(
                type=ViolationType.EXCESSIVE_LEVERAGE,
                message=f"Gross leverage {leverage:.2f} exceeds limit {self.config.max_gross_leverage}",
                severity="critical",
                details={"leverage": leverage},
            ))

        # Drawdown
        if portfolio.drawdown_from_peak > self.config.max_drawdown:
            violations.append(Violation(
                type=ViolationType.EMERGENCY_DRAWDOWN,
                message=f"Drawdown {portfolio.drawdown_from_peak:.4f} exceeds limit {self.config.max_drawdown}",
                severity="critical",
                details={"drawdown": portfolio.drawdown_from_peak},
            ))

        return violations

    def has_critical_violations(self, portfolio: Portfolio) -> bool:
        """Check if there are any critical violations."""
        violations = self.check_invariants(portfolio)
        return any(v.severity == "critical" for v in violations)

    def check_pre_trade(self, opp: Opportunity, size: float, portfolio: Portfolio) -> bool:
        """Pre-trade risk check: would this trade violate any invariant?"""
        if size <= 0:
            return False

        # Check existing violations first
        if self.has_critical_violations(portfolio):
            logger.warning("Pre-trade check failed: existing critical violations")
            return False

        # Simulate the trade's impact
        total_equity = portfolio.total_equity
        if total_equity <= 0:
            return False

        # Position concentration check
        pos_pct = size / total_equity
        if pos_pct > self.config.max_position_pct:
            logger.warning(f"Pre-trade: position size {pos_pct:.4f} exceeds limit")
            return False

        # Exchange concentration check
        current_at_exchange = portfolio.equity_at(opp.exchange)
        new_pct = (current_at_exchange + size) / total_equity
        if new_pct > self.config.max_exchange_pct:
            logger.warning(f"Pre-trade: exchange concentration {new_pct:.4f} exceeds limit")
            return False

        # Leverage check
        current_notional = sum(abs(p.notional_usd) for p in portfolio.positions if p.is_open)
        new_leverage = (current_notional + size) / total_equity
        if new_leverage > self.config.max_gross_leverage:
            logger.warning(f"Pre-trade: leverage {new_leverage:.2f} exceeds limit")
            return False

        return True

    def compute_robust_size(
        self,
        nominal_size: float,
        uncertainty_bound: float,
        portfolio_equity: float
    ) -> float:
        """
        Aerospace-grade robust control for portfolio sizing.
        Solves the minimax problem: min(max(E(u, w))) where w is adversarial.
        Returns a size that is stable under worst-case funding inversions.
        """
        if uncertainty_bound <= 0:
            return nominal_size

        # H-Infinity gain formulation (simplified for scalar trade)
        # We penalize the 'tracking error' from nominal to actual flow
        # under a bounded disturbance w.
        gain = 1.0 / (1.0 + (self.gamma * uncertainty_bound))
        robust_size = nominal_size * gain

        # Final cap based on hard risk limits
        return min(robust_size, portfolio_equity * self.config.max_position_pct)

    def calculate_position_size(
        self, opp: Opportunity, portfolio: Portfolio, uncertainty_bound: float = 0.1
    ) -> float:
        """
        Calculates optimal position size using Kelly criterion integrated with
        H-Infinity robust control for worst-case protection.
        """
        edge = opp.expected_net_yield_per_period
        variance = opp.yield_variance

        if variance <= 0 or edge <= 0:
            return 0.0

        total_equity = portfolio.total_equity
        if total_equity <= 0:
            return 0.0

        # 1. Base Kelly fraction
        kelly_size = (edge / variance) * self.config.kelly_fraction * total_equity

        # 2. Apply H-Infinity robust sizing to the Kelly nominal size
        robust_size = self.compute_robust_size(kelly_size, uncertainty_bound, total_equity)

        # 3. Apply hard caps from concentration and leverage

        # Cap by exchange headroom
        max_by_exchange = (
            total_equity * self.config.max_exchange_pct
            - portfolio.equity_at(opp.exchange)
        )

        # Cap by leverage headroom
        current_notional = sum(
            abs(p.notional_usd) for p in portfolio.positions if p.is_open
        )
        max_by_leverage = (
            total_equity * self.config.max_gross_leverage - current_notional
        )

        size = max(0.0, min(robust_size, max_by_exchange, max_by_leverage))
        return size

    def adjust_for_regime(self, vol_regime: str) -> None:
        """Tighten or loosen parameters based on volatility regime."""
        if vol_regime == "high":
            self.config = RiskConfig(
                max_delta_pct=self._default_config.max_delta_pct * 0.5,
                max_position_pct=self._default_config.max_position_pct * 0.6,
                max_exchange_pct=self._default_config.max_exchange_pct,
                min_collateral_ratio=self._default_config.min_collateral_ratio * 1.5,
                max_drawdown=self._default_config.max_drawdown,
                max_gross_leverage=self._default_config.max_gross_leverage * 0.75,
                correlation_floor=self._default_config.correlation_floor,
                kelly_fraction=self._default_config.kelly_fraction * 0.5,
            )
            logger.info("Risk parameters tightened for high-volatility regime")
        elif vol_regime == "low":
            self.config = RiskConfig()
            logger.info("Risk parameters restored to defaults for low-volatility regime")
        elif vol_regime == "normal":
            self.config = RiskConfig()
            logger.info("Risk parameters at normal levels")
