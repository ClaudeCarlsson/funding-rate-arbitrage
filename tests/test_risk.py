"""Tests for the risk management engine."""
import pytest
from funding_arb.models import (
    ArbitragePosition,
    MarginState,
    Opportunity,
    OrderResult,
    OrderSide,
    Portfolio,
    ViolationType,
    Cycle,
    GraphNode,
    PositionType,
)
from funding_arb.risk import RiskManager
from funding_arb.config import RiskConfig


class TestRiskInvariants:
    def make_portfolio(self, **kwargs) -> Portfolio:
        return Portfolio(**kwargs)

    def test_no_violations_healthy_portfolio(self):
        rm = RiskManager()
        portfolio = self.make_portfolio(
            equity_by_exchange={"binance": 2500, "bybit": 2500, "okx": 2500, "hyperliquid": 2500},
            peak_equity=10000,
        )
        violations = rm.check_invariants(portfolio)
        assert len(violations) == 0

    def test_delta_drift_violation(self):
        rm = RiskManager()
        pos = ArbitragePosition(
            id="test",
            leg_a=OrderResult(
                order_id="a", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.BUY, amount=1.0, avg_price=50000, fee=0,
                is_filled=True,
            ),
            # No leg_b -> all delta is from leg_a
        )
        portfolio = self.make_portfolio(
            positions=[pos],
            equity_by_exchange={"binance": 10000},
            peak_equity=10000,
        )
        violations = rm.check_invariants(portfolio)
        delta_violations = [v for v in violations if v.type == ViolationType.DELTA_DRIFT]
        assert len(delta_violations) > 0

    def test_drawdown_violation(self):
        rm = RiskManager()
        portfolio = self.make_portfolio(
            equity_by_exchange={"binance": 9000},
            peak_equity=10000,
        )
        violations = rm.check_invariants(portfolio)
        dd_violations = [v for v in violations if v.type == ViolationType.EMERGENCY_DRAWDOWN]
        assert len(dd_violations) > 0
        assert dd_violations[0].severity == "critical"

    def test_exchange_concentration(self):
        rm = RiskManager(RiskConfig(max_exchange_pct=0.30))
        portfolio = self.make_portfolio(
            equity_by_exchange={"binance": 8000, "bybit": 2000},
            peak_equity=10000,
        )
        violations = rm.check_invariants(portfolio)
        ex_violations = [v for v in violations if v.type == ViolationType.EXCHANGE_CONCENTRATION]
        assert len(ex_violations) > 0

    def test_leverage_violation(self):
        rm = RiskManager(RiskConfig(max_gross_leverage=2.0))
        pos = ArbitragePosition(
            id="test",
            leg_a=OrderResult(
                order_id="a", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.BUY, amount=1.0, avg_price=50000, fee=0,
                is_filled=True,
            ),
            leg_b=OrderResult(
                order_id="b", exchange="binance", symbol="BTC/USDT:USDT",
                side=OrderSide.SELL, amount=1.0, avg_price=50000, fee=0,
                is_filled=True,
            ),
        )
        portfolio = self.make_portfolio(
            positions=[pos],
            equity_by_exchange={"binance": 10000},
            peak_equity=10000,
        )
        violations = rm.check_invariants(portfolio)
        lev_violations = [v for v in violations if v.type == ViolationType.EXCESSIVE_LEVERAGE]
        assert len(lev_violations) > 0


class TestPositionSizing:
    def test_kelly_sizing(self):
        rm = RiskManager()
        portfolio = Portfolio(
            equity_by_exchange={"binance": 1000, "bybit": 3000, "okx": 3000, "hyperliquid": 3000},
            peak_equity=10000,
        )
        opp = Opportunity(
            cycle=Cycle(nodes=[], total_weight=0),
            expected_net_yield_per_period=0.002,
            yield_variance=0.001,
            capital_required=5000,
            risk_adjusted_yield=0.002,
            exchange="binance",
        )
        size = rm.calculate_position_size(opp, portfolio)
        assert size > 0
        assert size <= 10000 * 0.20  # max position concentration

    def test_zero_edge_gives_zero_size(self):
        rm = RiskManager()
        portfolio = Portfolio(equity_by_exchange={"binance": 10000})
        opp = Opportunity(
            cycle=Cycle(nodes=[], total_weight=0),
            expected_net_yield_per_period=0.0,
            yield_variance=0.001,
            capital_required=5000,
            risk_adjusted_yield=0.0,
            exchange="binance",
        )
        assert rm.calculate_position_size(opp, portfolio) == 0.0

    def test_negative_variance_gives_zero_size(self):
        rm = RiskManager()
        portfolio = Portfolio(equity_by_exchange={"binance": 10000})
        opp = Opportunity(
            cycle=Cycle(nodes=[], total_weight=0),
            expected_net_yield_per_period=0.002,
            yield_variance=-0.001,
            capital_required=5000,
            risk_adjusted_yield=0.002,
            exchange="binance",
        )
        assert rm.calculate_position_size(opp, portfolio) == 0.0


class TestRegimeAdjustment:
    def test_high_vol_tightens(self):
        rm = RiskManager()
        original_kelly = rm.config.kelly_fraction
        rm.adjust_for_regime("high")
        assert rm.config.kelly_fraction < original_kelly

    def test_low_vol_restores(self):
        rm = RiskManager()
        rm.adjust_for_regime("high")
        rm.adjust_for_regime("low")
        assert rm.config.kelly_fraction == 0.25

    def test_regime_preserves_custom_config(self):
        custom = RiskConfig(kelly_fraction=0.15, max_gross_leverage=2.0)
        rm = RiskManager(config=custom)
        rm.adjust_for_regime("high")
        assert rm.config.kelly_fraction == 0.15 * 0.5
        assert rm.config.max_gross_leverage == 2.0 * 0.75
        rm.adjust_for_regime("normal")
        assert rm.config.kelly_fraction == 0.15
        assert rm.config.max_gross_leverage == 2.0


class TestPreTradeCheck:
    def test_passes_healthy(self):
        rm = RiskManager()
        portfolio = Portfolio(
            equity_by_exchange={"binance": 1000, "bybit": 3000, "okx": 3000, "hyperliquid": 3000},
            peak_equity=10000,
        )
        opp = Opportunity(
            cycle=Cycle(nodes=[], total_weight=0),
            expected_net_yield_per_period=0.002,
            yield_variance=0.001,
            capital_required=5000,
            risk_adjusted_yield=0.002,
            exchange="binance",
        )
        assert rm.check_pre_trade(opp, 1000, portfolio) is True

    def test_fails_on_oversize(self):
        rm = RiskManager()
        portfolio = Portfolio(
            equity_by_exchange={"binance": 10000},
            peak_equity=10000,
        )
        opp = Opportunity(
            cycle=Cycle(nodes=[], total_weight=0),
            expected_net_yield_per_period=0.002,
            yield_variance=0.001,
            capital_required=5000,
            risk_adjusted_yield=0.002,
            exchange="binance",
        )
        # 3000 / 10000 = 30% > 20% limit
        assert rm.check_pre_trade(opp, 3000, portfolio) is False


class TestRobustControl:
    def test_h_infinity_risk_throttling(self):
        config = RiskConfig(max_position_pct=0.2)
        rm = RiskManager(config, gamma=2.0)

        nominal = 1000.0
        equity = 10000.0

        # Low uncertainty
        size_safe = rm.compute_robust_size(nominal, uncertainty_bound=0.1, portfolio_equity=equity)
        # High uncertainty
        size_risky = rm.compute_robust_size(nominal, uncertainty_bound=2.0, portfolio_equity=equity)

        assert size_risky < size_safe < nominal
