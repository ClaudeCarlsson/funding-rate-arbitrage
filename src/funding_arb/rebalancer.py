"""Balance rebalancer — monitors capital skew across exchanges and alerts."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Portfolio

logger = logging.getLogger(__name__)

DEFAULT_SKEW_ALERT_PCT = 0.65
DEFAULT_SKEW_CRITICAL_PCT = 0.80


@dataclass
class SkewAlert:
    exchange: str
    balance_usd: float
    total_usd: float
    pct: float
    severity: str  # "warning" or "critical"

    @property
    def message(self) -> str:
        return (
            f"Balance skew on {self.exchange}: "
            f"${self.balance_usd:,.2f} / ${self.total_usd:,.2f} "
            f"({self.pct*100:.1f}%) [{self.severity}]"
        )


class Rebalancer:
    """Monitors balance distribution across exchanges and generates alerts.

    Does NOT auto-execute transfers — alerts for manual action.
    Cross-exchange transfers are high-risk and not worth automating early on.
    """

    def __init__(
        self,
        skew_alert_pct: float = DEFAULT_SKEW_ALERT_PCT,
        skew_critical_pct: float = DEFAULT_SKEW_CRITICAL_PCT,
    ):
        self.skew_alert_pct = skew_alert_pct
        self.skew_critical_pct = skew_critical_pct

    def check_skew(self, portfolio: Portfolio) -> list[SkewAlert]:
        """Check if any exchange holds a disproportionate share of total capital.

        Returns list of SkewAlert for exchanges exceeding thresholds.
        """
        total = portfolio.total_equity
        if total <= 0:
            return []

        alerts: list[SkewAlert] = []
        for exchange, balance in portfolio.equity_by_exchange.items():
            pct = balance / total
            if pct >= self.skew_critical_pct:
                alerts.append(SkewAlert(
                    exchange=exchange,
                    balance_usd=balance,
                    total_usd=total,
                    pct=pct,
                    severity="critical",
                ))
            elif pct >= self.skew_alert_pct:
                alerts.append(SkewAlert(
                    exchange=exchange,
                    balance_usd=balance,
                    total_usd=total,
                    pct=pct,
                    severity="warning",
                ))

        return alerts

    def suggest_transfers(self, portfolio: Portfolio) -> list[dict]:
        """Suggest transfers to rebalance capital evenly across exchanges.

        Returns list of dicts with 'from', 'to', 'amount_usd' fields.
        These are suggestions only — never auto-executed.
        """
        total = portfolio.total_equity
        exchanges = list(portfolio.equity_by_exchange.keys())
        if len(exchanges) < 2 or total <= 0:
            return []

        target_per_exchange = total / len(exchanges)
        surplus: list[tuple[str, float]] = []
        deficit: list[tuple[str, float]] = []

        for ex, balance in portfolio.equity_by_exchange.items():
            diff = balance - target_per_exchange
            if diff > 0:
                surplus.append((ex, diff))
            elif diff < 0:
                deficit.append((ex, -diff))

        surplus.sort(key=lambda x: x[1], reverse=True)
        deficit.sort(key=lambda x: x[1], reverse=True)

        suggestions = []
        s_idx, d_idx = 0, 0
        while s_idx < len(surplus) and d_idx < len(deficit):
            from_ex, s_amount = surplus[s_idx]
            to_ex, d_amount = deficit[d_idx]
            transfer = min(s_amount, d_amount)

            if transfer > 10:  # minimum $10 to bother suggesting
                suggestions.append({
                    "from": from_ex,
                    "to": to_ex,
                    "amount_usd": round(transfer, 2),
                })

            surplus[s_idx] = (from_ex, s_amount - transfer)
            deficit[d_idx] = (to_ex, d_amount - transfer)

            if surplus[s_idx][1] < 1:
                s_idx += 1
            if deficit[d_idx][1] < 1:
                d_idx += 1

        return suggestions
