"""Main orchestrator - ties scanner, optimizer, risk manager, executor, alerter, and prediction together."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime

import pandas as pd

from .alerter import LogAlerter, TelegramAlerter
from .config import Config, load_config
from .database import Database
from .executor import TradeExecutor
from .health import HealthCheck
from .metrics import (
    BEST_YIELD_APR,
    EXCHANGES_SCANNED,
    FUNDING_RATES_SAVED,
    GRAPH_OPPORTUNITIES,
    KILL_SWITCH_ACTIVE,
    POSITIONS_OPENED,
    RISK_VIOLATIONS,
    SIMPLE_OPPORTUNITIES,
    STALE_EXCHANGES,
    TICK_COUNT,
    TICK_DURATION,
    TICK_ERRORS,
    start_metrics_server,
    update_portfolio_metrics,
)
from .models import Portfolio
from .optimizer import ArbitrageOptimizer
from .prediction import FundingPredictor, FundingRegime
from .rebalancer import Rebalancer
from .risk import RiskManager
from .scanner import FundingScanner

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main async loop that coordinates all system components."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.scanner = FundingScanner(self.config)
        self.optimizer = ArbitrageOptimizer(self.config.optimizer)
        self.risk_manager = RiskManager(self.config.risk)
        self.executor = TradeExecutor(self.config)
        self.rebalancer = Rebalancer()
        self.portfolio = Portfolio()
        self.predictor = FundingPredictor()
        self.database = Database(
            state_db_path=self.config.database.state_db_path,
            trades_db_path=self.config.database.trades_db_path,
            funding_db_path=self.config.database.funding_db_path,
            parquet_dir=self.config.database.parquet_dir,
        )

        # Use Telegram if credentials available, otherwise log
        if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
            self.alerter: TelegramAlerter | LogAlerter = TelegramAlerter()
        else:
            self.alerter = LogAlerter()

        self._running = False
        self._iteration = 0
        self._funding_history: dict[str, list[float]] = {}  # symbol -> recent rates
        self._last_daily_summary: datetime | None = None
        self._health = HealthCheck()

    async def start(self) -> None:
        """Start the main loop."""
        logger.info("Starting orchestrator...")
        await self.scanner.initialize()
        await self.executor.initialize()
        await self._health.start()
        start_metrics_server()
        await self.alerter.notify_system_start()
        self._running = True

        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self.config.scanner.poll_interval_s)
        except KeyboardInterrupt:
            logger.info("Shutting down (keyboard interrupt)...")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the orchestrator and clean up."""
        self._running = False
        await self.alerter.notify_system_stop("normal shutdown")
        await self._health.stop()
        await self.scanner.close()
        await self.executor.close()
        logger.info("Orchestrator stopped")

    async def _tick(self) -> None:
        """Single iteration of the main loop."""
        self._iteration += 1
        tick_start = datetime.now(UTC)
        TICK_COUNT.inc()
        KILL_SWITCH_ACTIVE.set(1 if self.executor.is_killed() else 0)

        try:
            # 1. Scan all exchanges
            snapshot = await self.scanner.scan()
            EXCHANGES_SCANNED.set(len(snapshot.exchanges))
            STALE_EXCHANGES.set(len(snapshot.stale_exchanges))
            logger.info(
                f"Tick {self._iteration}: scanned {len(snapshot.exchanges)} exchanges, "
                f"{len(snapshot.instruments)} instruments, "
                f"{len(snapshot.stale_exchanges)} stale"
            )

            # Alert on scan failures
            for ex_name, error in snapshot.stale_exchanges.items():
                await self.alerter.notify_scan_failure(ex_name, error)

            # 2. Save funding rates to database
            all_rates = []
            for ex_data in snapshot.exchange_data.values():
                all_rates.extend(ex_data.rates.values())
            if all_rates:
                self.database.save_funding_rates_batch(all_rates)
                FUNDING_RATES_SAVED.inc(len(all_rates))

            # 3. Update funding history and detect regime
            for rate in all_rates:
                key = f"{rate.exchange}:{rate.symbol}"
                if key not in self._funding_history:
                    self._funding_history[key] = []
                self._funding_history[key].append(rate.rate)
                # Keep last 100 observations
                if len(self._funding_history[key]) > 100:
                    self._funding_history[key] = self._funding_history[key][-100:]

            # 4. Regime detection and risk adjustment
            regime_counts: dict[FundingRegime, int] = {}
            for _key, history in self._funding_history.items():
                if len(history) >= 6:
                    series = pd.Series(history)
                    state = self.predictor.classify_regime(series)
                    regime_counts[state.current_regime] = regime_counts.get(state.current_regime, 0) + 1

            if regime_counts:
                dominant_regime = max(regime_counts, key=lambda r: regime_counts[r])
                if dominant_regime == FundingRegime.NEGATIVE:
                    self.risk_manager.adjust_for_regime("high")
                elif dominant_regime == FundingRegime.HIGH_POSITIVE:
                    self.risk_manager.adjust_for_regime("low")
                else:
                    self.risk_manager.adjust_for_regime("normal")

            # 5. Check margin health — flatten if critical
            margin_violations = self.risk_manager.check_margin_health(self.portfolio)
            for v in margin_violations:
                logger.warning(f"Margin alert: {v.message}")
                await self.alerter.notify_risk_violation(v)

            exchanges_to_flatten = self.risk_manager.exchanges_to_flatten(self.portfolio)
            if exchanges_to_flatten:
                logger.critical(f"Flattening positions on: {exchanges_to_flatten}")
                for pos in self.portfolio.positions:
                    if not pos.is_open:
                        continue
                    if pos.leg_a and pos.leg_a.exchange in exchanges_to_flatten:
                        await self.executor.close_position(pos)
                    elif pos.leg_b and pos.leg_b.exchange in exchanges_to_flatten:
                        await self.executor.close_position(pos)

            # 6. Check risk invariants on existing portfolio
            violations = self.risk_manager.check_invariants(self.portfolio)
            if violations:
                for v in violations:
                    logger.warning(f"Risk violation: {v.message}")
                    await self.alerter.notify_risk_violation(v)
                    RISK_VIOLATIONS.labels(severity=v.severity).inc()
                if any(v.severity == "critical" for v in violations):
                    logger.critical("Critical risk violations — skipping new trades")
                    return

            # 7. Check balance skew
            skew_alerts = self.rebalancer.check_skew(self.portfolio)
            for alert in skew_alerts:
                logger.warning(alert.message)
                if alert.severity == "critical":
                    suggestions = self.rebalancer.suggest_transfers(self.portfolio)
                    for s in suggestions:
                        logger.info(
                            f"  Suggested transfer: ${s['amount_usd']:,.2f} "
                            f"from {s['from']} to {s['to']}"
                        )

            # 8. Check kill switch before trading
            if self.executor.is_killed():
                logger.warning("Kill switch active — skipping trade execution")
                return

            # 9. Find opportunities (fast path: simple cross-exchange)
            simple_opps = self.optimizer.find_simple_opportunities(snapshot)
            SIMPLE_OPPORTUNITIES.set(len(simple_opps))
            if simple_opps:
                best_apr = max(o["annualized_yield"] for o in simple_opps) * 100
                BEST_YIELD_APR.set(best_apr)
                logger.info(f"Found {len(simple_opps)} simple opportunities:")
                for opp in simple_opps[:5]:
                    logger.info(
                        f"  {opp['instrument']}: short {opp['short_exchange']} "
                        f"({opp['short_rate']:.6f}) / long {opp['long_exchange']} "
                        f"({opp['long_rate']:.6f}) → net {opp['net_yield_per_period']:.6f}/period "
                        f"({opp['annualized_yield']*100:.1f}% APR)"
                    )
                    # Alert on top opportunity
                    if opp == simple_opps[0]:
                        await self.alerter.notify_opportunity(opp)
            else:
                BEST_YIELD_APR.set(0)

            # 10. Build graph and find complex opportunities
            graph = self.optimizer.build_graph(snapshot)
            graph_opps = self.optimizer.find_opportunities(
                graph, rate_history=self._funding_history
            )
            GRAPH_OPPORTUNITIES.set(len(graph_opps))
            if graph_opps:
                logger.info(f"Found {len(graph_opps)} graph-based opportunities")
                for opp in graph_opps[:3]:
                    logger.info(
                        f"  Cycle yield: {opp.expected_net_yield_per_period:.6f}/period, "
                        f"risk-adjusted: {opp.risk_adjusted_yield:.6f}"
                    )

            # 11. Size and execute top opportunity (respecting concurrent position cap)
            open_count = sum(1 for p in self.portfolio.positions if p.is_open)
            max_open = self.config.executor.max_open_positions

            if open_count >= max_open:
                logger.info(
                    f"At max open positions ({open_count}/{max_open}) — skipping new trades"
                )
            else:
                for opp in graph_opps[:1]:
                    size = self.risk_manager.calculate_position_size(opp, self.portfolio)
                    if size <= 0:
                        continue

                    if not self.risk_manager.check_pre_trade(opp, size, self.portfolio):
                        logger.info(f"Pre-trade risk check failed for size={size:.2f}")
                        continue

                    position = await self.executor.open_position(opp, size)
                    if position:
                        self.portfolio.add_position(position)
                        POSITIONS_OPENED.inc()

                        # Detect partial fill (one-legged position)
                        if position.leg_a and not position.leg_b:
                            logger.critical(
                                f"PARTIAL POSITION {position.id}: unhedged "
                                f"{position.leg_a.side.value} on {position.leg_a.exchange} — "
                                f"attempting immediate unwind"
                            )
                            await self.alerter.notify_emergency_unwind(
                                f"Partial position {position.id}: leg B failed. "
                                f"Unwinding leg A ({position.leg_a.side.value} on {position.leg_a.exchange})"
                            )
                            await self.executor.close_position(position)
                        elif self.config.executor.mode == "dry_run":
                            legs = []
                            if position.leg_a:
                                legs.append(position.leg_a.raw if position.leg_a.raw else {})
                            if position.leg_b:
                                legs.append(position.leg_b.raw if position.leg_b.raw else {})
                            await self.alerter.notify_dry_trade(
                                {"instrument": opp.cycle.nodes[0].instrument if opp.cycle.nodes else "N/A"},
                                legs,
                            )
                        else:
                            await self.alerter.notify_new_position(position)
                        logger.info(f"Opened position {position.id}: ${position.notional_usd:,.2f}")

            # 12. Update portfolio peak and metrics
            self.portfolio.update_peak()
            update_portfolio_metrics(self.portfolio)

            # 13. Daily summary (every 24h)
            now = datetime.now(UTC)
            if (
                self._last_daily_summary is None
                or (now - self._last_daily_summary).total_seconds() >= 86400
            ):
                await self.alerter.notify_daily_summary(self.portfolio)
                self._last_daily_summary = now

            elapsed = (datetime.now(UTC) - tick_start).total_seconds()
            TICK_DURATION.observe(elapsed)
            open_count = sum(1 for p in self.portfolio.positions if p.is_open)
            self._health.record_tick(success=True, open_positions=open_count)
            logger.info(f"Tick {self._iteration} completed in {elapsed:.2f}s")

        except Exception as e:
            TICK_ERRORS.inc()
            self._health.record_tick(success=False)
            logger.error(f"Tick {self._iteration} failed: {e}", exc_info=True)

    async def run_once(self) -> dict:
        """Run a single scan and return results (useful for testing/CLI)."""
        await self.scanner.initialize()
        try:
            snapshot = await self.scanner.scan()

            # Save rates
            all_rates = []
            for ex_data in snapshot.exchange_data.values():
                all_rates.extend(ex_data.rates.values())
            if all_rates:
                self.database.save_funding_rates_batch(all_rates)

            simple_opps = self.optimizer.find_simple_opportunities(snapshot)
            graph = self.optimizer.build_graph(snapshot)
            graph_opps = self.optimizer.find_opportunities(graph)

            return {
                "snapshot": snapshot,
                "simple_opportunities": simple_opps,
                "graph_opportunities": graph_opps,
                "violations": self.risk_manager.check_invariants(self.portfolio),
            }
        finally:
            await self.scanner.close()


async def main() -> None:
    """Entry point for the orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    orchestrator = Orchestrator(config)
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
