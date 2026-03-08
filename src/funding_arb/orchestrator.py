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
from .models import Portfolio
from .optimizer import ArbitrageOptimizer
from .prediction import FundingPredictor, FundingRegime
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

    async def start(self) -> None:
        """Start the main loop."""
        logger.info("Starting orchestrator...")
        await self.scanner.initialize()
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
        await self.scanner.close()
        logger.info("Orchestrator stopped")

    async def _tick(self) -> None:
        """Single iteration of the main loop."""
        self._iteration += 1
        tick_start = datetime.now(UTC)

        try:
            # 1. Scan all exchanges
            snapshot = await self.scanner.scan()
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

            # 5. Check risk invariants on existing portfolio
            violations = self.risk_manager.check_invariants(self.portfolio)
            if violations:
                for v in violations:
                    logger.warning(f"Risk violation: {v.message}")
                    await self.alerter.notify_risk_violation(v)
                if any(v.severity == "critical" for v in violations):
                    logger.critical("Critical risk violations — skipping new trades")
                    return

            # 6. Find opportunities (fast path: simple cross-exchange)
            simple_opps = self.optimizer.find_simple_opportunities(snapshot)
            if simple_opps:
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

            # 7. Build graph and find complex opportunities
            graph = self.optimizer.build_graph(snapshot)
            graph_opps = self.optimizer.find_opportunities(graph)
            if graph_opps:
                logger.info(f"Found {len(graph_opps)} graph-based opportunities")
                for opp in graph_opps[:3]:
                    logger.info(
                        f"  Cycle yield: {opp.expected_net_yield_per_period:.6f}/period, "
                        f"risk-adjusted: {opp.risk_adjusted_yield:.6f}"
                    )

            # 8. Size and execute (paper trading for now)
            for opp in graph_opps[:1]:  # only top opportunity
                size = self.risk_manager.calculate_position_size(opp, self.portfolio)
                if size > 0:
                    logger.info(f"Would open position: size={size:.2f}")
                    # Executor would be called here in live mode

            # 9. Update portfolio peak
            self.portfolio.update_peak()

            elapsed = (datetime.now(UTC) - tick_start).total_seconds()
            logger.info(f"Tick {self._iteration} completed in {elapsed:.2f}s")

        except Exception as e:
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
