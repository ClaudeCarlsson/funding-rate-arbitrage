"""CLI entry point for the funding rate arbitrage system."""
from __future__ import annotations

import argparse
import asyncio
import logging

from .config import load_config
from .hft_node import HFTNode
from .orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Funding Rate Arbitrage System"
    )
    parser.add_argument(
        "--config", "-c", default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )
    parser.add_argument(
        "--scan-once", action="store_true",
        help="Run a single scan and exit",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--node-role", choices=["brain", "execution"], default=None,
        help="Run as a specialized HFT mesh node (bypasses standard orchestrator)",
    )
    parser.add_argument(
        "--node-id", default="local_node",
        help="Identifier for the HFT node (e.g., 'tokyo_exec_1')",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.node_role:
        # Run the distributed bare-metal mesh
        node = HFTNode(role=args.node_role, node_id=args.node_id, peers=["peer1", "peer2"])
        try:
            asyncio.run(node.start())
        except KeyboardInterrupt:
            logging.info("Shutting down HFT Node...")
        finally:
            node.shutdown()
        return

    config = load_config(args.config)
    orchestrator = Orchestrator(config)

    if args.scan_once:
        result = asyncio.run(orchestrator.run_once())
        print(f"\nExchanges scanned: {len(result['snapshot'].exchanges)}")
        print(f"Instruments: {result['snapshot'].instruments}")
        print(f"Stale: {result['snapshot'].stale_exchanges}")
        print(f"\nSimple opportunities: {len(result['simple_opportunities'])}")
        for opp in result["simple_opportunities"][:5]:
            print(
                f"  {opp['instrument']}: "
                f"short {opp['short_exchange']} ({opp['short_rate']:.6f}) / "
                f"long {opp['long_exchange']} ({opp['long_rate']:.6f}) → "
                f"net {opp['net_yield_per_period']:.6f}/period "
                f"({opp['annualized_yield']*100:.1f}% APR)"
            )
        print(f"\nGraph opportunities: {len(result['graph_opportunities'])}")
        print(f"Risk violations: {len(result['violations'])}")
    else:
        asyncio.run(orchestrator.start())


if __name__ == "__main__":
    main()
