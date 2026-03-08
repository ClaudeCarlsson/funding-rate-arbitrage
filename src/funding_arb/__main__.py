"""CLI entry point for the funding rate arbitrage system."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .backup import backup_all
from .config import load_config
from .database import Database
from .executor import KILL_SWITCH_PATH
from .orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Funding Rate Arbitrage System"
    )
    parser.add_argument(
        "--config", "-c", default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )

    sub = parser.add_subparsers(dest="command")

    # Default: run the main loop
    sub.add_parser("run", help="Start the main trading loop")

    # Scan once and exit
    sub.add_parser("scan", help="Run a single scan and exit")

    # Show system status
    sub.add_parser("status", help="Show system status and open positions")

    # Database backup
    backup_parser = sub.add_parser("backup", help="Backup all SQLite databases")
    backup_parser.add_argument(
        "--dir", default="data/backups",
        help="Backup directory (default: data/backups)",
    )
    backup_parser.add_argument(
        "--keep", type=int, default=7,
        help="Number of backups to retain (default: 7)",
    )

    # Kill switch management
    kill_parser = sub.add_parser("kill", help="Manage the kill switch")
    kill_parser.add_argument("action", choices=["on", "off", "status"])

    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "kill":
        _handle_kill_switch(args.action)
        return

    config = load_config(args.config)

    if args.command == "backup":
        paths = backup_all(
            state_db=config.database.state_db_path,
            trades_db=config.database.trades_db_path,
            funding_db=config.database.funding_db_path,
            backup_dir=args.dir,
            max_backups=args.keep,
        )
        for p in paths:
            print(f"Backed up: {p}")
        if not paths:
            print("No databases found to backup")
        return

    if args.command == "status":
        _show_status(config)
        return

    if args.command == "scan":
        orchestrator = Orchestrator(config)
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
        return

    # Default: run the main loop
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())


def _handle_kill_switch(action: str) -> None:
    if action == "on":
        KILL_SWITCH_PATH.touch()
        print(f"Kill switch ACTIVATED at {KILL_SWITCH_PATH}")
    elif action == "off":
        KILL_SWITCH_PATH.unlink(missing_ok=True)
        print(f"Kill switch DEACTIVATED")
    elif action == "status":
        if KILL_SWITCH_PATH.exists():
            print(f"Kill switch is ON ({KILL_SWITCH_PATH})")
        else:
            print("Kill switch is OFF — trading is allowed")


def _show_status(config) -> None:
    db = Database(
        state_db_path=config.database.state_db_path,
        trades_db_path=config.database.trades_db_path,
        funding_db_path=config.database.funding_db_path,
        parquet_dir=config.database.parquet_dir,
    )

    # Kill switch
    kill_status = "ON (no new trades)" if KILL_SWITCH_PATH.exists() else "OFF (trading allowed)"
    print(f"Kill switch: {kill_status}")
    print(f"Mode: {config.executor.mode.upper()}")
    print(f"Max position: ${config.executor.max_position_usd:,.0f}")
    print(f"Max open positions: {config.executor.max_open_positions}")

    # Exchanges
    print(f"\nExchanges:")
    for name, ex in config.exchanges.items():
        status = "enabled" if ex.enabled else "disabled"
        has_key = "key set" if ex.api_key else "no key"
        print(f"  {name}: {status}, {has_key}, max ${ex.max_position_usd:,.0f}")

    # Open positions
    positions = db.get_open_positions()
    print(f"\nOpen positions: {len(positions)}")
    for pos in positions:
        legs = []
        if pos.leg_a:
            legs.append(f"{pos.leg_a.side.value} on {pos.leg_a.exchange}")
        if pos.leg_b:
            legs.append(f"{pos.leg_b.side.value} on {pos.leg_b.exchange}")
        print(
            f"  {pos.id}: ${pos.notional_usd:,.2f} | "
            f"funding=${pos.funding_collected:,.2f} | "
            f"{' / '.join(legs)}"
        )

    # Recent rates
    recent_rates = db.get_funding_rates(limit=10)
    if recent_rates:
        print(f"\nLast {len(recent_rates)} funding rates:")
        for r in recent_rates[:10]:
            ann = r.annualized * 100
            print(f"  {r.exchange}/{r.symbol}: {r.rate:.6f} ({ann:.1f}% APR) @ {r.timestamp}")


if __name__ == "__main__":
    main()
