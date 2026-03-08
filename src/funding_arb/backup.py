"""SQLite database backup utility with rotation."""
from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_BACKUPS = 7


def backup_database(
    db_path: str | Path,
    backup_dir: str | Path,
    max_backups: int = DEFAULT_MAX_BACKUPS,
) -> Path:
    """Create a timestamped SQLite backup using the online backup API.

    Args:
        db_path: Path to the SQLite database file.
        backup_dir: Directory to store backups.
        max_backups: Number of backups to retain (oldest deleted first).

    Returns:
        Path to the new backup file.
    """
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    backup_name = f"{db_path.stem}_{timestamp}.db"
    backup_path = backup_dir / backup_name

    # Use SQLite online backup API for consistency
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(backup_path)
    try:
        src.backup(dst)
        logger.info(f"Backed up {db_path.name} → {backup_path}")
    finally:
        dst.close()
        src.close()

    # Rotate old backups
    _rotate_backups(backup_dir, db_path.stem, max_backups)

    return backup_path


def _rotate_backups(backup_dir: Path, stem: str, max_backups: int) -> None:
    """Delete oldest backups exceeding max_backups."""
    pattern = f"{stem}_*.db"
    backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    while len(backups) > max_backups:
        oldest = backups.pop(0)
        oldest.unlink()
        logger.info(f"Rotated old backup: {oldest.name}")


def backup_all(
    state_db: str | Path,
    trades_db: str | Path,
    funding_db: str | Path,
    backup_dir: str | Path = "data/backups",
    max_backups: int = DEFAULT_MAX_BACKUPS,
) -> list[Path]:
    """Backup all three SQLite databases.

    Returns list of backup file paths.
    """
    backup_dir = Path(backup_dir)
    results = []

    for db in [state_db, trades_db, funding_db]:
        db_path = Path(db)
        if db_path.exists():
            result = backup_database(db_path, backup_dir, max_backups)
            results.append(result)
        else:
            logger.warning(f"Skipping backup of {db} (does not exist)")

    return results
