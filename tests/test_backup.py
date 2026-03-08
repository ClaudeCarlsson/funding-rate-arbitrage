"""Tests for the SQLite backup utility."""
from __future__ import annotations

import sqlite3

import pytest

from funding_arb.backup import backup_all, backup_database, _rotate_backups


class TestBackupDatabase:
    def test_creates_backup(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
        conn.close()

        backup_dir = tmp_path / "backups"
        result = backup_database(db, backup_dir)

        assert result.exists()
        assert result.parent == backup_dir
        assert result.name.startswith("test_")
        assert result.name.endswith(".db")

        # Verify data is intact in backup
        bk = sqlite3.connect(result)
        row = bk.execute("SELECT id FROM t").fetchone()
        bk.close()
        assert row[0] == 42

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            backup_database(tmp_path / "nonexistent.db", tmp_path / "bk")

    def test_rotation_keeps_max(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        backup_dir = tmp_path / "backups"
        paths = []
        for i in range(5):
            p = backup_database(db, backup_dir, max_backups=3)
            paths.append(p)

        remaining = list(backup_dir.glob("test_*.db"))
        assert len(remaining) == 3
        # The 3 newest should survive
        assert paths[-1].exists()
        assert paths[-2].exists()
        assert paths[-3].exists()

    def test_rotation_with_one_backup(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        backup_dir = tmp_path / "backups"
        p = backup_database(db, backup_dir, max_backups=1)
        assert p.exists()
        assert len(list(backup_dir.glob("test_*.db"))) == 1

    def test_backup_dir_created(self, tmp_path):
        db = tmp_path / "test.db"
        sqlite3.connect(db).close()

        deep = tmp_path / "a" / "b" / "c"
        backup_database(db, deep)
        assert deep.exists()


class TestBackupAll:
    def test_backs_up_existing_dbs(self, tmp_path):
        state = tmp_path / "state.db"
        trades = tmp_path / "trades.db"
        funding = tmp_path / "funding.db"

        for db in [state, trades, funding]:
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.commit()
            conn.close()

        backup_dir = tmp_path / "backups"
        results = backup_all(state, trades, funding, backup_dir)
        assert len(results) == 3
        for r in results:
            assert r.exists()

    def test_skips_missing_dbs(self, tmp_path):
        state = tmp_path / "state.db"
        sqlite3.connect(state).close()

        results = backup_all(
            state, tmp_path / "nope1.db", tmp_path / "nope2.db",
            backup_dir=tmp_path / "backups"
        )
        assert len(results) == 1

    def test_empty_all_missing(self, tmp_path):
        results = backup_all(
            tmp_path / "a.db", tmp_path / "b.db", tmp_path / "c.db",
            backup_dir=tmp_path / "backups"
        )
        assert results == []


class TestRotateBackups:
    def test_rotate_deletes_oldest(self, tmp_path):
        for i in range(5):
            (tmp_path / f"test_{i:04d}.db").write_text(str(i))

        _rotate_backups(tmp_path, "test", max_backups=2)
        remaining = sorted(tmp_path.glob("test_*.db"))
        assert len(remaining) == 2

    def test_rotate_noop_under_limit(self, tmp_path):
        for i in range(2):
            (tmp_path / f"test_{i:04d}.db").write_text(str(i))

        _rotate_backups(tmp_path, "test", max_backups=5)
        assert len(list(tmp_path.glob("test_*.db"))) == 2
