"""The daemon's one piece of durable state.

A job submitted to an async backend is identified there by a handle the submit
response returned. Holding it only in memory means a restart submits the job a
second time — and the first run keeps billing the operator. So the handle is
written here the moment it is known and deleted when the backend phase ends;
boot recovery resumes whatever is still on record.

Stdlib sqlite in autocommit mode: one row per in-flight job, written and
deleted from the event loop thread. No path means an in-memory table, which is
what a scheduler built in code (tests) gets; the YAML loader always names a file.

The schema is `vorqd/migrations/NNNN_*.sql`, applied in filename order on
open; `schema_migrations` records each file a state file has run, by name. A
change is the next numbered file — never an edit to a shipped one, whose
`IF NOT EXISTS` is what adopted files that predate the record. The rows are
handles a backend is still billing against, so a file is migrated in place and
never recreated.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from importlib import resources


def _load_migrations() -> tuple[tuple[str, str], ...]:
    """`(version, sql)` per `migrations/*.sql`, in filename order; the filename is the version."""
    files = sorted(
        (f for f in (resources.files("vorqd") / "migrations").iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )
    return tuple((f.name, f.read_text()) for f in files)


MIGRATIONS: tuple[tuple[str, str], ...] = _load_migrations()


def _statements(sql: str) -> list[str]:
    """Splits a file into complete statements; `execute` runs one at a time."""
    out, buf = [], ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf)
            buf = ""
    if any(line.strip() and not line.strip().startswith("--") for line in buf.splitlines()):
        raise RuntimeError(f"incomplete statement in migration: {buf.strip()[:60]}")
    return out


@dataclass(frozen=True)
class InflightRow:
    job_id: str
    model: str
    handle: str
    created_at: float


class InflightStore:
    def __init__(self, path: str | None, *, clock=time.time) -> None:
        self._clock = clock
        self._db = sqlite3.connect(path or ":memory:", isolation_level=None)
        if path:
            self._db.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        # IMMEDIATE takes the write lock before reading the record, so two
        # daemons opening one file cannot both run the same migration.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version TEXT PRIMARY KEY, applied_at REAL NOT NULL)"
            )
            applied = {row[0] for row in self._db.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                # executescript would COMMIT the open transaction first.
                for statement in _statements(sql):
                    self._db.execute(statement)
                self._db.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, time.time()),
                )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def put(self, job_id: str, model: str, handle: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO inflight (job_id, model, handle, created_at) VALUES (?, ?, ?, ?)",
            (job_id, model, handle, float(self._clock())),
        )

    def get(self, job_id: str) -> InflightRow | None:
        row = self._db.execute(
            "SELECT job_id, model, handle, created_at FROM inflight WHERE job_id = ?", (job_id,)
        ).fetchone()
        return InflightRow(*row) if row else None

    def delete(self, job_id: str) -> None:
        self._db.execute("DELETE FROM inflight WHERE job_id = ?", (job_id,))

    def all(self) -> list[InflightRow]:
        rows = self._db.execute(
            "SELECT job_id, model, handle, created_at FROM inflight ORDER BY created_at, job_id"
        ).fetchall()
        return [InflightRow(*row) for row in rows]

    def close(self) -> None:
        self._db.close()
