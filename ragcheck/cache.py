"""SQLite-backed cache for LLM judgments."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def make_key(*parts: str) -> str:
    """Build a stable SHA256 cache key from ordered string components.

    Callers pass (metric_name, prompt_version, question, context, answer, ...);
    components are length-prefixed so no two part sequences collide.
    """
    h = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
    return h.hexdigest()


class JudgmentCache:
    """Persistent key-value store for judge outputs.

    Hits are counted on ``hits`` / ``misses`` so the runner can report
    cache effectiveness on re-runs.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the cache database at ``path``."""
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        """Return the cached value for ``key``, or None on a miss."""
        row = self._conn.execute("SELECT value FROM judgments WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return str(row[0])

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``, overwriting any previous entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO judgments (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
