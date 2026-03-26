"""TTL-based response cache for tool results.

Caches tool responses to avoid redundant API calls. Uses aiosqlite
(already a project dependency) for persistence across server restarts.

Usage:
    from ..store.cache import response_cache

    # Check cache before calling API:
    cached = await response_cache.get("hn", {"action": "trending"})
    if cached:
        return cached

    # After getting fresh result:
    await response_cache.set("hn", {"action": "trending"}, result, ttl_category="trending")
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from ..config import Settings

logger = logging.getLogger("research-mcp-server")

# TTL categories in seconds
TTL_SECONDS = {
    "trending": 900,           # 15 minutes
    "search": 3600,            # 1 hour
    "stats": 86400,            # 24 hours
    "paper_metadata": 604800,  # 7 days
    "default": 1800,           # 30 minutes
}


def _make_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Create a stable cache key from tool name + arguments."""
    # Sort keys for stability, exclude non-deterministic params
    sorted_args = json.dumps(arguments, sort_keys=True, default=str)
    raw = f"{tool_name}:{sorted_args}"
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    """Async SQLite-backed response cache with TTL support."""

    def __init__(self) -> None:
        self._db_path: Optional[Path] = None
        self._initialized = False

    def _get_db_path(self) -> Path:
        if self._db_path is None:
            settings = Settings()
            self._db_path = settings.STORAGE_PATH / "response_cache.db"
        return self._db_path

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        if not self._initialized:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    response TEXT NOT NULL,
                    ttl_category TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON cache(expires_at)
            """)
            await db.commit()
            self._initialized = True

    async def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Optional[str]:
        """Get cached response if it exists and hasn't expired.

        Returns the raw JSON string, or None if cache miss/expired.
        """
        key = _make_key(tool_name, arguments)
        now = time.time()

        async with aiosqlite.connect(self._get_db_path()) as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT response FROM cache WHERE key = ? AND expires_at > ?",
                (key, now),
            )
            row = await cursor.fetchone()
            if row:
                logger.debug(f"Cache HIT for {tool_name}")
                return row[0]
            logger.debug(f"Cache MISS for {tool_name}")
            return None

    async def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        response: str,
        ttl_category: str = "default",
    ) -> None:
        """Store a response in the cache."""
        key = _make_key(tool_name, arguments)
        now = time.time()
        ttl = TTL_SECONDS.get(ttl_category, TTL_SECONDS["default"])
        expires_at = now + ttl

        async with aiosqlite.connect(self._get_db_path()) as db:
            await self._ensure_table(db)
            await db.execute(
                """INSERT OR REPLACE INTO cache (key, tool_name, response, ttl_category, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, tool_name, response, ttl_category, now, expires_at),
            )
            await db.commit()
            logger.debug(f"Cache SET for {tool_name} (ttl={ttl}s, category={ttl_category})")

    async def invalidate(self, tool_name: Optional[str] = None) -> int:
        """Invalidate cache entries. If tool_name given, only that tool's entries."""
        async with aiosqlite.connect(self._get_db_path()) as db:
            await self._ensure_table(db)
            if tool_name:
                cursor = await db.execute("DELETE FROM cache WHERE tool_name = ?", (tool_name,))
            else:
                cursor = await db.execute("DELETE FROM cache")
            await db.commit()
            count = cursor.rowcount
            logger.info(f"Cache invalidated: {count} entries" + (f" for {tool_name}" if tool_name else ""))
            return count

    async def cleanup(self) -> int:
        """Remove expired entries."""
        now = time.time()
        async with aiosqlite.connect(self._get_db_path()) as db:
            await self._ensure_table(db)
            cursor = await db.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            await db.commit()
            count = cursor.rowcount
            if count:
                logger.info(f"Cache cleanup: removed {count} expired entries")
            return count

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        async with aiosqlite.connect(self._get_db_path()) as db:
            await self._ensure_table(db)
            total = (await (await db.execute("SELECT COUNT(*) FROM cache")).fetchone())[0]
            active = (await (await db.execute("SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,))).fetchone())[0]
            expired = total - active

            # Per-category breakdown
            cursor = await db.execute(
                "SELECT ttl_category, COUNT(*) FROM cache WHERE expires_at > ? GROUP BY ttl_category",
                (now,),
            )
            by_category = {row[0]: row[1] for row in await cursor.fetchall()}

            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": expired,
                "by_category": by_category,
            }


# Singleton instance
response_cache = ResponseCache()
