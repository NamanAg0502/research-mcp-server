"""Stack Overflow API client.

Base URL: https://api.stackexchange.com/2.3
Auth: Optional STACKOVERFLOW_KEY for higher limits (10k req/day vs 300).
Rate limit: 30 req/s with key, lower without.
"""

import logging
import os
from typing import Any, Optional

import httpx

from ..utils.rate_limiter import so_limiter

logger = logging.getLogger("research-mcp-server")

SO_API = "https://api.stackexchange.com/2.3"


def _normalize_question(q: dict[str, Any]) -> dict[str, Any]:
    """Normalize an SO question."""
    return {
        "id": q.get("question_id", ""),
        "source": "stackoverflow",
        "title": q.get("title", ""),
        "link": q.get("link", ""),
        "score": q.get("score", 0),
        "answer_count": q.get("answer_count", 0),
        "view_count": q.get("view_count", 0),
        "is_answered": q.get("is_answered", False),
        "tags": q.get("tags", []),
        "creation_date": q.get("creation_date", 0),
        "author": (q.get("owner") or {}).get("display_name", ""),
    }


def _normalize_tag(t: dict[str, Any]) -> dict[str, Any]:
    """Normalize an SO tag."""
    return {
        "name": t.get("name", ""),
        "count": t.get("count", 0),
        "has_synonyms": t.get("has_synonyms", False),
        "is_moderator_only": t.get("is_moderator_only", False),
    }


class SOClient:
    """Async client for Stack Overflow API."""

    def __init__(self) -> None:
        self._key = os.environ.get("STACKOVERFLOW_KEY", "").strip() or None

    def _base_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"site": "stackoverflow"}
        if self._key:
            params["key"] = self._key
        return params

    async def _request(self, path: str, params: dict | None = None) -> Any:
        await so_limiter.wait()
        merged = {**self._base_params(), **(params or {})}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{SO_API}{path}", params=merged)
            resp.raise_for_status()
            return resp.json()

    async def search(
        self,
        query: str,
        tagged: str | None = None,
        sort: str = "relevance",
        max_results: int = 15,
    ) -> list[dict[str, Any]]:
        """Search SO questions."""
        params: dict[str, Any] = {
            "order": "desc",
            "sort": sort,
            "intitle": query,
            "pagesize": min(max_results, 30),
            "filter": "!nNPvSNdWme",  # Include body excerpt
        }
        if tagged:
            params["tagged"] = tagged
        data = await self._request("/search/advanced", params)
        return [_normalize_question(q) for q in data.get("items", [])]

    async def tag_info(self, tags: list[str]) -> list[dict[str, Any]]:
        """Get info about tags (question count = adoption proxy)."""
        tag_str = ";".join(tags[:5])
        data = await self._request(f"/tags/{tag_str}/info")
        return [_normalize_tag(t) for t in data.get("items", [])]

    async def trending_questions(
        self,
        tagged: str | None = None,
        max_results: int = 15,
    ) -> list[dict[str, Any]]:
        """Get recent hot questions, optionally filtered by tag."""
        params: dict[str, Any] = {
            "order": "desc",
            "sort": "activity",
            "pagesize": min(max_results, 30),
        }
        if tagged:
            params["tagged"] = tagged
        data = await self._request("/questions", params)
        return [_normalize_question(q) for q in data.get("items", [])]
