"""Context7 API client — live library documentation and code examples.

Base URL: https://context7.com/api/v2
Auth: Optional CONTEXT7_API_KEY for higher rate limits.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger("research-mcp-server")

CONTEXT7_API = "https://context7.com/api/v2"


def _normalize_library(lib: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Context7 library result."""
    return {
        "id": lib.get("id", ""),
        "name": lib.get("name", ""),
        "description": lib.get("description", ""),
        "total_snippets": lib.get("totalSnippets", 0),
        "trust_score": lib.get("trustScore", ""),
        "benchmark_score": lib.get("benchmarkScore", 0),
        "versions": lib.get("versions", []),
    }


class Context7Client:
    """Async client for Context7 documentation API."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("CONTEXT7_API_KEY", "").strip() or None
        self._headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"
            logger.info("Context7 client initialized with API key")
        else:
            logger.info("Context7 client initialized without API key (lower rate limits)")

    async def _request(self, path: str, params: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{CONTEXT7_API}{path}",
                params=params,
                headers=self._headers,
            )
            if resp.status_code == 202:
                return {"status": "processing", "message": "Library not finalized yet. Try again shortly."}
            if resp.status_code == 301:
                return {"status": "redirect", "redirect_url": resp.headers.get("Location", "")}
            resp.raise_for_status()
            return resp.json()

    async def resolve_library(
        self,
        library_name: str,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Resolve a library name to Context7-compatible IDs.

        Args:
            library_name: Library name (e.g., 'react', 'fastapi', 'drizzle').
            query: What you want to do (used for relevance ranking).

        Returns:
            List of matching libraries with IDs, descriptions, snippet counts.
        """
        data = await self._request("/libs/search", {
            "libraryName": library_name,
            "query": query or f"documentation for {library_name}",
        })
        if isinstance(data, dict) and data.get("status") in ("processing", "redirect"):
            return [data]
        if isinstance(data, list):
            return [_normalize_library(lib) for lib in data]
        return []

    async def query_docs(
        self,
        library_id: str,
        query: str,
        output_type: str = "json",
    ) -> dict[str, Any]:
        """Query documentation for a specific library.

        Args:
            library_id: Context7 library ID (e.g., '/facebook/react').
            query: What you want to know (e.g., 'how to set up authentication').
            output_type: 'json' for structured snippets, 'txt' for plain text.

        Returns:
            Documentation snippets matching the query.
        """
        data = await self._request("/context", {
            "libraryId": library_id,
            "query": query,
            "type": output_type,
        })
        if isinstance(data, dict) and data.get("status") in ("processing", "redirect"):
            return data
        if isinstance(data, list):
            return {
                "library_id": library_id,
                "query": query,
                "total_snippets": len(data),
                "snippets": data[:10],  # Cap response size
            }
        return {"library_id": library_id, "query": query, "snippets": [], "raw": data}
