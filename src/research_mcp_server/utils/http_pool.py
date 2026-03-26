"""Shared httpx.AsyncClient pool — reuses connections across tool calls.

Instead of creating/destroying an httpx.AsyncClient per API call (27 times across
the codebase), this pool maintains persistent clients per base URL with connection
multiplexing and keep-alive.

Usage:
    from ..utils.http_pool import http_pool

    # In any client method:
    resp = await http_pool.get("https://api.github.com/repos/foo/bar", headers={...})
    resp = await http_pool.post("https://api.anthropic.com/v1/messages", json={...}, headers={...})
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("research-mcp-server")


class HttpPool:
    """Singleton pool of httpx.AsyncClient instances, one per base URL.

    Benefits:
    - TCP connection reuse (keep-alive)
    - DNS caching
    - HTTP/2 multiplexing where supported
    - Reduced memory allocation per request
    """

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _get_base(self, url: str) -> str:
        """Extract base URL (scheme + host) from a full URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_client(
        self,
        url: str,
        timeout: float = 15.0,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        """Get or create an AsyncClient for the given URL's base."""
        base = self._get_base(url)
        if base not in self._clients:
            self._clients[base] = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=follow_redirects,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
            logger.debug(f"Created pooled client for {base}")
        return self._clients[base]

    async def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float = 15.0,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        """Pooled GET request."""
        client = self._get_client(url, timeout=timeout, follow_redirects=follow_redirects)
        return await client.get(url, params=params, headers=headers)

    async def post(
        self,
        url: str,
        json: Any = None,
        data: Any = None,
        headers: dict | None = None,
        auth: tuple | None = None,
        timeout: float = 15.0,
    ) -> httpx.Response:
        """Pooled POST request."""
        client = self._get_client(url, timeout=timeout)
        return await client.post(url, json=json, data=data, headers=headers, auth=auth)

    async def close(self) -> None:
        """Close all pooled clients. Call on server shutdown."""
        for base, client in self._clients.items():
            await client.aclose()
            logger.debug(f"Closed pooled client for {base}")
        self._clients.clear()


# Singleton instance
http_pool = HttpPool()
