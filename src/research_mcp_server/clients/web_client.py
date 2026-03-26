"""Web content fetcher — extracts structured content from URLs.

Primary: httpx + HTML tag stripping (no extra deps, handles most pages).
Future: Lightpanda CDP for JS-rendered pages.
"""

import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("research-mcp-server")

# Common user agent to avoid blocks
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-mcp-server/1.0"


def _strip_html(html: str) -> str:
    """Convert HTML to readable text by stripping tags and normalizing whitespace."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Convert common elements to text markers
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?p[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?div[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?h[1-6][^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>', '\n- ', text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')

    # Normalize whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _extract_title(html: str) -> str:
    """Extract page title from HTML."""
    match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_meta_description(html: str) -> str:
    """Extract meta description from HTML."""
    match = re.search(
        r'<meta\s+(?:[^>]*\s+)?(?:name|property)=["\'](?:description|og:description)["\'][^>]*\s+content=["\']([^"\']*)["\']',
        html, re.IGNORECASE
    )
    if not match:
        match = re.search(
            r'<meta\s+(?:[^>]*\s+)?content=["\']([^"\']*)["\'][^>]*\s+(?:name|property)=["\'](?:description|og:description)["\']',
            html, re.IGNORECASE
        )
    return match.group(1).strip() if match else ""


def _extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Extract links from HTML."""
    links = []
    for match in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE):
        href = match.group(1).strip()
        text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        if href and text and not href.startswith(('#', 'javascript:', 'mailto:')):
            # Make relative URLs absolute
            if href.startswith('/'):
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                href = f"{parsed.scheme}://{parsed.netloc}{href}"
            links.append({"text": text[:100], "url": href})
    return links[:50]  # Cap at 50 links


class WebClient:
    """Async web content fetcher."""

    async def fetch(
        self,
        url: str,
        extract: str = "article",
        max_length: int = 10000,
    ) -> dict[str, Any]:
        """Fetch and extract content from a URL.

        Args:
            url: URL to fetch.
            extract: Extraction mode — 'article' (main text), 'links', 'raw' (full text), 'metadata'.
            max_length: Max characters to return for text content.

        Returns:
            Dict with extracted content.
        """
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, max_redirects=5) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        result: dict[str, Any] = {
            "url": str(resp.url),  # Final URL after redirects
            "status": resp.status_code,
            "title": _extract_title(html),
            "description": _extract_meta_description(html),
        }

        if extract == "metadata":
            return result

        if extract == "links":
            result["links"] = _extract_links(html, str(resp.url))
            return result

        # article or raw
        text = _strip_html(html)
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[Truncated at {max_length} characters]"
        result["content"] = text
        result["content_length"] = len(text)

        return result
