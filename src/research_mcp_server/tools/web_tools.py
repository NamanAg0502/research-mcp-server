"""Web scraping tool — fetch and extract content from any URL."""

import json
import logging
from typing import Any, Dict, List

import mcp.types as types

from ..clients.web_client import WebClient

logger = logging.getLogger("research-mcp-server")

web_tool = types.Tool(
    name="web",
    description=(
        "Fetch and extract content from any URL. Handles HTML pages, blogs, "
        "documentation, and news articles. Use for sources without dedicated APIs: "
        "tech blogs, company engineering posts, conference pages, tech radar sites.\n\n"
        "Extract modes:\n"
        "- 'article': Main text content (default, strips nav/footer/scripts).\n"
        "- 'links': Extract all links from the page.\n"
        "- 'metadata': Just title + description (fast).\n"
        "- 'raw': Full page text without filtering."
    ),
    inputSchema={
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch. Must be a valid HTTP/HTTPS URL.",
            },
            "extract": {
                "type": "string",
                "enum": ["article", "links", "metadata", "raw"],
                "description": "Extraction mode. Default: article.",
            },
            "max_length": {
                "type": "integer",
                "minimum": 1000,
                "maximum": 50000,
                "description": "Max characters for text content. Default: 10000.",
            },
        },
    },
)


async def handle_web(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Fetch and extract web content."""
    url = arguments.get("url")
    if not url:
        return [types.TextContent(type="text", text="Error: 'url' is required.")]

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    client = WebClient()
    extract = arguments.get("extract", "article")
    max_length = arguments.get("max_length", 10000)

    try:
        result = await client.fetch(url=url, extract=extract, max_length=max_length)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Web tool error for {url}: {e}")
        return [types.TextContent(type="text", text=f"Error fetching {url}: {str(e)}")]
