"""Stack Overflow tool — search questions, check tag trends, hot topics."""

import json
import logging
from typing import Any, Dict, List

import mcp.types as types

from ..clients.so_client import SOClient

logger = logging.getLogger("research-mcp-server")

so_tool = types.Tool(
    name="stackoverflow",
    description=(
        "Search Stack Overflow questions and check tag popularity. "
        "Tag question counts are a proxy for technology adoption. "
        "Optional STACKOVERFLOW_KEY env var for higher rate limits.\n"
        "Actions:\n"
        "- 'search': Search questions by keyword, optionally filtered by tag.\n"
        "- 'tags': Get question counts for tags (adoption signal). E.g., tags=['fastapi','flask','django'].\n"
        "- 'trending': Get recent active questions, optionally by tag."
    ),
    inputSchema={
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "tags", "trending"],
                "description": "The operation to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search'). E.g., 'FastAPI authentication JWT'.",
            },
            "tagged": {
                "type": "string",
                "description": "Filter by tag (for 'search' and 'trending'). E.g., 'python', 'rust'. Semicolon-separated for multiple: 'python;fastapi'.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
                "description": "Tags to look up (for 'tags' action). Returns question counts per tag.",
            },
            "sort": {
                "type": "string",
                "enum": ["relevance", "votes", "creation", "activity"],
                "description": "Sort order for 'search'. Default: relevance.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "description": "Maximum results. Default: 15.",
            },
        },
    },
)


async def handle_so(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Dispatch Stack Overflow tool calls."""
    action = arguments.get("action")
    if not action:
        return [types.TextContent(type="text", text="Error: 'action' is required.")]

    client = SOClient()
    max_results = arguments.get("max_results", 15)

    try:
        if action == "search":
            query = arguments.get("query")
            if not query:
                return [types.TextContent(type="text", text="Error: 'query' is required for search.")]
            results = await client.search(
                query=query,
                tagged=arguments.get("tagged"),
                sort=arguments.get("sort", "relevance"),
                max_results=max_results,
            )
            return [types.TextContent(type="text", text=json.dumps({"total": len(results), "questions": results}, indent=2))]

        elif action == "tags":
            tags = arguments.get("tags")
            if not tags:
                return [types.TextContent(type="text", text="Error: 'tags' array is required.")]
            results = await client.tag_info(tags)
            return [types.TextContent(type="text", text=json.dumps({"tags": results}, indent=2))]

        elif action == "trending":
            results = await client.trending_questions(
                tagged=arguments.get("tagged"),
                max_results=max_results,
            )
            return [types.TextContent(type="text", text=json.dumps({"total": len(results), "questions": results}, indent=2))]

        else:
            return [types.TextContent(type="text", text=f"Error: Unknown action '{action}'.")]

    except Exception as e:
        logger.error(f"SO tool error: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
