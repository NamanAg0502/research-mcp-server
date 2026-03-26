"""Context7 tool — live library documentation and code examples."""

import json
import logging
from typing import Any, Dict, List

import mcp.types as types

from ..clients.context7_client import Context7Client

logger = logging.getLogger("research-mcp-server")

context7_tool = types.Tool(
    name="docs",
    description=(
        "Get up-to-date documentation and code examples for any library or framework "
        "via Context7. Useful when evaluating tools or needing current API references.\n"
        "Actions:\n"
        "- 'resolve': Find a library by name and get its Context7 ID.\n"
        "- 'query': Get documentation snippets for a specific question.\n"
        "- 'lookup': Resolve + query in one step (convenience)."
    ),
    inputSchema={
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["resolve", "query", "lookup"],
                "description": "The operation to perform.",
            },
            "library": {
                "type": "string",
                "description": "Library name (for 'resolve' and 'lookup'). E.g., 'fastapi', 'drizzle', 'react'.",
            },
            "library_id": {
                "type": "string",
                "description": "Context7 library ID (for 'query'). E.g., '/tiangolo/fastapi'. Get from 'resolve' action.",
            },
            "query": {
                "type": "string",
                "description": "What you want to know (for 'query' and 'lookup'). E.g., 'how to set up JWT authentication'.",
            },
        },
    },
)


async def handle_context7(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Dispatch Context7 tool calls."""
    action = arguments.get("action")
    if not action:
        return [types.TextContent(type="text", text="Error: 'action' is required.")]

    client = Context7Client()

    try:
        if action == "resolve":
            library = arguments.get("library")
            if not library:
                return [types.TextContent(type="text", text="Error: 'library' is required for resolve.")]
            results = await client.resolve_library(
                library_name=library,
                query=arguments.get("query", ""),
            )
            return [types.TextContent(type="text", text=json.dumps({"libraries": results}, indent=2))]

        elif action == "query":
            library_id = arguments.get("library_id")
            query = arguments.get("query")
            if not library_id or not query:
                return [types.TextContent(type="text", text="Error: 'library_id' and 'query' are required.")]
            result = await client.query_docs(library_id=library_id, query=query)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif action == "lookup":
            library = arguments.get("library")
            query = arguments.get("query")
            if not library or not query:
                return [types.TextContent(type="text", text="Error: 'library' and 'query' are required for lookup.")]

            # Step 1: Resolve
            libraries = await client.resolve_library(library_name=library, query=query)
            if not libraries or (isinstance(libraries[0], dict) and libraries[0].get("status")):
                return [types.TextContent(type="text", text=json.dumps({
                    "error": "Could not resolve library",
                    "resolve_result": libraries,
                }, indent=2))]

            # Pick best match (first result)
            best = libraries[0]
            library_id = best.get("id", "")
            if not library_id:
                return [types.TextContent(type="text", text=json.dumps({
                    "error": "No library ID found",
                    "candidates": libraries[:3],
                }, indent=2))]

            # Step 2: Query docs
            docs = await client.query_docs(library_id=library_id, query=query)
            return [types.TextContent(type="text", text=json.dumps({
                "library": best,
                "docs": docs,
            }, indent=2))]

        else:
            return [types.TextContent(type="text", text=f"Error: Unknown action '{action}'.")]

    except Exception as e:
        logger.error(f"Context7 tool error: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
