"""Tests for the Context7 docs tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.context7_tools import handle_context7, context7_tool

MODULE = "research_mcp_server.clients.context7_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert context7_tool.name == "docs"
        schema = context7_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "resolve" in actions
        assert "query" in actions
        assert "lookup" in actions


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_requires_library(self):
        result = await handle_context7({"action": "resolve"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_resolve_returns_libraries(self):
        with patch(f"{MODULE}.Context7Client.resolve_library", new_callable=AsyncMock, return_value=[
            {"id": "/tiangolo/fastapi", "name": "FastAPI", "description": "Modern Python web framework",
             "total_snippets": 500, "trust_score": "High", "benchmark_score": 85}
        ]):
            result = await handle_context7({"action": "resolve", "library": "fastapi"})
            data = _parse(result)
            assert len(data["libraries"]) == 1
            assert data["libraries"][0]["id"] == "/tiangolo/fastapi"


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_requires_both_params(self):
        result = await handle_context7({"action": "query", "library_id": "/a/b"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_query_returns_docs(self):
        with patch(f"{MODULE}.Context7Client.query_docs", new_callable=AsyncMock, return_value={
            "library_id": "/tiangolo/fastapi", "query": "JWT auth",
            "total_snippets": 3, "snippets": [{"title": "Security", "content": "Use OAuth2..."}]
        }):
            result = await handle_context7({"action": "query", "library_id": "/tiangolo/fastapi", "query": "JWT auth"})
            data = _parse(result)
            assert data["total_snippets"] == 3


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_both_params(self):
        result = await handle_context7({"action": "lookup", "library": "react"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_lookup_resolves_and_queries(self):
        with patch(f"{MODULE}.Context7Client.resolve_library", new_callable=AsyncMock, return_value=[
            {"id": "/facebook/react", "name": "React", "total_snippets": 1000}
        ]), patch(f"{MODULE}.Context7Client.query_docs", new_callable=AsyncMock, return_value={
            "library_id": "/facebook/react", "query": "hooks",
            "total_snippets": 5, "snippets": [{"title": "useState", "content": "..."}]
        }):
            result = await handle_context7({"action": "lookup", "library": "react", "query": "hooks"})
            data = _parse(result)
            assert "library" in data
            assert "docs" in data
            assert data["docs"]["total_snippets"] == 5


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_context7({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_context7({})
        assert "error" in result[0].text.lower()
