"""Tests for the Stack Overflow tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.so_tools import handle_so, so_tool

MODULE = "research_mcp_server.clients.so_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert so_tool.name == "stackoverflow"
        schema = so_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "search" in actions
        assert "tags" in actions
        assert "trending" in actions


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await handle_so({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_returns_questions(self):
        with patch(f"{MODULE}.SOClient.search", new_callable=AsyncMock, return_value=[
            {"id": 12345, "source": "stackoverflow", "title": "How to use FastAPI?",
             "score": 50, "answer_count": 3, "tags": ["python", "fastapi"]}
        ]):
            result = await handle_so({"action": "search", "query": "FastAPI authentication"})
            data = _parse(result)
            assert data["total"] == 1
            assert data["questions"][0]["score"] == 50


class TestTags:
    @pytest.mark.asyncio
    async def test_tags_requires_tags(self):
        result = await handle_so({"action": "tags"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_tags_returns_info(self):
        with patch(f"{MODULE}.SOClient.tag_info", new_callable=AsyncMock, return_value=[
            {"name": "python", "count": 2100000},
            {"name": "fastapi", "count": 15000},
        ]):
            result = await handle_so({"action": "tags", "tags": ["python", "fastapi"]})
            data = _parse(result)
            assert len(data["tags"]) == 2
            assert data["tags"][0]["name"] == "python"


class TestTrending:
    @pytest.mark.asyncio
    async def test_trending_returns_questions(self):
        with patch(f"{MODULE}.SOClient.trending_questions", new_callable=AsyncMock, return_value=[
            {"id": 99, "source": "stackoverflow", "title": "New Python feature?", "score": 20}
        ]):
            result = await handle_so({"action": "trending"})
            data = _parse(result)
            assert data["total"] == 1


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_so({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_so({})
        assert "error" in result[0].text.lower()
