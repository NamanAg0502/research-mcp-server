"""Tests for the Hacker News tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.hn_tools import handle_hn, hn_tool

MODULE = "research_mcp_server.clients.hn_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert hn_tool.name == "hn"
        schema = hn_tool.inputSchema
        assert "action" in schema["properties"]
        actions = schema["properties"]["action"]["enum"]
        assert "search" in actions
        assert "trending" in actions
        assert "discussion" in actions
        assert schema["required"] == ["action"]


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await handle_hn({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_stories = [
            {"objectID": "123", "title": "Test Story", "url": "https://example.com",
             "author": "user1", "points": 100, "num_comments": 50, "created_at": "2024-01-01"}
        ]
        with patch(f"{MODULE}.HNClient.search", new_callable=AsyncMock, return_value=[
            {"id": "123", "source": "hackernews", "title": "Test Story", "url": "https://example.com",
             "author": "user1", "points": 100, "num_comments": 50, "created_at": "2024-01-01",
             "hn_url": "https://news.ycombinator.com/item?id=123", "story_text": ""}
        ]):
            result = await handle_hn({"action": "search", "query": "test", "max_results": 5})
            data = _parse(result)
            assert data["total"] == 1
            assert data["stories"][0]["title"] == "Test Story"

    @pytest.mark.asyncio
    async def test_trending_returns_results(self):
        with patch(f"{MODULE}.HNClient.trending", new_callable=AsyncMock, return_value=[
            {"id": "456", "source": "hackernews", "title": "Trending Story", "url": "https://example.com",
             "author": "user2", "points": 200, "num_comments": 80, "created_at": "2024-01-01",
             "hn_url": "https://news.ycombinator.com/item?id=456"}
        ]):
            result = await handle_hn({"action": "trending", "max_results": 5})
            data = _parse(result)
            assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_discussion_requires_story_id(self):
        result = await handle_hn({"action": "discussion"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_hn({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_hn({})
        assert "error" in result[0].text.lower()
