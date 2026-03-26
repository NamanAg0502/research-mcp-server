"""Tests for the community tool (Dev.to + Lobsters)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.community_tools import handle_community, community_tool

DEVTO_MODULE = "research_mcp_server.clients.devto_client"
LOBSTERS_MODULE = "research_mcp_server.clients.lobsters_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert community_tool.name == "community"
        schema = community_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "search" in actions
        assert "trending" in actions
        assert "by_tag" in actions


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query_or_tag(self):
        result = await handle_community({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_with_query(self):
        with patch(f"{DEVTO_MODULE}.DevtoClient.search", new_callable=AsyncMock, return_value=[
            {"id": "1", "source": "devto", "title": "Test Article", "url": "https://dev.to/test",
             "author": "dev1", "tags": ["python"], "positive_reactions_count": 10, "comments_count": 5}
        ]):
            result = await handle_community({"action": "search", "query": "python"})
            data = _parse(result)
            assert data["source"] == "devto"
            assert data["total"] == 1


class TestTrending:
    @pytest.mark.asyncio
    async def test_trending_devto(self):
        with patch(f"{DEVTO_MODULE}.DevtoClient.trending", new_callable=AsyncMock, return_value=[
            {"id": "2", "source": "devto", "title": "Trending", "url": "https://dev.to/t"}
        ]):
            result = await handle_community({"action": "trending", "source_filter": "devto"})
            data = _parse(result)
            assert "devto" in data

    @pytest.mark.asyncio
    async def test_trending_lobsters(self):
        with patch(f"{LOBSTERS_MODULE}.LobstersClient.hottest", new_callable=AsyncMock, return_value=[
            {"id": "abc", "source": "lobsters", "title": "Hot Story", "score": 20}
        ]):
            result = await handle_community({"action": "trending", "source_filter": "lobsters"})
            data = _parse(result)
            assert "lobsters" in data


class TestByTag:
    @pytest.mark.asyncio
    async def test_by_tag_requires_tag(self):
        result = await handle_community({"action": "by_tag"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_by_tag_returns_results(self):
        with patch(f"{LOBSTERS_MODULE}.LobstersClient.by_tag", new_callable=AsyncMock, return_value=[
            {"id": "xyz", "source": "lobsters", "title": "Rust Story", "tags": ["rust"], "score": 15}
        ]):
            result = await handle_community({"action": "by_tag", "tag": "rust"})
            data = _parse(result)
            assert data["tag"] == "rust"
            assert data["total"] == 1


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_community({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_community({})
        assert "error" in result[0].text.lower()
