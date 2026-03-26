"""Tests for the Reddit tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.reddit_tools import handle_reddit, reddit_tool

MODULE = "research_mcp_server.clients.reddit_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert reddit_tool.name == "reddit"
        schema = reddit_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "search" in actions
        assert "trending" in actions
        assert "discussion" in actions


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await handle_reddit({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_returns_posts(self):
        with patch(f"{MODULE}.RedditClient.search", new_callable=AsyncMock, return_value=[
            {"id": "abc123", "source": "reddit", "subreddit": "MachineLearning",
             "title": "Best LLM framework?", "score": 150, "num_comments": 80,
             "permalink": "https://reddit.com/r/MachineLearning/abc123"}
        ]):
            result = await handle_reddit({"action": "search", "query": "LLM framework"})
            data = _parse(result)
            assert data["total"] == 1
            assert data["posts"][0]["subreddit"] == "MachineLearning"


class TestTrending:
    @pytest.mark.asyncio
    async def test_trending_returns_posts(self):
        with patch(f"{MODULE}.RedditClient.trending", new_callable=AsyncMock, return_value=[
            {"id": "def456", "source": "reddit", "subreddit": "programming",
             "title": "Hot Post", "score": 300, "num_comments": 120}
        ]):
            result = await handle_reddit({"action": "trending"})
            data = _parse(result)
            assert data["total"] == 1


class TestDiscussion:
    @pytest.mark.asyncio
    async def test_discussion_requires_post_id(self):
        result = await handle_reddit({"action": "discussion"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_discussion_returns_comments(self):
        with patch(f"{MODULE}.RedditClient.get_discussion", new_callable=AsyncMock, return_value={
            "post": {"id": "abc", "title": "Test Post", "score": 100},
            "top_comments": [
                {"id": "c1", "author": "user1", "body": "Great post!", "score": 50}
            ]
        }):
            result = await handle_reddit({"action": "discussion", "post_id": "abc"})
            data = _parse(result)
            assert len(data["top_comments"]) == 1


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_reddit({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_reddit({})
        assert "error" in result[0].text.lower()
