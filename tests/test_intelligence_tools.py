"""Tests for composite CTO intelligence tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest
import mcp.types as types

from research_mcp_server.tools.intelligence_tools import (
    handle_tech_pulse, tech_pulse_tool,
    handle_evaluate, evaluate_tool,
    handle_sentiment, sentiment_tool,
    handle_deep_research, deep_research_tool,
)


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


def _mock_handler(data):
    """Create an AsyncMock that returns a tool-like response."""
    return AsyncMock(return_value=[types.TextContent(type="text", text=json.dumps(data))])


# Since intelligence_tools uses lazy imports inside handler functions,
# we patch at the source module where the handlers actually live.
HN = "research_mcp_server.tools.hn_tools.handle_hn"
GITHUB = "research_mcp_server.tools.github_tools.handle_github"
COMMUNITY = "research_mcp_server.tools.community_tools.handle_community"
HF = "research_mcp_server.tools.hf_papers.handle_hf_trending"
REDDIT = "research_mcp_server.tools.reddit_tools.handle_reddit"
PACKAGES = "research_mcp_server.tools.package_tools.handle_packages"
SEARCH = "research_mcp_server.tools.search.handle_search"


class TestToolDefinitions:
    def test_tech_pulse(self):
        assert tech_pulse_tool.name == "tech_pulse"

    def test_evaluate(self):
        assert evaluate_tool.name == "evaluate"
        assert "items" in evaluate_tool.inputSchema["required"]

    def test_sentiment(self):
        assert sentiment_tool.name == "sentiment"
        assert "topic" in sentiment_tool.inputSchema["required"]

    def test_deep_research(self):
        assert deep_research_tool.name == "deep_research"
        assert "topic" in deep_research_tool.inputSchema["required"]


class TestTechPulse:
    @pytest.mark.asyncio
    async def test_returns_multi_source_results(self):
        mock = _mock_handler({"total": 1, "stories": [{"title": "Test"}]})
        with patch(HN, mock), patch(GITHUB, mock), patch(COMMUNITY, mock), patch(HF, mock):
            result = await handle_tech_pulse({"max_per_source": 3})
            data = _parse(result)
            assert data["sources_queried"] >= 1
            assert data["topic"] == "general tech"

    @pytest.mark.asyncio
    async def test_with_topic_filter(self):
        mock = _mock_handler({"total": 1, "stories": [{"title": "Rust"}]})
        with patch(HN, mock), patch(GITHUB, mock), patch(COMMUNITY, mock), patch(HF, mock):
            result = await handle_tech_pulse({"topic": "Rust", "max_per_source": 3})
            data = _parse(result)
            assert data["topic"] == "Rust"


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_returns_comparison(self):
        mock = _mock_handler({"total": 1, "repos": [{"name": "test"}]})
        with patch(GITHUB, mock), patch(REDDIT, mock), patch(HN, mock), patch(PACKAGES, mock):
            result = await handle_evaluate({"items": ["Drizzle", "Prisma"]})
            data = _parse(result)
            assert data["items"] == ["Drizzle", "Prisma"]


class TestSentiment:
    @pytest.mark.asyncio
    async def test_returns_sentiment_data(self):
        mock_reddit = _mock_handler({"total": 2, "posts": [
            {"title": "Test", "score": 100, "num_comments": 50},
            {"title": "Test2", "score": 50, "num_comments": 20},
        ]})
        mock_hn = _mock_handler({"total": 1, "stories": [
            {"title": "HN Test", "points": 200, "num_comments": 80},
        ]})
        with patch(REDDIT, mock_reddit), patch(HN, mock_hn):
            result = await handle_sentiment({"topic": "Bun"})
            data = _parse(result)
            assert data["topic"] == "Bun"
            assert "summary" in data
            assert data["summary"]["total_threads_found"] == 3


class TestDeepResearch:
    @pytest.mark.asyncio
    async def test_returns_multi_source_research(self):
        mock = _mock_handler({"total": 1, "results": [{"title": "Test"}]})
        with patch(SEARCH, mock), patch(GITHUB, mock), patch(HN, mock), \
             patch(REDDIT, mock), patch(COMMUNITY, mock), patch(PACKAGES, mock):
            result = await handle_deep_research({"topic": "WebTransport"})
            data = _parse(result)
            assert data["topic"] == "WebTransport"
            assert data["sources_queried"] >= 5
            assert data["sources_succeeded"] >= 1
