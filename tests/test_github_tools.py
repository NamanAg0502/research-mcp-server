"""Tests for the GitHub tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.github_tools import handle_github, github_tool

MODULE = "research_mcp_server.clients.github_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert github_tool.name == "github"
        schema = github_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "search" in actions
        assert "repo" in actions
        assert "compare" in actions
        assert "trending" in actions
        assert "releases" in actions


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await handle_github({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_returns_repos(self):
        with patch(f"{MODULE}.GitHubClient.search_repos", new_callable=AsyncMock, return_value=[
            {"name": "langchain-ai/langchain", "stars": 80000, "description": "LLM framework",
             "url": "https://github.com/langchain-ai/langchain", "language": "Python", "source": "github"}
        ]):
            result = await handle_github({"action": "search", "query": "LLM framework"})
            data = _parse(result)
            assert data["total"] == 1
            assert data["repos"][0]["stars"] == 80000


class TestRepo:
    @pytest.mark.asyncio
    async def test_repo_requires_owner_repo(self):
        result = await handle_github({"action": "repo"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_repo_returns_info(self):
        with patch(f"{MODULE}.GitHubClient.get_repo", new_callable=AsyncMock, return_value={
            "name": "vercel/next.js", "stars": 120000, "forks": 25000, "source": "github"
        }):
            result = await handle_github({"action": "repo", "owner_repo": "vercel/next.js"})
            data = _parse(result)
            assert data["name"] == "vercel/next.js"


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_requires_repos(self):
        result = await handle_github({"action": "compare", "repos": ["a/b"]})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_compare_returns_results(self):
        with patch(f"{MODULE}.GitHubClient.compare_repos", new_callable=AsyncMock, return_value=[
            {"name": "a/b", "stars": 100, "source": "github"},
            {"name": "c/d", "stars": 200, "source": "github"},
        ]):
            result = await handle_github({"action": "compare", "repos": ["a/b", "c/d"]})
            data = _parse(result)
            assert len(data["comparison"]) == 2


class TestTrending:
    @pytest.mark.asyncio
    async def test_trending_returns_repos(self):
        with patch(f"{MODULE}.GitHubClient.trending", new_callable=AsyncMock, return_value=[
            {"name": "trending/repo", "stars": 500, "source": "github"}
        ]):
            result = await handle_github({"action": "trending"})
            data = _parse(result)
            assert data["total"] == 1


class TestReleases:
    @pytest.mark.asyncio
    async def test_releases_requires_owner_repo(self):
        result = await handle_github({"action": "releases"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_releases_returns_results(self):
        with patch(f"{MODULE}.GitHubClient.get_releases", new_callable=AsyncMock, return_value=[
            {"repo": "a/b", "tag": "v1.0.0", "name": "Release 1.0", "published_at": "2024-01-01"}
        ]):
            result = await handle_github({"action": "releases", "owner_repo": "a/b"})
            data = _parse(result)
            assert data["repo"] == "a/b"
            assert len(data["releases"]) == 1


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_github({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_github({})
        assert "error" in result[0].text.lower()
