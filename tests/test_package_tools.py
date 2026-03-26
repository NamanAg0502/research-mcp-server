"""Tests for the packages tool (npm, PyPI, crates.io)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.package_tools import handle_packages, packages_tool

MODULE = "research_mcp_server.clients.package_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert packages_tool.name == "packages"
        schema = packages_tool.inputSchema
        actions = schema["properties"]["action"]["enum"]
        assert "stats" in actions
        assert "compare" in actions
        assert "search" in actions


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_requires_name(self):
        result = await handle_packages({"action": "stats"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_stats_returns_package_info(self):
        with patch(f"{MODULE}.PackageClient.get_package", new_callable=AsyncMock, return_value={
            "name": "fastapi", "registry": "pypi", "version": "0.100.0",
            "description": "FastAPI framework"
        }):
            result = await handle_packages({"action": "stats", "name": "fastapi", "registry": "pypi"})
            data = _parse(result)
            assert data["name"] == "fastapi"
            assert data["registry"] == "pypi"


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_requires_packages(self):
        result = await handle_packages({"action": "compare", "packages": [{"name": "a", "registry": "npm"}]})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_compare_returns_results(self):
        with patch(f"{MODULE}.PackageClient.get_package", new_callable=AsyncMock, side_effect=[
            {"name": "express", "registry": "npm", "version": "4.18.0"},
            {"name": "fastify", "registry": "npm", "version": "4.25.0"},
        ]):
            result = await handle_packages({
                "action": "compare",
                "packages": [
                    {"name": "express", "registry": "npm"},
                    {"name": "fastify", "registry": "npm"},
                ]
            })
            data = _parse(result)
            assert len(data["comparison"]) == 2


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await handle_packages({"action": "search"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_search_npm(self):
        with patch(f"{MODULE}.PackageClient.search_npm", new_callable=AsyncMock, return_value=[
            {"name": "express", "registry": "npm", "description": "Web framework", "version": "4.18.0"}
        ]):
            result = await handle_packages({"action": "search", "query": "web framework", "search_registry": "npm"})
            data = _parse(result)
            assert data["registry"] == "npm"
            assert data["total"] == 1


class TestErrors:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await handle_packages({"action": "invalid"})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_action(self):
        result = await handle_packages({})
        assert "error" in result[0].text.lower()
