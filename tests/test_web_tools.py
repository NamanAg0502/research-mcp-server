"""Tests for the web scraping tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from research_mcp_server.tools.web_tools import handle_web, web_tool

MODULE = "research_mcp_server.clients.web_client"


def _parse(result):
    assert len(result) == 1
    return json.loads(result[0].text)


class TestToolDefinition:
    def test_tool_exists(self):
        assert web_tool.name == "web"
        assert "url" in web_tool.inputSchema["required"]
        modes = web_tool.inputSchema["properties"]["extract"]["enum"]
        assert "article" in modes
        assert "links" in modes
        assert "metadata" in modes


class TestFetch:
    @pytest.mark.asyncio
    async def test_requires_url(self):
        result = await handle_web({})
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_article_mode(self):
        with patch(f"{MODULE}.WebClient.fetch", new_callable=AsyncMock, return_value={
            "url": "https://example.com", "status": 200, "title": "Example",
            "description": "An example page", "content": "Hello world", "content_length": 11
        }):
            result = await handle_web({"url": "https://example.com"})
            data = _parse(result)
            assert data["title"] == "Example"
            assert data["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_metadata_mode(self):
        with patch(f"{MODULE}.WebClient.fetch", new_callable=AsyncMock, return_value={
            "url": "https://example.com", "status": 200, "title": "Example",
            "description": "An example page"
        }):
            result = await handle_web({"url": "https://example.com", "extract": "metadata"})
            data = _parse(result)
            assert data["title"] == "Example"
            assert "content" not in data

    @pytest.mark.asyncio
    async def test_links_mode(self):
        with patch(f"{MODULE}.WebClient.fetch", new_callable=AsyncMock, return_value={
            "url": "https://example.com", "status": 200, "title": "Example",
            "description": "", "links": [{"text": "About", "url": "https://example.com/about"}]
        }):
            result = await handle_web({"url": "https://example.com", "extract": "links"})
            data = _parse(result)
            assert len(data["links"]) == 1

    @pytest.mark.asyncio
    async def test_prepends_https(self):
        with patch(f"{MODULE}.WebClient.fetch", new_callable=AsyncMock, return_value={
            "url": "https://example.com", "status": 200, "title": "Example",
            "description": "", "content": "test", "content_length": 4
        }) as mock_fetch:
            await handle_web({"url": "example.com"})
            mock_fetch.assert_awaited_once()
            call_kwargs = mock_fetch.call_args
            assert "https://example.com" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        with patch(f"{MODULE}.WebClient.fetch", new_callable=AsyncMock, side_effect=Exception("Connection refused")):
            result = await handle_web({"url": "https://bad.example.com"})
            assert "error" in result[0].text.lower()
