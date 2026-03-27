"""LLM-based sentiment analysis.

Uses Claude Code CLI (`claude -p`) if available — no API key needed, uses existing
Claude Code OAuth auth. Falls back to ANTHROPIC_API_KEY direct API if CLI not found.
"""

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("research-mcp-server")

HAIKU_MODEL = "claude-haiku-4-6"


class SentimentResult(BaseModel):
    overall_sentiment: Literal["positive", "negative", "mixed", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_praise: list[str]
    key_concerns: list[str]
    adoption_signal: Literal["growing", "stable", "declining", "emerging"]
    summary: str


def _build_prompt(topic: str, discussions: list[dict[str, Any]]) -> str:
    discussion_text = ""
    for d in discussions[:15]:
        title = d.get("title", "")
        body = d.get("selftext", d.get("body", d.get("story_text", "")))
        score = d.get("score", d.get("points", 0))
        source = d.get("source", "unknown")
        if body and len(body) > 200:
            body = body[:200] + "..."
        discussion_text += f"[{source}, score:{score}] {title}"
        if body:
            discussion_text += f"\n  {body}"
        discussion_text += "\n\n"

    return f"""Analyze the community sentiment about "{topic}" based on these developer discussions:

{discussion_text}

Respond in JSON format only (no markdown, no explanation):
{{
  "overall_sentiment": "positive" | "negative" | "mixed" | "neutral",
  "confidence": 0.0-1.0,
  "key_praise": ["list of things developers praise"],
  "key_concerns": ["list of concerns or criticisms"],
  "adoption_signal": "growing" | "stable" | "declining" | "emerging",
  "summary": "2-3 sentence summary of community sentiment"
}}

Be specific and evidence-based. Only include points actually mentioned in the discussions."""


def _parse_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


class SentimentAnalyzer:
    """LLM sentiment analysis — prefers Claude Code CLI, falls back to API key."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        self._cli = shutil.which("claude")
        self._available = bool(self._cli) or bool(self._api_key)

        if self._cli:
            logger.info("Sentiment analyzer using Claude Code CLI")
        elif self._api_key:
            logger.info("Sentiment analyzer using Anthropic API key")
        else:
            logger.info("Sentiment analyzer unavailable (no claude CLI or ANTHROPIC_API_KEY)")

    @property
    def available(self) -> bool:
        return self._available

    async def analyze(self, topic: str, discussions: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze sentiment across discussion threads."""
        if not self._available:
            return {
                "available": False,
                "note": "Install Claude Code CLI (`npm i -g @anthropic-ai/claude-code`) or set ANTHROPIC_API_KEY",
            }

        prompt = _build_prompt(topic, discussions)

        if self._cli:
            return await self._analyze_via_cli(prompt)
        return await self._analyze_via_api(prompt)

    async def _analyze_via_cli(self, prompt: str) -> dict[str, Any]:
        """Run sentiment analysis via `claude -p` subprocess."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli, "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            text = stdout.decode().strip()
            analysis = _parse_response(text)
            analysis["model"] = "claude-code-cli"
            analysis["available"] = True
            return analysis
        except asyncio.TimeoutError:
            logger.error("Claude Code CLI timed out")
            return {"error": "CLI timeout", "available": True}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse CLI response as JSON: {text[:200]}")
            return {"overall_sentiment": "unknown", "raw_response": text[:500], "available": True, "parse_error": True}
        except Exception as e:
            logger.error(f"CLI sentiment error: {e}")
            return {"error": str(e), "available": True}

    async def _analyze_via_api(self, prompt: str) -> dict[str, Any]:
        """Run sentiment analysis via Anthropic SDK + Instructor for structured output."""
        try:
            import instructor
            from anthropic import Anthropic

            client = instructor.from_anthropic(Anthropic(api_key=self._api_key))
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=500,
                    response_model=SentimentResult,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            analysis = result.model_dump()
            analysis["model"] = HAIKU_MODEL
            analysis["available"] = True
            return analysis
        except Exception as e:
            logger.error(f"API sentiment error: {e}")
            return {"error": str(e), "available": True}
