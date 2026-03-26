"""LLM-based sentiment analysis using Claude Haiku.

Requires ANTHROPIC_API_KEY env var. Falls back gracefully when not available.
Cost: ~$0.001 per analysis call (Haiku is very cheap).
"""

import json
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger("research-mcp-server")

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


class SentimentAnalyzer:
    """LLM-based sentiment analysis using Claude Haiku."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        self._available = bool(self._api_key)
        if self._available:
            logger.info("Sentiment analyzer initialized with Anthropic API key")
        else:
            logger.info("Sentiment analyzer unavailable (no ANTHROPIC_API_KEY)")

    @property
    def available(self) -> bool:
        return self._available

    async def analyze(
        self,
        topic: str,
        discussions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze sentiment across discussion threads using Haiku.

        Args:
            topic: Technology/topic being analyzed.
            discussions: List of dicts with 'title', 'body'/'selftext', 'score', 'source'.

        Returns:
            Structured sentiment analysis.
        """
        if not self._available:
            return {"error": "ANTHROPIC_API_KEY not set", "available": False}

        # Build discussion summary for the prompt (stay under ~2K tokens)
        discussion_text = ""
        for d in discussions[:15]:  # Cap at 15 threads
            title = d.get("title", "")
            body = d.get("selftext", d.get("body", d.get("story_text", "")))
            score = d.get("score", d.get("points", 0))
            source = d.get("source", "unknown")
            # Truncate body
            if body and len(body) > 200:
                body = body[:200] + "..."
            discussion_text += f"[{source}, score:{score}] {title}"
            if body:
                discussion_text += f"\n  {body}"
            discussion_text += "\n\n"

        prompt = f"""Analyze the community sentiment about "{topic}" based on these developer discussions:

{discussion_text}

Respond in JSON format:
{{
  "overall_sentiment": "positive" | "negative" | "mixed" | "neutral",
  "confidence": 0.0-1.0,
  "key_praise": ["list of things developers praise"],
  "key_concerns": ["list of concerns or criticisms"],
  "adoption_signal": "growing" | "stable" | "declining" | "emerging",
  "summary": "2-3 sentence summary of community sentiment"
}}

Be specific and evidence-based. Only include points actually mentioned in the discussions."""

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": HAIKU_MODEL,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(ANTHROPIC_API, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # Extract text from response
            text = data.get("content", [{}])[0].get("text", "")

            # Parse JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            analysis = json.loads(text)
            analysis["model"] = HAIKU_MODEL
            analysis["available"] = True
            return analysis

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Haiku response as JSON: {text[:200]}")
            return {
                "overall_sentiment": "unknown",
                "raw_response": text[:500],
                "available": True,
                "parse_error": True,
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e), "available": True}
