"""Query rewriting — generates 2-3 reformulations to improve retrieval recall.

Uses the same CLI-first / API-fallback pattern as sentiment_client.
Each reformulation targets a different aspect of the original query:
  - technical: precise terminology, abbreviations expanded
  - practitioner: how a developer would phrase it
  - conceptual: underlying concepts and problem framing

Merged via RRF at the search layer.
"""

import asyncio
import json
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger("research-mcp-server")

_REWRITE_PROMPT = """Generate 2 alternative search queries for the following research/developer query.
The alternatives should cover different phrasings and terminologies that might match relevant documents.

Original query: {query}

Return a JSON array of 2 strings (the alternative queries only, no explanation):
["alternative 1", "alternative 2"]

Rules:
- Each alternative must be ≤8 words
- Cover different angles: technical terms, practical phrasing, related concepts
- Do NOT repeat the original query
- Do NOT add markdown fences, just raw JSON"""


async def rewrite_query(query: str) -> list[str]:
    """Generate 2-3 query reformulations.

    Returns the original query + up to 2 alternatives. Always includes
    the original so callers can use the full list unconditionally.

    Falls back to [query] (original only) if LLM is unavailable or fails.
    """
    cli = shutil.which("claude")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None

    if not cli and not api_key:
        return [query]

    prompt = _REWRITE_PROMPT.format(query=query)

    try:
        if cli:
            alternatives = await _rewrite_via_cli(cli, prompt)
        else:
            alternatives = await _rewrite_via_api(api_key, prompt)

        # Deduplicate and combine with original
        seen = {query.lower()}
        result = [query]
        for alt in alternatives:
            if isinstance(alt, str) and alt.strip() and alt.lower() not in seen:
                seen.add(alt.lower())
                result.append(alt.strip())
        return result[:3]  # original + at most 2 alternatives

    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return [query]


async def _rewrite_via_cli(cli: str, prompt: str) -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        cli, "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    text = stdout.decode().strip()
    return _parse_alternatives(text)


async def _rewrite_via_api(api_key: str, prompt: str) -> list[str]:
    try:
        import instructor
        from anthropic import Anthropic
        from pydantic import BaseModel

        class QueryAlternatives(BaseModel):
            alternatives: list[str]

        client = instructor.from_anthropic(Anthropic(api_key=api_key))
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-haiku-4-6",
                max_tokens=200,
                response_model=QueryAlternatives,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        return result.alternatives
    except Exception:
        # Instructor not available — raw httpx fallback
        import httpx
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-haiku-4-6",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data.get("content", [{}])[0].get("text", "")
        return _parse_alternatives(text)


def _parse_alternatives(text: str) -> list[str]:
    """Parse JSON array of strings from LLM response."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [str(s) for s in parsed if s]
    return []
