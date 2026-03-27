"""Composite CTO intelligence tools.

These meta-tools orchestrate queries across multiple data sources to answer
high-level questions that no single source can answer alone.

Follows the orchestrator-worker pattern from Anthropic's multi-agent research system:
each tool spawns parallel sub-queries internally.
"""

import json
import logging
import asyncio
from typing import Any, Dict, List

import mcp.types as types
from pydantic import BaseModel


class EvidenceAssessment(BaseModel):
    """Structured assessment of whether retrieved evidence covers the query."""
    required_facts: List[str]   # key facets the query asks about
    confirmed_facts: List[str]  # facets found in retrieved results
    gaps: List[str]             # facets not yet covered
    sufficient: bool            # True when gaps is empty or coverage ≥ threshold
    sub_queries: List[str]      # targeted queries to fill the gaps

logger = logging.getLogger("research-mcp-server")


# =============================================================================
# Tool 1: tech_pulse — "What's trending this week?"
# =============================================================================

tech_pulse_tool = types.Tool(
    name="tech_pulse",
    description="What's trending in tech? Aggregates HN, GitHub, Dev.to, and HuggingFace in parallel.",
    inputSchema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Optional topic filter. E.g., 'AI', 'Rust', 'web frameworks'. Omit for general tech pulse.",
            },
            "max_per_source": {
                "type": "integer",
                "minimum": 3,
                "maximum": 15,
                "description": "Max items per source. Default: 5.",
            },
        },
    },
)


async def handle_tech_pulse(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Aggregate trending content from multiple sources."""
    # Import handlers lazily to avoid circular imports
    from .hn_tools import handle_hn
    from .community_tools import handle_community
    from .github_tools import handle_github
    from .hf_papers import handle_hf_trending

    topic = arguments.get("topic")
    short_query = _simplify_query(topic) if topic else None
    max_per = arguments.get("max_per_source", 5)
    results: Dict[str, Any] = {}
    errors: List[str] = []

    # Build parallel tasks
    async def fetch_source(name: str, coro):
        try:
            r = await coro
            data = json.loads(r[0].text)
            results[name] = data
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    tasks = []

    # HN: search if topic, else trending
    if short_query:
        tasks.append(fetch_source("hackernews", handle_hn({
            "action": "search", "query": short_query, "max_results": max_per, "time_range": "week",
        })))
    else:
        tasks.append(fetch_source("hackernews", handle_hn({
            "action": "trending", "max_results": max_per,
        })))

    # GitHub: trending, optionally filtered
    gh_args: Dict[str, Any] = {"action": "trending", "max_results": max_per, "since": "weekly"}
    if short_query:
        gh_args = {"action": "search", "query": short_query, "max_results": max_per, "sort": "stars"}
    tasks.append(fetch_source("github", handle_github(gh_args)))

    # Dev.to + Lobsters
    if short_query:
        tasks.append(fetch_source("community", handle_community({
            "action": "search", "query": short_query, "max_results": max_per,
        })))
    else:
        tasks.append(fetch_source("community", handle_community({
            "action": "trending", "source_filter": "both", "max_results": max_per,
        })))

    # HuggingFace trending (always relevant for AI/ML)
    tasks.append(fetch_source("huggingface", handle_hf_trending({
        "limit": max_per,
    })))

    await asyncio.gather(*tasks)

    output = {
        "topic": topic or "general tech",
        "sources_queried": len(results),
        **results,
    }
    if errors:
        output["errors"] = errors

    return [types.TextContent(type="text", text=json.dumps(output, indent=2))]


# =============================================================================
# Tool 2: evaluate — "Should we use X or Y?"
# =============================================================================

evaluate_tool = types.Tool(
    name="evaluate",
    description="Compare technologies with evidence — GitHub stats, Reddit discussions, HN threads, package data.",
    inputSchema={
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
                "description": "Technologies to compare. E.g., ['Drizzle', 'Prisma', 'Kysely'].",
            },
            "github_repos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional GitHub repos in owner/repo format. Auto-detected if omitted.",
            },
            "package_names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "registry": {"type": "string", "enum": ["npm", "pypi", "crates"]},
                    },
                    "required": ["name", "registry"],
                },
                "description": "Optional package names + registries. Auto-searched if omitted.",
            },
        },
    },
)


async def handle_evaluate(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Compare technologies using multiple data sources including live docs."""
    from .github_tools import handle_github
    from .reddit_tools import handle_reddit
    from .hn_tools import handle_hn
    from .package_tools import handle_packages
    from .context7_tools import handle_context7
    from .so_tools import handle_so

    items = arguments["items"]
    results: Dict[str, Any] = {"items": items}
    errors: List[str] = []

    async def safe_call(name, coro):
        try:
            r = await asyncio.wait_for(coro, timeout=30.0)
            return json.loads(r[0].text)
        except asyncio.TimeoutError:
            errors.append(f"{name}: timed out after 30s")
            return None
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            return None

    vs_query = " vs ".join(items[:3])

    # 1. GitHub comparison (if repos provided)
    github_repos = arguments.get("github_repos")
    if github_repos and len(github_repos) >= 2:
        gh_data = await safe_call("github", handle_github({
            "action": "compare", "repos": github_repos,
        }))
        if gh_data:
            results["github"] = gh_data

    # 2. Search GitHub for each item
    if not github_repos:
        gh_results = []
        for item in items:
            data = await safe_call(f"github:{item}", handle_github({
                "action": "search", "query": item, "max_results": 3,
            }))
            if data:
                gh_results.append({"item": item, "top_repos": data.get("repos", [])[:3]})
        if gh_results:
            results["github_search"] = gh_results

    # 3. Package stats (if provided)
    pkg_names = arguments.get("package_names")
    if pkg_names:
        pkg_data = await safe_call("packages", handle_packages({
            "action": "compare", "packages": pkg_names,
        }))
        if pkg_data:
            results["packages"] = pkg_data

    # Run Reddit, HN, SO, and docs in parallel
    parallel_tasks = {
        "reddit": safe_call("reddit", handle_reddit({
            "action": "search", "query": vs_query, "max_results": 5, "time_filter": "year",
        })),
        "hackernews": safe_call("hackernews", handle_hn({
            "action": "search", "query": vs_query, "max_results": 5, "time_range": "year",
        })),
        "stackoverflow_adoption": safe_call("stackoverflow_tags", handle_so({
            "action": "tags", "tags": items[:5],
        })),
        "stackoverflow_discussions": safe_call("stackoverflow_search", handle_so({
            "action": "search", "query": vs_query, "max_results": 5, "sort": "votes",
        })),
    }
    parallel_names = list(parallel_tasks.keys())
    parallel_results = await asyncio.gather(*parallel_tasks.values())
    for name, data in zip(parallel_names, parallel_results):
        if data is not None:
            results[name] = data

    # Context7 live docs — sequential per item (API constraint)
    docs_results = []
    for item in items:
        doc_data = await safe_call(f"docs:{item}", handle_context7({
            "action": "lookup", "library": item, "query": "getting started features overview",
        }))
        if doc_data and doc_data.get("content"):
            docs_results.append({"item": item, "docs": doc_data})
    if docs_results:
        results["live_documentation"] = docs_results

    if errors:
        results["errors"] = errors

    return [types.TextContent(type="text", text=json.dumps(results, indent=2))]


# =============================================================================
# Tool 3: sentiment — "What does the community think about X?"
# =============================================================================

sentiment_tool = types.Tool(
    name="sentiment",
    description="What does the dev community think about a technology? Gathers and analyzes Reddit + HN discussions.",
    inputSchema={
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {
                "type": "string",
                "description": "Technology or topic to analyze sentiment for.",
            },
            "subreddits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific subreddits to search. Default: broad dev subreddits.",
            },
            "time_range": {
                "type": "string",
                "enum": ["week", "month", "year"],
                "description": "How far back to look. Default: month.",
            },
            "max_threads": {
                "type": "integer",
                "minimum": 3,
                "maximum": 15,
                "description": "Max discussion threads to analyze. Default: 8.",
            },
        },
    },
)


async def handle_sentiment(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Gather community discussions and summarize sentiment."""
    from .reddit_tools import handle_reddit
    from .hn_tools import handle_hn

    topic = arguments["topic"]
    time_range = arguments.get("time_range", "month")
    max_threads = arguments.get("max_threads", 8)
    errors: List[str] = []

    discussions: Dict[str, Any] = {"topic": topic}

    async def fetch_source(name: str, coro):
        try:
            r = await asyncio.wait_for(coro, timeout=30.0)
            return name, json.loads(r[0].text)
        except asyncio.TimeoutError:
            errors.append(f"{name}: timed out after 30s")
            return name, None
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            return name, None

    # Fetch Reddit and HN in parallel
    results = await asyncio.gather(
        fetch_source("reddit", handle_reddit({
            "action": "search",
            "query": topic,
            "sort": "relevance",
            "time_filter": time_range,
            "max_results": max_threads,
        })),
        fetch_source("hackernews", handle_hn({
            "action": "search",
            "query": topic,
            "sort": "relevance",
            "time_range": time_range if time_range != "month" else "year",
            "max_results": max_threads,
        })),
    )

    for name, data in results:
        if data is None:
            continue
        if name == "reddit":
            discussions["reddit"] = {
                "total_posts": data.get("total", 0),
                "posts": data.get("posts", []),
            }
        elif name == "hackernews":
            discussions["hackernews"] = {
                "total_stories": data.get("total", 0),
                "stories": data.get("stories", []),
            }

    # Build summary metrics
    reddit_posts = discussions.get("reddit", {}).get("posts", [])
    hn_stories = discussions.get("hackernews", {}).get("stories", [])

    total_engagement = (
        sum(p.get("score", 0) for p in reddit_posts) +
        sum(s.get("points", 0) for s in hn_stories)
    )
    total_comments = (
        sum(p.get("num_comments", 0) for p in reddit_posts) +
        sum(s.get("num_comments", 0) for s in hn_stories)
    )

    discussions["summary"] = {
        "total_threads_found": len(reddit_posts) + len(hn_stories),
        "total_engagement_score": total_engagement,
        "total_comments": total_comments,
        "avg_engagement": round(total_engagement / max(len(reddit_posts) + len(hn_stories), 1), 1),
        "note": (
            "Review the posts and comments above for qualitative sentiment. "
            "High engagement + high upvote ratios = positive sentiment. "
            "Use the 'discussion' action on reddit/hn tools to read individual thread comments."
        ),
    }

    # LLM-based sentiment analysis (if available)
    from ..clients.sentiment_client import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    if analyzer.available:
        all_discussions = reddit_posts + [
            {"title": s.get("title", ""), "body": "", "score": s.get("points", 0), "source": "hackernews"}
            for s in hn_stories
        ]
        try:
            llm_result = await analyzer.analyze(topic, all_discussions)
            discussions["llm_analysis"] = llm_result
        except Exception as e:
            discussions["llm_analysis"] = {"error": str(e), "available": True}
    else:
        discussions["llm_analysis"] = {
            "available": False,
            "note": "Set ANTHROPIC_API_KEY for LLM-based sentiment analysis",
        }

    if errors:
        discussions["errors"] = errors

    return [types.TextContent(type="text", text=json.dumps(discussions, indent=2))]


# =============================================================================
# Tool 4: deep_research — "Everything about X"
# =============================================================================

deep_research_tool = types.Tool(
    name="deep_research",
    description="Comprehensive multi-source research — arXiv + GitHub + HN + Reddit + Dev.to + packages. Includes evidence sufficiency gate for gap-fill.",
    inputSchema={
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {
                "type": "string",
                "description": "Research topic. Be specific for better results.",
            },
            "max_per_source": {
                "type": "integer",
                "minimum": 3,
                "maximum": 10,
                "description": "Max results per source. Default: 5.",
            },
            "include_packages": {
                "type": "boolean",
                "description": "Search package registries for related packages. Default: true.",
            },
        },
    },
)


async def _assess_evidence(
    topic: str,
    results: Dict[str, Any],
) -> EvidenceAssessment | None:
    """Run a lightweight LLM evidence sufficiency check.

    Returns None if LLM is unavailable (graceful degradation).
    """
    import shutil, os
    cli = shutil.which("claude")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    if not cli and not api_key:
        return None

    # Summarize what was retrieved
    sources_summary = []
    for source, data in results.items():
        if source in ("topic", "search_query", "sources_queried", "sources_succeeded", "errors"):
            continue
        if isinstance(data, dict):
            count = data.get("total", data.get("count", len(data.get("papers", data.get("repos", data.get("stories", data.get("posts", [])))))))
            sources_summary.append(f"{source}: {count} results")
        elif isinstance(data, list):
            sources_summary.append(f"{source}: {len(data)} results")

    prompt = f"""You are assessing whether research results sufficiently cover a query.

Query: "{topic}"
Retrieved so far: {', '.join(sources_summary) if sources_summary else 'nothing yet'}

Return a JSON object:
{{
  "required_facts": ["list of 2-4 key aspects the query asks about"],
  "confirmed_facts": ["which of those are likely covered by the results above"],
  "gaps": ["which aspects are missing or underrepresented"],
  "sufficient": true/false,
  "sub_queries": ["1-2 targeted queries to fill the gaps, if any"]
}}

Rules: Be concise. If results look adequate, set sufficient=true and sub_queries=[].
Return raw JSON only, no markdown."""

    try:
        if cli:
            import asyncio
            proc = await asyncio.create_subprocess_exec(
                cli, "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            text = stdout.decode().strip()
        else:
            try:
                import instructor
                from anthropic import Anthropic

                client = instructor.from_anthropic(Anthropic(api_key=api_key))
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=400,
                        response_model=EvidenceAssessment,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                return result
            except Exception:
                import httpx
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400,
                              "messages": [{"role": "user", "content": prompt}]},
                        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                    )
                    resp.raise_for_status()
                    text = resp.json().get("content", [{}])[0].get("text", "")

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        return EvidenceAssessment(**data)

    except Exception as e:
        logger.debug(f"Evidence assessment skipped: {e}")
        return None


def _simplify_query(topic: str, max_words: int = 5) -> str:
    """Simplify a verbose topic into a short search query.

    GitHub/HN/Reddit APIs work best with 2-5 word queries.
    Long queries like "MCP server performance optimization connection pooling"
    return zero results on these APIs.
    """
    # Remove common filler words
    stopwords = {
        "and", "or", "the", "a", "an", "in", "on", "for", "to", "of", "with",
        "using", "via", "how", "what", "why", "best", "about", "between",
    }
    words = [w for w in topic.split() if w.lower() not in stopwords]
    return " ".join(words[:max_words])


async def handle_deep_research(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Comprehensive multi-source research including live docs and web content."""
    from .search import handle_search
    from .github_tools import handle_github
    from .hn_tools import handle_hn
    from .reddit_tools import handle_reddit
    from .community_tools import handle_community
    from .package_tools import handle_packages
    from .context7_tools import handle_context7
    from .so_tools import handle_so
    from .web_tools import handle_web

    topic = arguments["topic"]
    # Try LLM-based rewriting; fall back to heuristic truncation
    try:
        from ..clients.query_rewriter import rewrite_query
        query_variants = await rewrite_query(topic)
        short_query = query_variants[1] if len(query_variants) > 1 else _simplify_query(topic)
    except Exception:
        short_query = _simplify_query(topic)
    max_per = arguments.get("max_per_source", 5)
    include_packages = arguments.get("include_packages", True)
    results: Dict[str, Any] = {"topic": topic, "search_query": short_query}
    errors: List[str] = []

    async def safe_call(name, coro):
        try:
            r = await asyncio.wait_for(coro, timeout=30.0)
            return json.loads(r[0].text)
        except asyncio.TimeoutError:
            errors.append(f"{name}: timed out after 30s")
            return None
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            return None

    # Run all sources in parallel
    # arXiv gets the full topic (handles long queries well)
    # GitHub/HN/Reddit/community/SO get the simplified query
    tasks = {
        "arxiv": safe_call("arxiv", handle_search({
            "query": topic, "max_results": max_per, "sort_by": "relevance",
        })),
        "github": safe_call("github", handle_github({
            "action": "search", "query": short_query, "max_results": max_per,
        })),
        "hackernews": safe_call("hackernews", handle_hn({
            "action": "search", "query": short_query, "max_results": max_per, "time_range": "year",
        })),
        "reddit": safe_call("reddit", handle_reddit({
            "action": "search", "query": short_query, "max_results": max_per, "time_filter": "year",
        })),
        "community": safe_call("community", handle_community({
            "action": "search", "query": short_query, "max_results": max_per,
        })),
        "stackoverflow": safe_call("stackoverflow", handle_so({
            "action": "search", "query": short_query, "max_results": max_per, "sort": "votes",
        })),
        "documentation": safe_call("context7", handle_context7({
            "action": "lookup", "library": short_query, "query": topic,
        })),
    }

    if include_packages:
        tasks["packages_npm"] = safe_call("packages_npm", handle_packages({
            "action": "search", "query": short_query, "search_registry": "npm", "max_results": 3,
        }))

    # Execute all in parallel
    task_names = list(tasks.keys())
    task_coros = list(tasks.values())
    task_results = await asyncio.gather(*task_coros)

    for name, data in zip(task_names, task_results):
        if data is not None:
            results[name] = data

    # Fetch top HN story URLs to get full article content
    hn_stories = results.get("hackernews", {}).get("stories", [])
    urls_to_fetch = [
        s["url"] for s in hn_stories
        if s.get("url") and s["url"].startswith("http")
    ][:2]  # top 2 external links only

    if urls_to_fetch:
        web_pages = await asyncio.gather(*[
            safe_call(f"web:{url}", handle_web({
                "url": url, "extract": "article", "max_length": 3000,
            }))
            for url in urls_to_fetch
        ])
        web_results = [
            {"url": url, **page}
            for url, page in zip(urls_to_fetch, web_pages)
            if page
        ]
        if web_results:
            results["web_articles"] = web_results

    if errors:
        results["errors"] = errors

    results["sources_queried"] = len(task_names)
    results["sources_succeeded"] = len(task_names) - len(errors)

    # ── Evidence sufficiency gate (FAIR-RAG SEA pattern) ────────
    # Assess whether results cover the key facets of the query.
    # If gaps found, run a second targeted pass (max 1 retry).
    assessment = await _assess_evidence(topic, results)
    if assessment and not assessment.sufficient and assessment.sub_queries:
        logger.info(f"Evidence gaps detected: {assessment.gaps}. Running gap-fill pass.")
        gap_results: Dict[str, Any] = {}
        gap_errors: List[str] = []

        async def gap_call(name: str, coro):
            try:
                r = await asyncio.wait_for(coro, timeout=30.0)
                return json.loads(r[0].text)
            except Exception as e:
                gap_errors.append(f"{name}: {str(e)}")
                return None

        gap_tasks = {}
        for i, sub_q in enumerate(assessment.sub_queries[:2]):
            gap_tasks[f"gap_arxiv_{i}"] = gap_call(f"gap_arxiv_{i}", handle_search({
                "query": sub_q, "max_results": 3, "sort_by": "relevance",
            }))
            gap_tasks[f"gap_hn_{i}"] = gap_call(f"gap_hn_{i}", handle_hn({
                "action": "search", "query": _simplify_query(sub_q), "max_results": 3,
            }))

        gap_task_names = list(gap_tasks.keys())
        gap_task_results = await asyncio.gather(*gap_tasks.values())
        for name, data in zip(gap_task_names, gap_task_results):
            if data is not None:
                gap_results[name] = data

        if gap_results:
            results["gap_fill"] = gap_results
        results["evidence_assessment"] = assessment.model_dump()

    return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
