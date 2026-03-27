"""Semantic + keyword search across the personal knowledge base.

All operations are local — no external API calls. Supports keyword-only,
semantic-only (embedding similarity), or hybrid search modes.
"""

import json
import logging
from typing import Any, Dict, List

import bm25s
import numpy as np
import mcp.types as types

from .semantic_search import _load_model, MODEL_NAME, BGE_QUERY_PREFIX
from ..store.knowledge_base import KnowledgeBase
from ..clients.query_rewriter import rewrite_query

logger = logging.getLogger("research-mcp-server")

# Lazy singleton cross-encoder for reranking
_cross_encoder = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        except Exception as e:
            logger.warning(f"CrossEncoder unavailable: {e}")
    return _cross_encoder

kb_search_tool = types.Tool(
    name="kb_search",
    description="""Search your local knowledge base (papers saved via kb_save). Unlike search_papers/arxiv_semantic_search which search arXiv, this searches only YOUR saved papers. Fully local, no API calls.

Modes: "hybrid" (default, best results), "semantic" (meaning-based), "keyword" (text match). Filter by tags, categories, reading_status, or collection.

Examples: query="attention mechanisms" | query="RL for robotics", tags=["agents"], mode="semantic\"""",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
                "minLength": 1,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by any of these tags.",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by any of these categories (e.g., ['cs.AI', 'cs.LG']).",
            },
            "reading_status": {
                "type": "string",
                "description": "Filter by reading status: unread, reading, completed, archived.",
                "enum": ["unread", "reading", "completed", "archived"],
            },
            "collection": {
                "type": "string",
                "description": "Search within a specific collection.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10).",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
            "mode": {
                "type": "string",
                "description": "Search mode: 'hybrid' (default), 'semantic', or 'keyword'.",
                "default": "hybrid",
                "enum": ["hybrid", "semantic", "keyword"],
            },
        },
        "required": ["query"],
    },
)


def _apply_filters(
    paper: Dict[str, Any],
    *,
    tags: List[str] | None,
    categories: List[str] | None,
    reading_status: str | None,
) -> bool:
    """Return True if paper passes all provided filters."""
    if reading_status and paper.get("reading_status") != reading_status:
        return False
    if tags:
        paper_tags = paper.get("tags", [])
        if not any(t in paper_tags for t in tags):
            return False
    if categories:
        paper_cats = paper.get("categories", [])
        if not any(c in paper_cats for c in categories):
            return False
    return True


def _bm25_search(
    query: str,
    papers: List[Dict[str, Any]],
    *,
    tags: List[str] | None,
    categories: List[str] | None,
    reading_status: str | None,
    collection: str | None,
    top_k: int,
) -> List[tuple[Dict[str, Any], float]]:
    """Run BM25 over a paper corpus and return (paper, score) pairs.

    Uses title + abstract as the document text. Filters are applied
    post-scoring so the BM25 index covers the full corpus.
    """
    if not papers:
        return []

    corpus_texts = [
        f"{p.get('title', '')} {p.get('abstract', '') or ''}"
        for p in papers
    ]
    try:
        tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        tokenized_query = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(len(papers), top_k)
        results_idx, scores = retriever.retrieve(tokenized_query, k=k)
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []

    ranked: List[tuple[Dict[str, Any], float]] = []
    for idx, score in zip(results_idx[0], scores[0]):
        paper = papers[int(idx)]
        if not _apply_filters(
            paper, tags=tags, categories=categories, reading_status=reading_status
        ):
            continue
        if collection:
            if collection not in paper.get("collections", []):
                continue
        ranked.append((paper, float(score)))

    return ranked


async def handle_kb_search(
    arguments: Dict[str, Any],
) -> List[types.TextContent]:
    """Handle knowledge base search requests.

    Supports keyword, semantic, and hybrid search modes with optional
    tag/category/status/collection filters.

    Args:
        arguments: Tool input with query, optional filters, mode, max_results.

    Returns:
        List containing a single TextContent with JSON results.
    """
    try:
        query = arguments["query"]
        tags = arguments.get("tags")
        categories = arguments.get("categories")
        reading_status = arguments.get("reading_status")
        collection = arguments.get("collection")
        max_results = min(max(int(arguments.get("max_results", 10)), 1), 50)
        mode = arguments.get("mode", "hybrid")

        if mode not in ("hybrid", "semantic", "keyword"):
            mode = "hybrid"

        kb = KnowledgeBase()
        model_note: str | None = None

        # ── Query rewriting (hybrid only) ────────────────────────
        # Generate 2-3 reformulations; all are used for BM25/keyword retrieval.
        # Dense semantic search uses the original query only (embedding space is
        # robust to phrasing — rewriting there adds noise, not signal).
        queries = [query]
        if mode == "hybrid":
            try:
                queries = await rewrite_query(query)
            except Exception as _rw_err:
                logger.debug(f"Query rewrite skipped: {_rw_err}")

        # ── BM25 corpus (hybrid only — fetch all papers once) ───
        # Run BM25 for each query reformulation, merge all ranked lists via RRF.
        bm25_results: List[tuple[Dict[str, Any], float]] = []
        all_papers_for_bm25: List[Dict[str, Any]] = []
        if mode == "hybrid":
            all_papers_for_bm25 = await kb.list_papers(limit=10000)
            if all_papers_for_bm25:
                # RRF-merge BM25 results from all query reformulations
                bm25_rrf: Dict[str, Dict[str, Any]] = {}
                for q in queries:
                    ranked = _bm25_search(
                        q,
                        all_papers_for_bm25,
                        tags=tags,
                        categories=categories,
                        reading_status=reading_status,
                        collection=collection,
                        top_k=max_results * 5,
                    )
                    for rank, (paper, _score) in enumerate(ranked, start=1):
                        pid = paper["id"]
                        contribution = 1.0 / (60 + rank)
                        if pid in bm25_rrf:
                            bm25_rrf[pid]["score"] += contribution
                        else:
                            bm25_rrf[pid] = {"paper": paper, "score": contribution}
                bm25_results = [
                    (entry["paper"], entry["score"])
                    for entry in sorted(bm25_rrf.values(), key=lambda x: x["score"], reverse=True)
                ]

        # ── Keyword search ──────────────────────────────────────
        # Run SQL LIKE for each query reformulation, deduplicate by paper ID.
        keyword_results: List[Dict[str, Any]] = []
        if mode in ("keyword", "hybrid"):
            seen_kw: set[str] = set()
            for q in queries:
                rows = await kb.list_papers(
                    query=q,
                    tags=tags,
                    categories=categories,
                    reading_status=reading_status,
                    collection=collection,
                    limit=max_results * 5,
                )
                for paper in rows:
                    if paper["id"] not in seen_kw:
                        seen_kw.add(paper["id"])
                        keyword_results.append(paper)

        # ── Semantic search ─────────────────────────────────────
        semantic_results: List[tuple[Dict[str, Any], float]] = []
        if mode in ("semantic", "hybrid"):
            model = _load_model()
            if model is None:
                logger.warning(
                    "Embedding model unavailable, falling back to keyword-only"
                )
                model_note = (
                    "Embedding model could not be loaded. Results are ranked "
                    "by keyword matching only (not semantic similarity)."
                )
                if mode == "semantic":
                    # Pure semantic was requested but model is unavailable —
                    # fall back to keyword
                    keyword_results = await kb.list_papers(
                        query=query,
                        tags=tags,
                        categories=categories,
                        reading_status=reading_status,
                        collection=collection,
                        limit=max_results,
                    )
                    mode = "keyword"
                else:
                    # hybrid — just skip semantic component
                    mode = "keyword"
            else:
                # Get all papers with embeddings
                papers_with_embs = await kb.get_all_papers_with_embeddings(
                    MODEL_NAME
                )

                if papers_with_embs:
                    # Encode query (BGE models need instruction prefix for queries)
                    query_emb = model.encode(
                        [BGE_QUERY_PREFIX + query], normalize_embeddings=True
                    )[0]

                    for paper, emb_bytes in papers_with_embs:
                        emb = np.frombuffer(emb_bytes, dtype=np.float32)
                        sim = float(np.dot(query_emb, emb))

                        # Apply post-ranking filters
                        if not _apply_filters(
                            paper,
                            tags=tags,
                            categories=categories,
                            reading_status=reading_status,
                        ):
                            continue

                        # Collection filter — check if paper is in the collection
                        if collection:
                            paper_collections = paper.get("collections", [])
                            if collection not in paper_collections:
                                continue

                        semantic_results.append((paper, sim))

                    # Sort by similarity descending
                    semantic_results.sort(key=lambda x: x[1], reverse=True)

        # ── Combine results ─────────────────────────────────────
        final_papers: List[Dict[str, Any]]

        if mode == "keyword":
            # Pure keyword or fallback
            final_papers = []
            for paper in keyword_results[:max_results]:
                p = paper.copy()
                p["search_mode"] = "keyword"
                final_papers.append(p)

        elif mode == "semantic":
            # Pure semantic
            final_papers = []
            for paper, sim in semantic_results[:max_results]:
                p = paper.copy()
                p["semantic_score"] = round(sim, 4)
                p["search_mode"] = "semantic"
                final_papers.append(p)

        else:
            # Hybrid — merge keyword + BM25 + semantic via RRF (k=60)
            rrf_k = 60
            combined_scores: Dict[str, Dict[str, Any]] = {}

            def _rrf_add(pid: str, paper: Dict[str, Any], rank: int, source: str) -> None:
                contribution = 1.0 / (rrf_k + rank)
                if pid in combined_scores:
                    combined_scores[pid]["rrf_score"] += contribution
                    existing = combined_scores[pid]["found_by"]
                    if source not in existing:
                        combined_scores[pid]["found_by"] = f"{existing}+{source}"
                else:
                    combined_scores[pid] = {
                        "paper": paper,
                        "rrf_score": contribution,
                        "found_by": source,
                    }

            for rank, paper in enumerate(keyword_results, start=1):
                _rrf_add(paper["id"], paper, rank, "keyword")

            for rank, (paper, _) in enumerate(bm25_results, start=1):
                _rrf_add(paper["id"], paper, rank, "bm25")

            for rank, (paper, _) in enumerate(semantic_results, start=1):
                _rrf_add(paper["id"], paper, rank, "semantic")

            scored_papers: List[tuple[Dict[str, Any], float, str]] = [
                (entry["paper"], entry["rrf_score"], entry["found_by"])
                for entry in combined_scores.values()
            ]
            scored_papers.sort(key=lambda x: x[1], reverse=True)

            final_papers = []
            for paper, score, found_by in scored_papers[:max_results]:
                p = paper.copy()
                p["rrf_score"] = round(score, 6)
                p["found_by"] = found_by
                p["search_mode"] = "hybrid"
                p["scoring"] = "reciprocal_rank_fusion"
                final_papers.append(p)

        # ── Cross-encoder reranking (hybrid mode, ≥2 results) ───
        if mode == "hybrid" and len(final_papers) >= 2:
            cross_encoder = _get_cross_encoder()
            if cross_encoder is not None:
                try:
                    pairs = [
                        (query, f"{p.get('title', '')} {p.get('abstract', '') or ''}")
                        for p in final_papers
                    ]
                    ce_scores = cross_encoder.predict(pairs)
                    for paper, score in zip(final_papers, ce_scores):
                        paper["rerank_score"] = round(float(score), 4)
                    final_papers.sort(key=lambda p: p.get("rerank_score", 0), reverse=True)
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed: {e}")

        # ── Build response ──────────────────────────────────────
        response: Dict[str, Any] = {
            "total": len(final_papers),
            "mode": mode,
            "query": query,
            "papers": final_papers,
        }

        if model_note:
            response["note"] = model_note

        logger.info(
            f"KB search completed: mode={mode}, query='{query}', "
            f"results={len(final_papers)}"
        )

        return [
            types.TextContent(
                type="text", text=json.dumps(response, indent=2)
            )
        ]

    except Exception as e:
        logger.error(f"KB search error: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
