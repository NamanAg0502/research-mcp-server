# research-mcp-server

Multi-source research intelligence server for Claude via MCP.

## What It Does

Turns Claude into a research-aware technical advisor by combining academic papers, developer communities, package registries, and GitHub into a unified research interface. Ask a question, get answers synthesized across 17 data sources -- from arXiv papers to Hacker News discussions to npm download stats.

31 tools organized across 8 categories, all accessible through natural language in Claude Code or Claude Desktop.

## Tools

### Search and Discovery (3)

| Tool | Description |
|------|-------------|
| `search` | Search arXiv papers with structured query building, date filters, and category constraints |
| `semantic_search` | Embedding-based similarity search across your downloaded papers |
| `cross_search` | Search multiple academic sources simultaneously (arXiv, Semantic Scholar, OpenAlex, CrossRef) |

### Paper Management (4)

| Tool | Description |
|------|-------------|
| `download_paper` | Download and store arXiv papers locally as PDF |
| `list_papers` | List all downloaded papers with metadata |
| `read_paper` | Read full text content of a downloaded paper |
| `read_paper_chunks` | Read papers in paginated chunks for large documents |

### Analysis (5)

| Tool | Description |
|------|-------------|
| `citations` | Citation graph traversal via Semantic Scholar -- who cites what |
| `lineage` | Trace a paper's research lineage: influences and descendants |
| `compare` | Side-by-side comparison of multiple papers |
| `trends` | Publication trend analysis over time for a research topic |
| `digest` | Generate structured research digests with gap analysis |

### Knowledge and Memory (3)

| Tool | Description |
|------|-------------|
| `kb` | Personal knowledge base: save, search, tag, annotate, and organize papers into collections |
| `kg_query` | Query the auto-built knowledge graph of concepts, methods, and datasets |
| `memory` | Persistent research memory: sessions, open questions, findings |

### Academic Sources (6)

| Tool | Description |
|------|-------------|
| `hf_trending` | Trending papers and models from HuggingFace |
| `benchmarks` | Search Papers With Code for SOTA results and benchmarks |
| `model_benchmarks` | ML model benchmarks from Epoch AI |
| `venue_lookup` | Venue and publication metadata via DBLP |
| `patent_search` | Patent search via Lens.org |
| `export` | Export papers as BibTeX, markdown, or JSON |

### Practitioner Sources (6)

| Tool | Description |
|------|-------------|
| `hn` | Search and browse Hacker News stories and discussions |
| `community` | Developer content from Dev.to and Lobsters |
| `packages` | Package stats and comparison across npm, PyPI, and crates.io |
| `github` | GitHub repository search, trending repos, and repo details |
| `reddit` | Search and browse Reddit developer communities |
| `stackoverflow` | Search Stack Overflow questions and answers |

### CTO Intelligence (4)

| Tool | Description |
|------|-------------|
| `tech_pulse` | Aggregated view of what is trending across all practitioner sources |
| `evaluate` | Technology comparison with data from papers, packages, GitHub, and community sentiment |
| `sentiment` | Developer sentiment analysis for a technology across communities |
| `deep_research` | Multi-source deep dive on any technical topic |

### Meta (1)

| Tool | Description |
|------|-------------|
| `help` | Suggests relevant tools based on your query |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/NamanAg0502/arxiv-mcp-server.git
cd arxiv-mcp-server
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"
```

### Run

```bash
uv run research-mcp-server
```

## MCP Configuration

### Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "research": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory", "/path/to/arxiv-mcp-server",
        "run", "research-mcp-server"
      ],
      "env": {
        "GITHUB_TOKEN": "",
        "SEMANTIC_SCHOLAR_API_KEY": ""
      }
    }
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/arxiv-mcp-server",
        "run", "research-mcp-server"
      ]
    }
  }
}
```

## Data Sources

| Source | API | Auth | Provides |
|--------|-----|------|----------|
| arXiv | export.arxiv.org | None | Paper search, metadata, full-text PDFs |
| Semantic Scholar | api.semanticscholar.org | Optional key | Citations, references, batch paper lookup |
| OpenAlex | api.openalex.org | Optional email | Cross-disciplinary paper metadata |
| CrossRef | api.crossref.org | Optional email | DOI resolution, citation metadata |
| HuggingFace | huggingface.co/api | Optional token | Trending papers, models, datasets |
| Papers With Code | paperswithcode.com/api | None | SOTA benchmarks, code implementations |
| Epoch AI | epoch.ai | None | ML model benchmarks, notable AI models |
| DBLP | dblp.org | None | Venue metadata, publication records |
| Lens.org | api.lens.org | Required token | Patent search, scholarly-patent links |
| Hacker News | hn.algolia.com | None | Tech discussions, developer sentiment |
| Dev.to | dev.to/api | None | Developer articles, tutorials |
| Lobsters | lobste.rs | None | High-signal developer link aggregation |
| npm | registry.npmjs.org | None | Package stats, versions, downloads |
| PyPI | pypi.org/pypi | None | Python package metadata |
| crates.io | crates.io/api | None | Rust package metadata |
| GitHub | api.github.com | Optional token | Repository search, trending, stars |
| Reddit | oauth.reddit.com | Optional OAuth | Developer subreddit discussions |

## Environment Variables

All environment variables are optional. The server works with zero configuration, but API keys unlock higher rate limits and additional sources.

| Variable | Purpose |
|----------|---------|
| `SEMANTIC_SCHOLAR_API_KEY` | Higher S2 rate limits (free at [semanticscholar.org](https://www.semanticscholar.org/product/api)) |
| `GITHUB_TOKEN` | GitHub API: 5000 req/hr vs 60 req/hr unauthenticated |
| `HF_TOKEN` | HuggingFace API: higher rate limits |
| `LENS_API_TOKEN` | Required for patent search via Lens.org |
| `REDDIT_CLIENT_ID` | Reddit OAuth2 access (paired with secret) |
| `REDDIT_CLIENT_SECRET` | Reddit OAuth2 access (paired with client ID) |
| `OPENALEX_EMAIL` | Polite pool access for OpenAlex (faster responses) |
| `CROSSREF_EMAIL` | Polite pool access for CrossRef (faster responses) |
| `STACKOVERFLOW_KEY` | Stack Overflow API: higher rate limits |

## CTO Intelligence Examples

The composite intelligence tools orchestrate queries across multiple sources to answer high-level questions.

**"What's trending this week?"**

```
Use tech_pulse to see what developers are talking about right now.
Aggregates Hacker News, Reddit, GitHub trending, Dev.to, and HuggingFace.
```

**"Drizzle vs Prisma?"**

```
Use evaluate to compare technologies with data from npm downloads,
GitHub stars, Stack Overflow activity, community sentiment, and papers.
```

**"What do devs think about Bun?"**

```
Use sentiment to analyze developer opinion across Hacker News discussions,
Reddit threads, Dev.to articles, and Stack Overflow questions.
```

**"Everything about WebTransport"**

```
Use deep_research for a multi-source deep dive: papers, specs,
community discussions, package ecosystem, and GitHub implementations.
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Format
black src/ tests/

# Run server locally with custom storage path
uv run research-mcp-server --storage-path ~/.arxiv-papers
```

## License

Apache-2.0
