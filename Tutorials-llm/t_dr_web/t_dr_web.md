---
title: Domain Research (dr) — Live Web Search
---

This guide shows how to use `omicverse.llm.domain_research` with a live web search retriever to produce a comprehensive, cited report sourced from the internet.

Prerequisites:
- Optional but recommended: set `TAVILY_API_KEY` for the Tavily backend.
- For the DuckDuckGo backend, install `duckduckgo_search` and `beautifulsoup4` for better results.

Quick start (auto backend):

```python
from omicverse.llm.domain_research import ResearchManager

# Easiest: `vector_store="web"` auto-selects Tavily if TAVILY_API_KEY is set,
# otherwise falls back to DuckDuckGo.
rm = ResearchManager(vector_store="web")
report = rm.run("State-of-the-art methods for single-cell integration in 2024")
print(report)
```

Force backend or tweak retrieval:
- `vector_store="web:tavily"` or `vector_store="web:duckduckgo"` to force a backend.
- For manual control (e.g., `max_results`, `fetch_content`), import and instantiate `WebRetrieverStore` directly.

Tips:
- `fetch_content=True` fetches and extracts text from result URLs. Set to `False` to rely on snippets only.
- Combine with an LLM-backed synthesizer for a stronger executive summary:

```python
import os
from omicverse.llm.domain_research.write.synthesizer import PromptSynthesizer

synth = PromptSynthesizer(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY", ""),
)
rm = ResearchManager(vector_store="web", synthesizer=synth)
print(rm.run("Multi-omics integration benchmarks 2023–2025"))
```

Troubleshooting:
- Tavily: ensure `TAVILY_API_KEY` is set and valid.
- DuckDuckGo: for best stability use the `duckduckgo_search` package; otherwise the HTML fallback may be rate-limited or change over time.
- If pages are not HTML or are behind paywalls, the fetcher returns the raw response text (truncated) as the document body.
