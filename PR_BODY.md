# LLM Domain Research: live web retrieval, alias module, docs/tutorial updates, deprecate `ov.llm.dr`

## Summary
Adds live web-backed retrieval to the domain research pipeline and introduces a clearer import path `omicverse.llm.domain_research` (re-exporting the legacy `omicverse.llm.dr`). Importing `omicverse.llm.dr` now emits a deprecation warning guiding users to the new module path. Docs and tutorials are updated to reflect the new usage and web options.

## Changes
- Feature: Live web retriever
  - `omicverse/llm/dr/retrievers/web_store.py`: New `WebRetrieverStore` with two backends:
    - `tavily` (requires `TAVILY_API_KEY`) — reliable results/snippets
    - `duckduckgo` — works without keys; best with `duckduckgo_search` + `beautifulsoup4`
  - Optional `fetch_content` fetches pages and extracts text; otherwise uses snippets.

- Pipeline wiring
  - `omicverse/llm/dr/research_manager.py`:
    - Accepts `vector_store="web" | "web:tavily" | "web:duckduckgo"` to auto/force web retrieval.
    - Backward compatibility: existing object-based `vector_store` continues to work unchanged.

- New alias module
  - `omicverse/llm/domain_research/__init__.py`: Re-exports primary API from `..dr`.
  - `omicverse/llm/domain_research/retrievers/__init__.py`: Re-exports `WebRetrieverStore`.
  - `omicverse/llm/domain_research/write/synthesizer.py`: Re-exports synthesizer types.

- Deprecation
  - `omicverse/llm/dr/__init__.py`: Emits `DeprecationWarning` on import with a pointer to `omicverse.llm.domain_research`.

- Documentation
  - `README.md`: Examples updated to use `omicverse.llm.domain_research`; added live web usage.
  - `omicverse_guide/mkdocs.yml`: Adds “Domain Research (dr) — Live Web” to LLM section.
  - `omicverse_guide/docs/Tutorials-llm/t_dr_web.md`: New live web tutorial (auto/forced backend examples and LLM synthesis).
  - `omicverse_guide/docs/Tutorials-llm/t_dr.ipynb`:
    - Added preface on web options and env vars.
    - Added sections for auto web retrieval, forced backend, and web+LLM synthesis.
    - Imports updated to `omicverse.llm.domain_research`.

## Usage
- Auto-select backend based on env:
  ```python
  from omicverse.llm.domain_research import ResearchManager
  rm = ResearchManager(vector_store="web")  # Tavily if TAVILY_API_KEY else DuckDuckGo
  print(rm.run("Recent advances in single-cell RNA-seq batch correction"))
  ```
- Force backend:
  ```python
  ResearchManager(vector_store="web:tavily")      # requires TAVILY_API_KEY
  ResearchManager(vector_store="web:duckduckgo")  # no API key required
  ```
- LLM-backed synthesis:
  ```python
  from omicverse.llm.domain_research.write.synthesizer import PromptSynthesizer
  ```

## Migration & Deprecation
- Prefer `omicverse.llm.domain_research` going forward.
- `omicverse.llm.dr` remains functional but now triggers a `DeprecationWarning` upon import.

## Notes
- Tavily: set `TAVILY_API_KEY` in the environment.
- DuckDuckGo: for robustness, install `duckduckgo_search` and `beautifulsoup4`.
- Set `fetch_content=False` to rely on search snippets (fewer network calls).

## Testing
- The change is backward compatible at the API boundary. Live web retrieval relies on external services; unit tests for the "web" flag can mock HTTP calls if desired (not included here).

## Checklist
- [x] Backward compatible exports preserved via alias.
- [x] Added deprecation warning for `omicverse.llm.dr`.
- [x] README and tutorials updated; mkdocs nav updated.
- [ ] CI/docs build verification by maintainers (mkdocs build).
- [ ] Optional: add mocked unit tests for web retrieval flag.

