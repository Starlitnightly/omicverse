This PR unifies the Skill Seeker Jupyter API under `ov.Agent.seeker` and fixes critical issues in the new implementation.

Summary
- Primary API: `ov.Agent.seeker(...)` attached to the `Agent` factory
- Backward compatibility: `ov.agent.seeker(...)` kept as a deprecated alias with `DeprecationWarning`
- Fixed implementation bugs in `Agent.seeker`:
  - Correct builder signatures: `build_from_link(link, output_root, name, description, max_pages)` and `build_from_config(config_path, output_root)`
  - Proper output-root resolution for `target="skills"|"output"` and `out_dir`
  - Implemented packaging via `_zip_dir`; returns `{"slug","skill_dir","zip"}` when `package=True`
  - Return type normalized to `Dict[str, str]`
  - Selective optional-deps gating (docs path requires BeautifulSoup)
- Docs updated to reference `ov.Agent.seeker(...)` and note the deprecation of `ov.agent.seeker(...)`
- Tests added/updated for packaging, signature verification, result shape, and alias forwarding

Key Files
- omicverse/utils/smart_agent.py: adds `Agent.seeker` implementation and packaging
- omicverse/agent/__init__.py: deprecation alias + `_zip_dir` helper
- omicverse/ov_skill_seeker/README.md: API usage now `ov.Agent.seeker(...)`
- tests/utils/test_smart_agent.py: covers signatures/packaging/returns
- tests/test_ov_skill_seeker.py: CLI/builders/config validation and alias forwarding

Notes
- Optional extras remain under `[skillseeker]` (beautifulsoup4, PyGithub, PyMuPDF)
- Unified builder guards imports per-source and degrades by writing error references when deps are missing
- Follow-up tasks (separate PRs): security docs (URL/path validation), broader CLI/scraper tests, and CI smoke tests


