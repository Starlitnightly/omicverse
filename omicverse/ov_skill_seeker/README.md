OmicVerse Skill Seeker
======================

Deeper tooling to create, inspect, validate, and package OmicVerse Claude Agent skills. It also exposes a Jupyter-friendly API to scaffold new skills from link(s).

What it does
- List bundled skills discovered under `.claude/skills`
- Validate frontmatter and presence of important files
- Package skills into uploadable `.zip` archives
- Create a new skill from a single link (same‑domain crawl)
- Build a new skill from a unified JSON config (docs + GitHub + PDFs)
- Jupyter API: `ov.Agent.seeker(...)` for notebook workflows

Install optional extras (for full builders)
- pip install -e .[skillseeker]
  - beautifulsoup4: docs scraping
  - PyGithub: GitHub metadata/README extraction
  - PyMuPDF (fitz): PDF text extraction

CLI usage (from repo root)
- List skills
  python -m omicverse.ov_skill_seeker --list

- Validate all skills
  python -m omicverse.ov_skill_seeker --validate

- Package one skill by slug
  python -m omicverse.ov_skill_seeker --package bulk-combat-correction

- Package all skills to output/
  python -m omicverse.ov_skill_seeker --package-all

- Create a new skill from a single link (defaults to `.claude/skills`)
  python -m omicverse.ov_skill_seeker --create-from-link https://example.com/feature-doc \
      --name "New Analysis Function" \
      --description "Prototype capability not yet in OmicVerse" \
      --max-pages 30 \
      --package-after

- Build from a unified config into output/
  python -m omicverse.ov_skill_seeker --build-config ./my_unified_config.json --out-dir output

Jupyter API (notebook friendly)
- Single link
  >>> import omicverse as ov
  >>> ov.Agent.seeker("https://example.com/docs/feature", name="New Analysis", package=True)
  {'slug': 'new-analysis', 'skill_dir': '.../.claude/skills/new-analysis', 'zip': '.../output/new-analysis.zip'}

- Multiple links to output directory
  >>> ov.Agent.seeker([
  ...   "https://docs.site-a.com/",
  ...   "https://docs.site-b.com/guide"
  ... ], name="multi-source", target="output", package=True)
  {'slug': 'multi-source', 'skill_dir': '.../output/multi-source', 'zip': '.../output/multi-source.zip'}

Note: `ov.agent.seeker()` is deprecated. Use `ov.Agent.seeker()` instead.

Unified config format (JSON)
- Example (save as `my_unified_config.json`):
  {
    "name": "OmicVerse",
    "description": "OmicVerse docs + GitHub repository unified skill",
    "sources": [
      { "type": "documentation", "base_url": "https://omicverse.readthedocs.io/", "max_pages": 50 },
      { "type": "github", "repo": "Starlitnightly/omicverse" },
      { "type": "pdf", "path": "./docs/supplement.pdf" }
    ]
  }

Outputs and structure
- New skills are created under:
  - `.claude/skills/<slug>` when `target=skills` (default)
  - `./output/<slug>` when `target=output` or using `--out-dir`
- Each skill contains:
  - `SKILL.md` with YAML frontmatter: `name` (slug), `title`, `description`, source trace
  - `references/` with scraped or extracted content
- Packaging creates `<slug>.zip` alongside `output/` (or `--out-dir` / `package_dir` via API)

Behavior and prerequisites
- YAML-aware frontmatter parsing is used throughout; fallbacks handle environments without PyYAML.
- Network-restricted runs still scaffold SKILL.md and place error notes into `references/*-error.md` when a source cannot be fetched.
- GitHub extraction honors anonymous access; it uses `GITHUB_TOKEN` if present.

Testing
- Offline tests exist for builders, CLI, and config validation:
  pytest -q omicverse/tests/test_ov_skill_seeker.py
- These tests stub network calls and optional dependencies.

Tips
- Pick concise slugs; the system trims to 64 chars and de-duplicates on disk when necessary.
- Use the Jupyter API to quickly bootstrap a skill while exploring missing features; iterate on the references and SKILL.md text before packaging.

Roadmap
- Add example notebooks and CI smoke tests for the CLI.
- Expand docs with richer frontmatter examples and guidance for best practices.
