# OmicVerse Agent Interface

This directory contains the Streamlit application that powers the OmicVerse "Agentic" experience. The
app bundles a retrieval-augmented generation (RAG) pipeline, the skill router, and multiple operator
modes (paper checker, computer-use agent, and the standard document assistant).

## Installation

1. Create and activate a Python 3.11 environment.
2. Install the core OmicVerse project dependencies.
3. Install the agent-specific extras:

```bash
pip install -r OvIntelligence/requirements.txt
```

4. No Pantheon dependency is required. The smart agent now uses an internal backend.

## Configuring document packages

The agent indexes tutorial notebooks and annotated scripts from local directories. Paths can be
customised via the `OvIntelligence/config.json` file or by setting the `OV_PACKAGE_BASE_DIR`
environment variable before launching Streamlit.

When the configuration file is absent, the application falls back to the following defaults:

- `package_base_dir`: `~/omicverse_packages`
- `packages`: predefined package names and their relative sub-directories (for example
  `6O_json_files/cellrank_notebooks` and `annotated_scripts/cellrank_notebooks`).

Each entry can specify either absolute paths (`converted_jsons_directory` and
`annotated_scripts_directory`) or relative sub-directories (`converted_jsons_subdir` and
`annotated_scripts_subdir`). Missing or inaccessible paths are logged and ignored during startup, and
no RAG context will be built until at least one package resolves successfully.

Example `config.json` snippet:

```json
{
  "package_base_dir": "/data/omicverse/tutorials",
  "packages": [
    {
      "name": "cellrank_notebooks",
      "converted_jsons_subdir": "6O_json_files/cellrank_notebooks",
      "annotated_scripts_subdir": "annotated_scripts/cellrank_notebooks"
    },
    {
      "name": "custom_workflow",
      "converted_jsons_directory": "/opt/omicverse/custom/json",
      "annotated_scripts_directory": "/opt/omicverse/custom/scripts"
    }
  ]
}
```

Any edits made through the UI are persisted to the same configuration file.

## API keys

Set the following environment variables prior to launching the Streamlit application to enable LLM
features:

- `GEMINI_API_KEY` – enables Gemini powered search and code refinement
- `OV_PACKAGE_BASE_DIR` *(optional)* – overrides the default location for package data

## Security considerations

The OmicVerse smart agent executes generated Python inside a restricted sandbox. The sandbox limits
available built-ins and only allows importing whitelisted modules, but it is **not** a perfect
security boundary. You should run the agent inside an environment you trust (for example, a local
machine or a locked-down container) and review generated code when working with untrusted prompts.

For additional isolation consider:

- Running the Streamlit app inside a disposable container or VM
- Mounting data directories as read-only volumes
- Auditing the agent logs stored in `OvIntelligence/logs/`

## Using skills

Project-specific skills live in the `.claude/skills/` directory at the repository root. During
startup the Streamlit app builds a skill registry and exposes the discovered skills in the sidebar.
From the chat interface you can request:

```
List all available skills
```

or inspect a particular skill with:

```
Show details for the cellrank trajectory skill
```

These commands render the metadata extracted from the `SKILL.md` files, helping operators verify the
available workflows.

## Troubleshooting

- **"No document packages are configured"** – verify your `config.json` values or export
  `OV_PACKAGE_BASE_DIR` to point at the directory that contains the JSON summaries and annotated
  scripts.
- **"GEMINI_API_KEY not set" warnings** – set the environment variable if you want to use Gemini
  powered web search and code refinement. The rest of the UI will continue to function without it.
