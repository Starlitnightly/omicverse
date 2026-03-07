from pathlib import Path

from omicverse.utils.ovagent.workflow import (
    WorkflowConfig,
    load_workflow_document,
)


def test_load_workflow_document_parses_front_matter(tmp_path: Path):
    workflow_path = tmp_path / "WORKFLOW.md"
    workflow_path.write_text(
        "---\n"
        "domain: bioinformatics\n"
        "default_tools:\n"
        "  - inspect_data\n"
        "  - execute_code\n"
        "max_turns: 9\n"
        "completion_criteria:\n"
        "  - Save outputs\n"
        "---\n"
        "\n"
        "# Policy\n"
        "\n"
        "Use concise summaries.\n",
        encoding="utf-8",
    )

    workflow = load_workflow_document(workflow_path=workflow_path)

    assert workflow.config.domain == "bioinformatics"
    assert workflow.config.default_tools == ["inspect_data", "execute_code"]
    assert workflow.config.max_turns == 9
    assert workflow.config.completion_criteria == ["Save outputs"]
    assert "Use concise summaries." in workflow.body


def test_workflow_config_rejects_unknown_domain():
    config = WorkflowConfig(domain="general-coding")

    issues = config.validate()

    assert issues
    assert "domain must be one of" in issues[0]
