from pathlib import Path

from omicverse.utils.ovagent.run_store import RunStore
from omicverse.utils.ovagent.workflow import WorkflowConfig, WorkflowDocument


def test_run_store_start_and_finish(tmp_path: Path):
    workflow = WorkflowDocument(
        path=tmp_path / "WORKFLOW.md",
        config=WorkflowConfig(),
        body="Use provenance-aware analysis.",
        raw_text="---\ndomain: bioinformatics\n---\nUse provenance-aware analysis.\n",
    )
    store = RunStore(root=tmp_path / "runs")

    run = store.start_run(
        request="Analyze GSE dataset",
        model="gpt-5.2",
        provider="openai",
        session_id="ses_demo",
        workflow=workflow,
    )
    store.finish_run(
        run.run_id,
        status="success",
        summary="Completed analysis.",
        trace_id="trace_demo",
        artifacts=[{"kind": "file", "label": "summary", "path": "/tmp/summary.md"}],
    )

    bundle = store.build_bundle(run.run_id)
    reloaded = store.load_run(run.run_id)

    assert reloaded.status == "success"
    assert reloaded.trace_ids == ["trace_demo"]
    assert reloaded.summary == "Completed analysis."
    assert bundle["paths"]["bundle"].endswith("bundle.json")
    assert bundle["paths"]["summary"].endswith("summary.md")
