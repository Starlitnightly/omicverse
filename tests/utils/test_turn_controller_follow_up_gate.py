from __future__ import annotations

from omicverse.utils.ovagent.turn_controller import FollowUpGate


def test_visualization_request_requires_tool_action_for_image_send() -> None:
    assert FollowUpGate.request_requires_tool_action("把分析后的图发送出来", None) is True
    assert FollowUpGate.request_requires_tool_action("Please plot the UMAP and send the image", None) is True


def test_visualization_text_only_claim_triggers_follow_up_when_no_tool_called() -> None:
    should_continue = FollowUpGate.should_continue_after_text(
        request="把分析后的图发送出来",
        response_content="已给你绘制并发送 T细胞 marker 的 UMAP 图。",
        adata=None,
        had_meaningful_tool_call=False,
    )

    assert should_continue is True


def test_inspect_data_does_not_count_as_meaningful_progress() -> None:
    assert FollowUpGate.tool_counts_as_meaningful_progress("inspect_data") is False
    assert FollowUpGate.tool_counts_as_meaningful_progress("InspectData") is False
    assert FollowUpGate.tool_counts_as_meaningful_progress("run_snippet") is False


def test_execute_code_counts_as_meaningful_progress() -> None:
    assert FollowUpGate.tool_counts_as_meaningful_progress("execute_code") is True
    assert FollowUpGate.tool_counts_as_meaningful_progress("web_fetch") is True
