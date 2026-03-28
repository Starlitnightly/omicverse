"""Execution tool handlers: execute_code, run_snippet, bash, agent, inspect, search, finish.

Extracted from ``tool_runtime.py`` during Phase 3 decomposition (pass 2).
These handler functions receive explicit dependencies (ctx, executor,
subagent_controller) as parameters rather than accessing ``self`` on
``ToolRuntime``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..harness.runtime_state import runtime_state
from ..harness.tool_catalog import resolve_tool_search
from ..._registry import _global_registry

if TYPE_CHECKING:
    from .analysis_executor import AnalysisExecutor
    from .protocol import AgentContext

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int) -> str:
    """Truncate text for LLM-facing tool output while preserving total length."""
    if limit <= 0 or len(text) <= limit:
        return text
    suffix = f"\n... (truncated, total_chars={len(text)})"
    keep = max(0, limit - len(suffix))
    return text[:keep] + suffix


# ------------------------------------------------------------------
# Data inspection
# ------------------------------------------------------------------

def handle_inspect_data(adata: Any, aspect: str) -> str:
    """Inspect an AnnData/MuData object and return a structured summary."""
    if adata is None:
        return (
            "No dataset is loaded. Use web_download to download a "
            "dataset first, then load it with execute_code."
        )
    try:
        parts: List[str] = []
        dtype = type(adata).__name__
        is_mudata = dtype == "MuData"

        if is_mudata and aspect in ("full", "shape"):
            parts.append("Type: MuData")
            if hasattr(adata, "mod"):
                mod_keys = list(adata.mod.keys())
                parts.append(f"Modalities: {mod_keys}")
                for mk in mod_keys:
                    mod = adata.mod[mk]
                    layers_keys = (
                        list(mod.layers.keys())
                        if getattr(mod, "layers", None) is not None
                        else []
                    )
                    parts.append(
                        f"  {mk}: {mod.shape[0]} cells x "
                        f"{mod.shape[1]} features, layers={layers_keys}"
                    )
            if hasattr(adata, "shape"):
                parts.append(
                    f"Combined shape: {adata.shape[0]} obs x "
                    f"{adata.shape[1]} vars"
                )

        if aspect in ("shape", "full") and not is_mudata:
            parts.append(
                f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes"
            )
        if aspect in ("obs", "full"):
            cols = list(adata.obs.columns)
            parts.append(f"obs columns ({len(cols)}): {cols}")
            try:
                parts.append(
                    f"obs.head(3):\n{adata.obs.head(3).to_string()}"
                )
            except Exception as exc:
                logger.debug("Failed to format obs.head(3): %s", exc)
        if aspect in ("var", "full") and not is_mudata:
            cols = list(adata.var.columns)
            parts.append(f"var columns ({len(cols)}): {cols}")
            try:
                parts.append(
                    f"var.head(3):\n{adata.var.head(3).to_string()}"
                )
            except Exception as exc:
                logger.debug("Failed to format var.head(3): %s", exc)
        if aspect in ("obsm", "full"):
            keys = (
                list(adata.obsm.keys()) if hasattr(adata, "obsm") else []
            )
            parts.append(f"obsm keys: {keys}")
            for k in keys:
                try:
                    parts.append(f"  {k}: shape {adata.obsm[k].shape}")
                except Exception as exc:
                    logger.debug("Failed to get shape for obsm[%s]: %s", k, exc)
        if aspect in ("uns", "full"):
            keys = (
                list(adata.uns.keys()) if hasattr(adata, "uns") else []
            )
            parts.append(f"uns keys: {keys}")
        if aspect in ("layers", "full"):
            layers = getattr(adata, "layers", None)
            keys = list(layers.keys()) if layers is not None else []
            parts.append(f"layers: {keys}")
        return (
            "\n".join(parts) if parts else f"Unknown aspect: {aspect}"
        )
    except Exception as e:
        return f"Error inspecting data: {e}"


# ------------------------------------------------------------------
# Code execution
# ------------------------------------------------------------------

def handle_execute_code(
    ctx: "AgentContext",
    executor: "AnalysisExecutor",
    code: str,
    description: str,
    adata: Any,
) -> dict:
    """Execute transformed Python code through the structured repair loop."""
    from .analysis_executor import ProactiveCodeTransformer
    from .repair_loop import ExecutionRepairLoop

    code = ProactiveCodeTransformer().transform(code)
    if getattr(ctx, "_code_only_mode", False):
        capture = getattr(ctx, "_capture_code_only_snippet", None)
        if callable(capture):
            capture(code, description=description)
        return {
            "adata": adata,
            "output": (
                "CODE ONLY MODE: captured generated Python code without "
                "executing it."
            ),
        }

    prereq_warnings = executor.check_code_prerequisites(code, adata)

    executor._notebook_fallback_error = None

    # --- Structured self-healing loop ---
    repair_loop = ExecutionRepairLoop(executor, max_retries=3)
    extract_code_fn = getattr(ctx, "_extract_python_code", None)

    import asyncio as _asyncio

    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            repair_result = pool.submit(
                _asyncio.run,
                repair_loop.run(code, adata, phase="execution",
                               extract_code_fn=extract_code_fn),
            ).result()
    else:
        repair_result = _asyncio.run(
            repair_loop.run(code, adata, phase="execution",
                           extract_code_fn=extract_code_fn)
        )

    if repair_result.success:
        result = repair_result.exec_result
        stdout = result.get("stdout", "") if isinstance(result, dict) else ""
        result_adata = (
            result.get("adata", adata)
            if isinstance(result, dict)
            else (result if result is not None else adata)
        )

        output_parts: List[str] = []
        debug_output_parts: List[str] = []
        notebook_err = getattr(
            executor, "_notebook_fallback_error", None
        )
        if notebook_err:
            notebook_msg = (
                f"WARNING: notebook session execution failed with error:\n{notebook_err}\n"
                f"Fell back to in-process execution. Please fix the code to avoid this error."
            )
            output_parts.append(notebook_msg)
            debug_output_parts.append(notebook_msg)

        # Annotate recovered attempts
        if len(repair_result.attempts) > 1:
            strategies = [
                a.strategy for a in repair_result.attempts if not a.success
            ]
            recovery_msg = (
                f"RECOVERED after {len(repair_result.attempts)} attempt(s) "
                f"(strategies: {', '.join(strategies)})"
            )
            output_parts.append(recovery_msg)
            debug_output_parts.append(recovery_msg)

        if prereq_warnings:
            warning_msg = f"PREREQUISITE WARNINGS: {prereq_warnings}"
            output_parts.append(warning_msg)
            debug_output_parts.append(warning_msg)
        if stdout.strip():
            output_parts.append(
                f"stdout:\n{_truncate_text(stdout, 3000)}"
            )
            debug_output_parts.append(f"stdout:\n{stdout}")
        try:
            result_msg = (
                f"Result adata shape: {result_adata.shape[0]} cells x "
                f"{result_adata.shape[1]} features"
            )
            output_parts.append(result_msg)
            debug_output_parts.append(result_msg)
        except Exception:
            result_msg = f"Result type: {type(result_adata).__name__}"
            output_parts.append(result_msg)
            debug_output_parts.append(result_msg)

        llm_output = (
            "\n".join(output_parts)
            if output_parts
            else "Code executed successfully (no output)."
        )
        debug_output = (
            "\n".join(debug_output_parts)
            if debug_output_parts
            else "Code executed successfully (no output)."
        )
        return {
            "adata": result_adata,
            "output": llm_output,
            "debug_output": debug_output,
            "stdout": stdout,
        }

    # All repair attempts failed — return structured diagnostic
    envelope = repair_result.final_envelope
    if envelope is not None:
        error_output = (
            f"ERROR: {envelope.exception}: {envelope.summary}\n\n"
            f"Phase: {envelope.phase}\n"
            f"Attempts: {envelope.retry_count + 1}/{repair_loop.max_retries + 1}\n"
            f"Traceback (last {len(envelope.traceback_excerpt)} chars):\n"
            f"{envelope.traceback_excerpt}"
        )
        if envelope.repair_hints:
            error_output += (
                "\nRepair hints:\n"
                + "\n".join(f"  - {h}" for h in envelope.repair_hints)
            )
    else:
        error_output = "ERROR: execution failed with no diagnostic envelope"

    debug_error_output = error_output
    if prereq_warnings:
        error_output = (
            f"PREREQUISITE WARNINGS: {prereq_warnings}\n\n"
            f"{error_output}"
        )
        debug_error_output = (
            f"PREREQUISITE WARNINGS: {prereq_warnings}\n\n"
            f"{debug_error_output}"
        )
    return {
        "adata": adata,
        "output": error_output,
        "debug_output": debug_error_output,
        "stdout": "",
    }


def handle_run_snippet(executor: "AnalysisExecutor", code: str, adata: Any) -> str:
    """Read-only snippet — no adata copy, no serialisation round-trip."""
    try:
        return executor.execute_snippet_readonly(code, adata)
    except Exception as e:
        return f"ERROR: {e}"


# ------------------------------------------------------------------
# Function search
# ------------------------------------------------------------------

def handle_search_functions(ctx: "AgentContext", query: str) -> str:
    """Search the OmicVerse function registry (runtime + static)."""
    query = (query or "").strip()
    if not query:
        return "Please provide a non-empty function search query."

    matches: List[Dict[str, Any]] = []
    seen_full_names: set[str] = set()

    def _extend(entries: List[Dict[str, Any]]) -> None:
        for entry in entries or []:
            full_name = str(
                entry.get("full_name")
                or entry.get("short_name")
                or entry.get("name")
                or ""
            )
            if not full_name or full_name in seen_full_names:
                continue
            seen_full_names.add(full_name)
            matches.append(entry)

    try:
        runtime_matches = list(_global_registry.find(query))
    except Exception:
        runtime_matches = []
    _extend(runtime_matches)

    static_search = getattr(ctx, "_collect_static_registry_entries", None)
    if callable(static_search):
        try:
            static_matches = list(static_search(query, max_entries=20))
        except TypeError:
            static_matches = list(static_search(query))
        except Exception:
            static_matches = []
        _extend(static_matches)

    if not matches:
        return f"No functions found matching '{query}'. Try broader keywords."

    results: List[str] = []
    for m in matches[:20]:
        fname = m.get("full_name", m.get("short_name", ""))
        sig = m.get("signature", "")
        desc = m.get("description", "")[:300]

        entry_text = f"  {fname}({sig})\n    {desc}"

        branch_parameter = m.get("branch_parameter")
        branch_value = m.get("branch_value")
        if branch_parameter and branch_value:
            entry_text += f"\n    Branch: {branch_parameter}='{branch_value}'"

        prereqs = m.get("prerequisites", {})
        req_funcs = prereqs.get("functions", [])
        if req_funcs:
            entry_text += "\n    Must run first: " + ", ".join(req_funcs)

        requires = m.get("requires", {})
        if requires:
            req_items = [
                f"{k}['{v}']"
                for k, vals in requires.items()
                for v in vals
            ]
            entry_text += "\n    Requires: " + ", ".join(req_items)

        produces = m.get("produces", {})
        if produces:
            prod_items = [
                f"{k}['{v}']"
                for k, vals in produces.items()
                for v in vals
            ]
            entry_text += "\n    Produces: " + ", ".join(prod_items)

        examples = m.get("examples", [])
        code_examples = [
            ex
            for ex in examples
            if ex.strip().startswith(("ov.", "sc."))
        ]
        if code_examples:
            entry_text += "\n    Example: " + code_examples[0]
        elif examples:
            entry_text += "\n    Example: " + examples[0]

        results.append(entry_text)
        if len(results) >= 10:
            break

    return (
        f"Found {len(results)} matching functions:\n"
        + "\n".join(results)
    )


# ------------------------------------------------------------------
# Tool search (deferred loading)
# ------------------------------------------------------------------

def handle_tool_search(
    ctx: "AgentContext", query: str, max_results: int = 5
) -> str:
    """Resolve a tool search query and load selected tools into the session."""
    session_id = ctx._get_runtime_session_id()
    payload = resolve_tool_search(
        query,
        loaded_tools=runtime_state.get_loaded_tools(session_id),
        max_results=max_results,
    )
    selected = payload.get("selected_tools", [])
    loaded_tools: tuple = ()
    if selected:
        loaded_tools = runtime_state.load_tools(session_id, selected)
    payload["loaded_tools"] = list(
        runtime_state.get_loaded_tools(session_id)
    )
    payload["newly_loaded"] = list(loaded_tools)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# Bash (foreground + background)
# ------------------------------------------------------------------

def background_bash_worker(
    ctx: "AgentContext",
    task_id: str,
    *,
    command: str,
    cwd: str,
    timeout_ms: int,
) -> None:
    """Stream output from a background bash process into runtime state."""
    session_id = ctx._get_runtime_session_id()
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime_state.bind_process(session_id, task_id, proc)
    assert proc.stdout is not None
    start = time.time()
    for line in proc.stdout:
        runtime_state.append_task_output(
            session_id, task_id, line.rstrip("\n")
        )
        if time.time() - start > timeout_ms / 1000:
            proc.terminate()
            runtime_state.update_task(
                session_id,
                task_id,
                status="failed",
                summary="Background command timed out",
            )
            return
    return_code = proc.wait()
    status = "completed" if return_code == 0 else "failed"
    runtime_state.update_task(
        session_id,
        task_id,
        status=status,
        summary=f"Exit code {return_code}",
    )


def handle_bash(
    ctx: "AgentContext",
    tool_blocked_fn: Callable[[str], bool],
    command: str,
    description: str = "",
    timeout: int = 120000,
    run_in_background: bool = False,
    dangerouslyDisableSandbox: bool = False,
) -> str:
    """Execute a bash command (foreground or background)."""
    ctx._ensure_server_tool_mode("Bash")
    if tool_blocked_fn("Bash"):
        return "Bash is blocked while the session is in plan mode."
    reason = description or f"Run shell command: {command[:120]}"
    ctx._request_tool_approval(
        "Bash",
        reason=reason,
        payload={
            "command": command,
            "dangerouslyDisableSandbox": dangerouslyDisableSandbox,
        },
    )
    cwd = ctx._refresh_runtime_working_directory()
    if run_in_background:
        task = runtime_state.create_task(
            ctx._get_runtime_session_id(),
            title=description or command[:80],
            description=command,
            kind="background_command",
            status="in_progress",
            background=True,
            tool_name="Bash",
            metadata={"cwd": cwd, "command": command},
        )
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        runtime_state.attach_background_process(
            ctx._get_runtime_session_id(),
            task.task_id,
            proc,
        )
        return json.dumps(
            {
                "task_id": task.task_id,
                "status": "in_progress",
                "cwd": cwd,
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        )

    proc_result = subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout)) / 1000,
        check=False,
    )
    output = (proc_result.stdout or "") + (
        (proc_result.stderr or "") if proc_result.stderr else ""
    )
    return json.dumps(
        {
            "cwd": cwd,
            "command": command,
            "returncode": proc_result.returncode,
            "stdout": proc_result.stdout,
            "stderr": proc_result.stderr,
            "output": output.strip(),
        },
        ensure_ascii=False,
        indent=2,
    )


# ------------------------------------------------------------------
# Agent delegation
# ------------------------------------------------------------------

async def handle_agent(
    subagent_controller: Any,
    agent_type: str,
    task: str,
    adata: Any,
    context: str = "",
) -> Any:
    """Delegate to a subagent (explore, plan, or execute)."""
    logger.info(
        "delegation_started agent_type=%s task=%s",
        agent_type,
        task[:80],
    )
    sub_result = await subagent_controller.run_subagent(
        agent_type=agent_type,
        task=task,
        adata=adata,
        context=context,
    )
    if agent_type == "execute":
        return {
            "adata": sub_result["adata"],
            "output": sub_result["result"],
        }
    return sub_result["result"]


# ------------------------------------------------------------------
# Terminal
# ------------------------------------------------------------------

def handle_finish(summary: str) -> dict:
    """Return the terminal finish payload."""
    return {"finished": True, "summary": summary}
