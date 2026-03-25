"""ExecutionRepairLoop — structured self-healing for code execution failures.

Replaces the former regex-first recovery path with a normalized failure
envelope and bounded retry loop.  Domain-specific regex transforms
(``ProactiveCodeTransformer``, ``apply_execution_error_fix``) are still
applied as *optional early guardrails*, but they no longer own the primary
repair path.  When guardrails cannot fix the code, the loop delegates to
LLM-guided diagnosis with a consistent structured diagnostic payload.

Failure envelope contract
-------------------------
Every execution failure is represented as a :class:`FailureEnvelope` that
carries:

* **phase** — which pipeline stage failed (``"execution"``, ``"transform"``,
  ``"validation"``, …)
* **exception** — the exception type name
* **summary** — a concise human-readable error summary
* **traceback_excerpt** — tail of the traceback (bounded length)
* **retry_count** — how many repair attempts have been made so far
* **repair_hints** — list of domain-specific hints (from guardrail matches)
* **retry_safe** — whether an automatic retry is considered safe
"""

from __future__ import annotations

import logging
import traceback as _traceback_mod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .analysis_executor import AnalysisExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure envelope
# ---------------------------------------------------------------------------

_MAX_TRACEBACK_CHARS = 2000


@dataclass
class FailureEnvelope:
    """Normalized diagnostic payload for a single execution failure.

    Conforms to the interface contract:
    phase + exception + summary + traceback excerpt + retry count +
    repair hints + retry safety flag.
    """

    phase: str
    exception: str
    summary: str
    traceback_excerpt: str
    retry_count: int
    repair_hints: List[str] = field(default_factory=list)
    retry_safe: bool = True

    # Optional rich context attached before LLM diagnosis
    code: str = ""
    dataset_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for logging / LLM prompt injection."""
        return {
            "phase": self.phase,
            "exception": self.exception,
            "summary": self.summary,
            "traceback_excerpt": self.traceback_excerpt,
            "retry_count": self.retry_count,
            "repair_hints": list(self.repair_hints),
            "retry_safe": self.retry_safe,
            "code": self.code,
            "dataset_context": self.dataset_context,
        }

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        phase: str = "execution",
        retry_count: int = 0,
        code: str = "",
        dataset_context: str = "",
        repair_hints: Optional[List[str]] = None,
        retry_safe: bool = True,
    ) -> "FailureEnvelope":
        """Build an envelope from a caught exception."""
        tb = _traceback_mod.format_exception(type(exc), exc, exc.__traceback__)
        tb_text = "".join(tb)
        return cls(
            phase=phase,
            exception=type(exc).__name__,
            summary=str(exc)[:500],
            traceback_excerpt=tb_text[-_MAX_TRACEBACK_CHARS:],
            retry_count=retry_count,
            repair_hints=list(repair_hints or []),
            retry_safe=retry_safe,
            code=code,
            dataset_context=dataset_context,
        )


# ---------------------------------------------------------------------------
# Repair attempt result
# ---------------------------------------------------------------------------


@dataclass
class RepairAttempt:
    """Record of a single repair attempt within the loop."""

    attempt: int
    strategy: str  # "guardrail" | "llm" | "passthrough"
    code: str
    success: bool
    envelope: Optional[FailureEnvelope] = None


# ---------------------------------------------------------------------------
# Repair result
# ---------------------------------------------------------------------------


@dataclass
class RepairResult:
    """Outcome of the full repair loop."""

    success: bool
    final_code: str
    exec_result: Any = None
    attempts: List[RepairAttempt] = field(default_factory=list)
    final_envelope: Optional[FailureEnvelope] = None


# ---------------------------------------------------------------------------
# Dataset context builder
# ---------------------------------------------------------------------------


def build_dataset_context(adata: Any) -> str:
    """Build a short dataset description for LLM diagnosis prompts."""
    if adata is None:
        return ""
    parts: List[str] = []
    if hasattr(adata, "shape"):
        parts.append(f"Shape: {adata.shape[0]} obs x {adata.shape[1]} vars")
    if hasattr(adata, "obs") and hasattr(adata.obs, "columns"):
        cols = list(adata.obs.columns[:20])
        parts.append(f"obs columns: {cols}")
    if hasattr(adata, "var") and hasattr(adata.var, "columns"):
        vcols = list(adata.var.columns[:10])
        parts.append(f"var columns: {vcols}")
    if hasattr(adata, "obsm") and hasattr(adata.obsm, "keys"):
        try:
            parts.append(f"obsm keys: {list(adata.obsm.keys())[:10]}")
        except Exception:
            pass
    return "; ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# LLM repair prompt builder
# ---------------------------------------------------------------------------


def build_llm_repair_prompt(envelope: FailureEnvelope) -> str:
    """Build a structured prompt for LLM-guided code repair.

    The prompt provides code + error + traceback + dataset context in a
    consistent format so the LLM can diagnose and fix the issue.
    """
    sections = [
        "The following OmicVerse agent-generated Python code failed during execution.",
        "",
        f"--- PHASE ---\n{envelope.phase}",
        "",
        f"--- EXCEPTION ---\n{envelope.exception}: {envelope.summary}",
        "",
        f"--- TRACEBACK ---\n{envelope.traceback_excerpt}",
        "",
        f"--- CODE ---\n{envelope.code}",
    ]

    if envelope.dataset_context:
        sections.extend(["", f"--- DATASET ---\n{envelope.dataset_context}"])

    if envelope.repair_hints:
        hints_text = "\n".join(f"  - {h}" for h in envelope.repair_hints)
        sections.extend(["", f"--- REPAIR HINTS ---\n{hints_text}"])

    sections.extend([
        "",
        f"--- RETRY COUNT ---\n{envelope.retry_count}",
        "",
        "Your task:",
        "1. Diagnose the root cause of the error.",
        "2. Generate a CORRECTED version of the full code that fixes the issue.",
        "3. Wrap the corrected code in ```python ... ``` markers.",
        "",
        "Important rules:",
        "- Fix ONLY the error. Do not change logic that already works.",
        "- If a variable is undefined, define it or remove the reference.",
        "- If a module is unavailable, use an alternative or add a try/except.",
        "- Preserve all file output operations (savefig, to_csv, json.dump, etc.).",
    ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# ExecutionRepairLoop
# ---------------------------------------------------------------------------

# Default maximum number of repair attempts before giving up.
DEFAULT_MAX_RETRIES = 3


class ExecutionRepairLoop:
    """Bounded self-healing loop for code execution failures.

    The loop runs up to *max_retries* repair attempts using this strategy
    order:

    1. **Guardrail pass** — ``AnalysisExecutor.apply_execution_error_fix``
       applies fast regex-based transforms.  If the guardrail produces
       different code, re-execute immediately.  This is an *optional* early
       stage; if it does not match, the loop falls through.
    2. **LLM diagnosis** — build a structured :class:`FailureEnvelope`,
       call the LLM with a consistent diagnostic prompt, extract repaired
       code, and re-execute.
    3. If all attempts are exhausted, return the final
       :class:`FailureEnvelope` to the caller.

    Parameters
    ----------
    executor : AnalysisExecutor
        Provides ``apply_execution_error_fix``, ``diagnose_error_with_llm``,
        and ``execute_generated_code``.
    max_retries : int
        Upper bound on repair attempts (default 3).
    """

    def __init__(
        self,
        executor: "AnalysisExecutor",
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._executor = executor
        self._max_retries = max_retries

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def run(
        self,
        code: str,
        adata: Any,
        *,
        exec_fn: Optional[Callable[..., Any]] = None,
        extract_code_fn: Optional[Callable[[str], str]] = None,
        phase: str = "execution",
    ) -> RepairResult:
        """Execute *code* and attempt bounded self-healing on failure.

        Parameters
        ----------
        code : str
            The code to execute.
        adata : Any
            The dataset object (typically AnnData).
        exec_fn : callable, optional
            ``(code, adata) -> result_dict``.  Defaults to
            ``executor.execute_generated_code(code, adata, capture_stdout=True)``.
        extract_code_fn : callable, optional
            ``(llm_response) -> code_str``.  Used to extract Python from an
            LLM repair response.  Defaults to
            ``executor._ctx._extract_python_code``.
        phase : str
            Label for the pipeline phase (used in envelopes).

        Returns
        -------
        RepairResult
            Contains success flag, final code, exec result, list of
            attempts, and (on failure) the final envelope.
        """
        if exec_fn is None:
            def exec_fn(c, a):
                return self._executor.execute_generated_code(
                    c, a, capture_stdout=True
                )

        if extract_code_fn is None:
            ctx = self._executor._ctx
            extract_code_fn = getattr(ctx, "_extract_python_code", None)

        attempts: List[RepairAttempt] = []
        current_code = code
        dataset_ctx = build_dataset_context(adata)

        for attempt_num in range(self._max_retries + 1):
            # --- Execute current code ---
            try:
                result = exec_fn(current_code, adata)
                attempts.append(RepairAttempt(
                    attempt=attempt_num,
                    strategy="passthrough" if attempt_num == 0 else attempts[-1].strategy if attempts else "passthrough",
                    code=current_code,
                    success=True,
                ))
                return RepairResult(
                    success=True,
                    final_code=current_code,
                    exec_result=result,
                    attempts=attempts,
                )
            except Exception as exc:
                envelope = FailureEnvelope.from_exception(
                    exc,
                    phase=phase,
                    retry_count=attempt_num,
                    code=current_code,
                    dataset_context=dataset_ctx,
                )

                # Exhausted all retries?
                if attempt_num >= self._max_retries:
                    attempts.append(RepairAttempt(
                        attempt=attempt_num,
                        strategy="exhausted",
                        code=current_code,
                        success=False,
                        envelope=envelope,
                    ))
                    return RepairResult(
                        success=False,
                        final_code=current_code,
                        attempts=attempts,
                        final_envelope=envelope,
                    )

                logger.info(
                    "repair_loop attempt=%d/%d phase=%s exception=%s",
                    attempt_num + 1,
                    self._max_retries,
                    phase,
                    envelope.exception,
                )

                # --- Stage 1: Guardrail (regex) pass ---
                guardrail_code = self._executor.apply_execution_error_fix(
                    current_code, str(exc)
                )
                if guardrail_code and guardrail_code != current_code:
                    envelope.repair_hints.append(
                        "guardrail: regex transform produced different code"
                    )
                    attempts.append(RepairAttempt(
                        attempt=attempt_num,
                        strategy="guardrail",
                        code=current_code,
                        success=False,
                        envelope=envelope,
                    ))
                    current_code = guardrail_code
                    continue

                # --- Stage 2: LLM-guided repair ---
                llm_code = await self._try_llm_repair(
                    envelope, extract_code_fn
                )
                if llm_code and llm_code.strip() and llm_code != current_code:
                    envelope.repair_hints.append(
                        "llm: diagnosis produced replacement code"
                    )
                    attempts.append(RepairAttempt(
                        attempt=attempt_num,
                        strategy="llm",
                        code=current_code,
                        success=False,
                        envelope=envelope,
                    ))
                    current_code = llm_code
                    continue

                # Neither guardrail nor LLM could produce new code — mark
                # envelope as not retry-safe and break early.
                envelope.retry_safe = False
                envelope.repair_hints.append(
                    "no repair strategy produced new code"
                )
                attempts.append(RepairAttempt(
                    attempt=attempt_num,
                    strategy="none",
                    code=current_code,
                    success=False,
                    envelope=envelope,
                ))
                return RepairResult(
                    success=False,
                    final_code=current_code,
                    attempts=attempts,
                    final_envelope=envelope,
                )

        # Should not be reached, but guard against it
        return RepairResult(  # pragma: no cover
            success=False,
            final_code=current_code,
            attempts=attempts,
        )

    async def _try_llm_repair(
        self,
        envelope: FailureEnvelope,
        extract_code_fn: Optional[Callable[[str], str]],
    ) -> Optional[str]:
        """Attempt LLM-guided repair and return extracted code or None."""
        try:
            diagnosed = await self._executor.diagnose_error_with_llm(
                envelope.code,
                envelope.summary,
                envelope.traceback_excerpt,
                None,  # adata not passed to LLM diagnosis — context is in envelope
            )
            if diagnosed and extract_code_fn is not None:
                try:
                    return extract_code_fn(diagnosed)
                except Exception:
                    # extract_code_fn may itself fail; fall back to raw diagnosed
                    return diagnosed
            return diagnosed
        except Exception as exc:
            logger.debug("LLM repair failed: %s", exc)
            return None
