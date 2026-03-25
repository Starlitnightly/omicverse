"""RepairLoop — structured multi-attempt self-healing for execution failures.

Replaces the single regex-retry path in ``tool_runtime._tool_execute_code``
with a strategy-ordered repair loop that:

1. Normalizes raw exceptions into ``ExecutionFailureEnvelope`` objects.
2. Runs ordered ``RepairStrategy`` instances (regex pattern fix is a fallback,
   not the primary path).
3. Supports configurable max retries with per-attempt tracking.
4. Feeds structured failure context back to each strategy so the LLM (or
   any future strategy) can reason about the error.

The ``ProactiveCodeTransformer`` remains as a *pre-execution* guard and is
not part of the repair loop itself.
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .contracts import (
    ExecutionFailureEnvelope,
    FailurePhase,
    RepairHint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repair strategy protocol
# ---------------------------------------------------------------------------


class RepairStrategy(Protocol):
    """Interface for a single repair strategy.

    Each strategy receives a failure envelope and the original code, and
    returns either a repaired code string or ``None`` to signal that it
    cannot handle this failure.
    """

    @property
    def name(self) -> str: ...

    def attempt_repair(
        self,
        envelope: ExecutionFailureEnvelope,
        code: str,
        adata: Any,
    ) -> Optional[str]:
        """Return repaired code or ``None`` if this strategy cannot help."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class RegexRepairStrategy:
    """Wraps ``AnalysisExecutor.apply_execution_error_fix`` as a fallback.

    This is the *demoted* regex-first path — it now runs as a later-priority
    strategy instead of being the only recovery mechanism.
    """

    name = "regex_pattern_fix"

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def attempt_repair(
        self,
        envelope: ExecutionFailureEnvelope,
        code: str,
        adata: Any,
    ) -> Optional[str]:
        fixed = self._executor.apply_execution_error_fix(
            code, envelope.message,
        )
        if fixed and fixed != code:
            logger.debug(
                "RegexRepairStrategy: produced fix for %s",
                envelope.exception_type,
            )
            return fixed
        return None


class HintDrivenRepairStrategy:
    """Apply repair hints attached to the failure envelope.

    Repair hints are machine-readable suggestions produced during failure
    normalization (e.g. "add missing import", "remove assignment from
    in-place call").  This strategy applies simple, deterministic
    transformations based on hint metadata.
    """

    name = "hint_driven_repair"

    _HINT_HANDLERS: Dict[str, Callable[["HintDrivenRepairStrategy", RepairHint, str], Optional[str]]] = {}

    def attempt_repair(
        self,
        envelope: ExecutionFailureEnvelope,
        code: str,
        adata: Any,
    ) -> Optional[str]:
        for hint in envelope.repair_hints:
            handler = self._HINT_HANDLERS.get(hint.strategy)
            if handler is not None:
                result = handler(self, hint, code)
                if result is not None and result != code:
                    logger.debug(
                        "HintDrivenRepairStrategy: applied hint %s",
                        hint.strategy,
                    )
                    return result
        return None

    # -- hint handlers (registered via dict for extensibility) --

    def _handle_add_import(self, hint: RepairHint, code: str) -> Optional[str]:
        import_line = hint.metadata.get("import_line", "")
        if import_line and import_line not in code:
            return import_line + "\n" + code
        return None

    def _handle_rename_attribute(self, hint: RepairHint, code: str) -> Optional[str]:
        old = hint.metadata.get("old", "")
        new = hint.metadata.get("new", "")
        if old and new and old in code:
            return code.replace(old, new)
        return None

    def _handle_remove_assignment(self, hint: RepairHint, code: str) -> Optional[str]:
        pattern = hint.metadata.get("pattern", "")
        replacement = hint.metadata.get("replacement", "")
        if pattern:
            fixed = re.sub(pattern, replacement, code)
            if fixed != code:
                return fixed
        return None


# Register hint handlers
HintDrivenRepairStrategy._HINT_HANDLERS = {
    "add_import": HintDrivenRepairStrategy._handle_add_import,
    "rename_attribute": HintDrivenRepairStrategy._handle_rename_attribute,
    "remove_assignment": HintDrivenRepairStrategy._handle_remove_assignment,
}


# ---------------------------------------------------------------------------
# Failure normalization
# ---------------------------------------------------------------------------


def _classify_phase(exception: BaseException) -> FailurePhase:
    """Classify which execution phase the exception belongs to."""
    exc_type = type(exception).__name__
    msg = str(exception).lower()

    if "timeout" in msg or "timed out" in msg:
        return FailurePhase.TIMEOUT
    if "security" in msg or "sandbox" in msg or "permission" in msg:
        return FailurePhase.PRE_EXEC
    return FailurePhase.EXECUTION


def _generate_repair_hints(
    exception: BaseException, code: str
) -> List[RepairHint]:
    """Generate machine-readable repair hints from exception context."""
    hints: List[RepairHint] = []
    msg = str(exception)
    msg_lower = msg.lower()

    # Missing module → suggest auto-install
    mod_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
    if mod_match:
        pkg = mod_match.group(1).split(".")[0]
        hints.append(RepairHint(
            strategy="auto_install",
            description=f"Missing package '{pkg}' — auto-install may resolve",
            confidence=0.8,
            metadata={"package": pkg},
        ))

    # .dtype vs .dtypes
    if "has no attribute 'dtype'" in msg_lower or "'dtype'" in msg_lower:
        hints.append(RepairHint(
            strategy="rename_attribute",
            description="Use .dtypes instead of .dtype for DataFrames",
            confidence=0.9,
            metadata={"old": ".dtype", "new": ".dtypes"},
        ))

    # NoneType from in-place assignment
    if "'nonetype' object has no attribute" in msg_lower:
        inplace_funcs = [
            "pca", "scale", "neighbors", "leiden", "umap",
            "tsne", "sude", "scrublet", "mde",
        ]
        pattern = (
            r"adata\s*=\s*(ov\.pp\.(?:"
            + "|".join(inplace_funcs)
            + r")\s*\([^)]*\))"
        )
        if re.search(pattern, code):
            hints.append(RepairHint(
                strategy="remove_assignment",
                description="Remove assignment from in-place function call",
                confidence=0.85,
                metadata={"pattern": pattern, "replacement": r"\1"},
            ))

    # seurat_v3 LOESS error
    if any(kw in msg_lower for kw in ("extrapolation", "loess", "blending")):
        if "seurat_v3" in code:
            hints.append(RepairHint(
                strategy="rename_attribute",
                description="Use flavor='seurat' instead of 'seurat_v3'",
                confidence=0.7,
                metadata={
                    "old": "flavor='seurat_v3'",
                    "new": "flavor='seurat'",
                },
            ))

    # Categorical errors
    if "cannot setitem on a categorical" in msg_lower or (
        "nan" in msg_lower
        and ("batch" in msg_lower or "categorical" in msg_lower)
    ):
        hints.append(RepairHint(
            strategy="add_import",
            description="Add categorical batch column fix preamble",
            confidence=0.7,
            metadata={
                "import_line": (
                    "import pandas as pd\n"
                    "if 'batch' in adata.obs.columns:\n"
                    "    if pd.api.types.is_categorical_dtype(adata.obs['batch']):\n"
                    "        adata.obs['batch'] = adata.obs['batch'].astype(str)\n"
                    "    adata.obs['batch'] = adata.obs['batch'].fillna('unknown')\n"
                    "    adata.obs['batch'] = adata.obs['batch'].astype('category')"
                ),
            },
        ))

    # NameError → suggest the missing name
    if isinstance(exception, NameError):
        name_match = re.search(r"name '(\w+)' is not defined", msg)
        if name_match:
            hints.append(RepairHint(
                strategy="undefined_name",
                description=f"Name '{name_match.group(1)}' is not defined",
                confidence=0.5,
                metadata={"name": name_match.group(1)},
            ))

    return hints


def normalize_execution_failure(
    *,
    tool_name: str,
    exception: BaseException,
    code: str,
    retry_count: int = 0,
    max_retries: int = 3,
) -> ExecutionFailureEnvelope:
    """Build a structured ``ExecutionFailureEnvelope`` from a raw exception.

    This is the single entry point for converting arbitrary execution errors
    into the structured format that the repair loop and LLM can reason about.
    """
    tb_str = traceback.format_exception(
        type(exception), exception, exception.__traceback__
    )
    tb_summary = "".join(tb_str)

    # Extract stderr-like content (last N chars of traceback)
    stderr_summary = tb_summary[-1500:] if len(tb_summary) > 1500 else tb_summary

    return ExecutionFailureEnvelope(
        tool_name=tool_name,
        phase=_classify_phase(exception),
        exception_type=type(exception).__name__,
        message=str(exception),
        stderr_summary=stderr_summary,
        traceback_summary=tb_summary[-2000:],
        retry_count=retry_count,
        max_retries=max_retries,
        repair_hints=_generate_repair_hints(exception, code),
    )


# ---------------------------------------------------------------------------
# Repair result
# ---------------------------------------------------------------------------


@dataclass
class RepairAttempt:
    """Record of a single repair attempt."""
    strategy_name: str
    succeeded: bool
    repaired_code: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RepairResult:
    """Outcome of the full repair loop."""
    success: bool = False
    final_code: Optional[str] = None
    final_output: Optional[str] = None
    final_adata: Any = None
    attempts: List[RepairAttempt] = field(default_factory=list)
    final_envelope: Optional[ExecutionFailureEnvelope] = None

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    @property
    def winning_strategy(self) -> Optional[str]:
        for attempt in self.attempts:
            if attempt.succeeded:
                return attempt.strategy_name
        return None


# ---------------------------------------------------------------------------
# RepairLoop
# ---------------------------------------------------------------------------


class RepairLoop:
    """Multi-attempt structured self-healing loop for execution failures.

    The loop tries each strategy in order for each retry attempt.  Strategies
    are ordered so that structured/hint-driven repairs run before the legacy
    regex fallback.  The proactive code transformer is *not* part of this
    loop — it runs before execution as a pre-guard.

    Parameters
    ----------
    strategies : sequence of RepairStrategy
        Ordered list of strategies to try.  First match wins per attempt.
    executor_fn : callable
        ``(code, adata) -> result_dict`` — the sandbox execution function.
    max_retries : int
        Maximum number of repair attempts (default 3).
    """

    def __init__(
        self,
        strategies: Sequence[RepairStrategy],
        executor_fn: Callable[..., dict],
        max_retries: int = 3,
    ) -> None:
        self._strategies = list(strategies)
        self._executor_fn = executor_fn
        self._max_retries = max_retries

    @property
    def strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]

    def run(
        self,
        *,
        code: str,
        adata: Any,
        initial_exception: BaseException,
        tool_name: str = "execute_code",
    ) -> RepairResult:
        """Execute the repair loop.

        Tries each strategy in order.  If a strategy produces repaired code,
        re-executes it.  If re-execution succeeds, returns success.  If it
        fails again, normalizes the new failure and continues to the next
        attempt (cycling through strategies again with updated context).

        Returns a ``RepairResult`` with the outcome.
        """
        result = RepairResult()
        current_code = code
        current_exception = initial_exception

        for attempt_idx in range(self._max_retries):
            envelope = normalize_execution_failure(
                tool_name=tool_name,
                exception=current_exception,
                code=current_code,
                retry_count=attempt_idx,
                max_retries=self._max_retries,
            )
            result.final_envelope = envelope

            logger.info(
                "repair_loop attempt=%d/%d phase=%s exception=%s hints=%d",
                attempt_idx + 1,
                self._max_retries,
                envelope.phase.value,
                envelope.exception_type,
                len(envelope.repair_hints),
            )

            repaired_code = None
            strategy_name = ""

            for strategy in self._strategies:
                try:
                    candidate = strategy.attempt_repair(
                        envelope, current_code, adata,
                    )
                except Exception as strat_exc:
                    logger.warning(
                        "repair_strategy_error strategy=%s error=%s",
                        strategy.name,
                        strat_exc,
                    )
                    result.attempts.append(RepairAttempt(
                        strategy_name=strategy.name,
                        succeeded=False,
                        error=str(strat_exc),
                    ))
                    continue

                if candidate is not None and candidate != current_code:
                    repaired_code = candidate
                    strategy_name = strategy.name
                    break

            if repaired_code is None:
                # No strategy could produce a fix
                logger.info(
                    "repair_loop no_strategy_matched attempt=%d",
                    attempt_idx + 1,
                )
                result.attempts.append(RepairAttempt(
                    strategy_name="none",
                    succeeded=False,
                    error="no strategy produced a repair",
                ))
                break

            # Try executing the repaired code
            try:
                exec_result = self._executor_fn(repaired_code, adata)
                result.attempts.append(RepairAttempt(
                    strategy_name=strategy_name,
                    succeeded=True,
                    repaired_code=repaired_code,
                ))
                result.success = True
                result.final_code = repaired_code
                result.final_output = exec_result.get("output", "")
                result.final_adata = exec_result.get("adata", adata)
                logger.info(
                    "repair_loop success attempt=%d strategy=%s",
                    attempt_idx + 1,
                    strategy_name,
                )
                return result
            except Exception as retry_exc:
                logger.info(
                    "repair_loop retry_failed attempt=%d strategy=%s "
                    "error=%s",
                    attempt_idx + 1,
                    strategy_name,
                    retry_exc,
                )
                result.attempts.append(RepairAttempt(
                    strategy_name=strategy_name,
                    succeeded=False,
                    repaired_code=repaired_code,
                    error=str(retry_exc),
                ))
                current_code = repaired_code
                current_exception = retry_exc

        result.success = False
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_repair_loop(
    executor: Any,
    executor_fn: Callable[..., dict],
    max_retries: int = 3,
) -> RepairLoop:
    """Build a ``RepairLoop`` with the default strategy ordering.

    Strategy order (structured first, regex fallback last):
    1. HintDrivenRepairStrategy — deterministic fixes from failure hints
    2. RegexRepairStrategy — legacy pattern-based fixes (demoted)
    """
    strategies: List[RepairStrategy] = [
        HintDrivenRepairStrategy(),
        RegexRepairStrategy(executor),
    ]
    return RepairLoop(
        strategies=strategies,
        executor_fn=executor_fn,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "RepairStrategy",
    "RegexRepairStrategy",
    "HintDrivenRepairStrategy",
    "RepairAttempt",
    "RepairResult",
    "RepairLoop",
    "build_default_repair_loop",
    "normalize_execution_failure",
]
