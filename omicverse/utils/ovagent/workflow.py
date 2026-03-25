"""Repo-owned workflow contract for OVAgent analysis runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


ALLOWED_WORKFLOW_DOMAINS = {"data-science", "bioinformatics"}


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    return raw.strip("\"'")


def _parse_front_matter_minimal(front_matter: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in front_matter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            parsed.setdefault(current_key, []).append(_parse_scalar(stripped[2:]))
            continue
        if ":" not in line:
            current_key = None
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            parsed[key] = []
            current_key = key
            continue
        parsed[key] = _parse_scalar(value)
        current_key = None
    return parsed


def _parse_front_matter(front_matter: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            payload = yaml.safe_load(front_matter) or {}
            if isinstance(payload, dict):
                return payload
        except Exception as e:
            logger.debug("_parse_front_matter: yaml.safe_load failed (%s), falling back to minimal parser", e)
    return _parse_front_matter_minimal(front_matter)


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return {}, text
    front_matter = match.group(1)
    body = match.group(2)
    return _parse_front_matter(front_matter), body.strip()


@dataclass(frozen=True)
class WorkflowConfig:
    domain: str = "bioinformatics"
    default_tools: list[str] = field(default_factory=list)
    approval_policy: str = "guarded"
    max_turns: int = 15
    execution_mode: str = "notebook_preferred"
    required_artifacts: list[str] = field(default_factory=lambda: ["summary.md", "bundle.json", "trace_linkage"])
    validation_commands: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    compaction_policy: str = ""

    @classmethod
    def from_mapping(cls, payload: Optional[dict[str, Any]] = None) -> "WorkflowConfig":
        data = dict(payload or {})
        return cls(
            domain=str(data.get("domain") or "bioinformatics"),
            default_tools=_coerce_list(data.get("default_tools")),
            approval_policy=str(data.get("approval_policy") or "guarded"),
            max_turns=int(data.get("max_turns") or 15),
            execution_mode=str(data.get("execution_mode") or "notebook_preferred"),
            required_artifacts=_coerce_list(data.get("required_artifacts") or ["summary.md", "bundle.json", "trace_linkage"]),
            validation_commands=_coerce_list(data.get("validation_commands")),
            completion_criteria=_coerce_list(data.get("completion_criteria")),
            compaction_policy=str(data.get("compaction_policy") or ""),
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.domain not in ALLOWED_WORKFLOW_DOMAINS:
            issues.append(
                f"domain must be one of {sorted(ALLOWED_WORKFLOW_DOMAINS)}, got '{self.domain}'"
            )
        if self.max_turns < 1:
            issues.append("max_turns must be >= 1")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowDocument:
    path: Path
    config: WorkflowConfig
    body: str
    raw_text: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "sha256": self.sha256,
            "config": self.config.to_dict(),
            "body": self.body,
        }

    def build_prompt_block(self) -> str:
        lines = [
            "REPOSITORY WORKFLOW POLICY:",
            f"- Domain: {self.config.domain}",
            f"- Approval policy: {self.config.approval_policy}",
            f"- Max turns: {self.config.max_turns}",
            f"- Execution mode: {self.config.execution_mode}",
        ]
        if self.config.default_tools:
            lines.append("- Default tools: " + ", ".join(self.config.default_tools))
        if self.config.required_artifacts:
            lines.append("- Required artifacts: " + ", ".join(self.config.required_artifacts))
        if self.config.completion_criteria:
            lines.append("- Completion criteria:")
            lines.extend(f"  - {item}" for item in self.config.completion_criteria)
        if self.body:
            lines.append("")
            lines.append(self.body.strip())
        return "\n".join(lines).strip()


def resolve_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def load_workflow_document(
    repo_root: Optional[Path] = None,
    workflow_path: Optional[Path] = None,
) -> WorkflowDocument:
    if workflow_path is not None:
        path = workflow_path.expanduser().resolve()
    else:
        root = repo_root or resolve_repo_root()
        if root is None:
            root = Path.cwd()
        path = root / "WORKFLOW.md"
    if not path.exists():
        return WorkflowDocument(path=path, config=WorkflowConfig(), body="", raw_text="")
    raw_text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw_text)
    config = WorkflowConfig.from_mapping(front_matter)
    return WorkflowDocument(path=path, config=config, body=body, raw_text=raw_text)
