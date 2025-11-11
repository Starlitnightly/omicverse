"""
Data structures for validation results.

This module defines the data structures used throughout the DataStateInspector
system to represent validation results, data checks, and suggestions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal
from enum import Enum


class ComplexityLevel(Enum):
    """Workflow complexity classification."""
    LOW = "low"  # 0-1 missing prerequisites
    MEDIUM = "medium"  # 2-3 missing prerequisites
    HIGH = "high"  # 4+ missing prerequisites OR includes qc/preprocess


@dataclass
class ObsCheckResult:
    """Result of checking adata.obs columns."""

    is_valid: bool
    required_columns: List[str]
    missing_columns: List[str]
    present_columns: List[str]
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"obs check: ✓ All {len(self.required_columns)} columns present"
        return f"obs check: ✗ Missing {len(self.missing_columns)} columns: {self.missing_columns}"


@dataclass
class ObsmCheckResult:
    """Result of checking adata.obsm keys."""

    is_valid: bool
    required_keys: List[str]
    missing_keys: List[str]
    present_keys: List[str]
    shape_info: Dict[str, tuple] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"obsm check: ✓ All {len(self.required_keys)} keys present"
        return f"obsm check: ✗ Missing {len(self.missing_keys)} keys: {self.missing_keys}"


@dataclass
class ObspCheckResult:
    """Result of checking adata.obsp keys."""

    is_valid: bool
    required_keys: List[str]
    missing_keys: List[str]
    present_keys: List[str]
    is_sparse: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"obsp check: ✓ All {len(self.required_keys)} keys present"
        return f"obsp check: ✗ Missing {len(self.missing_keys)} keys: {self.missing_keys}"


@dataclass
class UnsCheckResult:
    """Result of checking adata.uns keys."""

    is_valid: bool
    required_keys: List[str]
    missing_keys: List[str]
    present_keys: List[str]
    nested_structure: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"uns check: ✓ All {len(self.required_keys)} keys present"
        return f"uns check: ✗ Missing {len(self.missing_keys)} keys: {self.missing_keys}"


@dataclass
class LayersCheckResult:
    """Result of checking adata.layers keys."""

    is_valid: bool
    required_keys: List[str]
    missing_keys: List[str]
    present_keys: List[str]
    shape_info: Dict[str, tuple] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"layers check: ✓ All {len(self.required_keys)} keys present"
        return f"layers check: ✗ Missing {len(self.missing_keys)} keys: {self.missing_keys}"


@dataclass
class DataCheckResult:
    """Comprehensive result of all data structure checks."""

    is_valid: bool
    obs_result: Optional[ObsCheckResult] = None
    obsm_result: Optional[ObsmCheckResult] = None
    obsp_result: Optional[ObspCheckResult] = None
    uns_result: Optional[UnsCheckResult] = None
    layers_result: Optional[LayersCheckResult] = None

    @property
    def all_missing_structures(self) -> Dict[str, List[str]]:
        """Get all missing structures across all categories."""
        missing = {}

        if self.obs_result and self.obs_result.missing_columns:
            missing['obs'] = self.obs_result.missing_columns
        if self.obsm_result and self.obsm_result.missing_keys:
            missing['obsm'] = self.obsm_result.missing_keys
        if self.obsp_result and self.obsp_result.missing_keys:
            missing['obsp'] = self.obsp_result.missing_keys
        if self.uns_result and self.uns_result.missing_keys:
            missing['uns'] = self.uns_result.missing_keys
        if self.layers_result and self.layers_result.missing_keys:
            missing['layers'] = self.layers_result.missing_keys

        return missing

    def __str__(self) -> str:
        if self.is_valid:
            return "Data check: ✓ All required structures present"

        missing = self.all_missing_structures
        parts = []
        for category, keys in missing.items():
            parts.append(f"{category}: {keys}")

        return f"Data check: ✗ Missing structures - {', '.join(parts)}"


@dataclass
class Suggestion:
    """A suggested fix for missing prerequisites."""

    priority: Literal['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    suggestion_type: Literal['direct_fix', 'workflow_guidance', 'alternative', 'optimization']

    description: str
    code: str
    explanation: str

    estimated_time: str
    estimated_time_seconds: Optional[int] = None

    prerequisites: List[str] = field(default_factory=list)
    impact: str = ""

    auto_executable: bool = False

    def __str__(self) -> str:
        return f"[{self.priority}] {self.description}\n  Code: {self.code}\n  {self.explanation}"


@dataclass
class ValidationResult:
    """Result of prerequisite validation for a function."""

    function_name: str
    is_valid: bool
    message: str

    # Missing components
    missing_prerequisites: List[str] = field(default_factory=list)
    missing_data_structures: Dict[str, List[str]] = field(default_factory=dict)

    # Current state
    executed_functions: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Suggestions
    suggestions: List[Suggestion] = field(default_factory=list)

    # Data check details
    data_check_result: Optional[DataCheckResult] = None

    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        if self.is_valid:
            return f"✓ {self.function_name}: {self.message}"

        lines = [f"✗ {self.function_name}: {self.message}"]

        if self.missing_prerequisites:
            lines.append(f"  Missing prerequisites: {', '.join(self.missing_prerequisites)}")

        if self.missing_data_structures:
            for category, keys in self.missing_data_structures.items():
                lines.append(f"  Missing {category}: {', '.join(keys)}")

        if self.suggestions:
            lines.append(f"  Suggestions: {len(self.suggestions)} available")

        return '\n'.join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a dictionary summary suitable for LLM consumption."""
        return {
            'function': self.function_name,
            'valid': self.is_valid,
            'message': self.message,
            'missing_prerequisites': self.missing_prerequisites,
            'missing_data_structures': self.missing_data_structures,
            'suggestions': [
                {
                    'priority': s.priority,
                    'type': s.suggestion_type,
                    'description': s.description,
                    'code': s.code,
                    'auto_executable': s.auto_executable,
                }
                for s in self.suggestions
            ],
            'timestamp': self.validation_timestamp.isoformat(),
        }


@dataclass
class ExecutionEvidence:
    """Evidence that a function was executed."""

    function_name: str
    confidence: float  # 0.0 to 1.0
    evidence_type: Literal['metadata_marker', 'output_signature', 'distribution_pattern', 'distribution_analysis']

    # Additional fields for detailed evidence
    location: str = ''  # Where the evidence was found (e.g., 'adata.uns["pca"]')
    description: str = ''  # Human-readable description of the evidence

    evidence_details: Dict[str, Any] = field(default_factory=dict)
    detected_outputs: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    def __str__(self) -> str:
        return f"{self.function_name}: {self.confidence:.2f} confidence ({self.evidence_type})"


@dataclass
class ExecutionState:
    """State of detected function executions."""

    detected_functions: Dict[str, ExecutionEvidence] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    confidence_summary: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        high_confidence = [
            f for f, conf in self.confidence_summary.items()
            if conf >= 0.7
        ]
        return f"Execution state: {len(high_confidence)} functions detected with high confidence"
