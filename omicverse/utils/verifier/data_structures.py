"""
Data structures for the OmicVerse Skills Verifier.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SkillDescription:
    """
    Minimal skill information using progressive disclosure approach.

    Only name and description are loaded at startup (like Claude Code),
    full SKILL.md content is loaded on-demand.
    """
    name: str
    description: str

    def __post_init__(self):
        """Validate that name and description are non-empty."""
        if not self.name or not self.name.strip():
            raise ValueError("Skill name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Skill description cannot be empty")


@dataclass
class NotebookTask:
    """
    A task extracted from a tutorial notebook.

    Represents a natural language task description that should trigger
    specific skill selections when given to an LLM.
    """
    task_id: str
    notebook_path: str
    task_description: str
    expected_skills: List[str]
    expected_order: List[str]
    category: str  # e.g., "bulk", "single-cell", "spatial"
    difficulty: str = "single"  # "single", "workflow", "complex", "ambiguous"
    context: Dict[str, Any] = field(default_factory=dict)
    alternate_acceptable: List[str] = field(default_factory=list)  # For ambiguous cases

    def __post_init__(self):
        """Validate task fields."""
        if not self.task_id or not self.task_description:
            raise ValueError("task_id and task_description are required")
        if not self.expected_skills:
            raise ValueError("expected_skills cannot be empty")
        if not self.expected_order:
            # Default to expected_skills order if not specified
            self.expected_order = self.expected_skills.copy()


@dataclass
class LLMSelectionResult:
    """
    Result of LLM-based skill selection.

    Contains the skills selected by the LLM and its reasoning,
    mimicking Claude Code's autonomous skill selection.
    """
    task_id: str
    selected_skills: List[str]
    skill_order: List[str]
    reasoning: str
    confidence: Optional[float] = None
    raw_response: Optional[str] = None  # Full LLM response for debugging

    def __post_init__(self):
        """Validate and normalize selection results."""
        if not self.selected_skills:
            self.selected_skills = []
        if not self.skill_order:
            # Default to selected_skills order if not specified
            self.skill_order = self.selected_skills.copy()


@dataclass
class VerificationResult:
    """
    Comparison of LLM skill selection against ground truth.

    Contains accuracy metrics for skill selection and ordering.
    """
    task_id: str
    passed: bool
    llm_selection: LLMSelectionResult
    expected_skills: List[str]
    expected_order: List[str]
    precision: float  # Correct skills / total selected
    recall: float  # Correct skills / total expected
    f1_score: float
    ordering_accuracy: float  # 1.0 if exact match, partial credit otherwise
    error_message: Optional[str] = None

    @classmethod
    def calculate(
        cls,
        task: NotebookTask,
        llm_result: LLMSelectionResult,
        ordering_threshold: float = 1.0
    ) -> "VerificationResult":
        """
        Calculate verification metrics by comparing LLM selection to ground truth.

        Args:
            task: The ground truth task
            llm_result: The LLM's selection result
            ordering_threshold: Minimum ordering accuracy to pass (default 1.0 = exact match)

        Returns:
            VerificationResult with calculated metrics
        """
        expected_set = set(task.expected_skills)
        selected_set = set(llm_result.selected_skills)

        # Also consider alternate acceptable skills for ambiguous tasks
        if task.alternate_acceptable:
            # If any alternate is selected, consider it valid
            for alt in task.alternate_acceptable:
                if alt in selected_set:
                    expected_set.add(alt)

        # Calculate precision and recall
        correct_selections = expected_set & selected_set

        precision = (
            len(correct_selections) / len(selected_set)
            if selected_set else 0.0
        )
        recall = (
            len(correct_selections) / len(expected_set)
            if expected_set else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # Calculate ordering accuracy
        ordering_accuracy = cls._calculate_ordering_accuracy(
            llm_result.skill_order,
            task.expected_order
        )

        # Task passes if:
        # 1. All expected skills are selected (recall = 1.0)
        # 2. No extra skills are selected (precision = 1.0)
        # 3. Ordering is correct (>= threshold)
        passed = (
            precision == 1.0 and
            recall == 1.0 and
            ordering_accuracy >= ordering_threshold
        )

        error_message = None
        if not passed:
            errors = []
            if precision < 1.0:
                extra = selected_set - expected_set
                errors.append(f"Extra skills: {extra}")
            if recall < 1.0:
                missing = expected_set - selected_set
                errors.append(f"Missing skills: {missing}")
            if ordering_accuracy < ordering_threshold:
                errors.append(f"Ordering accuracy {ordering_accuracy:.2f} < {ordering_threshold}")
            error_message = "; ".join(errors)

        return cls(
            task_id=task.task_id,
            passed=passed,
            llm_selection=llm_result,
            expected_skills=task.expected_skills,
            expected_order=task.expected_order,
            precision=precision,
            recall=recall,
            f1_score=f1,
            ordering_accuracy=ordering_accuracy,
            error_message=error_message
        )

    @staticmethod
    def _calculate_ordering_accuracy(
        actual_order: List[str],
        expected_order: List[str]
    ) -> float:
        """
        Calculate ordering accuracy.

        Returns 1.0 for exact match, partial credit for partially correct order.
        Uses normalized Kendall tau distance.
        """
        if not actual_order or not expected_order:
            return 0.0

        # Exact match
        if actual_order == expected_order:
            return 1.0

        # If different lengths, penalize
        if len(actual_order) != len(expected_order):
            # Only compare common elements
            common = set(actual_order) & set(expected_order)
            if not common:
                return 0.0

            # Filter to common elements only
            actual_filtered = [s for s in actual_order if s in common]
            expected_filtered = [s for s in expected_order if s in common]

            # Calculate on filtered lists
            return VerificationResult._kendall_tau_accuracy(
                actual_filtered,
                expected_filtered
            )

        return VerificationResult._kendall_tau_accuracy(actual_order, expected_order)

    @staticmethod
    def _kendall_tau_accuracy(list1: List[str], list2: List[str]) -> float:
        """
        Calculate normalized Kendall tau correlation (0 to 1).

        1.0 = perfect agreement
        0.0 = maximum disagreement
        """
        if not list1 or not list2 or list1 == list2:
            return 1.0 if list1 == list2 else 0.0

        n = len(list1)
        if n != len(list2):
            return 0.0

        # Create position mappings
        pos1 = {item: i for i, item in enumerate(list1)}
        pos2 = {item: i for i, item in enumerate(list2)}

        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                item_i = list1[i]
                item_j = list1[j]

                # Check if both items exist in list2
                if item_i not in pos2 or item_j not in pos2:
                    continue

                # Compare relative orderings
                order1 = pos1[item_i] < pos1[item_j]
                order2 = pos2[item_i] < pos2[item_j]

                if order1 == order2:
                    concordant += 1
                else:
                    discordant += 1

        total_pairs = concordant + discordant
        if total_pairs == 0:
            return 1.0

        # Normalize to [0, 1]
        tau = (concordant - discordant) / total_pairs
        return (tau + 1) / 2  # Convert from [-1, 1] to [0, 1]
