"""
End-to-End Verifier - Phase 5

Tests the complete workflow: notebook → task extraction → LLM selection → verification.
Generates comprehensive reports on skill selection accuracy.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from .data_structures import (
    SkillDescription,
    NotebookTask,
    LLMSelectionResult,
    VerificationResult,
)
from .skill_description_loader import SkillDescriptionLoader
from .llm_skill_selector import LLMSkillSelector
from .notebook_task_extractor import NotebookTaskExtractor


@dataclass
class VerificationRunConfig:
    """Configuration for a verification run."""

    notebooks_dir: str
    notebook_pattern: str = "**/*.ipynb"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_concurrent_tasks: int = 5
    skip_notebooks: List[str] = field(default_factory=list)
    only_categories: Optional[List[str]] = None  # e.g., ['bulk', 'single-cell']


@dataclass
class VerificationSummary:
    """Summary of verification run results."""

    # Run metadata
    run_id: str
    timestamp: str
    config: VerificationRunConfig

    # Overall metrics
    total_tasks: int
    tasks_verified: int
    tasks_passed: int
    tasks_failed: int

    # Selection accuracy
    avg_precision: float
    avg_recall: float
    avg_f1_score: float

    # Ordering accuracy
    avg_ordering_accuracy: float

    # Coverage
    notebooks_tested: int
    skills_tested: int
    skills_not_tested: List[str]

    # Category breakdown
    category_metrics: Dict[str, Dict[str, float]]

    # Difficulty breakdown
    difficulty_metrics: Dict[str, Dict[str, float]]

    # Details
    verification_results: List[VerificationResult]
    failed_tasks: List[Dict[str, Any]]

    def passed_criteria(self, f1_threshold: float = 0.90, ordering_threshold: float = 0.85) -> bool:
        """Check if verification passed success criteria."""
        return (
            self.avg_f1_score >= f1_threshold and
            self.avg_ordering_accuracy >= ordering_threshold
        )


class EndToEndVerifier:
    """
    End-to-end verification system.

    Tests the complete workflow:
    1. Extract tasks from notebooks
    2. Use LLM to select skills
    3. Compare against ground truth
    4. Generate comprehensive reports
    """

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        llm_selector: Optional[LLMSkillSelector] = None,
        task_extractor: Optional[NotebookTaskExtractor] = None,
    ):
        """
        Initialize end-to-end verifier.

        Args:
            skills_dir: Directory containing skill descriptions (optional)
            llm_selector: Pre-configured LLM selector (optional)
            task_extractor: Pre-configured task extractor (optional)
        """
        # Load skills
        self.loader = SkillDescriptionLoader(skills_dir=skills_dir)
        self.skills = self.loader.load_all_descriptions()

        # Setup components
        self.llm_selector = llm_selector
        self.task_extractor = task_extractor or NotebookTaskExtractor()

    def _create_llm_selector(self, model: str, temperature: float) -> LLMSkillSelector:
        """Create LLM selector with specified config."""
        from omicverse.utils.agent_backend import OmicVerseLLMBackend

        backend = OmicVerseLLMBackend(
            system_prompt="You are an expert at selecting relevant skills based on task descriptions.",
            model=model,
            temperature=temperature,
        )

        return LLMSkillSelector(
            llm_backend=backend,
            skill_descriptions=self.skills,
        )

    async def verify_task_async(
        self,
        task: NotebookTask,
        selector: LLMSkillSelector,
    ) -> Tuple[LLMSelectionResult, VerificationResult]:
        """
        Verify a single task.

        Args:
            task: NotebookTask to verify
            selector: LLMSkillSelector to use

        Returns:
            Tuple of (LLM selection result, Verification result)
        """
        # Get LLM selection
        llm_result = await selector.select_skills_async(task)

        # Compare against ground truth
        verification = VerificationResult.calculate(
            task=task,
            llm_result=llm_result,
        )

        return llm_result, verification

    def verify_task(
        self,
        task: NotebookTask,
        selector: LLMSkillSelector,
    ) -> Tuple[LLMSelectionResult, VerificationResult]:
        """Sync wrapper for verify_task_async."""
        return asyncio.run(self.verify_task_async(task, selector))

    async def verify_tasks_async(
        self,
        tasks: List[NotebookTask],
        selector: LLMSkillSelector,
        max_concurrent: int = 5,
    ) -> List[Tuple[NotebookTask, LLMSelectionResult, VerificationResult]]:
        """
        Verify multiple tasks in parallel.

        Args:
            tasks: List of NotebookTask objects
            selector: LLMSkillSelector to use
            max_concurrent: Maximum concurrent verifications

        Returns:
            List of tuples (task, llm_result, verification_result)
        """
        results = []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_with_semaphore(task):
            async with semaphore:
                llm_result, verification = await self.verify_task_async(task, selector)
                return (task, llm_result, verification)

        # Run verifications in parallel
        verification_tasks = [verify_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Filter out exceptions and return successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Warning: Task {tasks[i].task_id} failed: {result}")
            else:
                successful_results.append(result)

        return successful_results

    def verify_tasks(
        self,
        tasks: List[NotebookTask],
        selector: LLMSkillSelector,
        max_concurrent: int = 5,
    ) -> List[Tuple[NotebookTask, LLMSelectionResult, VerificationResult]]:
        """Sync wrapper for verify_tasks_async."""
        return asyncio.run(self.verify_tasks_async(tasks, selector, max_concurrent))

    async def run_verification_async(
        self,
        config: VerificationRunConfig,
    ) -> VerificationSummary:
        """
        Run complete end-to-end verification.

        Args:
            config: Verification run configuration

        Returns:
            VerificationSummary with all results
        """
        run_id = f"verification-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Extract tasks from notebooks
        print(f"Extracting tasks from {config.notebooks_dir}...")
        all_tasks = self.task_extractor.extract_from_directory(
            config.notebooks_dir,
            pattern=config.notebook_pattern,
        )

        # Filter tasks
        tasks = all_tasks
        if config.skip_notebooks:
            tasks = [
                t for t in tasks
                if not any(skip in t.notebook_path for skip in config.skip_notebooks)
            ]

        if config.only_categories:
            tasks = [t for t in tasks if t.category in config.only_categories]

        print(f"Found {len(tasks)} tasks to verify")

        # Create LLM selector
        if self.llm_selector is None:
            selector = self._create_llm_selector(config.model, config.temperature)
        else:
            selector = self.llm_selector

        # Run verifications
        print(f"Running verifications (max {config.max_concurrent_tasks} concurrent)...")
        verification_results = await self.verify_tasks_async(
            tasks,
            selector,
            max_concurrent=config.max_concurrent_tasks,
        )

        # Aggregate results
        print("Aggregating results...")
        summary = self._create_summary(
            run_id=run_id,
            config=config,
            tasks=tasks,
            results=verification_results,
        )

        return summary

    def run_verification(self, config: VerificationRunConfig) -> VerificationSummary:
        """Sync wrapper for run_verification_async."""
        return asyncio.run(self.run_verification_async(config))

    def _create_summary(
        self,
        run_id: str,
        config: VerificationRunConfig,
        tasks: List[NotebookTask],
        results: List[Tuple[NotebookTask, LLMSelectionResult, VerificationResult]],
    ) -> VerificationSummary:
        """Create verification summary from results."""

        # Basic counts
        total_tasks = len(tasks)
        tasks_verified = len(results)

        verification_results = [r[2] for r in results]
        tasks_passed = sum(1 for v in verification_results if v.passed)
        tasks_failed = tasks_verified - tasks_passed

        # Overall metrics
        if verification_results:
            avg_precision = sum(v.precision for v in verification_results) / len(verification_results)
            avg_recall = sum(v.recall for v in verification_results) / len(verification_results)
            avg_f1_score = sum(v.f1_score for v in verification_results) / len(verification_results)
            avg_ordering_accuracy = sum(v.ordering_accuracy for v in verification_results) / len(verification_results)
        else:
            avg_precision = avg_recall = avg_f1_score = avg_ordering_accuracy = 0.0

        # Coverage
        unique_notebooks = set(t.notebook_path for t in tasks)
        notebooks_tested = len(unique_notebooks)

        skills_in_tasks = set()
        for task in tasks:
            skills_in_tasks.update(task.expected_skills)
        skills_in_tasks.discard('unknown')

        all_skill_names = set(s.name for s in self.skills)
        skills_tested = len(skills_in_tasks & all_skill_names)
        skills_not_tested = sorted(list(all_skill_names - skills_in_tasks))

        # Category breakdown
        category_metrics = self._calculate_category_metrics(results)

        # Difficulty breakdown
        difficulty_metrics = self._calculate_difficulty_metrics(results)

        # Failed tasks details
        failed_tasks = []
        for task, llm_result, verification in results:
            if not verification.passed:
                failed_tasks.append({
                    'task_id': task.task_id,
                    'task_description': task.task_description,
                    'expected': task.expected_skills,
                    'selected': llm_result.selected_skills,
                    'precision': verification.precision,
                    'recall': verification.recall,
                    'f1_score': verification.f1_score,
                    'reasoning': llm_result.reasoning,
                })

        return VerificationSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            total_tasks=total_tasks,
            tasks_verified=tasks_verified,
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1_score,
            avg_ordering_accuracy=avg_ordering_accuracy,
            notebooks_tested=notebooks_tested,
            skills_tested=skills_tested,
            skills_not_tested=skills_not_tested,
            category_metrics=category_metrics,
            difficulty_metrics=difficulty_metrics,
            verification_results=verification_results,
            failed_tasks=failed_tasks,
        )

    def _calculate_category_metrics(
        self,
        results: List[Tuple[NotebookTask, LLMSelectionResult, VerificationResult]],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per category."""

        category_results = {}
        for task, llm_result, verification in results:
            category = task.category
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(verification)

        category_metrics = {}
        for category, verifications in category_results.items():
            if verifications:
                category_metrics[category] = {
                    'count': len(verifications),
                    'passed': sum(1 for v in verifications if v.passed),
                    'precision': sum(v.precision for v in verifications) / len(verifications),
                    'recall': sum(v.recall for v in verifications) / len(verifications),
                    'f1_score': sum(v.f1_score for v in verifications) / len(verifications),
                    'ordering_accuracy': sum(v.ordering_accuracy for v in verifications) / len(verifications),
                }

        return category_metrics

    def _calculate_difficulty_metrics(
        self,
        results: List[Tuple[NotebookTask, LLMSelectionResult, VerificationResult]],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per difficulty level."""

        difficulty_results = {}
        for task, llm_result, verification in results:
            difficulty = task.difficulty
            if difficulty not in difficulty_results:
                difficulty_results[difficulty] = []
            difficulty_results[difficulty].append(verification)

        difficulty_metrics = {}
        for difficulty, verifications in difficulty_results.items():
            if verifications:
                difficulty_metrics[difficulty] = {
                    'count': len(verifications),
                    'passed': sum(1 for v in verifications if v.passed),
                    'precision': sum(v.precision for v in verifications) / len(verifications),
                    'recall': sum(v.recall for v in verifications) / len(verifications),
                    'f1_score': sum(v.f1_score for v in verifications) / len(verifications),
                    'ordering_accuracy': sum(v.ordering_accuracy for v in verifications) / len(verifications),
                }

        return difficulty_metrics

    def generate_report(
        self,
        summary: VerificationSummary,
        detailed: bool = True,
    ) -> str:
        """
        Generate human-readable verification report.

        Args:
            summary: VerificationSummary object
            detailed: Include detailed breakdown

        Returns:
            Formatted report string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("OmicVerse Skills Verifier - End-to-End Verification Report")
        lines.append("=" * 80)
        lines.append(f"Run ID: {summary.run_id}")
        lines.append(f"Timestamp: {summary.timestamp}")
        lines.append(f"Model: {summary.config.model}")
        lines.append("")

        # Overall Results
        lines.append("OVERALL RESULTS")
        lines.append("-" * 80)
        lines.append(f"Total Tasks: {summary.total_tasks}")
        lines.append(f"Tasks Verified: {summary.tasks_verified}")
        if summary.tasks_verified > 0:
            lines.append(f"Tasks Passed: {summary.tasks_passed} ({summary.tasks_passed/summary.tasks_verified*100:.1f}%)")
            lines.append(f"Tasks Failed: {summary.tasks_failed} ({summary.tasks_failed/summary.tasks_verified*100:.1f}%)")
        else:
            lines.append(f"Tasks Passed: {summary.tasks_passed} (N/A)")
            lines.append(f"Tasks Failed: {summary.tasks_failed} (N/A)")
        lines.append("")

        # Metrics
        lines.append("SELECTION ACCURACY METRICS")
        lines.append("-" * 80)
        lines.append(f"Average Precision: {summary.avg_precision:.3f}")
        lines.append(f"Average Recall: {summary.avg_recall:.3f}")
        lines.append(f"Average F1-Score: {summary.avg_f1_score:.3f} (Target: ≥0.90)")
        lines.append("")
        lines.append("ORDERING ACCURACY METRICS")
        lines.append("-" * 80)
        lines.append(f"Average Ordering Accuracy: {summary.avg_ordering_accuracy:.3f} (Target: ≥0.85)")
        lines.append("")

        # Success Criteria
        passed_criteria = summary.passed_criteria()
        lines.append("SUCCESS CRITERIA")
        lines.append("-" * 80)
        lines.append(f"Status: {'✅ PASSED' if passed_criteria else '❌ FAILED'}")
        lines.append(f"F1-Score Target (≥0.90): {'✅' if summary.avg_f1_score >= 0.90 else '❌'} {summary.avg_f1_score:.3f}")
        lines.append(f"Ordering Target (≥0.85): {'✅' if summary.avg_ordering_accuracy >= 0.85 else '❌'} {summary.avg_ordering_accuracy:.3f}")
        lines.append("")

        # Coverage
        lines.append("COVERAGE")
        lines.append("-" * 80)
        lines.append(f"Notebooks Tested: {summary.notebooks_tested}")
        lines.append(f"Skills Tested: {summary.skills_tested}")
        if summary.skills_not_tested:
            lines.append(f"Skills Not Tested: {', '.join(summary.skills_not_tested)}")
        lines.append("")

        if detailed:
            # Category breakdown
            if summary.category_metrics:
                lines.append("CATEGORY BREAKDOWN")
                lines.append("-" * 80)
                for category, metrics in sorted(summary.category_metrics.items()):
                    lines.append(f"{category}:")
                    lines.append(f"  Tasks: {metrics['count']}")
                    if metrics['count'] > 0:
                        lines.append(f"  Passed: {metrics['passed']}/{metrics['count']} ({metrics['passed']/metrics['count']*100:.1f}%)")
                    else:
                        lines.append(f"  Passed: {metrics['passed']}/{metrics['count']} (N/A)")
                    lines.append(f"  F1-Score: {metrics['f1_score']:.3f}")
                    lines.append(f"  Ordering: {metrics['ordering_accuracy']:.3f}")
                    lines.append("")

            # Difficulty breakdown
            if summary.difficulty_metrics:
                lines.append("DIFFICULTY BREAKDOWN")
                lines.append("-" * 80)
                for difficulty, metrics in sorted(summary.difficulty_metrics.items()):
                    lines.append(f"{difficulty}:")
                    lines.append(f"  Tasks: {metrics['count']}")
                    if metrics['count'] > 0:
                        lines.append(f"  Passed: {metrics['passed']}/{metrics['count']} ({metrics['passed']/metrics['count']*100:.1f}%)")
                    else:
                        lines.append(f"  Passed: {metrics['passed']}/{metrics['count']} (N/A)")
                    lines.append(f"  F1-Score: {metrics['f1_score']:.3f}")
                    lines.append(f"  Ordering: {metrics['ordering_accuracy']:.3f}")
                    lines.append("")

            # Failed tasks
            if summary.failed_tasks:
                lines.append("FAILED TASKS")
                lines.append("-" * 80)
                for failed in summary.failed_tasks[:10]:  # Show first 10
                    lines.append(f"Task ID: {failed['task_id']}")
                    lines.append(f"  Description: {failed['task_description'][:100]}...")
                    lines.append(f"  Expected: {failed['expected']}")
                    lines.append(f"  Selected: {failed['selected']}")
                    lines.append(f"  F1-Score: {failed['f1_score']:.3f}")
                    lines.append("")

                if len(summary.failed_tasks) > 10:
                    lines.append(f"... and {len(summary.failed_tasks) - 10} more")
                    lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(
        self,
        summary: VerificationSummary,
        output_path: str,
        detailed: bool = True,
    ):
        """
        Save verification report to file.

        Args:
            summary: VerificationSummary object
            output_path: Path to save report
            detailed: Include detailed breakdown
        """
        report = self.generate_report(summary, detailed=detailed)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)


def create_verifier(
    skills_dir: Optional[str] = None,
    llm_selector: Optional[LLMSkillSelector] = None,
) -> EndToEndVerifier:
    """
    Convenience function to create an end-to-end verifier.

    Args:
        skills_dir: Optional custom skills directory
        llm_selector: Optional pre-configured LLM selector

    Returns:
        Configured EndToEndVerifier
    """
    return EndToEndVerifier(
        skills_dir=skills_dir,
        llm_selector=llm_selector,
    )
