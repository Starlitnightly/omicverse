"""
Tests for End-to-End Verifier - Phase 5

Tests the complete verification workflow from notebooks to reports.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

from omicverse.utils.verifier import (
    EndToEndVerifier,
    VerificationRunConfig,
    VerificationSummary,
    create_verifier,
    SkillDescription,
    NotebookTask,
    LLMSelectionResult,
    VerificationResult,
    LLMSkillSelector,
    NotebookTaskExtractor,
)


# Fixtures

@pytest.fixture
def mock_skills():
    """Create mock skill descriptions."""
    return [
        SkillDescription(
            name="bulk-deg-analysis",
            description="Guide Claude through omicverse's bulk RNA-seq DEG pipeline."
        ),
        SkillDescription(
            name="single-preprocessing",
            description="Preprocess single-cell data with QC filtering and normalization."
        ),
        SkillDescription(
            name="single-clustering",
            description="Cluster single-cell data using leiden or louvain algorithms."
        ),
    ]


@pytest.fixture
def mock_tasks():
    """Create mock notebook tasks."""
    return [
        NotebookTask(
            task_id="task-001",
            notebook_path="test_notebooks/t_deg.ipynb",
            task_description="Perform differential expression analysis",
            expected_skills=["bulk-deg-analysis"],
            expected_order=["bulk-deg-analysis"],
            category="bulk",
            difficulty="single",
        ),
        NotebookTask(
            task_id="task-002",
            notebook_path="test_notebooks/t_preprocess.ipynb",
            task_description="Preprocess and cluster single-cell data",
            expected_skills=["single-preprocessing", "single-clustering"],
            expected_order=["single-preprocessing", "single-clustering"],
            category="single-cell",
            difficulty="workflow",
        ),
        NotebookTask(
            task_id="task-003",
            notebook_path="test_notebooks/t_cluster.ipynb",
            task_description="Cluster single-cell data",
            expected_skills=["single-clustering"],
            expected_order=["single-clustering"],
            category="single-cell",
            difficulty="single",
        ),
    ]


@pytest.fixture
def mock_llm_selector(mock_skills):
    """Create mock LLM selector."""
    selector = Mock(spec=LLMSkillSelector)

    # Mock select_skills_async to return appropriate results
    async def mock_select_async(task):
        if isinstance(task, str):
            task_id = "mock-task"
            task_desc = task
        else:
            task_id = task.task_id
            task_desc = task.task_description

        # Simple mock logic based on task description
        if "differential expression" in task_desc.lower():
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=["bulk-deg-analysis"],
                skill_order=["bulk-deg-analysis"],
                reasoning="Task requires DEG analysis",
            )
        elif "preprocess" in task_desc.lower() and "cluster" in task_desc.lower():
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=["single-preprocessing", "single-clustering"],
                skill_order=["single-preprocessing", "single-clustering"],
                reasoning="Task requires preprocessing followed by clustering",
            )
        elif "cluster" in task_desc.lower():
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=["single-clustering"],
                skill_order=["single-clustering"],
                reasoning="Task requires clustering",
            )
        else:
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=[],
                skill_order=[],
                reasoning="No matching skills",
            )

    selector.select_skills_async = AsyncMock(side_effect=mock_select_async)

    return selector


@pytest.fixture
def mock_task_extractor(mock_tasks):
    """Create mock task extractor."""
    extractor = Mock(spec=NotebookTaskExtractor)
    extractor.extract_from_directory.return_value = mock_tasks
    return extractor


# Basic Creation Tests

def test_create_verifier():
    """Test creating verifier with factory function."""
    verifier = create_verifier()
    assert isinstance(verifier, EndToEndVerifier)
    assert verifier.skills is not None
    assert len(verifier.skills) > 0


def test_verifier_initialization(mock_llm_selector, mock_task_extractor):
    """Test verifier initialization with components."""
    verifier = EndToEndVerifier(
        llm_selector=mock_llm_selector,
        task_extractor=mock_task_extractor,
    )

    assert verifier.llm_selector == mock_llm_selector
    assert verifier.task_extractor == mock_task_extractor
    assert len(verifier.skills) > 0


# Single Task Verification Tests

@pytest.mark.asyncio
async def test_verify_single_task_async(mock_tasks, mock_llm_selector):
    """Test verifying a single task asynchronously."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    task = mock_tasks[0]  # DEG analysis task

    llm_result, verification = await verifier.verify_task_async(task, mock_llm_selector)

    assert isinstance(llm_result, LLMSelectionResult)
    assert isinstance(verification, VerificationResult)
    assert verification.task_id == task.task_id
    assert verification.passed  # Should pass as mock returns correct result


def test_verify_single_task_sync(mock_tasks, mock_llm_selector):
    """Test verifying a single task synchronously."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    task = mock_tasks[0]

    llm_result, verification = verifier.verify_task(task, mock_llm_selector)

    assert isinstance(llm_result, LLMSelectionResult)
    assert isinstance(verification, VerificationResult)


# Multiple Task Verification Tests

@pytest.mark.asyncio
async def test_verify_multiple_tasks_async(mock_tasks, mock_llm_selector):
    """Test verifying multiple tasks in parallel."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    results = await verifier.verify_tasks_async(
        mock_tasks,
        mock_llm_selector,
        max_concurrent=2,
    )

    assert len(results) == len(mock_tasks)
    for task, llm_result, verification in results:
        assert isinstance(task, NotebookTask)
        assert isinstance(llm_result, LLMSelectionResult)
        assert isinstance(verification, VerificationResult)


def test_verify_multiple_tasks_sync(mock_tasks, mock_llm_selector):
    """Test verifying multiple tasks synchronously."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    results = verifier.verify_tasks(mock_tasks, mock_llm_selector)

    assert len(results) == len(mock_tasks)


@pytest.mark.asyncio
async def test_verify_tasks_with_failure(mock_tasks, mock_llm_selector):
    """Test that verification handles task failures gracefully."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    # Store original function
    original_select = mock_llm_selector.select_skills_async

    # Make one task fail
    async def failing_select(task):
        if task.task_id == "task-002":
            raise ValueError("Simulated failure")
        # Call original for other tasks
        return await original_select(task)

    mock_llm_selector.select_skills_async = AsyncMock(side_effect=failing_select)

    results = await verifier.verify_tasks_async(
        mock_tasks,
        mock_llm_selector,
        max_concurrent=2,
    )

    # Should have 2 successful results (out of 3)
    assert len(results) == 2


# Verification Run Tests

@pytest.mark.asyncio
async def test_run_verification_async(mock_llm_selector, mock_task_extractor):
    """Test complete verification run."""
    verifier = EndToEndVerifier(
        llm_selector=mock_llm_selector,
        task_extractor=mock_task_extractor,
    )

    config = VerificationRunConfig(
        notebooks_dir="/fake/path",
        model="gpt-4o-mini",
        max_concurrent_tasks=2,
    )

    summary = await verifier.run_verification_async(config)

    assert isinstance(summary, VerificationSummary)
    assert summary.total_tasks == 3
    assert summary.tasks_verified == 3
    assert summary.tasks_passed == 3  # All should pass with our mock


def test_run_verification_sync(mock_llm_selector, mock_task_extractor):
    """Test complete verification run (sync)."""
    verifier = EndToEndVerifier(
        llm_selector=mock_llm_selector,
        task_extractor=mock_task_extractor,
    )

    config = VerificationRunConfig(
        notebooks_dir="/fake/path",
        model="gpt-4o-mini",
    )

    summary = verifier.run_verification(config)

    assert isinstance(summary, VerificationSummary)
    assert summary.total_tasks > 0


@pytest.mark.asyncio
async def test_run_verification_with_filters(mock_llm_selector, mock_tasks):
    """Test verification with category filters."""
    extractor = Mock(spec=NotebookTaskExtractor)
    extractor.extract_from_directory.return_value = mock_tasks

    verifier = EndToEndVerifier(
        llm_selector=mock_llm_selector,
        task_extractor=extractor,
    )

    config = VerificationRunConfig(
        notebooks_dir="/fake/path",
        only_categories=["bulk"],  # Only bulk tasks
    )

    summary = await verifier.run_verification_async(config)

    # Should only verify bulk tasks
    assert summary.tasks_verified == 1  # Only task-001


@pytest.mark.asyncio
async def test_run_verification_with_skip(mock_llm_selector, mock_tasks):
    """Test verification with notebook skipping."""
    extractor = Mock(spec=NotebookTaskExtractor)
    extractor.extract_from_directory.return_value = mock_tasks

    verifier = EndToEndVerifier(
        llm_selector=mock_llm_selector,
        task_extractor=extractor,
    )

    config = VerificationRunConfig(
        notebooks_dir="/fake/path",
        skip_notebooks=["t_deg.ipynb"],
    )

    summary = await verifier.run_verification_async(config)

    # Should skip task-001
    assert summary.tasks_verified == 2


# Summary and Metrics Tests

def test_verification_summary_creation(mock_tasks, mock_llm_selector):
    """Test creating verification summary."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    # Create mock results
    llm_result_1 = LLMSelectionResult(
        task_id="task-001",
        selected_skills=["bulk-deg-analysis"],
        skill_order=["bulk-deg-analysis"],
        reasoning="DEG analysis",
    )
    llm_result_2 = LLMSelectionResult(
        task_id="task-002",
        selected_skills=["single-preprocessing", "single-clustering"],
        skill_order=["single-preprocessing", "single-clustering"],
        reasoning="Preprocess and cluster",
    )

    results = [
        (
            mock_tasks[0],
            llm_result_1,
            VerificationResult(
                task_id="task-001",
                passed=True,
                llm_selection=llm_result_1,
                expected_skills=mock_tasks[0].expected_skills,
                expected_order=mock_tasks[0].expected_order,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                ordering_accuracy=1.0,
            ),
        ),
        (
            mock_tasks[1],
            llm_result_2,
            VerificationResult(
                task_id="task-002",
                passed=True,
                llm_selection=llm_result_2,
                expected_skills=mock_tasks[1].expected_skills,
                expected_order=mock_tasks[1].expected_order,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                ordering_accuracy=1.0,
            ),
        ),
    ]

    config = VerificationRunConfig(notebooks_dir="/fake")

    summary = verifier._create_summary(
        run_id="test-run",
        config=config,
        tasks=mock_tasks[:2],
        results=results,
    )

    assert summary.run_id == "test-run"
    assert summary.total_tasks == 2
    assert summary.tasks_verified == 2
    assert summary.tasks_passed == 2
    assert summary.tasks_failed == 0
    assert summary.avg_precision == 1.0
    assert summary.avg_recall == 1.0
    assert summary.avg_f1_score == 1.0
    assert summary.avg_ordering_accuracy == 1.0


def test_verification_summary_with_failures(mock_tasks, mock_llm_selector):
    """Test summary with some failed tasks."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    llm_result_1 = LLMSelectionResult(
        task_id="task-001",
        selected_skills=["bulk-deg-analysis"],
        skill_order=["bulk-deg-analysis"],
        reasoning="DEG analysis",
    )
    llm_result_2 = LLMSelectionResult(
        task_id="task-002",
        selected_skills=["single-clustering"],  # Missing preprocessing
        skill_order=["single-clustering"],
        reasoning="Just cluster",
    )

    results = [
        (
            mock_tasks[0],
            llm_result_1,
            VerificationResult(
                task_id="task-001",
                passed=True,
                llm_selection=llm_result_1,
                expected_skills=mock_tasks[0].expected_skills,
                expected_order=mock_tasks[0].expected_order,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                ordering_accuracy=1.0,
            ),
        ),
        (
            mock_tasks[1],
            llm_result_2,
            VerificationResult(
                task_id="task-002",
                passed=False,
                llm_selection=llm_result_2,
                expected_skills=mock_tasks[1].expected_skills,
                expected_order=mock_tasks[1].expected_order,
                precision=1.0,
                recall=0.5,  # Only got 1 of 2 skills
                f1_score=0.67,
                ordering_accuracy=0.5,
            ),
        ),
    ]

    config = VerificationRunConfig(notebooks_dir="/fake")

    summary = verifier._create_summary(
        run_id="test-run",
        config=config,
        tasks=mock_tasks[:2],
        results=results,
    )

    assert summary.tasks_passed == 1
    assert summary.tasks_failed == 1
    assert len(summary.failed_tasks) == 1
    assert summary.failed_tasks[0]['task_id'] == "task-002"


def test_passed_criteria():
    """Test success criteria checking."""
    # Create a passing summary
    summary_pass = VerificationSummary(
        run_id="test",
        timestamp="2024-01-01",
        config=VerificationRunConfig(notebooks_dir="/fake"),
        total_tasks=10,
        tasks_verified=10,
        tasks_passed=10,
        tasks_failed=0,
        avg_precision=0.95,
        avg_recall=0.95,
        avg_f1_score=0.95,
        avg_ordering_accuracy=0.90,
        notebooks_tested=5,
        skills_tested=10,
        skills_not_tested=[],
        category_metrics={},
        difficulty_metrics={},
        verification_results=[],
        failed_tasks=[],
    )

    assert summary_pass.passed_criteria()

    # Create a failing summary
    summary_fail = VerificationSummary(
        run_id="test",
        timestamp="2024-01-01",
        config=VerificationRunConfig(notebooks_dir="/fake"),
        total_tasks=10,
        tasks_verified=10,
        tasks_passed=8,
        tasks_failed=2,
        avg_precision=0.85,
        avg_recall=0.80,
        avg_f1_score=0.82,  # Below 0.90 threshold
        avg_ordering_accuracy=0.90,
        notebooks_tested=5,
        skills_tested=10,
        skills_not_tested=[],
        category_metrics={},
        difficulty_metrics={},
        verification_results=[],
        failed_tasks=[],
    )

    assert not summary_fail.passed_criteria()


def test_category_metrics_calculation(mock_tasks, mock_llm_selector):
    """Test category metrics calculation."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    llm_result_1 = Mock()
    llm_result_2 = Mock()
    llm_result_3 = Mock()

    results = [
        (
            mock_tasks[0],  # bulk
            llm_result_1,
            VerificationResult(
                task_id="task-001",
                passed=True,
                llm_selection=llm_result_1,
                expected_skills=mock_tasks[0].expected_skills,
                expected_order=mock_tasks[0].expected_order,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                ordering_accuracy=1.0,
            ),
        ),
        (
            mock_tasks[1],  # single-cell
            llm_result_2,
            VerificationResult(
                task_id="task-002",
                passed=True,
                llm_selection=llm_result_2,
                expected_skills=mock_tasks[1].expected_skills,
                expected_order=mock_tasks[1].expected_order,
                precision=0.9,
                recall=0.9,
                f1_score=0.9,
                ordering_accuracy=0.9,
            ),
        ),
        (
            mock_tasks[2],  # single-cell
            llm_result_3,
            VerificationResult(
                task_id="task-003",
                passed=False,
                llm_selection=llm_result_3,
                expected_skills=mock_tasks[2].expected_skills,
                expected_order=mock_tasks[2].expected_order,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                ordering_accuracy=0.8,
            ),
        ),
    ]

    category_metrics = verifier._calculate_category_metrics(results)

    assert "bulk" in category_metrics
    assert "single-cell" in category_metrics

    assert category_metrics["bulk"]["count"] == 1
    assert category_metrics["bulk"]["passed"] == 1
    assert category_metrics["bulk"]["f1_score"] == 1.0

    assert category_metrics["single-cell"]["count"] == 2
    assert category_metrics["single-cell"]["passed"] == 1
    assert category_metrics["single-cell"]["f1_score"] == pytest.approx(0.825)


def test_difficulty_metrics_calculation(mock_tasks, mock_llm_selector):
    """Test difficulty metrics calculation."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    llm_result_1 = Mock()
    llm_result_2 = Mock()

    results = [
        (
            mock_tasks[0],  # single
            llm_result_1,
            VerificationResult(
                task_id="task-001",
                passed=True,
                llm_selection=llm_result_1,
                expected_skills=mock_tasks[0].expected_skills,
                expected_order=mock_tasks[0].expected_order,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                ordering_accuracy=1.0,
            ),
        ),
        (
            mock_tasks[1],  # workflow
            llm_result_2,
            VerificationResult(
                task_id="task-002",
                passed=True,
                llm_selection=llm_result_2,
                expected_skills=mock_tasks[1].expected_skills,
                expected_order=mock_tasks[1].expected_order,
                precision=0.9,
                recall=0.9,
                f1_score=0.9,
                ordering_accuracy=0.9,
            ),
        ),
    ]

    difficulty_metrics = verifier._calculate_difficulty_metrics(results)

    assert "single" in difficulty_metrics
    assert "workflow" in difficulty_metrics

    assert difficulty_metrics["single"]["count"] == 1
    assert difficulty_metrics["workflow"]["count"] == 1


# Report Generation Tests

def test_generate_report_basic(mock_llm_selector):
    """Test basic report generation."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    summary = VerificationSummary(
        run_id="test-run",
        timestamp="2024-01-01T10:00:00",
        config=VerificationRunConfig(notebooks_dir="/fake", model="gpt-4o-mini"),
        total_tasks=10,
        tasks_verified=10,
        tasks_passed=9,
        tasks_failed=1,
        avg_precision=0.95,
        avg_recall=0.93,
        avg_f1_score=0.94,
        avg_ordering_accuracy=0.92,
        notebooks_tested=5,
        skills_tested=8,
        skills_not_tested=["skill-a", "skill-b"],
        category_metrics={
            "bulk": {
                "count": 5,
                "passed": 5,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "ordering_accuracy": 1.0,
            }
        },
        difficulty_metrics={
            "single": {
                "count": 8,
                "passed": 7,
                "precision": 0.95,
                "recall": 0.93,
                "f1_score": 0.94,
                "ordering_accuracy": 0.95,
            }
        },
        verification_results=[],
        failed_tasks=[
            {
                "task_id": "task-001",
                "task_description": "Some task",
                "expected": ["skill-a"],
                "selected": ["skill-b"],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "reasoning": "Wrong skill",
            }
        ],
    )

    report = verifier.generate_report(summary, detailed=True)

    assert "OmicVerse Skills Verifier" in report
    assert "test-run" in report
    assert "Total Tasks: 10" in report
    assert "Tasks Passed: 9" in report
    assert "Average F1-Score: 0.940" in report
    assert "PASSED" in report  # Should pass criteria
    assert "CATEGORY BREAKDOWN" in report
    assert "bulk:" in report
    assert "FAILED TASKS" in report


def test_generate_report_non_detailed(mock_llm_selector):
    """Test non-detailed report generation."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    summary = VerificationSummary(
        run_id="test-run",
        timestamp="2024-01-01T10:00:00",
        config=VerificationRunConfig(notebooks_dir="/fake"),
        total_tasks=10,
        tasks_verified=10,
        tasks_passed=10,
        tasks_failed=0,
        avg_precision=1.0,
        avg_recall=1.0,
        avg_f1_score=1.0,
        avg_ordering_accuracy=1.0,
        notebooks_tested=5,
        skills_tested=10,
        skills_not_tested=[],
        category_metrics={},
        difficulty_metrics={},
        verification_results=[],
        failed_tasks=[],
    )

    report = verifier.generate_report(summary, detailed=False)

    assert "OmicVerse Skills Verifier" in report
    assert "CATEGORY BREAKDOWN" not in report
    assert "FAILED TASKS" not in report


def test_save_report(mock_llm_selector, tmp_path):
    """Test saving report to file."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    summary = VerificationSummary(
        run_id="test-run",
        timestamp="2024-01-01T10:00:00",
        config=VerificationRunConfig(notebooks_dir="/fake"),
        total_tasks=5,
        tasks_verified=5,
        tasks_passed=5,
        tasks_failed=0,
        avg_precision=1.0,
        avg_recall=1.0,
        avg_f1_score=1.0,
        avg_ordering_accuracy=1.0,
        notebooks_tested=3,
        skills_tested=5,
        skills_not_tested=[],
        category_metrics={},
        difficulty_metrics={},
        verification_results=[],
        failed_tasks=[],
    )

    output_path = tmp_path / "report.txt"
    verifier.save_report(summary, str(output_path))

    assert output_path.exists()
    content = output_path.read_text()
    assert "OmicVerse Skills Verifier" in content


# Integration Tests

@pytest.mark.integration
def test_integration_with_real_skills():
    """Integration test with real skill descriptions."""
    verifier = create_verifier()

    assert len(verifier.skills) > 0
    assert verifier.loader is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_verify_single_task():
    """Integration test verifying a single task."""
    verifier = create_verifier()

    # Create a realistic task
    task = NotebookTask(
        task_id="integration-test",
        notebook_path="test.ipynb",
        task_description="Perform differential expression analysis on bulk RNA-seq data",
        expected_skills=["bulk-deg-analysis"],
        expected_order=["bulk-deg-analysis"],
        category="bulk",
        difficulty="single",
    )

    # Mock LLM selector for integration test
    mock_selector = Mock(spec=LLMSkillSelector)

    async def mock_select(t):
        return LLMSelectionResult(
            task_id=t.task_id,
            selected_skills=["bulk-deg-analysis"],
            skill_order=["bulk-deg-analysis"],
            reasoning="Task requires DEG analysis",
        )

    mock_selector.select_skills_async = AsyncMock(side_effect=mock_select)

    llm_result, verification = await verifier.verify_task_async(task, mock_selector)

    assert verification.passed
    assert verification.f1_score == 1.0


# Edge Cases

def test_empty_task_list(mock_llm_selector):
    """Test verification with no tasks."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    results = verifier.verify_tasks([], mock_llm_selector)

    assert len(results) == 0


def test_verification_config_defaults():
    """Test default configuration values."""
    config = VerificationRunConfig(notebooks_dir="/fake/path")

    assert config.notebook_pattern == "**/*.ipynb"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.max_concurrent_tasks == 5
    assert config.skip_notebooks == []
    assert config.only_categories is None


@pytest.mark.asyncio
async def test_concurrency_limit(mock_tasks, mock_llm_selector):
    """Test that concurrency limit is respected."""
    verifier = EndToEndVerifier(llm_selector=mock_llm_selector)

    # Track concurrent calls
    concurrent_count = 0
    max_concurrent = 0

    original_select = mock_llm_selector.select_skills_async

    async def tracked_select(task):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)

        result = await original_select(task)

        concurrent_count -= 1
        return result

    mock_llm_selector.select_skills_async = AsyncMock(side_effect=tracked_select)

    # Run with max_concurrent=2
    await verifier.verify_tasks_async(
        mock_tasks,
        mock_llm_selector,
        max_concurrent=2,
    )

    # Should not exceed limit
    assert max_concurrent <= 2
