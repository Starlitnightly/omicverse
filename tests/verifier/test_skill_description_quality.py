"""
Tests for SkillDescriptionQualityChecker

Tests the quality checker that validates skill descriptions are effective for LLM matching.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from omicverse.utils.verifier import (
    SkillDescriptionQualityChecker,
    QualityMetrics,
    EffectivenessResult,
    ComparisonResult,
    SkillDescription,
    LLMSkillSelector,
    create_quality_checker,
)


class TestQualityMetricsCalculation:
    """Test quality metrics calculation."""

    @pytest.fixture
    def checker(self):
        """Create quality checker without LLM selector."""
        return SkillDescriptionQualityChecker()

    def test_good_description(self, checker):
        """Test quality metrics for a good description."""
        skill = SkillDescription(
            name="test-skill",
            description="Analyze bulk RNA-seq data for differential expression. Use when you have bulk count matrices and need to find genes that vary between conditions."
        )

        metrics = checker.check_quality(skill)

        assert metrics.has_what is True  # "Analyze"
        assert metrics.has_when is True  # "Use when"
        assert metrics.is_concise is True  # < 100 words
        assert metrics.completeness_score >= 0.66  # Has what and when
        assert metrics.overall_score > 0.6
        assert len(metrics.warnings) == 0

    def test_missing_what(self, checker):
        """Test description missing 'what' it does."""
        skill = SkillDescription(
            name="test-skill",
            description="Use when you want to do something with your data."
        )

        metrics = checker.check_quality(skill)

        assert metrics.has_what is False
        assert "doesn't clearly state what" in " ".join(metrics.warnings).lower()
        assert any("action verbs" in rec.lower() for rec in metrics.recommendations)

    def test_missing_when(self, checker):
        """Test description missing 'when' to use it."""
        skill = SkillDescription(
            name="test-skill",
            description="Processes data and generates results and visualizations."
        )

        metrics = checker.check_quality(skill)

        assert metrics.has_when is False
        assert "doesn't state when" in " ".join(metrics.warnings).lower()
        assert any("use when" in rec.lower() or "for" in rec.lower() for rec in metrics.recommendations)

    def test_too_long_description(self, checker):
        """Test description that is too long."""
        long_desc = " ".join(["word"] * 150)
        skill = SkillDescription(
            name="test-skill",
            description=long_desc
        )

        metrics = checker.check_quality(skill)

        assert metrics.word_count == 150
        assert metrics.is_concise is False
        assert any("long" in w.lower() for w in metrics.warnings)
        assert any("reduce" in rec.lower() for rec in metrics.recommendations)

    def test_concise_description(self, checker):
        """Test a concise, well-structured description."""
        skill = SkillDescription(
            name="test-skill",
            description="Perform differential expression analysis. Use for bulk RNA-seq data."
        )

        metrics = checker.check_quality(skill)

        assert metrics.is_concise is True
        assert metrics.word_count < 100
        assert metrics.token_estimate < 80

    def test_has_use_cases(self, checker):
        """Test detection of use cases/examples."""
        skill = SkillDescription(
            name="test-skill",
            description="Analyze data including single-cell and spatial transcriptomics, for example PBMC datasets."
        )

        metrics = checker.check_quality(skill)

        assert metrics.has_use_cases is True  # "including", "for example"

    def test_token_estimation(self, checker):
        """Test token count estimation."""
        skill = SkillDescription(
            name="test-skill",
            description="Short description with few words"  # 5 words
        )

        metrics = checker.check_quality(skill)

        # Estimate: ~5 words * 1.3 = ~6.5 tokens
        assert 5 <= metrics.token_estimate <= 10
        assert metrics.is_token_efficient is True

    def test_sentence_counting(self, checker):
        """Test sentence counting."""
        skill = SkillDescription(
            name="test-skill",
            description="First sentence. Second sentence! Third sentence?"
        )

        metrics = checker.check_quality(skill)

        assert metrics.sentence_count == 3


class TestCompletenessChecks:
    """Test completeness checking logic."""

    @pytest.fixture
    def checker(self):
        return SkillDescriptionQualityChecker()

    def test_what_detection_action_verbs(self, checker):
        """Test detection of action verbs."""
        test_cases = [
            ("analyze data", True),
            ("process files", True),
            ("calculate statistics", True),
            ("does something", False),
            ("this is a skill", False),
        ]

        for text, expected in test_cases:
            result = checker._has_what(text)
            assert result == expected, f"Failed for '{text}'"

    def test_when_detection(self, checker):
        """Test detection of 'when' indicators."""
        test_cases = [
            ("use when you have data", True),
            ("for bulk RNA-seq analysis", True),
            ("if you need clustering", True),
            ("helps you process data", True),
            ("this does analysis", False),
        ]

        for text, expected in test_cases:
            result = checker._has_when(text)
            assert result == expected, f"Failed for '{text}'"

    def test_use_case_detection(self, checker):
        """Test detection of examples and use cases."""
        test_cases = [
            ("for example, PBMC data", True),
            ("such as clustering", True),
            ("including scRNA-seq", True),
            ("e.g. differential expression", True),
            ("does analysis", False),
        ]

        for text, expected in test_cases:
            result = checker._has_use_cases(text)
            assert result == expected, f"Failed for '{text}'"


class TestEffectivenessTesting:
    """Test effectiveness testing with LLM."""

    @pytest.fixture
    def mock_selector(self):
        """Create mock LLM selector."""
        selector = Mock(spec=LLMSkillSelector)
        selector.set_skill_descriptions = Mock()
        return selector

    @pytest.fixture
    def sample_skills(self):
        """Sample skills for testing."""
        return [
            SkillDescription(
                name="bulk-deg",
                description="Analyze differential expression in bulk RNA-seq"
            ),
            SkillDescription(
                name="single-prep",
                description="Preprocess single-cell data"
            ),
        ]

    @pytest.mark.asyncio
    async def test_effectiveness_perfect_match(self, mock_selector, sample_skills):
        """Test effectiveness when LLM always selects correctly."""
        # Mock LLM responses
        async def mock_select(task):
            result = Mock()
            if "bulk" in task.lower() or "differential" in task.lower():
                result.selected_skills = ["bulk-deg"]
            else:
                result.selected_skills = []
            return result

        mock_selector.select_skills_async = mock_select

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)

        skill = sample_skills[0]  # bulk-deg
        positive_tasks = [
            "Find differentially expressed genes in bulk RNA-seq",
            "Analyze bulk RNA-seq for DE genes"
        ]
        negative_tasks = [
            "Preprocess single-cell data",
            "Cluster cells"
        ]

        result = await checker.test_effectiveness_async(
            skill, positive_tasks, negative_tasks, sample_skills
        )

        assert result.skill_name == "bulk-deg"
        assert result.correct_matches == 2  # Both positive tasks matched
        assert result.false_matches == 0  # No false positives
        assert result.missed_matches == 0  # No false negatives
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    @pytest.mark.asyncio
    async def test_effectiveness_with_false_positives(self, mock_selector, sample_skills):
        """Test effectiveness when LLM has false positives."""
        # Mock LLM that over-selects
        async def mock_select(task):
            result = Mock()
            result.selected_skills = ["bulk-deg"]  # Always select
            return result

        mock_selector.select_skills_async = mock_select

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)

        skill = sample_skills[0]
        positive_tasks = ["Analyze bulk RNA-seq"]
        negative_tasks = ["Preprocess single-cell", "Cluster cells"]

        result = await checker.test_effectiveness_async(
            skill, positive_tasks, negative_tasks, sample_skills
        )

        assert result.correct_matches == 1
        assert result.false_matches == 2  # Selected for negative tasks
        assert result.precision < 1.0  # 1 / (1 + 2) = 0.33
        assert result.recall == 1.0  # Didn't miss any positives

    @pytest.mark.asyncio
    async def test_effectiveness_with_false_negatives(self, mock_selector, sample_skills):
        """Test effectiveness when LLM misses some matches."""
        call_count = [0]

        async def mock_select(task):
            result = Mock()
            # Only select for first positive task
            if call_count[0] == 0 and "bulk" in task.lower():
                result.selected_skills = ["bulk-deg"]
            else:
                result.selected_skills = []
            call_count[0] += 1
            return result

        mock_selector.select_skills_async = mock_select

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)

        skill = sample_skills[0]
        positive_tasks = ["Analyze bulk RNA-seq", "Find DE genes in bulk"]
        negative_tasks = ["Preprocess single-cell"]

        result = await checker.test_effectiveness_async(
            skill, positive_tasks, negative_tasks, sample_skills
        )

        assert result.correct_matches == 1
        assert result.missed_matches == 1  # Missed one positive
        assert result.recall < 1.0  # 1 / (1 + 1) = 0.5
        assert result.precision == 1.0  # No false positives

    def test_effectiveness_sync_wrapper(self, mock_selector, sample_skills):
        """Test synchronous wrapper for effectiveness testing."""
        async def mock_select(task):
            result = Mock()
            result.selected_skills = ["bulk-deg"] if "bulk" in task.lower() else []
            return result

        mock_selector.select_skills_async = mock_select

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)

        skill = sample_skills[0]
        result = checker.test_effectiveness(
            skill,
            positive_tasks=["Analyze bulk RNA-seq"],
            negative_tasks=["Preprocess cells"],
            all_skills=sample_skills
        )

        assert isinstance(result, EffectivenessResult)
        assert result.skill_name == "bulk-deg"

    def test_effectiveness_requires_llm_selector(self, sample_skills):
        """Test that effectiveness testing requires LLM selector."""
        checker = SkillDescriptionQualityChecker()  # No selector

        skill = sample_skills[0]

        with pytest.raises(ValueError, match="LLM selector required"):
            checker.test_effectiveness(
                skill,
                positive_tasks=["test"],
                negative_tasks=["test"],
                all_skills=sample_skills
            )


class TestComparisonABTesting:
    """Test A/B comparison of descriptions."""

    @pytest.fixture
    def mock_selector(self):
        selector = Mock(spec=LLMSkillSelector)
        selector.set_skill_descriptions = Mock()
        return selector

    @pytest.mark.asyncio
    async def test_comparison_modified_is_better(self, mock_selector):
        """Test comparison when modified description is better."""
        # Mock: modified performs better
        async def mock_test_effectiveness(skill, pos, neg, all_skills):
            if "Improved" in skill.description:
                # Modified version - significantly better (>5% improvement)
                return EffectivenessResult(
                    skill_name=skill.name,
                    total_tasks=10,
                    correct_matches=10,
                    false_matches=0,
                    missed_matches=0,
                    precision=1.0,
                    recall=1.0,
                    f1_score=1.0,  # Perfect score
                    accuracy=1.0
                )
            else:
                # Original version - mediocre
                return EffectivenessResult(
                    skill_name=skill.name,
                    total_tasks=10,
                    correct_matches=7,
                    false_matches=1,
                    missed_matches=3,
                    precision=0.875,
                    recall=0.7,
                    f1_score=0.778,  # Below threshold
                    accuracy=0.7
                )

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)
        checker.test_effectiveness_async = mock_test_effectiveness

        original = SkillDescription(
            name="test-skill",
            description="Does analysis"
        )
        modified = SkillDescription(
            name="test-skill",
            description="Improved analysis description"
        )

        result = await checker.compare_descriptions_async(
            original,
            modified,
            test_tasks=["task1", "task2", "task3"],
            positive_task_indices=[0, 1],
            all_skills=[]
        )

        assert result.winner == "modified"
        assert result.improvement > 0  # Positive improvement
        assert result.modified_effectiveness.f1_score > result.original_effectiveness.f1_score

    @pytest.mark.asyncio
    async def test_comparison_original_is_better(self, mock_selector):
        """Test comparison when original description is better."""
        async def mock_test_effectiveness(skill, pos, neg, all_skills):
            if "worse" in skill.description:
                # Modified version - significantly worse
                return EffectivenessResult(
                    skill_name=skill.name,
                    total_tasks=10,
                    correct_matches=5,
                    false_matches=2,
                    missed_matches=5,
                    precision=0.714,
                    recall=0.5,
                    f1_score=0.588,  # Poor score
                    accuracy=0.5
                )
            else:
                # Original version - excellent
                return EffectivenessResult(
                    skill_name=skill.name,
                    total_tasks=10,
                    correct_matches=10,
                    false_matches=0,
                    missed_matches=0,
                    precision=1.0,
                    recall=1.0,
                    f1_score=1.0,  # Perfect score
                    accuracy=1.0
                )

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)
        checker.test_effectiveness_async = mock_test_effectiveness

        original = SkillDescription(name="test", description="Good description")
        modified = SkillDescription(name="test", description="Modified but worse")

        result = await checker.compare_descriptions_async(
            original, modified, ["task"], [0], []
        )

        assert result.winner == "original"
        assert result.improvement < 0  # Negative (worse)

    @pytest.mark.asyncio
    async def test_comparison_tie(self, mock_selector):
        """Test comparison when descriptions perform similarly."""
        async def mock_test_effectiveness(skill, pos, neg, all_skills):
            return EffectivenessResult(
                skill_name=skill.name,
                total_tasks=10,
                correct_matches=8,
                false_matches=1,
                missed_matches=2,
                precision=0.889,
                recall=0.8,
                f1_score=0.842,
                accuracy=0.8
            )

        checker = SkillDescriptionQualityChecker(llm_selector=mock_selector)
        checker.test_effectiveness_async = mock_test_effectiveness

        original = SkillDescription(name="test", description="Desc A")
        modified = SkillDescription(name="test", description="Desc B")

        result = await checker.compare_descriptions_async(
            original, modified, ["task"], [0], []
        )

        assert result.winner == "tie"
        assert abs(result.improvement) < 5  # Within 5% threshold


class TestBulkOperations:
    """Test bulk operations on multiple skills."""

    @pytest.fixture
    def checker(self):
        return SkillDescriptionQualityChecker()

    @pytest.fixture
    def sample_skills(self):
        return [
            SkillDescription(
                name="good-skill",
                description="Analyze data with advanced methods. Use when you need statistical analysis."
            ),
            SkillDescription(
                name="bad-skill",
                description="Does stuff"
            ),
            SkillDescription(
                name="okay-skill",
                description="Process data files"
            ),
        ]

    def test_check_all_skills(self, checker, sample_skills):
        """Test checking quality for all skills."""
        results = checker.check_all_skills(sample_skills)

        assert len(results) == 3
        assert "good-skill" in results
        assert "bad-skill" in results
        assert "okay-skill" in results

        # Good skill should score high
        assert results["good-skill"].overall_score > 0.6

        # Bad skill should score low
        assert results["bad-skill"].overall_score < 0.5

    def test_quality_summary(self, checker, sample_skills):
        """Test summary statistics generation."""
        summary = checker.get_quality_summary(sample_skills)

        assert summary['total_skills'] == 3
        assert 0.0 <= summary['avg_overall_score'] <= 1.0
        assert 0.0 <= summary['avg_completeness'] <= 1.0
        assert 0.0 <= summary['avg_clarity'] <= 1.0
        assert summary['skills_with_warnings'] >= 0
        assert summary['total_warnings'] >= 0

    def test_quality_summary_empty(self, checker):
        """Test summary with no skills."""
        summary = checker.get_quality_summary([])

        assert summary['total_skills'] == 0
        assert summary['avg_overall_score'] == 0.0

    def test_generate_report(self, checker, sample_skills):
        """Test report generation."""
        report = checker.generate_report(sample_skills)

        assert "SKILL DESCRIPTION QUALITY REPORT" in report
        assert "Total skills analyzed: 3" in report
        assert "good-skill" in report
        assert "bad-skill" in report
        assert "okay-skill" in report

    def test_generate_report_with_recommendations(self, checker, sample_skills):
        """Test report includes recommendations."""
        report = checker.generate_report(sample_skills, show_recommendations=True)

        assert "💡 Recommendations:" in report or "Recommendations:" in report

    def test_generate_report_without_recommendations(self, checker, sample_skills):
        """Test report without recommendations."""
        report = checker.generate_report(sample_skills, show_recommendations=False)

        # Should still show warnings but not recommendations
        assert "⚠ Warnings:" in report or "Warnings:" in report


class TestConvenienceFunction:
    """Test convenience function."""

    def test_create_quality_checker(self):
        """Test creating quality checker with convenience function."""
        checker = create_quality_checker()

        assert isinstance(checker, SkillDescriptionQualityChecker)
        assert checker.llm_selector is None

    def test_create_quality_checker_with_selector(self):
        """Test creating quality checker with LLM selector."""
        mock_selector = Mock(spec=LLMSkillSelector)
        checker = create_quality_checker(llm_selector=mock_selector)

        assert isinstance(checker, SkillDescriptionQualityChecker)
        assert checker.llm_selector == mock_selector


class TestRealSkillsQuality:
    """Test with real OmicVerse skills."""

    @pytest.fixture
    def real_skills(self):
        """Load real OmicVerse skills if available."""
        from omicverse.utils.verifier import SkillDescriptionLoader

        skills_path = Path.cwd() / ".claude" / "skills"
        if not skills_path.exists():
            pytest.skip("Skills directory not found")

        loader = SkillDescriptionLoader()
        return loader.load_all_descriptions()

    def test_check_real_skills_quality(self, real_skills):
        """Test quality checking on real skills."""
        checker = SkillDescriptionQualityChecker()

        results = checker.check_all_skills(real_skills)

        # Should have results for all skills
        assert len(results) >= 20

        # All should have metrics
        for skill_name, metrics in results.items():
            assert isinstance(metrics, QualityMetrics)
            assert 0.0 <= metrics.overall_score <= 1.0

    def test_real_skills_summary(self, real_skills):
        """Test summary statistics on real skills."""
        checker = SkillDescriptionQualityChecker()

        summary = checker.get_quality_summary(real_skills)

        print(f"\nReal skills summary:")
        print(f"  Total: {summary['total_skills']}")
        print(f"  Avg overall score: {summary['avg_overall_score']:.2f}")
        print(f"  Avg completeness: {summary['avg_completeness']:.2f}")
        print(f"  Avg clarity: {summary['avg_clarity']:.2f}")
        print(f"  With warnings: {summary['skills_with_warnings']}")
        print(f"  Needing improvement: {summary['skills_needing_improvement']}")

        assert summary['total_skills'] >= 20

    def test_generate_real_skills_report(self, real_skills):
        """Test generating report for real skills."""
        checker = SkillDescriptionQualityChecker()

        report = checker.generate_report(real_skills[:5])  # First 5 skills

        assert len(report) > 0
        assert "SKILL DESCRIPTION QUALITY REPORT" in report
