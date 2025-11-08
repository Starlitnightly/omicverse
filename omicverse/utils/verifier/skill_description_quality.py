"""
Skill Description Quality Checker - Phase 3

Verifies that skill descriptions are effective for LLM matching.

Tests:
1. Completeness - has what/when/examples
2. Clarity - concise and unambiguous
3. Effectiveness - LLM selects correctly
4. Token efficiency - not too verbose
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .data_structures import SkillDescription
from .llm_skill_selector import LLMSkillSelector


@dataclass
class QualityMetrics:
    """Quality metrics for a skill description."""
    # Completeness
    has_what: bool  # Contains action verbs describing what it does
    has_when: bool  # States when to use it
    has_use_cases: bool  # Includes use cases or examples

    # Clarity
    word_count: int
    token_estimate: int
    is_concise: bool  # < 100 words
    is_token_efficient: bool  # < 80 tokens

    # Readability
    avg_word_length: float
    sentence_count: int

    # Overall scores
    completeness_score: float  # 0.0 to 1.0
    clarity_score: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EffectivenessResult:
    """Result of effectiveness testing."""
    skill_name: str
    total_tasks: int
    correct_matches: int  # True positives
    false_matches: int  # False positives
    missed_matches: int  # False negatives
    precision: float  # Correct / (Correct + False positives)
    recall: float  # Correct / (Correct + False negatives)
    f1_score: float
    accuracy: float


@dataclass
class ComparisonResult:
    """Result of A/B testing two descriptions."""
    original_skill: str
    modified_skill: str
    original_effectiveness: EffectivenessResult
    modified_effectiveness: EffectivenessResult
    improvement: float  # Percentage improvement in F1 score
    winner: str  # "original", "modified", or "tie"
    recommendation: str


class SkillDescriptionQualityChecker:
    """
    Verify that skill descriptions are effective for LLM matching.

    Checks:
    1. Completeness (what, when, examples)
    2. Clarity (concise, unambiguous)
    3. Effectiveness (LLM selection accuracy)
    4. Token efficiency
    """

    # Keywords for detecting "what" (action verbs)
    WHAT_KEYWORDS = [
        'analyze', 'process', 'calculate', 'extract', 'generate',
        'create', 'perform', 'run', 'execute', 'build', 'compute',
        'identify', 'detect', 'find', 'discover', 'cluster',
        'annotate', 'preprocess', 'normalize', 'filter', 'transform',
        'visualize', 'plot', 'export', 'import', 'load'
    ]

    # Keywords for detecting "when"
    WHEN_KEYWORDS = [
        'use when', 'when you', 'when the', 'when a',
        'for', 'if you need', 'to help', 'helps you',
        'applicable when', 'suitable for', 'designed for'
    ]

    # Keywords for use cases/examples
    USE_CASE_KEYWORDS = [
        'example', 'e.g.', 'such as', 'like', 'including',
        'for instance', 'use case', 'scenario'
    ]

    def __init__(self, llm_selector: Optional[LLMSkillSelector] = None):
        """
        Initialize quality checker.

        Args:
            llm_selector: LLMSkillSelector for effectiveness testing (optional)
        """
        self.llm_selector = llm_selector

    def check_quality(self, skill: SkillDescription) -> QualityMetrics:
        """
        Comprehensive quality check for a skill description.

        Args:
            skill: SkillDescription to check

        Returns:
            QualityMetrics with all quality scores
        """
        description = skill.description.lower()

        # Completeness checks
        has_what = self._has_what(description)
        has_when = self._has_when(description)
        has_use_cases = self._has_use_cases(description)

        # Clarity checks
        word_count = len(skill.description.split())
        token_estimate = self._estimate_tokens(skill)
        is_concise = word_count <= 100
        is_token_efficient = token_estimate <= 80

        # Readability
        avg_word_length = self._avg_word_length(skill.description)
        sentence_count = self._count_sentences(skill.description)

        # Calculate scores
        completeness_score = sum([has_what, has_when, has_use_cases]) / 3.0

        clarity_score = (
            (0.4 if is_concise else 0.2) +
            (0.4 if is_token_efficient else 0.2) +
            (0.2 if sentence_count >= 2 else 0.1)  # Multiple sentences good
        )

        overall_score = (completeness_score * 0.6) + (clarity_score * 0.4)

        # Generate warnings and recommendations
        warnings = []
        recommendations = []

        if not has_what:
            warnings.append("Description doesn't clearly state what the skill does")
            recommendations.append("Add action verbs describing the skill's functionality")

        if not has_when:
            warnings.append("Description doesn't state when to use this skill")
            recommendations.append("Add 'Use when...' or 'For...' to clarify usage context")

        if not has_use_cases:
            recommendations.append("Consider adding examples or use cases")

        if word_count > 100:
            warnings.append(f"Description is long ({word_count} words)")
            recommendations.append("Reduce to < 100 words for better clarity")

        if token_estimate > 80:
            warnings.append(f"Token count high ({token_estimate} estimated tokens)")
            recommendations.append("Aim for < 80 tokens for efficiency")

        if sentence_count < 2:
            recommendations.append("Consider using 2-3 sentences for better structure")

        return QualityMetrics(
            has_what=has_what,
            has_when=has_when,
            has_use_cases=has_use_cases,
            word_count=word_count,
            token_estimate=token_estimate,
            is_concise=is_concise,
            is_token_efficient=is_token_efficient,
            avg_word_length=avg_word_length,
            sentence_count=sentence_count,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            overall_score=overall_score,
            warnings=warnings,
            recommendations=recommendations
        )

    def _has_what(self, description: str) -> bool:
        """Check if description has action verbs (what it does)."""
        return any(keyword in description for keyword in self.WHAT_KEYWORDS)

    def _has_when(self, description: str) -> bool:
        """Check if description states when to use it."""
        return any(keyword in description for keyword in self.WHEN_KEYWORDS)

    def _has_use_cases(self, description: str) -> bool:
        """Check if description includes use cases or examples."""
        return any(keyword in description for keyword in self.USE_CASE_KEYWORDS)

    def _estimate_tokens(self, skill: SkillDescription) -> int:
        """Estimate token count (rough approximation: 1.3 tokens per word)."""
        text = f"{skill.name} {skill.description}"
        word_count = len(text.split())
        return int(word_count * 1.3)

    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)

    def _count_sentences(self, text: str) -> int:
        """Count sentences (rough approximation)."""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)

    async def test_effectiveness_async(
        self,
        skill: SkillDescription,
        positive_tasks: List[str],
        negative_tasks: List[str],
        all_skills: List[SkillDescription]
    ) -> EffectivenessResult:
        """
        Test how well LLM selects this skill for appropriate tasks.

        Args:
            skill: The skill to test
            positive_tasks: Tasks that SHOULD match this skill
            negative_tasks: Tasks that should NOT match this skill
            all_skills: All available skills (for context)

        Returns:
            EffectivenessResult with metrics
        """
        if not self.llm_selector:
            raise ValueError("LLM selector required for effectiveness testing")

        # Ensure the skill is in the selector's skills list
        self.llm_selector.set_skill_descriptions(all_skills)

        correct_matches = 0
        false_matches = 0
        missed_matches = 0

        # Test positive tasks (should match)
        for task in positive_tasks:
            result = await self.llm_selector.select_skills_async(task)
            if skill.name in result.selected_skills:
                correct_matches += 1  # True positive
            else:
                missed_matches += 1  # False negative

        # Test negative tasks (should NOT match)
        for task in negative_tasks:
            result = await self.llm_selector.select_skills_async(task)
            if skill.name in result.selected_skills:
                false_matches += 1  # False positive

        # Calculate metrics
        total_tasks = len(positive_tasks) + len(negative_tasks)

        precision = (
            correct_matches / (correct_matches + false_matches)
            if (correct_matches + false_matches) > 0 else 0.0
        )

        recall = (
            correct_matches / (correct_matches + missed_matches)
            if (correct_matches + missed_matches) > 0 else 0.0
        )

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        accuracy = (
            (correct_matches + (len(negative_tasks) - false_matches)) / total_tasks
            if total_tasks > 0 else 0.0
        )

        return EffectivenessResult(
            skill_name=skill.name,
            total_tasks=total_tasks,
            correct_matches=correct_matches,
            false_matches=false_matches,
            missed_matches=missed_matches,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy
        )

    def test_effectiveness(
        self,
        skill: SkillDescription,
        positive_tasks: List[str],
        negative_tasks: List[str],
        all_skills: List[SkillDescription]
    ) -> EffectivenessResult:
        """Sync wrapper for test_effectiveness_async."""
        return asyncio.run(self.test_effectiveness_async(
            skill, positive_tasks, negative_tasks, all_skills
        ))

    async def compare_descriptions_async(
        self,
        original: SkillDescription,
        modified: SkillDescription,
        test_tasks: List[str],
        positive_task_indices: List[int],
        all_skills: List[SkillDescription]
    ) -> ComparisonResult:
        """
        A/B test two skill descriptions.

        Args:
            original: Original skill description
            modified: Modified skill description
            test_tasks: All test tasks
            positive_task_indices: Indices of tasks that should match
            all_skills: All skills except the one being tested

        Returns:
            ComparisonResult showing which is better
        """
        if not self.llm_selector:
            raise ValueError("LLM selector required for comparison testing")

        # Separate positive and negative tasks
        positive_tasks = [test_tasks[i] for i in positive_task_indices]
        negative_tasks = [
            test_tasks[i] for i in range(len(test_tasks))
            if i not in positive_task_indices
        ]

        # Test original
        original_all_skills = [s for s in all_skills if s.name != modified.name] + [original]
        original_result = await self.test_effectiveness_async(
            original, positive_tasks, negative_tasks, original_all_skills
        )

        # Test modified
        modified_all_skills = [s for s in all_skills if s.name != original.name] + [modified]
        modified_result = await self.test_effectiveness_async(
            modified, positive_tasks, negative_tasks, modified_all_skills
        )

        # Calculate improvement
        improvement = (
            ((modified_result.f1_score - original_result.f1_score) / original_result.f1_score * 100)
            if original_result.f1_score > 0 else 0.0
        )

        # Determine winner
        if modified_result.f1_score > original_result.f1_score * 1.05:  # 5% improvement
            winner = "modified"
            recommendation = f"Use modified description (F1: {modified_result.f1_score:.2f} vs {original_result.f1_score:.2f})"
        elif original_result.f1_score > modified_result.f1_score * 1.05:
            winner = "original"
            recommendation = f"Keep original description (F1: {original_result.f1_score:.2f} vs {modified_result.f1_score:.2f})"
        else:
            winner = "tie"
            recommendation = f"Both descriptions perform similarly (F1 diff: {abs(modified_result.f1_score - original_result.f1_score):.3f})"

        return ComparisonResult(
            original_skill=original.name,
            modified_skill=modified.name,
            original_effectiveness=original_result,
            modified_effectiveness=modified_result,
            improvement=improvement,
            winner=winner,
            recommendation=recommendation
        )

    def compare_descriptions(
        self,
        original: SkillDescription,
        modified: SkillDescription,
        test_tasks: List[str],
        positive_task_indices: List[int],
        all_skills: List[SkillDescription]
    ) -> ComparisonResult:
        """Sync wrapper for compare_descriptions_async."""
        return asyncio.run(self.compare_descriptions_async(
            original, modified, test_tasks, positive_task_indices, all_skills
        ))

    def check_all_skills(
        self,
        skills: List[SkillDescription]
    ) -> Dict[str, QualityMetrics]:
        """
        Check quality for all skills.

        Args:
            skills: List of skills to check

        Returns:
            Dictionary mapping skill name to QualityMetrics
        """
        results = {}
        for skill in skills:
            results[skill.name] = self.check_quality(skill)
        return results

    def get_quality_summary(
        self,
        skills: List[SkillDescription]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a set of skills.

        Args:
            skills: List of skills to analyze

        Returns:
            Summary statistics
        """
        all_metrics = self.check_all_skills(skills)

        if not all_metrics:
            return {
                'total_skills': 0,
                'avg_overall_score': 0.0,
                'avg_completeness': 0.0,
                'avg_clarity': 0.0,
                'skills_with_warnings': 0,
                'total_warnings': 0,
            }

        overall_scores = [m.overall_score for m in all_metrics.values()]
        completeness_scores = [m.completeness_score for m in all_metrics.values()]
        clarity_scores = [m.clarity_score for m in all_metrics.values()]

        skills_with_warnings = sum(1 for m in all_metrics.values() if m.warnings)
        total_warnings = sum(len(m.warnings) for m in all_metrics.values())

        return {
            'total_skills': len(skills),
            'avg_overall_score': sum(overall_scores) / len(overall_scores),
            'avg_completeness': sum(completeness_scores) / len(completeness_scores),
            'avg_clarity': sum(clarity_scores) / len(clarity_scores),
            'skills_with_warnings': skills_with_warnings,
            'total_warnings': total_warnings,
            'skills_needing_improvement': sum(
                1 for m in all_metrics.values() if m.overall_score < 0.7
            ),
        }

    def generate_report(
        self,
        skills: List[SkillDescription],
        show_recommendations: bool = True
    ) -> str:
        """
        Generate a text report of skill quality.

        Args:
            skills: Skills to analyze
            show_recommendations: Include recommendations in report

        Returns:
            Formatted text report
        """
        all_metrics = self.check_all_skills(skills)
        summary = self.get_quality_summary(skills)

        lines = []
        lines.append("=" * 60)
        lines.append("SKILL DESCRIPTION QUALITY REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total skills analyzed: {summary['total_skills']}")
        lines.append(f"Average overall score: {summary['avg_overall_score']:.2f}")
        lines.append(f"Average completeness: {summary['avg_completeness']:.2f}")
        lines.append(f"Average clarity: {summary['avg_clarity']:.2f}")
        lines.append(f"Skills with warnings: {summary['skills_with_warnings']}")
        lines.append(f"Skills needing improvement: {summary['skills_needing_improvement']}")
        lines.append("")

        # Individual skill details
        for skill_name, metrics in sorted(all_metrics.items()):
            lines.append("-" * 60)
            lines.append(f"Skill: {skill_name}")
            lines.append(f"Overall Score: {metrics.overall_score:.2f}")
            lines.append(f"Completeness: {metrics.completeness_score:.2f} | Clarity: {metrics.clarity_score:.2f}")
            lines.append(f"Words: {metrics.word_count} | Tokens: {metrics.token_estimate}")

            if metrics.warnings:
                lines.append("⚠ Warnings:")
                for warning in metrics.warnings:
                    lines.append(f"  - {warning}")

            if show_recommendations and metrics.recommendations:
                lines.append("💡 Recommendations:")
                for rec in metrics.recommendations:
                    lines.append(f"  - {rec}")

            lines.append("")

        return "\n".join(lines)


def create_quality_checker(
    llm_selector: Optional[LLMSkillSelector] = None
) -> SkillDescriptionQualityChecker:
    """
    Convenience function to create a quality checker.

    Args:
        llm_selector: LLMSkillSelector for effectiveness testing (optional)

    Returns:
        Configured SkillDescriptionQualityChecker
    """
    return SkillDescriptionQualityChecker(llm_selector=llm_selector)
