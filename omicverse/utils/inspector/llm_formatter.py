"""
LLMFormatter - Format validation results for LLM consumption.

This module provides the LLMFormatter class which formats validation results
into natural language and LLM-friendly formats with prompt templates.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum

from .data_structures import ValidationResult, Suggestion
from .prerequisite_checker import DetectionResult


class OutputFormat(Enum):
    """Output format for LLM consumption."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    PROMPT = "prompt"


@dataclass
class LLMPrompt:
    """A prompt template for LLM agents.

    Attributes:
        system_prompt: System-level instructions for the LLM.
        user_prompt: User-level prompt with validation details.
        context: Additional context about the validation.
        suggestions: Formatted suggestions for the LLM.
    """

    system_prompt: str
    user_prompt: str
    context: Dict[str, Any]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'system': self.system_prompt,
            'user': self.user_prompt,
            'context': self.context,
            'suggestions': self.suggestions,
        }


class LLMFormatter:
    """Format validation results for LLM consumption.

    This class converts ValidationResult objects into natural language
    and LLM-friendly formats. It provides:
    - Natural language explanations
    - Prompt templates for LLM agents
    - Markdown/plain text formatting
    - JSON structured output

    Attributes:
        output_format: Default output format (markdown, plain_text, json, prompt).
        verbose: Include detailed explanations.

    Example:
        >>> formatter = LLMFormatter(output_format=OutputFormat.MARKDOWN)
        >>> formatted = formatter.format_validation_result(result)
        >>> print(formatted)
    """

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        verbose: bool = True,
    ):
        """Initialize LLM formatter.

        Args:
            output_format: Default output format.
            verbose: Include detailed explanations.
        """
        self.output_format = output_format
        self.verbose = verbose

    def format_validation_result(
        self,
        result: ValidationResult,
        format_override: Optional[OutputFormat] = None,
    ) -> str:
        """Format a validation result for LLM consumption.

        Args:
            result: ValidationResult to format.
            format_override: Override default output format.

        Returns:
            Formatted string in the specified format.

        Example:
            >>> formatted = formatter.format_validation_result(result)
        """
        fmt = format_override or self.output_format

        if fmt == OutputFormat.MARKDOWN:
            return self._format_markdown(result)
        elif fmt == OutputFormat.PLAIN_TEXT:
            return self._format_plain_text(result)
        elif fmt == OutputFormat.JSON:
            return self._format_json(result)
        elif fmt == OutputFormat.PROMPT:
            return self._format_prompt(result)
        else:
            return self._format_markdown(result)

    def create_agent_prompt(
        self,
        result: ValidationResult,
        task: str = "Fix the validation errors",
    ) -> LLMPrompt:
        """Create a prompt for an LLM agent.

        Args:
            result: ValidationResult with validation details.
            task: Task description for the agent.

        Returns:
            LLMPrompt with system and user prompts.

        Example:
            >>> prompt = formatter.create_agent_prompt(result, "Fix preprocessing issues")
            >>> print(prompt.system_prompt)
            >>> print(prompt.user_prompt)
        """
        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Build user prompt with validation details
        user_prompt = self._build_user_prompt(result, task)

        # Build context
        context = {
            'function_name': result.function_name,
            'is_valid': result.is_valid,
            'missing_prerequisites': result.missing_prerequisites,
            'missing_data_structures': result.missing_data_structures,
            'confidence_scores': result.confidence_scores,
        }

        # Format suggestions
        suggestions = [
            self._format_suggestion_for_agent(s) for s in result.suggestions
        ]

        return LLMPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            suggestions=suggestions,
        )

    def format_suggestion(
        self,
        suggestion: Suggestion,
        include_code: bool = True,
        include_explanation: bool = True,
    ) -> str:
        """Format a single suggestion for display.

        Args:
            suggestion: Suggestion to format.
            include_code: Include executable code.
            include_explanation: Include explanation.

        Returns:
            Formatted suggestion string.
        """
        lines = []

        # Priority and description
        lines.append(f"[{suggestion.priority}] {suggestion.description}")

        # Code (if requested)
        if include_code and suggestion.code:
            lines.append(f"\nCode:")
            lines.append(f"```python")
            lines.append(suggestion.code)
            lines.append(f"```")

        # Explanation (if requested)
        if include_explanation and suggestion.explanation:
            lines.append(f"\nWhy: {suggestion.explanation}")

        # Time estimate
        if suggestion.estimated_time:
            lines.append(f"Estimated time: {suggestion.estimated_time}")

        # Impact
        if suggestion.impact:
            lines.append(f"Impact: {suggestion.impact}")

        return "\n".join(lines)

    def _format_markdown(self, result: ValidationResult) -> str:
        """Format as markdown."""
        lines = []

        # Header
        status = "✅ Valid" if result.is_valid else "❌ Invalid"
        lines.append(f"# Validation Result: {result.function_name}")
        lines.append(f"\n**Status**: {status}")
        lines.append(f"\n**Message**: {result.message}")

        # Missing prerequisites
        if result.missing_prerequisites:
            lines.append(f"\n## Missing Prerequisites ({len(result.missing_prerequisites)})")
            for prereq in result.missing_prerequisites:
                confidence = result.confidence_scores.get(prereq, 0.0)
                lines.append(f"- `{prereq}` (confidence: {confidence:.2f})")

        # Missing data structures
        if result.missing_data_structures:
            lines.append(f"\n## Missing Data Structures")
            for struct_type, keys in result.missing_data_structures.items():
                lines.append(f"\n### {struct_type}")
                for key in keys:
                    lines.append(f"- `{key}`")

        # Executed functions
        if result.executed_functions:
            lines.append(f"\n## Executed Functions ({len(result.executed_functions)})")
            for func in result.executed_functions:
                confidence = result.confidence_scores.get(func, 0.0)
                lines.append(f"- `{func}` (confidence: {confidence:.2f})")

        # Suggestions
        if result.suggestions:
            lines.append(f"\n## Suggestions ({len(result.suggestions)})")
            for i, suggestion in enumerate(result.suggestions, 1):
                lines.append(f"\n### {i}. {suggestion.description}")
                lines.append(f"\n**Priority**: {suggestion.priority}")
                lines.append(f"**Type**: {suggestion.suggestion_type}")

                if suggestion.code:
                    lines.append(f"\n**Code**:")
                    lines.append(f"```python")
                    lines.append(suggestion.code)
                    lines.append(f"```")

                if suggestion.explanation and self.verbose:
                    lines.append(f"\n**Explanation**: {suggestion.explanation}")

                if suggestion.estimated_time:
                    lines.append(f"\n**Time**: {suggestion.estimated_time}")

        return "\n".join(lines)

    def _format_plain_text(self, result: ValidationResult) -> str:
        """Format as plain text."""
        lines = []

        # Header
        status = "VALID" if result.is_valid else "INVALID"
        lines.append(f"Validation Result: {result.function_name}")
        lines.append(f"Status: {status}")
        lines.append(f"Message: {result.message}")
        lines.append("")

        # Missing prerequisites
        if result.missing_prerequisites:
            lines.append(f"Missing Prerequisites ({len(result.missing_prerequisites)}):")
            for prereq in result.missing_prerequisites:
                confidence = result.confidence_scores.get(prereq, 0.0)
                lines.append(f"  - {prereq} (confidence: {confidence:.2f})")
            lines.append("")

        # Missing data structures
        if result.missing_data_structures:
            lines.append("Missing Data Structures:")
            for struct_type, keys in result.missing_data_structures.items():
                lines.append(f"  {struct_type}:")
                for key in keys:
                    lines.append(f"    - {key}")
            lines.append("")

        # Suggestions
        if result.suggestions:
            lines.append(f"Suggestions ({len(result.suggestions)}):")
            for i, suggestion in enumerate(result.suggestions, 1):
                lines.append(f"\n{i}. [{suggestion.priority}] {suggestion.description}")
                if suggestion.code:
                    lines.append(f"   Code: {suggestion.code}")
                if suggestion.estimated_time:
                    lines.append(f"   Time: {suggestion.estimated_time}")

        return "\n".join(lines)

    def _format_json(self, result: ValidationResult) -> str:
        """Format as JSON."""
        import json

        data = {
            'function_name': result.function_name,
            'is_valid': result.is_valid,
            'message': result.message,
            'missing_prerequisites': result.missing_prerequisites,
            'missing_data_structures': result.missing_data_structures,
            'executed_functions': result.executed_functions,
            'confidence_scores': result.confidence_scores,
            'suggestions': [
                {
                    'priority': s.priority,
                    'type': s.suggestion_type,
                    'description': s.description,
                    'code': s.code,
                    'explanation': s.explanation,
                    'estimated_time': s.estimated_time,
                    'estimated_time_seconds': s.estimated_time_seconds,
                    'impact': s.impact,
                }
                for s in result.suggestions
            ],
        }

        return json.dumps(data, indent=2)

    def _format_prompt(self, result: ValidationResult) -> str:
        """Format as LLM prompt."""
        prompt = self.create_agent_prompt(result)
        lines = []

        lines.append("=== SYSTEM PROMPT ===")
        lines.append(prompt.system_prompt)
        lines.append("\n=== USER PROMPT ===")
        lines.append(prompt.user_prompt)
        lines.append("\n=== SUGGESTIONS ===")
        for i, suggestion in enumerate(prompt.suggestions, 1):
            lines.append(f"\n{i}. {suggestion}")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM agent."""
        return """You are an expert bioinformatics assistant helping users with single-cell RNA-seq analysis using OmicVerse.

Your role is to:
1. Analyze validation results for preprocessing pipelines
2. Explain why certain steps are required
3. Suggest the correct sequence of operations
4. Provide executable Python code
5. Help users understand their data analysis workflow

When suggesting code:
- Use OmicVerse functions (ov.pp.*, ov.single.*, ov.bulk.*)
- Provide complete, executable examples
- Explain prerequisites and dependencies
- Consider computational time and complexity
- Suggest alternatives when appropriate

Be concise but thorough. Focus on helping the user successfully complete their analysis."""

    def _build_user_prompt(self, result: ValidationResult, task: str) -> str:
        """Build user prompt with validation details."""
        lines = []

        lines.append(f"Task: {task}")
        lines.append(f"\nFunction: {result.function_name}")
        lines.append(f"Status: {'Valid ✓' if result.is_valid else 'Invalid ✗'}")

        if not result.is_valid:
            lines.append(f"\nIssues Found:")

            if result.missing_prerequisites:
                lines.append(f"\nMissing Prerequisites ({len(result.missing_prerequisites)}):")
                for prereq in result.missing_prerequisites:
                    confidence = result.confidence_scores.get(prereq, 0.0)
                    lines.append(f"  - {prereq} (detection confidence: {confidence:.2f})")

            if result.missing_data_structures:
                lines.append(f"\nMissing Data Structures:")
                for struct_type, keys in result.missing_data_structures.items():
                    lines.append(f"  - {struct_type}: {', '.join(keys)}")

            lines.append(f"\nPlease analyze these issues and provide:")
            lines.append(f"1. A clear explanation of what's wrong")
            lines.append(f"2. The correct sequence of steps to fix it")
            lines.append(f"3. Executable Python code")
            lines.append(f"4. Any important considerations or warnings")

        return "\n".join(lines)

    def _format_suggestion_for_agent(self, suggestion: Suggestion) -> str:
        """Format suggestion for agent consumption."""
        lines = []

        lines.append(f"[{suggestion.priority}] {suggestion.description}")

        if suggestion.code:
            lines.append(f"Code: {suggestion.code}")

        if suggestion.explanation:
            lines.append(f"Why: {suggestion.explanation}")

        if suggestion.estimated_time:
            lines.append(f"Time: {suggestion.estimated_time}")

        if suggestion.impact:
            lines.append(f"Impact: {suggestion.impact}")

        return "\n".join(lines)

    def format_natural_language(self, result: ValidationResult) -> str:
        """Format as natural language explanation.

        Args:
            result: ValidationResult to explain.

        Returns:
            Natural language explanation suitable for users.

        Example:
            >>> explanation = formatter.format_natural_language(result)
            >>> print(explanation)
        """
        lines = []

        if result.is_valid:
            lines.append(f"✅ All requirements are satisfied for {result.function_name}!")
            if result.executed_functions:
                lines.append(f"\nDetected {len(result.executed_functions)} prerequisite function(s) already executed:")
                for func in result.executed_functions:
                    confidence = result.confidence_scores.get(func, 0.0)
                    lines.append(f"  • {func} (confidence: {confidence:.1%})")
        else:
            lines.append(f"❌ Cannot run {result.function_name} yet.")

            if result.missing_prerequisites:
                lines.append(f"\nYou need to run {len(result.missing_prerequisites)} prerequisite function(s) first:")
                for prereq in result.missing_prerequisites:
                    confidence = result.confidence_scores.get(prereq, 0.0)
                    detection_status = "not detected" if confidence < 0.5 else f"uncertain (confidence: {confidence:.1%})"
                    lines.append(f"  • {prereq} - {detection_status}")

            if result.missing_data_structures:
                total_missing = sum(len(keys) for keys in result.missing_data_structures.values())
                lines.append(f"\nMissing {total_missing} required data structure(s):")
                for struct_type, keys in result.missing_data_structures.items():
                    lines.append(f"  • {struct_type}: {', '.join(keys)}")

            if result.suggestions:
                lines.append(f"\n📋 Recommendations:")

                # Group by priority
                by_priority = {}
                for s in result.suggestions:
                    if s.priority not in by_priority:
                        by_priority[s.priority] = []
                    by_priority[s.priority].append(s)

                for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    if priority in by_priority:
                        suggestions = by_priority[priority]
                        lines.append(f"\n{priority} Priority ({len(suggestions)}):")
                        for s in suggestions[:2]:  # Show top 2 per priority
                            lines.append(f"  • {s.description}")
                            if s.estimated_time:
                                lines.append(f"    Time: {s.estimated_time}")

        return "\n".join(lines)

    def format_for_llm_agent(
        self,
        result: ValidationResult,
        agent_type: Literal["code_generator", "explainer", "debugger"] = "code_generator",
    ) -> Dict[str, Any]:
        """Format specifically for different types of LLM agents.

        Args:
            result: ValidationResult to format.
            agent_type: Type of agent (code_generator, explainer, debugger).

        Returns:
            Dictionary with agent-specific formatting.

        Example:
            >>> formatted = formatter.format_for_llm_agent(result, "code_generator")
            >>> print(formatted['task'])
        """
        base_context = {
            'function': result.function_name,
            'is_valid': result.is_valid,
            'missing_prerequisites': result.missing_prerequisites,
            'missing_data': result.missing_data_structures,
        }

        if agent_type == "code_generator":
            return {
                'task': f"Generate executable Python code to satisfy requirements for {result.function_name}",
                'context': base_context,
                'requirements': [
                    f"Must run: {', '.join(result.missing_prerequisites)}" if result.missing_prerequisites else None,
                    f"Must create: {', '.join(k for keys in result.missing_data_structures.values() for k in keys)}" if result.missing_data_structures else None,
                ],
                'code_templates': [s.code for s in result.suggestions if s.code],
                'constraints': [
                    'Use OmicVerse functions (ov.pp.*, ov.single.*, ov.bulk.*)',
                    'Ensure correct execution order',
                    'Include error handling',
                ],
            }

        elif agent_type == "explainer":
            return {
                'task': f"Explain what's needed to run {result.function_name}",
                'context': base_context,
                'explanation_points': [
                    'Why each prerequisite is needed',
                    'What data structures are missing and why',
                    'The recommended workflow order',
                    'Common pitfalls to avoid',
                ],
                'suggestions': [s.explanation for s in result.suggestions if s.explanation],
            }

        elif agent_type == "debugger":
            return {
                'task': f"Debug why {result.function_name} cannot run",
                'context': base_context,
                'diagnostic_info': {
                    'executed_functions': result.executed_functions,
                    'confidence_scores': result.confidence_scores,
                    'missing_prerequisites': result.missing_prerequisites,
                    'missing_data': result.missing_data_structures,
                },
                'debug_steps': [
                    'Check if prerequisites were executed',
                    'Verify required data structures exist',
                    'Examine confidence scores for uncertainty',
                    'Suggest recovery actions',
                ],
            }

        return base_context
