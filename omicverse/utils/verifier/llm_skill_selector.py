"""
LLM Skill Selector - Phase 2

Uses pure LLM reasoning to select skills based on task descriptions,
mimicking how Claude Code autonomously selects skills.

Key principle: No algorithmic routing, embeddings, or classifiers.
The LLM reads skill descriptions and uses language understanding to match tasks.
"""

import asyncio
import json
import re
from typing import List, Optional, Union

from ..agent_backend import OmicVerseLLMBackend
from .data_structures import SkillDescription, NotebookTask, LLMSelectionResult


class LLMSkillSelector:
    """
    Uses LLM reasoning to select skills (mimics Claude Code's behavior).

    This selector uses pure language understanding to match tasks against skill
    descriptions. It does NOT use:
    - Algorithmic routing or keyword matching
    - Embeddings or vector similarity
    - Classifiers or pattern matching

    The LLM reads the list of skill descriptions and autonomously decides
    which skills to use based on the task description.
    """

    def __init__(
        self,
        llm_backend: Optional[OmicVerseLLMBackend] = None,
        skill_descriptions: Optional[List[SkillDescription]] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,  # Deterministic for testing
    ):
        """
        Initialize the LLM skill selector.

        Args:
            llm_backend: Pre-configured LLM backend (optional)
            skill_descriptions: List of available skills
            model: Model to use if creating new backend (default: gpt-4o-mini)
            api_key: API key if creating new backend
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.skill_descriptions = skill_descriptions or []

        if llm_backend is not None:
            self.llm = llm_backend
        else:
            # Create a new backend with skill selection system prompt
            system_prompt = self._build_system_prompt()
            self.llm = OmicVerseLLMBackend(
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=2000,  # Enough for JSON response
            )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for skill selection."""
        return """You are a skill selection assistant for OmicVerse, a bioinformatics analysis framework.

Your task is to analyze user requests and select the appropriate skills needed to complete the task.

You will be given:
1. A list of available skills (name + description)
2. A user task description

You must:
1. Understand what the user wants to accomplish
2. Select the skill(s) needed to complete the task
3. Order the skills in the correct execution sequence (if multiple)
4. Explain your reasoning

Guidelines:
- Only select skills that are necessary for the task
- If multiple skills are needed, order them by dependencies (e.g., preprocessing before clustering)
- If the task is ambiguous, choose the most likely interpretation
- If no skills match, return an empty skills list

Always respond in JSON format:
{
  "skills": ["skill-name-1", "skill-name-2"],
  "order": ["skill-name-1", "skill-name-2"],
  "reasoning": "Brief explanation of why these skills were selected and in this order"
}"""

    def set_skill_descriptions(self, skills: List[SkillDescription]):
        """Update the list of available skills."""
        self.skill_descriptions = skills

    def _format_skills_for_prompt(self) -> str:
        """Format skill descriptions for the LLM prompt."""
        if not self.skill_descriptions:
            return "No skills available."

        lines = ["Available skills:"]
        for skill in sorted(self.skill_descriptions, key=lambda s: s.name):
            lines.append(f"- {skill.name}: {skill.description}")

        return "\n".join(lines)

    def _build_selection_prompt(self, task_description: str) -> str:
        """Build the user prompt for skill selection."""
        skills_text = self._format_skills_for_prompt()

        prompt = f"""{skills_text}

User task: {task_description}

Which skill(s) should be used to complete this task? If multiple skills are needed, in what order should they be executed?

Respond in JSON format:
{{
  "skills": ["skill-name-1", "skill-name-2"],
  "order": ["skill-name-1", "skill-name-2"],
  "reasoning": "Brief explanation"
}}"""

        return prompt

    def _parse_llm_response(self, response_text: str, task_id: str) -> LLMSelectionResult:
        """
        Parse LLM response into LLMSelectionResult.

        Handles various response formats:
        - Clean JSON
        - JSON wrapped in markdown code blocks
        - Partial JSON
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                # No JSON found - return empty result
                return LLMSelectionResult(
                    task_id=task_id,
                    selected_skills=[],
                    skill_order=[],
                    reasoning="Failed to parse LLM response - no JSON found",
                    raw_response=response_text
                )

        try:
            parsed = json.loads(json_text)

            # Extract fields with defaults
            selected_skills = parsed.get('skills', [])
            skill_order = parsed.get('order', selected_skills.copy())
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            # Validate types
            if not isinstance(selected_skills, list):
                selected_skills = [str(selected_skills)] if selected_skills else []
            if not isinstance(skill_order, list):
                skill_order = selected_skills.copy()

            # Ensure all elements are strings
            selected_skills = [str(s) for s in selected_skills]
            skill_order = [str(s) for s in skill_order]

            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=selected_skills,
                skill_order=skill_order,
                reasoning=str(reasoning),
                raw_response=response_text
            )

        except json.JSONDecodeError as e:
            # JSON parsing failed - return empty result
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=[],
                skill_order=[],
                reasoning=f"Failed to parse JSON: {e}",
                raw_response=response_text
            )

    async def select_skills_async(
        self,
        task: Union[NotebookTask, str]
    ) -> LLMSelectionResult:
        """
        Select skills using LLM reasoning (async).

        Args:
            task: NotebookTask or task description string

        Returns:
            LLMSelectionResult with selected skills and reasoning
        """
        # Extract task description and ID
        if isinstance(task, NotebookTask):
            task_description = task.task_description
            task_id = task.task_id
        else:
            task_description = task
            task_id = "adhoc"

        # Validate input
        if not task_description or not task_description.strip():
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=[],
                skill_order=[],
                reasoning="Empty task description",
                raw_response=""
            )

        if not self.skill_descriptions:
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=[],
                skill_order=[],
                reasoning="No skills available",
                raw_response=""
            )

        # Build prompt
        prompt = self._build_selection_prompt(task_description)

        # Call LLM
        try:
            response_text = await self.llm.run(prompt)
        except Exception as e:
            return LLMSelectionResult(
                task_id=task_id,
                selected_skills=[],
                skill_order=[],
                reasoning=f"LLM call failed: {e}",
                raw_response=""
            )

        # Parse response
        result = self._parse_llm_response(response_text, task_id)

        return result

    def select_skills(
        self,
        task: Union[NotebookTask, str]
    ) -> LLMSelectionResult:
        """
        Select skills using LLM reasoning (synchronous wrapper).

        Args:
            task: NotebookTask or task description string

        Returns:
            LLMSelectionResult with selected skills and reasoning
        """
        return asyncio.run(self.select_skills_async(task))

    async def select_skills_batch_async(
        self,
        tasks: List[Union[NotebookTask, str]]
    ) -> List[LLMSelectionResult]:
        """
        Select skills for multiple tasks in parallel (async).

        Args:
            tasks: List of NotebookTask or task description strings

        Returns:
            List of LLMSelectionResult
        """
        # Create tasks for parallel execution
        coroutines = [self.select_skills_async(task) for task in tasks]

        # Run in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_id = tasks[i].task_id if isinstance(tasks[i], NotebookTask) else f"batch-{i}"
                final_results.append(LLMSelectionResult(
                    task_id=task_id,
                    selected_skills=[],
                    skill_order=[],
                    reasoning=f"Error: {result}",
                    raw_response=""
                ))
            else:
                final_results.append(result)

        return final_results

    def select_skills_batch(
        self,
        tasks: List[Union[NotebookTask, str]]
    ) -> List[LLMSelectionResult]:
        """
        Select skills for multiple tasks in parallel (synchronous wrapper).

        Args:
            tasks: List of NotebookTask or task description strings

        Returns:
            List of LLMSelectionResult
        """
        return asyncio.run(self.select_skills_batch_async(tasks))


def create_skill_selector(
    skill_descriptions: List[SkillDescription],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0
) -> LLMSkillSelector:
    """
    Convenience function to create a skill selector.

    Args:
        skill_descriptions: List of available skills
        model: Model to use (default: gpt-4o-mini)
        api_key: API key (optional, will use environment variable)
        temperature: Sampling temperature (0.0 for deterministic)

    Returns:
        Configured LLMSkillSelector
    """
    return LLMSkillSelector(
        skill_descriptions=skill_descriptions,
        model=model,
        api_key=api_key,
        temperature=temperature
    )
