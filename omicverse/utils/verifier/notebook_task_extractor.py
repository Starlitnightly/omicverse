"""
Notebook Task Extractor - Phase 4

Extracts task descriptions from Jupyter notebooks (.ipynb files).
Maps tasks to skills based on code analysis and manual ground truth.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

from .data_structures import NotebookTask, SkillDescription


@dataclass
class ExtractedTask:
    """Intermediate representation of an extracted task."""
    description: str
    section_title: Optional[str]
    code_snippets: List[str]
    omicverse_functions: List[str]  # OV functions called


class NotebookTaskExtractor:
    """
    Extract task descriptions from Jupyter notebooks.

    Analyzes notebooks to identify:
    1. Task descriptions from markdown cells
    2. Code patterns that map to skills
    3. Ground truth skill mappings
    """

    # Task indicator patterns in markdown
    TASK_INDICATORS = [
        r"in this tutorial",
        r"we will",
        r"the goal is",
        r"this tutorial demonstrates",
        r"we demonstrate",
        r"here we",
        r"to (?:perform|do|analyze|process)",
        r"an important task",
    ]

    # OmicVerse function patterns for skill mapping
    SKILL_FUNCTION_PATTERNS = {
        'bulk-deg-analysis': [
            r'ov\.bulk\.pyDEG',
            r'dds\.deg_analysis',
            r'ov\.bulk\.geneset_enrichment',
            r'dds\.normalize',
            r'dds\.foldchange_set',
        ],
        'bulk-deseq2-analysis': [
            r'ov\.bulk\.pyDESeq2',
            r'dds\.deseq2',
        ],
        'bulk-wgcna-analysis': [
            r'ov\.bulk\.pyWGCNA',
            r'wgcna\.preprocess',
            r'wgcna\.calculate_soft_threshold',
        ],
        'bulk-combat-correction': [
            r'ov\.bulk\.combat',
            r'pyComBat',
        ],
        'single-preprocessing': [
            r'ov\.pp\.qc',
            r'ov\.pp\.preprocess',
            r'sc\.pp\.highly_variable_genes',
            r'adata\.raw',
        ],
        'single-clustering': [
            r'ov\.single\.cluster',
            r'ov\.single\.leiden',
            r'ov\.single\.louvain',
            r'sc\.tl\.leiden',
        ],
        'single-annotation': [
            r'ov\.single\.CellVote',
            r'ov\.single\.SCSA',
            r'ov\.single\.GPTAnno',
        ],
        'spatial-tutorials': [
            r'ov\.space',
            r'sq\.gr\.',
        ],
        'tcga-preprocessing': [
            r'ov\.bulk\.TCGASurvival',
            r'ov\.bulk\.TCGA',
        ],
    }

    def __init__(self):
        """Initialize notebook task extractor."""
        if not NBFORMAT_AVAILABLE:
            raise ImportError(
                "nbformat is required for notebook parsing. "
                "Install with: pip install nbformat"
            )

        self.ground_truth = self._build_ground_truth_mapping()

    def _build_ground_truth_mapping(self) -> Dict[str, List[str]]:
        """
        Build ground truth mapping: notebook basename → expected skills.

        This is based on the skill descriptions' references to notebooks.
        """
        mapping = {
            # Bulk tutorials
            't_deg.ipynb': ['bulk-deg-analysis'],
            't_deseq2.ipynb': ['bulk-deseq2-analysis'],
            't_wgcna.ipynb': ['bulk-wgcna-analysis'],
            't_bulk_combat.ipynb': ['bulk-combat-correction'],
            't_network.ipynb': ['bulk-stringdb-ppi'],

            # Single-cell tutorials
            't_preprocess.ipynb': ['single-preprocessing'],
            't_preprocess_cpu.ipynb': ['single-preprocessing'],
            't_preprocess_gpu.ipynb': ['single-preprocessing'],
            't_cluster.ipynb': ['single-clustering'],
            't_single_batch.ipynb': ['single-clustering'],

            # Annotation tutorials
            't_cellvote.ipynb': ['single-annotation'],
            't_scsa.ipynb': ['single-annotation'],
            't_gptanno.ipynb': ['single-annotation'],

            # Spatial tutorials
            't_cellpose.ipynb': ['spatial-tutorials'],
            't_spatial.ipynb': ['spatial-tutorials'],

            # TCGA tutorials
            't_tcga.ipynb': ['tcga-preprocessing'],

            # Multi-skill workflows
            't_single_batch.ipynb': ['single-preprocessing', 'single-clustering'],
        }
        return mapping

    def extract_from_notebook(
        self,
        notebook_path: str,
        category: Optional[str] = None
    ) -> List[NotebookTask]:
        """
        Extract tasks from a Jupyter notebook.

        Args:
            notebook_path: Path to .ipynb file
            category: Category hint (bulk, single-cell, spatial, etc.)

        Returns:
            List of NotebookTask objects
        """
        path = Path(notebook_path)
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        # Read notebook
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Extract tasks
        tasks = []

        # Get main task from title and introduction
        main_task = self._extract_main_task(nb, path)
        if main_task:
            tasks.append(main_task)

        # Get sub-tasks from sections
        sub_tasks = self._extract_sub_tasks(nb, path)
        tasks.extend(sub_tasks)

        # Infer category if not provided
        if category is None:
            category = self._infer_category(path)

        # Set category for all tasks
        for task in tasks:
            task.category = category

        return tasks

    def _extract_main_task(self, nb, notebook_path: Path) -> Optional[NotebookTask]:
        """Extract the main task from notebook title and introduction."""
        # Get first few markdown cells
        markdown_cells = [
            cell for cell in nb.cells
            if cell.cell_type == 'markdown'
        ][:5]  # First 5 markdown cells

        if not markdown_cells:
            return None

        # Extract title (usually first H1)
        title = None
        description_parts = []

        for cell in markdown_cells:
            source = cell.source
            lines = source.split('\n')

            for line in lines:
                # H1 heading
                if line.startswith('# ') and not title:
                    title = line[2:].strip()
                    continue

                # Look for task descriptions
                for pattern in self.TASK_INDICATORS:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Clean up the line
                        cleaned = re.sub(r'<.*?>', '', line)  # Remove HTML
                        cleaned = re.sub(r'\*\*', '', cleaned)  # Remove bold
                        cleaned = cleaned.strip()
                        if cleaned and len(cleaned) > 20:
                            description_parts.append(cleaned)
                        break

        if not title:
            return None

        # Combine title and description
        if description_parts:
            full_description = f"{title}. {' '.join(description_parts[:2])}"
        else:
            full_description = title

        # Get code patterns
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        all_code = '\n'.join([cell.source for cell in code_cells])
        matched_skills = self._match_skills_from_code(all_code)

        # Get ground truth
        basename = notebook_path.name
        expected_skills = self.ground_truth.get(basename, matched_skills)

        if not expected_skills:
            expected_skills = matched_skills if matched_skills else ['unknown']

        # Create task
        task_id = f"{basename.replace('.ipynb', '')}-main"

        return NotebookTask(
            task_id=task_id,
            notebook_path=str(notebook_path),
            task_description=full_description,
            expected_skills=expected_skills,
            expected_order=expected_skills,  # Main task assumes sequential order
            category=self._infer_category(notebook_path),
            difficulty='workflow' if len(expected_skills) > 1 else 'single'
        )

    def _extract_sub_tasks(self, nb, notebook_path: Path) -> List[NotebookTask]:
        """Extract sub-tasks from notebook sections."""
        tasks = []
        current_section = None
        section_code = []

        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                source = cell.source
                # Check for H2 or H3 headings
                if source.startswith('## ') or source.startswith('### '):
                    # Save previous section if it exists
                    if current_section and section_code:
                        task = self._create_task_from_section(
                            current_section,
                            section_code,
                            notebook_path
                        )
                        if task:
                            tasks.append(task)

                    # Start new section
                    if source.startswith('## '):
                        current_section = source[3:].strip()
                    else:
                        current_section = source[4:].strip()
                    section_code = []

            elif cell.cell_type == 'code' and current_section:
                section_code.append(cell.source)

        # Save last section
        if current_section and section_code:
            task = self._create_task_from_section(
                current_section,
                section_code,
                notebook_path
            )
            if task:
                tasks.append(task)

        return tasks

    def _create_task_from_section(
        self,
        section_title: str,
        code_cells: List[str],
        notebook_path: Path
    ) -> Optional[NotebookTask]:
        """Create a task from a notebook section."""
        # Skip certain sections
        skip_patterns = [
            'import', 'installation', 'reference', 'citation',
            'optional', 'acknowledgment'
        ]

        if any(pattern in section_title.lower() for pattern in skip_patterns):
            return None

        # Get matched skills from code
        all_code = '\n'.join(code_cells)
        matched_skills = self._match_skills_from_code(all_code)

        if not matched_skills:
            return None

        # Create task description
        task_description = f"{section_title} in {notebook_path.stem}"

        task_id = f"{notebook_path.stem}-{section_title.lower().replace(' ', '-')[:30]}"

        return NotebookTask(
            task_id=task_id,
            notebook_path=str(notebook_path),
            task_description=task_description,
            expected_skills=matched_skills,
            expected_order=matched_skills,
            category=self._infer_category(notebook_path),
            difficulty='single' if len(matched_skills) == 1 else 'workflow'
        )

    def _match_skills_from_code(self, code: str) -> List[str]:
        """Match skills based on function calls in code."""
        matched = set()

        for skill_name, patterns in self.SKILL_FUNCTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code):
                    matched.add(skill_name)
                    break  # One match per skill is enough

        return sorted(list(matched))

    def _infer_category(self, notebook_path: Path) -> str:
        """Infer category from notebook path."""
        path_str = str(notebook_path).lower()

        if 'bulk' in path_str:
            return 'bulk'
        elif 'single' in path_str:
            return 'single-cell'
        elif 'space' in path_str or 'spatial' in path_str:
            return 'spatial'
        elif 'tcga' in path_str:
            return 'tcga'
        elif 'plot' in path_str:
            return 'plotting'
        else:
            return 'other'

    def extract_from_directory(
        self,
        directory: str,
        pattern: str = "**/*.ipynb"
    ) -> List[NotebookTask]:
        """
        Extract tasks from all notebooks in a directory.

        Args:
            directory: Path to directory containing notebooks
            pattern: Glob pattern for finding notebooks

        Returns:
            List of all extracted NotebookTask objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_tasks = []
        notebook_files = sorted(dir_path.glob(pattern))

        for notebook_path in notebook_files:
            try:
                tasks = self.extract_from_notebook(str(notebook_path))
                all_tasks.extend(tasks)
            except Exception as e:
                print(f"Warning: Failed to extract from {notebook_path}: {e}")
                continue

        return all_tasks

    def get_coverage_statistics(
        self,
        tasks: List[NotebookTask],
        all_skills: List[SkillDescription]
    ) -> Dict[str, any]:
        """
        Get coverage statistics for extracted tasks.

        Args:
            tasks: List of extracted tasks
            all_skills: List of all available skills

        Returns:
            Coverage statistics
        """
        if not tasks:
            return {
                'total_tasks': 0,
                'total_notebooks': 0,
                'skills_covered': 0,
                'skills_not_covered': 0,
                'coverage_percentage': 0.0,
            }

        # Count unique notebooks
        unique_notebooks = set(task.notebook_path for task in tasks)

        # Count skills covered
        skills_in_tasks = set()
        for task in tasks:
            skills_in_tasks.update(task.expected_skills)

        # Remove 'unknown'
        skills_in_tasks.discard('unknown')

        all_skill_names = set(skill.name for skill in all_skills)
        skills_covered = skills_in_tasks & all_skill_names
        skills_not_covered = all_skill_names - skills_covered

        coverage_pct = (
            len(skills_covered) / len(all_skill_names) * 100
            if all_skill_names else 0.0
        )

        return {
            'total_tasks': len(tasks),
            'total_notebooks': len(unique_notebooks),
            'unique_skills_in_tasks': len(skills_in_tasks),
            'skills_covered': len(skills_covered),
            'skills_not_covered': len(skills_not_covered),
            'coverage_percentage': coverage_pct,
            'covered_skills': sorted(list(skills_covered)),
            'not_covered_skills': sorted(list(skills_not_covered)),
        }

    def save_tasks_to_json(
        self,
        tasks: List[NotebookTask],
        output_path: str
    ):
        """
        Save extracted tasks to JSON file.

        Args:
            tasks: List of NotebookTask objects
            output_path: Path to output JSON file
        """
        tasks_data = []
        for task in tasks:
            tasks_data.append({
                'task_id': task.task_id,
                'notebook_path': task.notebook_path,
                'task_description': task.task_description,
                'expected_skills': task.expected_skills,
                'expected_order': task.expected_order,
                'category': task.category,
                'difficulty': task.difficulty,
                'context': task.context,
                'alternate_acceptable': task.alternate_acceptable,
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'tasks': tasks_data}, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_tasks_from_json(json_path: str) -> List[NotebookTask]:
        """
        Load tasks from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            List of NotebookTask objects
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        for task_data in data.get('tasks', []):
            task = NotebookTask(
                task_id=task_data['task_id'],
                notebook_path=task_data['notebook_path'],
                task_description=task_data['task_description'],
                expected_skills=task_data['expected_skills'],
                expected_order=task_data['expected_order'],
                category=task_data['category'],
                difficulty=task_data.get('difficulty', 'single'),
                context=task_data.get('context', {}),
                alternate_acceptable=task_data.get('alternate_acceptable', []),
            )
            tasks.append(task)

        return tasks


def create_task_extractor() -> NotebookTaskExtractor:
    """
    Convenience function to create a notebook task extractor.

    Returns:
        Configured NotebookTaskExtractor
    """
    return NotebookTaskExtractor()
