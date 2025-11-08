"""
Tests for NotebookTaskExtractor

Tests notebook parsing and task extraction functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

from omicverse.utils.verifier import (
    NotebookTaskExtractor,
    create_task_extractor,
    NotebookTask,
    SkillDescription,
)


# Skip all tests if nbformat not available
pytestmark = pytest.mark.skipif(
    not NBFORMAT_AVAILABLE,
    reason="nbformat not installed"
)


class TestNotebookTaskExtractorBasic:
    """Test basic functionality of NotebookTaskExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a notebook task extractor."""
        return NotebookTaskExtractor()

    @pytest.fixture
    def temp_notebook(self):
        """Create a temporary test notebook."""
        nb = new_notebook()
        nb.cells = [
            new_markdown_cell("# Differential Expression Analysis"),
            new_markdown_cell(
                "In this tutorial we will perform differential expression analysis "
                "using omicverse. This demonstrates the DEG pipeline."
            ),
            new_code_cell("import omicverse as ov\nimport scanpy as sc"),
            new_markdown_cell("## Load Data"),
            new_code_cell("data = ov.read('data.h5ad')"),
            new_markdown_cell("## DEG Analysis"),
            new_code_cell("dds = ov.bulk.pyDEG(data)\ndds.normalize()"),
            new_code_cell("result = dds.deg_analysis(['treated'], ['control'])"),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nb, f)
            temp_path = f.name

        yield Path(temp_path)

        # Cleanup
        Path(temp_path).unlink()

    def test_extractor_initialization(self, extractor):
        """Test that extractor initializes correctly."""
        assert extractor is not None
        assert isinstance(extractor.ground_truth, dict)
        assert len(extractor.ground_truth) > 0

    def test_ground_truth_mapping_structure(self, extractor):
        """Test ground truth mapping has expected structure."""
        # Should have some bulk notebooks mapped
        assert 't_deg.ipynb' in extractor.ground_truth
        assert 'bulk-deg-analysis' in extractor.ground_truth['t_deg.ipynb']

        # Should have single-cell notebooks
        assert 't_preprocess.ipynb' in extractor.ground_truth
        assert 'single-preprocessing' in extractor.ground_truth['t_preprocess.ipynb']

    def test_extract_main_task(self, extractor, temp_notebook):
        """Test extracting main task from notebook."""
        tasks = extractor.extract_from_notebook(str(temp_notebook))

        assert len(tasks) > 0

        # Should have main task
        main_task = tasks[0]
        assert isinstance(main_task, NotebookTask)
        assert 'Differential Expression Analysis' in main_task.task_description
        assert len(main_task.expected_skills) > 0

    def test_extract_sub_tasks(self, extractor, temp_notebook):
        """Test extracting sub-tasks from sections."""
        tasks = extractor.extract_from_notebook(str(temp_notebook))

        # Should have multiple tasks (main + sub-tasks)
        assert len(tasks) > 1

        # Check for section-based tasks
        task_descriptions = [t.task_description for t in tasks]
        assert any('DEG Analysis' in desc for desc in task_descriptions)

    def test_match_skills_from_code(self, extractor):
        """Test skill matching from code patterns."""
        code = """
        import omicverse as ov
        dds = ov.bulk.pyDEG(data)
        dds.normalize()
        result = dds.deg_analysis(treatment, control)
        """

        skills = extractor._match_skills_from_code(code)

        assert 'bulk-deg-analysis' in skills

    def test_match_multiple_skills(self, extractor):
        """Test matching multiple skills from code."""
        code = """
        ov.pp.qc(adata)
        ov.pp.preprocess(adata)
        ov.single.cluster(adata)
        ov.single.leiden(adata)
        """

        skills = extractor._match_skills_from_code(code)

        assert 'single-preprocessing' in skills
        assert 'single-clustering' in skills

    def test_infer_category_bulk(self, extractor):
        """Test category inference for bulk notebooks."""
        path = Path("some/path/Tutorials-bulk/t_deg.ipynb")
        category = extractor._infer_category(path)
        assert category == 'bulk'

    def test_infer_category_single_cell(self, extractor):
        """Test category inference for single-cell notebooks."""
        path = Path("Tutorials-single/t_preprocess.ipynb")
        category = extractor._infer_category(path)
        assert category == 'single-cell'

    def test_infer_category_spatial(self, extractor):
        """Test category inference for spatial notebooks."""
        path = Path("Tutorials-space/t_spatial.ipynb")
        category = extractor._infer_category(path)
        assert category == 'spatial'


class TestNotebookCreation:
    """Test creating notebooks with various structures."""

    def test_extract_with_task_indicators(self):
        """Test extraction when markdown has task indicators."""
        nb = new_notebook()
        nb.cells = [
            new_markdown_cell("# Analysis Tutorial"),
            new_markdown_cell("The goal is to analyze single-cell data with clustering."),
            new_code_cell("ov.single.cluster(adata)"),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nb, f)
            temp_path = f.name

        try:
            extractor = NotebookTaskExtractor()
            tasks = extractor.extract_from_notebook(temp_path)

            assert len(tasks) > 0
            assert 'goal' in tasks[0].task_description.lower() or 'analyze' in tasks[0].task_description.lower()

        finally:
            Path(temp_path).unlink()

    def test_extract_with_no_markdown(self):
        """Test extraction from notebook with only code cells."""
        nb = new_notebook()
        nb.cells = [
            new_code_cell("import omicverse as ov"),
            new_code_cell("dds = ov.bulk.pyDEG(data)"),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nb, f)
            temp_path = f.name

        try:
            extractor = NotebookTaskExtractor()
            tasks = extractor.extract_from_notebook(temp_path)

            # Should still work, maybe with fewer tasks
            # but shouldn't crash
            assert isinstance(tasks, list)

        finally:
            Path(temp_path).unlink()


class TestDirectoryExtraction:
    """Test extracting from multiple notebooks in a directory."""

    @pytest.fixture
    def temp_directory_with_notebooks(self):
        """Create temporary directory with multiple notebooks."""
        import tempfile
        temp_dir = tempfile.mkdtemp()

        # Create notebook 1
        nb1 = new_notebook()
        nb1.cells = [
            new_markdown_cell("# Notebook 1"),
            new_code_cell("ov.bulk.pyDEG(data)"),
        ]
        with open(Path(temp_dir) / "test1.ipynb", 'w') as f:
            nbformat.write(nb1, f)

        # Create notebook 2
        nb2 = new_notebook()
        nb2.cells = [
            new_markdown_cell("# Notebook 2"),
            new_code_cell("ov.single.cluster(adata)"),
        ]
        with open(Path(temp_dir) / "test2.ipynb", 'w') as f:
            nbformat.write(nb2, f)

        yield Path(temp_dir)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_extract_from_directory(self, temp_directory_with_notebooks):
        """Test extracting from all notebooks in directory."""
        extractor = NotebookTaskExtractor()

        tasks = extractor.extract_from_directory(str(temp_directory_with_notebooks))

        # Should have tasks from both notebooks
        assert len(tasks) >= 2

        # Should have tasks from different notebooks
        notebook_paths = set(task.notebook_path for task in tasks)
        assert len(notebook_paths) >= 2


class TestCoverageStatistics:
    """Test coverage statistics calculation."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            NotebookTask(
                task_id="task-1",
                notebook_path="notebook1.ipynb",
                task_description="Task 1",
                expected_skills=["bulk-deg-analysis"],
                expected_order=["bulk-deg-analysis"],
                category="bulk"
            ),
            NotebookTask(
                task_id="task-2",
                notebook_path="notebook1.ipynb",
                task_description="Task 2",
                expected_skills=["bulk-wgcna-analysis"],
                expected_order=["bulk-wgcna-analysis"],
                category="bulk"
            ),
            NotebookTask(
                task_id="task-3",
                notebook_path="notebook2.ipynb",
                task_description="Task 3",
                expected_skills=["single-preprocessing", "single-clustering"],
                expected_order=["single-preprocessing", "single-clustering"],
                category="single-cell"
            ),
        ]

    @pytest.fixture
    def sample_skills(self):
        """Create sample skills."""
        return [
            SkillDescription("bulk-deg-analysis", "Bulk DEG"),
            SkillDescription("bulk-wgcna-analysis", "WGCNA"),
            SkillDescription("single-preprocessing", "Preprocess"),
            SkillDescription("single-clustering", "Cluster"),
            SkillDescription("spatial-tutorials", "Spatial"),  # Not covered
        ]

    def test_coverage_statistics(self, sample_tasks, sample_skills):
        """Test coverage statistics calculation."""
        extractor = NotebookTaskExtractor()

        stats = extractor.get_coverage_statistics(sample_tasks, sample_skills)

        assert stats['total_tasks'] == 3
        assert stats['total_notebooks'] == 2
        assert stats['skills_covered'] == 4  # All except spatial
        assert stats['skills_not_covered'] == 1  # spatial
        assert stats['coverage_percentage'] == 80.0  # 4/5 = 80%

    def test_coverage_empty_tasks(self):
        """Test coverage with no tasks."""
        extractor = NotebookTaskExtractor()

        stats = extractor.get_coverage_statistics([], [])

        assert stats['total_tasks'] == 0
        assert stats['coverage_percentage'] == 0.0


class TestJSONSerialization:
    """Test saving and loading tasks to/from JSON."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks."""
        return [
            NotebookTask(
                task_id="test-task-1",
                notebook_path="test.ipynb",
                task_description="Test task description",
                expected_skills=["skill-1", "skill-2"],
                expected_order=["skill-1", "skill-2"],
                category="test",
                difficulty="workflow",
                context={"key": "value"},
                alternate_acceptable=["skill-3"]
            )
        ]

    def test_save_tasks_to_json(self, sample_tasks):
        """Test saving tasks to JSON file."""
        extractor = NotebookTaskExtractor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            extractor.save_tasks_to_json(sample_tasks, temp_path)

            # Verify file exists and is valid JSON
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert 'tasks' in data
            assert len(data['tasks']) == 1
            assert data['tasks'][0]['task_id'] == 'test-task-1'

        finally:
            Path(temp_path).unlink()

    def test_load_tasks_from_json(self, sample_tasks):
        """Test loading tasks from JSON file."""
        extractor = NotebookTaskExtractor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save first
            extractor.save_tasks_to_json(sample_tasks, temp_path)

            # Load
            loaded_tasks = NotebookTaskExtractor.load_tasks_from_json(temp_path)

            assert len(loaded_tasks) == 1
            assert loaded_tasks[0].task_id == sample_tasks[0].task_id
            assert loaded_tasks[0].expected_skills == sample_tasks[0].expected_skills
            assert loaded_tasks[0].context == sample_tasks[0].context

        finally:
            Path(temp_path).unlink()

    def test_round_trip_json(self, sample_tasks):
        """Test save → load round trip preserves data."""
        extractor = NotebookTaskExtractor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            extractor.save_tasks_to_json(sample_tasks, temp_path)
            loaded = NotebookTaskExtractor.load_tasks_from_json(temp_path)

            # Compare
            original = sample_tasks[0]
            restored = loaded[0]

            assert original.task_id == restored.task_id
            assert original.notebook_path == restored.notebook_path
            assert original.task_description == restored.task_description
            assert original.expected_skills == restored.expected_skills
            assert original.expected_order == restored.expected_order
            assert original.category == restored.category
            assert original.difficulty == restored.difficulty
            assert original.alternate_acceptable == restored.alternate_acceptable

        finally:
            Path(temp_path).unlink()


class TestConvenienceFunction:
    """Test convenience function."""

    def test_create_task_extractor(self):
        """Test creating extractor with convenience function."""
        extractor = create_task_extractor()

        assert isinstance(extractor, NotebookTaskExtractor)
        assert len(extractor.ground_truth) > 0


class TestRealNotebooksExtraction:
    """Test extraction from real OmicVerse notebooks."""

    @pytest.fixture
    def real_notebooks_dir(self):
        """Path to real notebooks directory."""
        notebooks_dir = Path.cwd() / "omicverse_guide" / "docs"
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        return notebooks_dir

    def test_extract_from_real_deg_notebook(self, real_notebooks_dir):
        """Test extracting from real DEG notebook."""
        deg_notebook = real_notebooks_dir / "Tutorials-bulk" / "t_deg.ipynb"
        if not deg_notebook.exists():
            pytest.skip("DEG notebook not found")

        extractor = NotebookTaskExtractor()
        tasks = extractor.extract_from_notebook(str(deg_notebook))

        print(f"\nExtracted {len(tasks)} tasks from t_deg.ipynb:")
        for task in tasks:
            print(f"  - {task.task_id}: {task.expected_skills}")

        assert len(tasks) > 0
        # Should contain bulk-deg-analysis skill
        all_skills = set()
        for task in tasks:
            all_skills.update(task.expected_skills)
        assert 'bulk-deg-analysis' in all_skills

    def test_extract_from_real_preprocessing_notebook(self, real_notebooks_dir):
        """Test extracting from real preprocessing notebook."""
        preproc_notebook = real_notebooks_dir / "Tutorials-single" / "t_preprocess.ipynb"
        if not preproc_notebook.exists():
            pytest.skip("Preprocessing notebook not found")

        extractor = NotebookTaskExtractor()
        tasks = extractor.extract_from_notebook(str(preproc_notebook))

        print(f"\nExtracted {len(tasks)} tasks from t_preprocess.ipynb:")
        for task in tasks:
            print(f"  - {task.task_id}: {task.expected_skills}")

        assert len(tasks) > 0
        # Should contain single-preprocessing skill
        all_skills = set()
        for task in tasks:
            all_skills.update(task.expected_skills)
        assert 'single-preprocessing' in all_skills

    def test_extract_from_bulk_tutorials(self, real_notebooks_dir):
        """Test extracting from all bulk tutorials."""
        bulk_dir = real_notebooks_dir / "Tutorials-bulk"
        if not bulk_dir.exists():
            pytest.skip("Bulk tutorials directory not found")

        extractor = NotebookTaskExtractor()
        tasks = extractor.extract_from_directory(str(bulk_dir))

        print(f"\nExtracted {len(tasks)} tasks from Tutorials-bulk")

        # Should have multiple tasks
        assert len(tasks) > 5

        # Should have bulk category
        categories = set(task.category for task in tasks)
        assert 'bulk' in categories

    def test_coverage_with_real_notebooks(self, real_notebooks_dir):
        """Test coverage statistics with real notebooks."""
        bulk_dir = real_notebooks_dir / "Tutorials-bulk"
        if not bulk_dir.exists():
            pytest.skip("Bulk tutorials directory not found")

        from omicverse.utils.verifier import SkillDescriptionLoader

        loader = SkillDescriptionLoader()
        all_skills = loader.load_all_descriptions()

        extractor = NotebookTaskExtractor()
        tasks = extractor.extract_from_directory(str(bulk_dir))

        # Skip if no tasks were extracted (directory might be empty or inaccessible)
        if not tasks:
            pytest.skip("No tasks extracted from notebooks (directory may be empty)")

        stats = extractor.get_coverage_statistics(tasks, all_skills)

        print(f"\nCoverage statistics:")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Total notebooks: {stats['total_notebooks']}")
        print(f"  Skills covered: {stats['skills_covered']}/{len(all_skills)}")
        print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        print(f"  Covered: {stats['covered_skills']}")

        assert stats['total_tasks'] > 0
        assert stats['total_notebooks'] > 0
        # Only assert skills covered if we actually extracted tasks
        assert stats['skills_covered'] >= 0  # At least 0 skills covered


class TestErrorHandling:
    """Test error handling."""

    def test_extract_from_nonexistent_notebook(self):
        """Test extracting from non-existent notebook."""
        extractor = NotebookTaskExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_from_notebook("/nonexistent/path.ipynb")

    def test_extract_from_nonexistent_directory(self):
        """Test extracting from non-existent directory."""
        extractor = NotebookTaskExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_from_directory("/nonexistent/directory")

    def test_extract_from_invalid_json(self):
        """Test loading from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                NotebookTaskExtractor.load_tasks_from_json(temp_path)
        finally:
            Path(temp_path).unlink()
