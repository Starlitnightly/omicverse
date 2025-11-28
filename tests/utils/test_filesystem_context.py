"""Tests for FilesystemContextManager - filesystem-based context engineering."""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import the filesystem context module
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omicverse.utils.filesystem_context import (
    FilesystemContextManager,
    ContextNote,
    ExecutionPlan,
    ContextSearchResult,
)


class TestContextNote:
    """Tests for the ContextNote dataclass."""

    def test_create_note(self):
        """Test creating a context note."""
        note = ContextNote(
            key="test_key",
            content="Test content",
            category="notes",
        )
        assert note.key == "test_key"
        assert note.content == "Test content"
        assert note.category == "notes"
        assert note.compressed is False

    def test_note_to_dict(self):
        """Test converting note to dictionary."""
        note = ContextNote(
            key="test_key",
            content={"data": [1, 2, 3]},
            category="results",
            metadata={"function": "pca"},
        )
        data = note.to_dict()
        assert data["key"] == "test_key"
        assert data["content"] == {"data": [1, 2, 3]}
        assert data["category"] == "results"
        assert data["metadata"] == {"function": "pca"}

    def test_note_from_dict(self):
        """Test creating note from dictionary."""
        data = {
            "key": "test_key",
            "content": "Test content",
            "category": "notes",
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {},
            "compressed": False,
        }
        note = ContextNote.from_dict(data)
        assert note.key == "test_key"
        assert note.content == "Test content"


class TestExecutionPlan:
    """Tests for the ExecutionPlan dataclass."""

    def test_create_plan(self):
        """Test creating an execution plan."""
        steps = [
            {"description": "Run QC", "status": "pending"},
            {"description": "Normalize data", "status": "pending"},
        ]
        plan = ExecutionPlan(steps=steps)
        assert len(plan.steps) == 2
        assert plan.current_step == 0
        assert plan.status == "pending"

    def test_mark_step_complete(self):
        """Test marking a step as complete."""
        steps = [
            {"description": "Step 1", "status": "pending"},
            {"description": "Step 2", "status": "pending"},
        ]
        plan = ExecutionPlan(steps=steps)
        plan.mark_step_complete(0)
        assert 0 in plan.completed_steps
        assert plan.current_step == 1
        assert plan.status == "in_progress"

    def test_plan_completion(self):
        """Test plan completion status."""
        steps = [{"description": "Step 1", "status": "pending"}]
        plan = ExecutionPlan(steps=steps)
        plan.mark_step_complete(0)
        assert plan.status == "completed"


class TestFilesystemContextManager:
    """Tests for FilesystemContextManager."""

    @pytest.fixture
    def temp_base_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def ctx_manager(self, temp_base_dir):
        """Create a context manager with a temp directory."""
        return FilesystemContextManager(base_dir=temp_base_dir)

    def test_initialization(self, ctx_manager, temp_base_dir):
        """Test context manager initialization."""
        assert ctx_manager.session_id.startswith("session_")
        assert ctx_manager._workspace_dir.exists()
        assert (ctx_manager._workspace_dir / "notes").exists()
        assert (ctx_manager._workspace_dir / "results").exists()

    def test_write_note(self, ctx_manager):
        """Test writing a note."""
        path = ctx_manager.write_note(
            key="test_note",
            content="This is a test note",
            category="notes",
        )
        assert Path(path).exists()

        # Read back
        note = ctx_manager.read_note("test_note", "notes")
        assert note is not None
        assert note.content == "This is a test note"

    def test_write_note_with_dict_content(self, ctx_manager):
        """Test writing a note with dictionary content."""
        content = {"n_clusters": 8, "resolution": 1.0}
        path = ctx_manager.write_note(
            key="clustering_result",
            content=content,
            category="results",
        )
        assert Path(path).exists()

        note = ctx_manager.read_note("clustering_result", "results")
        assert note.content == content

    def test_write_note_with_metadata(self, ctx_manager):
        """Test writing a note with metadata."""
        ctx_manager.write_note(
            key="pca_result",
            content="PCA completed",
            category="results",
            metadata={"function": "pca", "n_pcs": 50},
        )

        note = ctx_manager.read_note("pca_result", "results")
        assert note.metadata["function"] == "pca"
        assert note.metadata["n_pcs"] == 50

    def test_read_nonexistent_note(self, ctx_manager):
        """Test reading a note that doesn't exist."""
        note = ctx_manager.read_note("nonexistent", "notes")
        assert note is None

    def test_search_context_glob(self, ctx_manager):
        """Test searching context with glob pattern."""
        # Write some notes
        ctx_manager.write_note("pca_result", "PCA done", "results")
        ctx_manager.write_note("pca_params", {"n_pcs": 50}, "results")
        ctx_manager.write_note("clustering_result", "Clustering done", "results")

        # Search with glob
        results = ctx_manager.search_context("pca*", match_type="glob")
        assert len(results) == 2
        keys = [r.key for r in results]
        assert "pca_result" in keys
        assert "pca_params" in keys

    def test_search_context_grep(self, ctx_manager):
        """Test searching context with grep pattern."""
        ctx_manager.write_note("note1", "resolution is 1.0", "notes")
        ctx_manager.write_note("note2", "n_clusters is 8", "notes")
        ctx_manager.write_note("note3", "resolution is 0.5", "notes")

        results = ctx_manager.search_context("resolution", match_type="grep")
        assert len(results) == 2

    def test_write_plan(self, ctx_manager):
        """Test writing an execution plan."""
        steps = [
            {"description": "Run QC", "status": "pending"},
            {"description": "Normalize", "status": "pending"},
            {"description": "Cluster", "status": "pending"},
        ]
        path = ctx_manager.write_plan(steps)
        assert Path(path).exists()

        plan = ctx_manager.load_plan()
        assert plan is not None
        assert len(plan.steps) == 3

    def test_update_plan_step(self, ctx_manager):
        """Test updating a plan step."""
        steps = [
            {"description": "Step 1", "status": "pending"},
            {"description": "Step 2", "status": "pending"},
        ]
        ctx_manager.write_plan(steps)

        ctx_manager.update_plan_step(0, "completed", "Done successfully")

        plan = ctx_manager.load_plan()
        assert plan.steps[0]["status"] == "completed"
        assert plan.steps[0]["result"] == "Done successfully"
        assert plan.current_step == 1

    def test_write_snapshot(self, ctx_manager):
        """Test writing a data snapshot."""
        snapshot_data = {
            "shape": (1000, 2000),
            "obsm": ["X_pca", "X_umap"],
            "obs_columns": ["leiden", "cell_type"],
        }
        path = ctx_manager.write_snapshot(
            snapshot_data,
            step_number=1,
            description="After PCA",
        )
        assert Path(path).exists()

    def test_get_relevant_context(self, ctx_manager):
        """Test getting relevant context for a query."""
        # Write some notes
        ctx_manager.write_note("clustering_params", {"resolution": 1.0}, "results")
        ctx_manager.write_note("clustering_obs", "Found 8 clusters", "notes")
        ctx_manager.write_note("pca_done", "PCA completed with 50 PCs", "notes")

        context = ctx_manager.get_relevant_context("clustering", max_tokens=500)
        assert len(context) > 0
        # Should find clustering-related notes
        assert "clustering" in context.lower() or "cluster" in context.lower()

    def test_list_notes(self, ctx_manager):
        """Test listing notes."""
        ctx_manager.write_note("note1", "Content 1", "notes")
        ctx_manager.write_note("note2", "Content 2", "notes")
        ctx_manager.write_note("result1", "Result 1", "results")

        all_notes = ctx_manager.list_notes()
        assert len(all_notes) == 3

        notes_only = ctx_manager.list_notes(category="notes")
        assert len(notes_only) == 2

    def test_compress_notes(self, ctx_manager):
        """Test compressing notes."""
        # Write many notes
        for i in range(15):
            ctx_manager.write_note(f"note_{i}", f"Content {i}", "notes")

        # Compress, keeping only 5 recent
        summary_path = ctx_manager.compress_notes("notes", keep_recent=5)

        # Check that summary was created
        assert summary_path != ""
        assert Path(summary_path).exists()

        # Check remaining notes
        notes = ctx_manager.list_notes(category="notes")
        # Should have 5 recent + 1 summary
        assert len(notes) <= 6

    def test_sub_agent_context(self, ctx_manager):
        """Test creating sub-agent context."""
        # Write something in parent
        ctx_manager.write_note("parent_note", "From parent", "notes")

        # Create sub-agent context
        sub_ctx = ctx_manager.create_sub_agent_context("sub_1")

        # Sub-agent should share workspace
        assert sub_ctx.parent_session_id == ctx_manager.session_id
        assert sub_ctx._workspace_dir == ctx_manager._workspace_dir

        # Sub-agent should be able to read parent's notes
        note = sub_ctx.read_note("parent_note", "notes")
        assert note is not None
        assert note.content == "From parent"

        # Sub-agent can write
        sub_ctx.write_note("sub_note", "From sub-agent", "notes")

        # Parent can read sub-agent's notes
        note = ctx_manager.read_note("sub_note", "notes")
        assert note is not None

    def test_shared_workspace(self, ctx_manager):
        """Test shared workspace functionality."""
        # Write to shared
        path = ctx_manager.write_to_shared("global_config", {"setting": "value"})
        assert Path(path).exists()

        # Read from shared
        note = ctx_manager.read_from_shared("global_config")
        assert note is not None
        assert note.content == {"setting": "value"}

    def test_get_session_summary(self, ctx_manager):
        """Test getting session summary."""
        ctx_manager.write_note("note1", "Content 1", "notes")
        ctx_manager.write_plan([{"description": "Step 1", "status": "pending"}])

        summary = ctx_manager.get_session_summary()
        assert ctx_manager.session_id in summary
        assert "notes" in summary.lower() or "Notes" in summary

    def test_get_workspace_stats(self, ctx_manager):
        """Test getting workspace statistics."""
        ctx_manager.write_note("note1", "Content 1", "notes")
        ctx_manager.write_note("result1", {"data": [1, 2, 3]}, "results")

        stats = ctx_manager.get_workspace_stats()
        assert stats["session_id"] == ctx_manager.session_id
        assert stats["total_notes"] >= 2
        assert "notes" in stats["categories"]
        assert "results" in stats["categories"]

    def test_sanitize_filename(self, ctx_manager):
        """Test filename sanitization."""
        # Write note with special characters in key
        ctx_manager.write_note("test/key:with<special>chars", "Content", "notes")

        # Should still be able to read it back
        note = ctx_manager.read_note("test/key:with<special>chars", "notes")
        # Key is sanitized but content is preserved
        assert note is not None
        assert note.content == "Content"

    def test_auto_compress_threshold(self, temp_base_dir):
        """Test auto-compression when threshold is reached."""
        ctx = FilesystemContextManager(
            base_dir=temp_base_dir,
            auto_compress_threshold=10,
            max_notes_per_category=20,
        )

        # Write notes up to threshold
        for i in range(12):
            ctx.write_note(f"note_{i}", f"Content {i}", "notes")

        # Auto-compression should have triggered
        notes = ctx.list_notes(category="notes")
        # Should have compressed some notes
        assert len(notes) <= 12

    def test_cleanup_session(self, ctx_manager):
        """Test session cleanup."""
        ctx_manager.write_note("note1", "Content", "notes")

        ctx_manager.cleanup_session(keep_summary=True)

        # Summary file should exist
        summary_file = ctx_manager._workspace_dir / "final_summary.md"
        assert summary_file.exists()


class TestFilesystemContextIntegration:
    """Integration tests for FilesystemContextManager."""

    @pytest.fixture
    def temp_base_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_workflow(self, temp_base_dir):
        """Test a complete workflow with plans, notes, and context retrieval."""
        ctx = FilesystemContextManager(base_dir=temp_base_dir)

        # 1. Create a plan
        ctx.write_plan([
            {"description": "Quality control", "status": "pending"},
            {"description": "Normalization", "status": "pending"},
            {"description": "PCA", "status": "pending"},
            {"description": "Clustering", "status": "pending"},
        ])

        # 2. Execute steps and write notes
        ctx.update_plan_step(0, "in_progress")
        ctx.write_note("qc_start", "Starting QC", "notes")
        ctx.write_note("qc_result", {"n_cells": 5000, "mito_pct": 0.05}, "results")
        ctx.update_plan_step(0, "completed", "QC removed 500 cells")

        ctx.update_plan_step(1, "in_progress")
        ctx.write_note("norm_result", {"method": "log1p", "target_sum": 10000}, "results")
        ctx.update_plan_step(1, "completed")

        ctx.update_plan_step(2, "in_progress")
        ctx.write_snapshot({"obsm": ["X_pca"], "n_pcs": 50}, step_number=2)
        ctx.update_plan_step(2, "completed")

        # 3. Get relevant context for clustering
        context = ctx.get_relevant_context("clustering", max_tokens=1000)
        assert len(context) > 0

        # 4. Check plan status
        plan = ctx.load_plan()
        assert plan.status == "in_progress"
        assert len(plan.completed_steps) == 3

        # 5. Get summary
        summary = ctx.get_session_summary()
        assert "3/4" in summary or "3" in summary

    def test_multi_agent_workflow(self, temp_base_dir):
        """Test workflow with parent and sub-agents."""
        parent = FilesystemContextManager(base_dir=temp_base_dir)

        # Parent creates plan
        parent.write_plan([
            {"description": "Preprocessing", "status": "pending"},
            {"description": "Analysis", "status": "pending"},
        ])

        # Sub-agent for preprocessing
        preproc_agent = parent.create_sub_agent_context("preprocessing")
        preproc_agent.write_note("preproc_done", "Preprocessing completed", "notes")
        parent.update_plan_step(0, "completed")

        # Sub-agent for analysis
        analysis_agent = parent.create_sub_agent_context("analysis")
        analysis_agent.write_note("analysis_result", {"clusters": 8}, "results")
        parent.update_plan_step(1, "completed")

        # Parent can see all notes
        all_notes = parent.list_notes()
        keys = [n["key"] for n in all_notes]
        assert "preproc_done" in keys
        assert "analysis_result" in keys

        # Plan should be complete
        plan = parent.load_plan()
        assert plan.status == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
