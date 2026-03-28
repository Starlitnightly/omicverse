"""
Tests for optional-adata support in AgentContextInjector.

Covers AC-001: no-dataset construction, safe inspection skipping,
and preserved dataset-aware behavior.
"""

import sys
import importlib.util
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Import machinery — mirrors the existing standalone test pattern to avoid
# circular imports via the package __init__.py.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _import(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, str(PROJECT_ROOT / rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_import(
    "omicverse.utils.inspector.data_structures",
    "omicverse/utils/inspector/data_structures.py",
)
_import(
    "omicverse.utils.inspector.validators",
    "omicverse/utils/inspector/validators.py",
)
_import(
    "omicverse.utils.inspector.prerequisite_checker",
    "omicverse/utils/inspector/prerequisite_checker.py",
)
_import(
    "omicverse.utils.inspector.suggestion_engine",
    "omicverse/utils/inspector/suggestion_engine.py",
)
_import(
    "omicverse.utils.inspector.llm_formatter",
    "omicverse/utils/inspector/llm_formatter.py",
)
_import(
    "omicverse.utils.inspector.inspector",
    "omicverse/utils/inspector/inspector.py",
)
_aci_mod = _import(
    "omicverse.utils.inspector.agent_context_injector",
    "omicverse/utils/inspector/agent_context_injector.py",
)

AgentContextInjector = _aci_mod.AgentContextInjector
ConversationState = _aci_mod.ConversationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockRegistry:
    """Minimal mock registry used by tests that supply a dataset."""

    def __init__(self):
        self.functions = {
            "pca": {
                "prerequisites": {"required": ["preprocess"], "optional": []},
                "requires": {},
                "produces": {"obsm": ["X_pca"], "uns": ["pca"]},
                "auto_fix": "escalate",
            },
            "neighbors": {
                "prerequisites": {"required": ["pca"], "optional": []},
                "requires": {"obsm": ["X_pca"]},
                "produces": {
                    "obsp": ["connectivities", "distances"],
                    "uns": ["neighbors"],
                },
                "auto_fix": "auto",
            },
            "leiden": {
                "prerequisites": {"required": ["neighbors"], "optional": []},
                "requires": {"obsp": ["connectivities"]},
                "produces": {"obs": ["leiden"]},
                "auto_fix": "auto",
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


def _make_adata(with_pca=False, with_neighbors=False):
    from anndata import AnnData

    np.random.seed(42)
    adata = AnnData(np.random.rand(50, 30))
    adata.obs_names = [f"C{i}" for i in range(50)]
    adata.var_names = [f"G{i}" for i in range(30)]
    if with_pca:
        adata.obsm["X_pca"] = np.random.rand(50, 30)
        adata.uns["pca"] = {"variance_ratio": np.random.rand(30)}
    if with_neighbors:
        adata.obsp["connectivities"] = np.random.rand(50, 50)
        adata.obsp["distances"] = np.random.rand(50, 50)
        adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}
    return adata


# ===================================================================
# 1. No-dataset construction
# ===================================================================


class TestNoDatasetConstruction:
    """AgentContextInjector(adata=None) must not raise."""

    def test_construct_none(self):
        injector = AgentContextInjector(adata=None, registry=None)
        assert injector.adata is None
        assert injector.inspector is None
        assert not injector.has_dataset

    def test_construct_no_args(self):
        """Default arguments (all None) produce a valid no-dataset injector."""
        injector = AgentContextInjector()
        assert not injector.has_dataset
        assert injector.inspector is None

    def test_no_initial_snapshot(self):
        """No snapshot should be taken when there is no dataset."""
        injector = AgentContextInjector()
        assert len(injector.conversation_state.data_snapshots) == 0


# ===================================================================
# 2. No-dataset mode skips inspection safely
# ===================================================================


class TestNoDatasetSafeSkipping:
    """All inspection pathways must return safe defaults without errors."""

    def test_inject_context_no_dataset(self):
        injector = AgentContextInjector()
        prompt = "You are a helpful assistant."
        enhanced = injector.inject_context(prompt)

        assert prompt in enhanced
        assert "knowledge-query" in enhanced or "No dataset" in enhanced

    def test_general_state_section_no_dataset(self):
        injector = AgentContextInjector()
        section = injector._build_general_state_section()
        assert "No dataset" in section
        assert "knowledge-query" in section

    def test_function_specific_section_no_dataset(self):
        injector = AgentContextInjector()
        section = injector._build_function_specific_section("leiden")
        assert "Target Function: leiden" in section
        assert "No dataset loaded" in section

    def test_detect_executed_functions_no_dataset(self):
        injector = AgentContextInjector()
        assert injector._detect_executed_functions() == {}

    def test_get_current_state_no_dataset(self):
        injector = AgentContextInjector()
        state = injector._get_current_state()
        assert state["shape"] == (0, 0)
        assert state["obsm"] == []
        assert state["obsp"] == []

    def test_update_after_execution_no_dataset(self):
        """Execution history should still be tracked without a dataset."""
        injector = AgentContextInjector()
        injector.update_after_execution("pca")
        assert "pca" in injector.conversation_state.executed_functions
        assert len(injector.conversation_state.execution_history) == 1
        # A snapshot is taken even in no-dataset mode (with empty state)
        assert len(injector.conversation_state.data_snapshots) == 1

    def test_conversation_summary_no_dataset(self):
        injector = AgentContextInjector()
        summary = injector.get_conversation_summary()
        assert "No functions executed" in summary

    def test_conversation_summary_after_exec_no_dataset(self):
        injector = AgentContextInjector()
        injector.update_after_execution("pca")
        summary = injector.get_conversation_summary()
        assert "pca" in summary

    def test_clear_conversation_state_no_dataset(self):
        injector = AgentContextInjector()
        injector.update_after_execution("pca")
        injector.clear_conversation_state()
        assert len(injector.conversation_state.executed_functions) == 0
        # No snapshot should be taken on clear in no-dataset mode
        assert len(injector.conversation_state.data_snapshots) == 0

    def test_create_sub_agent_injector_no_dataset(self):
        """Sub-agent injector inherits no-dataset mode from parent."""
        parent = AgentContextInjector()
        child = parent.create_sub_agent_injector()
        assert not child.has_dataset
        assert child.inspector is None

    def test_inject_context_selective_no_dataset(self):
        injector = AgentContextInjector()
        prompt = "base"
        enhanced = injector.inject_context(
            prompt,
            target_function="leiden",
            include_general_state=True,
            include_function_specific=True,
            include_instructions=True,
        )
        assert "base" in enhanced
        assert "No dataset" in enhanced
        assert "Target Function: leiden" in enhanced

    def test_filesystem_methods_no_dataset(self):
        """Filesystem helpers should not fail in no-dataset mode."""
        injector = AgentContextInjector(enable_filesystem_context=False)
        assert injector.write_to_context("k", "v") is None
        assert injector.search_context("*") == []
        assert injector.save_execution_plan([]) is None
        assert injector.save_data_snapshot() is None
        assert "disabled" in injector.get_workspace_summary().lower()


# ===================================================================
# 3. Dataset-aware behavior unchanged
# ===================================================================


class TestDatasetPresentBehavior:
    """Existing dataset-aware paths must be unaffected by the refactor."""

    def test_construct_with_adata(self):
        adata = _make_adata()
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        assert injector.has_dataset
        assert injector.inspector is not None
        # Initial snapshot taken
        assert len(injector.conversation_state.data_snapshots) == 1

    def test_inject_context_with_adata(self):
        adata = _make_adata()
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        enhanced = injector.inject_context("You are a helpful assistant.")
        assert "Current AnnData State" in enhanced
        assert "Prerequisite Handling Instructions" in enhanced

    def test_general_state_with_pca(self):
        adata = _make_adata(with_pca=True, with_neighbors=True)
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        section = injector._build_general_state_section()
        assert "X_pca" in section
        assert "connectivities" in section

    def test_function_specific_with_adata(self):
        adata = _make_adata(with_pca=True)
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        section = injector._build_function_specific_section("leiden")
        assert "Target Function: leiden" in section
        assert "neighbors" in section.lower()

    def test_update_and_summary_with_adata(self):
        adata = _make_adata()
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        injector.update_after_execution("pca")
        injector.update_after_execution("neighbors")
        summary = injector.get_conversation_summary()
        assert "pca" in summary
        assert "neighbors" in summary

    def test_clear_state_with_adata(self):
        adata = _make_adata()
        injector = AgentContextInjector(adata=adata, registry=_MockRegistry())
        injector.update_after_execution("pca")
        injector.clear_conversation_state()
        assert len(injector.conversation_state.executed_functions) == 0
        # Fresh snapshot should be present
        assert len(injector.conversation_state.data_snapshots) == 1

    def test_sub_agent_inherits_adata(self):
        adata = _make_adata()
        parent = AgentContextInjector(
            adata=adata,
            registry=_MockRegistry(),
            enable_filesystem_context=False,
        )
        child = parent.create_sub_agent_injector()
        assert child.has_dataset
        assert child.adata is adata

    def test_sub_agent_override_adata(self):
        adata1 = _make_adata()
        adata2 = _make_adata(with_pca=True)
        parent = AgentContextInjector(
            adata=adata1,
            registry=_MockRegistry(),
            enable_filesystem_context=False,
        )
        child = parent.create_sub_agent_injector(adata=adata2)
        assert child.adata is adata2

    def test_has_dataset_property(self):
        assert AgentContextInjector().has_dataset is False
        assert (
            AgentContextInjector(
                adata=_make_adata(), registry=_MockRegistry()
            ).has_dataset
            is True
        )
