"""End-to-end lifecycle tests for class-backed tools (Phase 2).

Tests full create → action → results → destroy lifecycle using mock classes.
"""

from __future__ import annotations

import pytest


class TestPyDEGLifecycle:
    """Full lifecycle: create → run → results → destroy."""

    def test_full_lifecycle(self, class_adapter_setup):
        adapter, store, adata_id, specs = class_adapter_setup

        entry = {
            "tool_name": "ov.bulk.pydeg",
            "full_name": "omicverse.bulk._Deseq2.pyDEG",
            "execution_class": "class",
        }

        # 1. Create
        result = adapter.invoke(entry, {
            "action": "create",
            "adata_id": adata_id,
        }, store)
        assert result["ok"] is True
        inst_id = result["outputs"][0]["ref_id"]

        # 2. Run
        result = adapter.invoke(entry, {
            "action": "run",
            "instance_id": inst_id,
            "treatment_groups": ["S1", "S2"],
            "control_groups": ["C1", "C2"],
            "method": "ttest",
        }, store)
        assert result["ok"] is True
        assert result["outputs"][0]["data"]["method"] == "ttest"

        # 3. Results
        result = adapter.invoke(entry, {
            "action": "results",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True
        assert result["outputs"][0]["data"]["n_deg"] == 42

        # 4. Destroy
        result = adapter.invoke(entry, {
            "action": "destroy",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True

        # 5. Verify instance is gone
        with pytest.raises(KeyError):
            store.get_instance(inst_id)

        # 6. Trying to use destroyed instance fails
        result = adapter.invoke(entry, {
            "action": "run",
            "instance_id": inst_id,
            "treatment_groups": ["S1"],
            "control_groups": ["C1"],
        }, store)
        assert result["ok"] is False
        assert result["error_code"] == "missing_session_object"


class TestPySCSALifecycle:
    """Full lifecycle: create → annotate → destroy."""

    def test_full_lifecycle(self, class_adapter_setup):
        adapter, store, adata_id, specs = class_adapter_setup

        entry = {
            "tool_name": "ov.single.pyscsa",
            "full_name": "omicverse.single._anno.pySCSA",
            "execution_class": "class",
        }

        # 1. Create with custom params
        result = adapter.invoke(entry, {
            "action": "create",
            "adata_id": adata_id,
            "foldchange": 2.0,
            "species": "Mouse",
        }, store)
        assert result["ok"] is True
        inst_id = result["outputs"][0]["ref_id"]

        # Verify params were passed
        instance = store.get_instance(inst_id)
        assert instance.foldchange == 2.0
        assert instance.species == "Mouse"

        # 2. Annotate
        result = adapter.invoke(entry, {
            "action": "annotate",
            "instance_id": inst_id,
            "clustertype": "leiden",
        }, store)
        assert result["ok"] is True
        data = result["outputs"][0]["data"]
        assert data["n_clusters"] == 5
        assert data["clustertype"] == "leiden"

        # 3. Destroy
        result = adapter.invoke(entry, {
            "action": "destroy",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True


class TestMetaCellLifecycle:
    """Full lifecycle: create → train → predict → destroy."""

    def test_full_lifecycle(self, class_adapter_setup):
        adapter, store, adata_id, specs = class_adapter_setup

        entry = {
            "tool_name": "ov.single.metacell",
            "full_name": "omicverse.single._metacell.MetaCell",
            "execution_class": "class",
        }

        # 1. Create
        result = adapter.invoke(entry, {
            "action": "create",
            "adata_id": adata_id,
            "use_rep": "X_pca",
        }, store)
        assert result["ok"] is True
        inst_id = result["outputs"][0]["ref_id"]

        # 2. Train
        result = adapter.invoke(entry, {
            "action": "train",
            "instance_id": inst_id,
            "max_iter": 100,
        }, store)
        assert result["ok"] is True
        assert result["outputs"][0]["data"]["converged"] is True

        # 3. Predict
        result = adapter.invoke(entry, {
            "action": "predict",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True
        assert len(result["outputs"]) >= 1

        # 4. Destroy
        result = adapter.invoke(entry, {
            "action": "destroy",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True


class TestInstanceIdReuse:
    """Verify that instance_id is correctly reused across multiple actions."""

    def test_instance_id_stable_across_actions(self, class_adapter_setup):
        adapter, store, adata_id, specs = class_adapter_setup

        entry = {
            "tool_name": "ov.bulk.pydeg",
            "full_name": "omicverse.bulk._Deseq2.pyDEG",
            "execution_class": "class",
        }

        # Create
        result = adapter.invoke(entry, {
            "action": "create",
            "adata_id": adata_id,
        }, store)
        inst_id = result["outputs"][0]["ref_id"]

        # Run — same instance_id
        result = adapter.invoke(entry, {
            "action": "run",
            "instance_id": inst_id,
            "treatment_groups": ["S1"],
            "control_groups": ["C1"],
        }, store)
        assert result["ok"] is True

        # Results — same instance_id
        result = adapter.invoke(entry, {
            "action": "results",
            "instance_id": inst_id,
        }, store)
        assert result["ok"] is True

        # Instance should still be in store
        assert store.get_instance(inst_id) is not None

    def test_multiple_instances_independent(self, class_adapter_setup):
        adapter, store, adata_id, specs = class_adapter_setup

        entry = {
            "tool_name": "ov.bulk.pydeg",
            "full_name": "omicverse.bulk._Deseq2.pyDEG",
            "execution_class": "class",
        }

        # Create two instances
        r1 = adapter.invoke(entry, {"action": "create", "adata_id": adata_id}, store)
        r2 = adapter.invoke(entry, {"action": "create", "adata_id": adata_id}, store)

        id1 = r1["outputs"][0]["ref_id"]
        id2 = r2["outputs"][0]["ref_id"]
        assert id1 != id2

        # Destroy one — other should still work
        adapter.invoke(entry, {"action": "destroy", "instance_id": id1}, store)

        result = adapter.invoke(entry, {
            "action": "run",
            "instance_id": id2,
            "treatment_groups": ["S1"],
            "control_groups": ["C1"],
        }, store)
        assert result["ok"] is True

        # First should be gone
        with pytest.raises(KeyError):
            store.get_instance(id1)
