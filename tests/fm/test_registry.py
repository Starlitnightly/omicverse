"""Tests for ``omicverse.fm.registry``."""

import pytest

from omicverse.fm.registry import (
    GeneIDScheme,
    HardwareRequirements,
    ModelRegistry,
    ModelSpec,
    Modality,
    OutputKeys,
    SkillReadyStatus,
    TaskType,
    get_registry,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestTaskTypeEnum:
    def test_task_values(self):
        expected = {"embed", "annotate", "integrate", "perturb", "spatial", "drug_response"}
        actual = {t.value for t in TaskType}
        assert actual == expected

    def test_task_from_string(self):
        assert TaskType("embed") == TaskType.EMBED
        assert TaskType("annotate") == TaskType.ANNOTATE


class TestModalityEnum:
    def test_modality_values(self):
        expected = {"RNA", "ATAC", "Spatial", "Protein", "Multi-omics"}
        actual = {m.value for m in Modality}
        assert actual == expected


class TestGeneIDSchemeEnum:
    def test_gene_scheme_values(self):
        expected = {"symbol", "ensembl", "custom"}
        actual = {g.value for g in GeneIDScheme}
        assert actual == expected


# ---------------------------------------------------------------------------
# ModelSpec tests
# ---------------------------------------------------------------------------


class TestModelSpec:
    @pytest.fixture()
    def sample_spec(self):
        return ModelSpec(
            name="test_model",
            version="1.0.0",
            tasks=[TaskType.EMBED, TaskType.INTEGRATE],
            modalities=[Modality.RNA],
            species=["human", "mouse"],
            output_keys=OutputKeys(embedding_key="X_test"),
            hardware=HardwareRequirements(),
        )

    def test_supports_task(self, sample_spec):
        assert sample_spec.supports_task(TaskType.EMBED)
        assert sample_spec.supports_task(TaskType.INTEGRATE)
        assert not sample_spec.supports_task(TaskType.ANNOTATE)

    def test_supports_species(self, sample_spec):
        assert sample_spec.supports_species("human")
        assert sample_spec.supports_species("mouse")
        assert not sample_spec.supports_species("zebrafish")

    def test_supports_modality(self, sample_spec):
        assert sample_spec.supports_modality(Modality.RNA)
        assert not sample_spec.supports_modality(Modality.ATAC)

    def test_to_dict(self, sample_spec):
        d = sample_spec.to_dict()
        assert d["name"] == "test_model"
        assert d["version"] == "1.0.0"
        assert "embed" in d["tasks"]
        assert "RNA" in d["modalities"]
        assert isinstance(d["hardware"], dict)
        assert isinstance(d["output_keys"], dict)


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_registry_initialization(self):
        registry = get_registry()
        models = registry.list_models()
        assert len(models) >= 22

    def test_list_models(self):
        registry = get_registry()
        models = registry.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, ModelSpec) for m in models)

    def test_list_models_skill_ready_only(self):
        registry = get_registry()
        ready = registry.list_models(skill_ready_only=True)
        assert all(m.skill_ready == SkillReadyStatus.READY for m in ready)
        ready_names = {m.name for m in ready}
        assert {"scgpt", "geneformer", "uce", "scfoundation", "cellplm"} <= ready_names

    def test_get_model(self):
        registry = get_registry()
        spec = registry.get("scgpt")
        assert spec is not None
        assert spec.name == "scgpt"

    def test_get_model_case_insensitive(self):
        registry = get_registry()
        spec = registry.get("ScGPT")
        assert spec is not None
        assert spec.name == "scgpt"

    def test_get_nonexistent_model(self):
        registry = get_registry()
        assert registry.get("nonexistent_model_xyz") is None

    def test_find_models_by_task(self):
        registry = get_registry()
        embed_models = registry.find_models(task=TaskType.EMBED)
        assert len(embed_models) > 0
        assert all(
            any(t == TaskType.EMBED for t in m.tasks) for m in embed_models
        )

    def test_find_models_by_species(self):
        registry = get_registry()
        human_models = registry.find_models(species="human")
        assert len(human_models) > 0
        assert all("human" in m.species for m in human_models)

    def test_find_models_by_gene_scheme(self):
        registry = get_registry()
        ensembl_models = registry.find_models(gene_scheme=GeneIDScheme.ENSEMBL)
        assert len(ensembl_models) > 0
        assert all(m.gene_id_scheme == GeneIDScheme.ENSEMBL for m in ensembl_models)

    def test_find_models_max_vram(self):
        registry = get_registry()
        low_vram = registry.find_models(max_vram_gb=4)
        for m in low_vram:
            assert m.hardware.min_vram_gb <= 4 or m.hardware.cpu_fallback

    def test_find_models_zero_shot(self):
        registry = get_registry()
        zs_models = registry.find_models(task=TaskType.EMBED, zero_shot=True)
        assert all(m.zero_shot_embedding for m in zs_models)


class TestSpecificModels:
    """Verify key properties of specific model specs."""

    def test_scgpt_spec(self):
        spec = get_registry().get("scgpt")
        assert spec.skill_ready == SkillReadyStatus.READY
        assert TaskType.EMBED in spec.tasks
        assert "human" in spec.species
        assert spec.gene_id_scheme == GeneIDScheme.SYMBOL

    def test_geneformer_spec(self):
        spec = get_registry().get("geneformer")
        assert spec.skill_ready == SkillReadyStatus.READY
        assert spec.gene_id_scheme == GeneIDScheme.ENSEMBL
        assert spec.hardware.cpu_fallback is True

    def test_uce_spec(self):
        spec = get_registry().get("uce")
        assert spec.skill_ready == SkillReadyStatus.READY
        assert spec.embedding_dim == 1280
        assert len(spec.species) >= 5  # multiple species

    def test_atacformer_spec(self):
        spec = get_registry().get("atacformer")
        assert Modality.ATAC in spec.modalities

    def test_scplantllm_spec(self):
        spec = get_registry().get("scplantllm")
        assert "plant" in spec.species

    def test_genept_spec(self):
        spec = get_registry().get("genept")
        assert spec.hardware.gpu_required is False
