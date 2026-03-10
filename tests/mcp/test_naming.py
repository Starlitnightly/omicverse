"""Tests for canonical tool naming."""

import pytest
from omicverse.mcp.naming import (
    build_tool_name,
    normalize_symbol_name,
    resolve_name_collision,
    build_search_aliases,
    validate_tool_name,
)


class TestBuildToolName:
    def test_pp_pca(self):
        assert build_tool_name("omicverse.pp._preprocess.pca") == "ov.pp.pca"

    def test_utils_read(self):
        assert build_tool_name("omicverse.utils._data.read") == "ov.utils.read"

    def test_io_read_override(self):
        assert build_tool_name("omicverse.io.single._read.read") == "ov.utils.read"

    def test_single_find_markers(self):
        assert build_tool_name("omicverse.single._markers.find_markers") == "ov.single.find_markers"

    def test_pl_embedding(self):
        assert build_tool_name("omicverse.pl._single.embedding") == "ov.pl.embedding"

    def test_bulk_pydeg(self):
        assert build_tool_name("omicverse.bulk._Deseq2.pyDEG") == "ov.bulk.pydeg"

    def test_deterministic(self):
        """Same input always produces same output."""
        name1 = build_tool_name("omicverse.pp._preprocess.pca")
        name2 = build_tool_name("omicverse.pp._preprocess.pca")
        assert name1 == name2


class TestNormalizeSymbolName:
    def test_lowercase(self):
        assert normalize_symbol_name("PCA") == "pca"

    def test_mixed_case(self):
        assert normalize_symbol_name("featureCount") == "featurecount"

    def test_underscore_preserved(self):
        assert normalize_symbol_name("Cal_Spatial_Net") == "cal_spatial_net"

    def test_gpu_suffix(self):
        assert normalize_symbol_name("anndata_to_GPU") == "anndata_to_gpu"


class TestResolveNameCollision:
    def test_appends_module_segment(self):
        entry = {"full_name": "omicverse.pp._preprocess.pca"}
        seen = {"ov.pp.pca": {}}
        result = resolve_name_collision("ov.pp.pca", entry, seen)
        assert result != "ov.pp.pca"
        assert "preprocess" in result

    def test_unique_result(self):
        entry = {"full_name": "omicverse.pp._preprocess.pca"}
        seen = {"ov.pp.pca": {}, "ov.pp.pca_preprocess": {}}
        result = resolve_name_collision("ov.pp.pca", entry, seen)
        assert result not in seen


class TestBuildSearchAliases:
    def test_includes_aliases(self):
        entry = {
            "short_name": "pca",
            "full_name": "omicverse.pp._preprocess.pca",
            "aliases": ["PCA", "主成分分析"],
        }
        aliases = build_search_aliases(entry)
        assert "pca" in aliases
        assert "主成分分析" in aliases

    def test_no_empty_strings(self):
        entry = {"short_name": "", "full_name": "", "aliases": []}
        aliases = build_search_aliases(entry)
        assert "" not in aliases


class TestValidateToolName:
    def test_valid_name(self):
        validate_tool_name("ov.pp.pca")  # should not raise

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError):
            validate_tool_name("ov.pp.PCA")

    def test_invalid_no_prefix(self):
        with pytest.raises(ValueError):
            validate_tool_name("pp.pca")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError):
            validate_tool_name("ov.pp.my tool")
