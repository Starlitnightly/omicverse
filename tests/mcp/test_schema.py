"""Tests for JSON Schema generation."""

import pytest
from omicverse.mcp.schema import (
    build_parameter_schema,
    signature_to_schema,
    annotation_to_json_schema,
    inject_session_params,
    apply_schema_overrides,
    build_empty_schema,
)


class TestSignatureToSchema:
    def test_simple_function(self):
        def func(x: int = 1, y: str = "hello"): pass
        schema = signature_to_schema(func)
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "string"

    def test_required_params(self):
        def func(x: int, y: str = "default"): pass
        schema = signature_to_schema(func)
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_kwargs_allows_additional(self):
        def func(x: int = 1, **kwargs): pass
        schema = signature_to_schema(func)
        assert schema.get("additionalProperties") is True

    def test_skips_self(self):
        def func(self, x: int = 1): pass
        schema = signature_to_schema(func)
        assert "self" not in schema["properties"]


class TestAnnotationToJsonSchema:
    def test_str(self):
        assert annotation_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert annotation_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert annotation_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert annotation_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert annotation_to_json_schema(list) == {"type": "array"}

    def test_dict(self):
        assert annotation_to_json_schema(dict) == {"type": "object"}


class TestInjectSessionParams:
    def test_adds_adata_id(self):
        schema = {
            "type": "object",
            "properties": {"adata": {"type": "string"}, "n_pcs": {"type": "integer"}},
            "required": ["adata"],
        }
        entry = {"execution_class": "adata"}
        result = inject_session_params(schema, entry)
        assert "adata_id" in result["properties"]
        assert "adata" not in result["properties"]
        assert "adata_id" in result["required"]
        assert "adata" not in result["required"]

    def test_preserves_other_params(self):
        schema = {
            "type": "object",
            "properties": {"adata": {}, "n_pcs": {"type": "integer"}},
            "required": ["adata"],
        }
        result = inject_session_params(schema, {})
        assert "n_pcs" in result["properties"]


class TestApplySchemaOverrides:
    def test_merges_properties(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        override = {"properties": {"y": {"type": "string"}}}
        result = apply_schema_overrides(schema, override)
        assert "x" in result["properties"]
        assert "y" in result["properties"]

    def test_override_wins(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        override = {"properties": {"x": {"type": "string"}}, "required": ["x", "y"]}
        result = apply_schema_overrides(schema, override)
        assert result["properties"]["x"]["type"] == "string"
        assert result["required"] == ["x", "y"]


class TestBuildParameterSchema:
    def test_stateless_no_adata_id(self):
        def func(x: int = 1): pass
        entry = {"function": func, "execution_class": "stateless"}
        schema = build_parameter_schema(entry)
        assert "adata_id" not in schema.get("properties", {})

    def test_adata_has_adata_id(self):
        def func(adata, n: int = 10): pass
        entry = {"function": func, "execution_class": "adata"}
        schema = build_parameter_schema(entry)
        assert "adata_id" in schema["properties"]

    def test_override_applied(self):
        def func(adata, **kwargs): pass
        entry = {"function": func, "execution_class": "adata"}
        override = {"properties": {"mode": {"type": "string"}}, "required": ["adata_id"]}
        schema = build_parameter_schema(entry, overrides=override)
        assert "mode" in schema["properties"]


class TestBuildEmptySchema:
    def test_shape(self):
        schema = build_empty_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True
