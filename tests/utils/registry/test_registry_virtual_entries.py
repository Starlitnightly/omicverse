from omicverse._registry import FunctionRegistry
from omicverse._registry import _global_registry
from omicverse.mcp.manifest import build_manifest_entry, ensure_registry_populated


def test_registry_register_derives_branch_entries():
    registry = FunctionRegistry()

    class Annotation:
        """Annotation manager."""

        def annotate(self, method="celltypist"):
            if method == "celltypist":
                import celltypist  # noqa: F401
            elif method == "scsa":
                from somewhere import scsa  # type: ignore # noqa: F401
            return method

    registry.register(
        func=Annotation,
        aliases=["annotation", "cell annotation"],
        category="single",
        description="Annotation manager",
        examples=["anno.annotate(method='celltypist')"],
    )

    results = registry.find("celltypist")

    assert any(
        item.get("virtual_entry") is True
        and item.get("branch_value") == "celltypist"
        and "Annotation.annotate" in item.get("full_name", "")
        for item in results
    )


def test_virtual_registry_entries_are_skipped_by_manifest_builder():
    entry = {
        "full_name": "omicverse.single.Annotation.annotate[method=celltypist]",
        "virtual_entry": True,
        "function": lambda: None,
        "category": "single",
    }

    item = build_manifest_entry(entry, overrides=type("O", (), {
        "get_schema_override": staticmethod(lambda full_name: {}),
        "get_rollout_phase": staticmethod(lambda full_name: "P0"),
        "get_manifest_override": staticmethod(lambda full_name: {}),
    })(), seen_names={})

    assert item is None
def test_scenic_registration_exposes_regdiffusion_branch():
    ensure_registry_populated()

    results = _global_registry.find("regdiffusion")

    assert any(
        item.get("full_name") == "omicverse.single._scenic.SCENIC.cal_grn[method=regdiffusion]"
        for item in results
    )
