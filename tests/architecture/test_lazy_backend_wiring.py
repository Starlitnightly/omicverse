import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse(path: str) -> ast.Module:
    source_path = REPO_ROOT / path
    return ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _function_calls(tree: ast.AST) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def _string_literals(tree: ast.AST) -> set[str]:
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def test_modified_backend_helpers_are_wired_to_call_sites():
    expected_helper_calls = {
        "omicverse/bulk/_Gene_module.py": {"_get_pywgcna_backend"},
        "omicverse/single/_cefcon.py": {"_get_cefcon_backend"},
        "omicverse/single/_metacell.py": {"_get_seacells_backend"},
        "omicverse/single/_mofa.py": {"_get_mofa_entry_point"},
        "omicverse/single/_tosica.py": {"_get_tosica_backend"},
        "omicverse/single/_traj.py": {"_get_palantir_backend"},
        "omicverse/single/_via.py": {"_load_via_modules"},
        "omicverse/space/_cluster.py": {"_get_stagate_backend"},
        "omicverse/space/_commot.py": {"_get_summarize_cluster_gpu"},
        "omicverse/space/_gaston.py": {"_get_gaston_modules"},
        "omicverse/space/_integrate.py": {"_get_staligner_backend"},
        "omicverse/space/_spatrio.py": {"_get_spatrio_functions"},
        "omicverse/space/_starfysh.py": {"_get_starfysh_modules"},
        "omicverse/space/_stt.py": {"_get_stt_modules"},
    }

    for relative_path, helper_names in expected_helper_calls.items():
        tree = _parse(relative_path)
        calls = _function_calls(tree)
        for helper_name in helper_names:
            assert helper_name in calls, f"{relative_path} defines {helper_name} but does not call it"


def test_bulk_lazy_exports_are_preserved():
    bulk_init_tree = _parse("omicverse/bulk/__init__.py")
    bulk_gene_tree = _parse("omicverse/bulk/_Gene_module.py")
    bulk_init_source = _read("omicverse/bulk/__init__.py")

    bulk_init_strings = _string_literals(bulk_init_tree)
    bulk_gene_strings = _string_literals(bulk_gene_tree)

    assert "pyWGCNA" in bulk_init_strings
    assert "readWGCNA" in bulk_init_strings
    assert "from . import _Gene_module as gene_module" in bulk_init_source

    assert "pyWGCNA" in bulk_gene_strings
    assert "readWGCNA" in bulk_gene_strings
    assert "_get_pywgcna_backend" in _function_calls(bulk_gene_tree)


def test_single_lazy_exports_are_preserved():
    single_init_tree = _parse("omicverse/single/__init__.py")
    cnmf_tree = _parse("omicverse/single/_cnmf.py")
    single_init_source = _read("omicverse/single/__init__.py")

    single_strings = _string_literals(single_init_tree)
    cnmf_strings = _string_literals(cnmf_tree)

    assert "popv" in single_strings
    assert "cNMF" in single_strings
    assert "Hotspot" in single_strings
    assert "from . import _cnmf as cnmf_module" in single_init_source

    assert "Hotspot" in cnmf_strings
    assert "..external.cnmf.cnmf" in cnmf_strings
