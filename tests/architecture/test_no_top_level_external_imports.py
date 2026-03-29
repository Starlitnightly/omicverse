import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "omicverse"
TARGET_DIRS = ("alignment", "single", "bulk", "io", "space")


def _iter_python_files():
    for dirname in TARGET_DIRS:
        yield from sorted((PACKAGE_ROOT / dirname).rglob("*.py"))


def _is_external_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "external" or ".external" in alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "external" or ".external" in module:
            return True
        if node.level and module.startswith("external"):
            return True
    return False


class _TopLevelExternalImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[tuple[int, int, str]] = []
        self._scope_depth = 0

    def _record_if_needed(self, node: ast.AST) -> None:
        if self._scope_depth == 0 and _is_external_import(node):
            source = ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__
            self.violations.append((node.lineno, node.col_offset, source))

    def visit_Import(self, node: ast.Import) -> None:
        self._record_if_needed(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_if_needed(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1


@pytest.mark.parametrize("path", list(_iter_python_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_top_level_external_imports_in_core_domains(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _TopLevelExternalImportVisitor()
    visitor.visit(tree)

    if visitor.violations:
        details = "\n".join(
            f"{path.relative_to(REPO_ROOT)}:{lineno}:{col} -> {stmt}"
            for lineno, col, stmt in visitor.violations
        )
        pytest.fail(
            "Top-level external imports are forbidden in alignment/single/bulk/io/space.\n"
            f"{details}"
        )
