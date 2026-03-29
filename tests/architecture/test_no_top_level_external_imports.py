import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "omicverse"
TARGET_DIRS = (
    "alignment",
    "single",
    "bulk",
    "io",
    "space",
    "pp",
    "pl",
    "llm",
    "mcp",
    "fm",
    "bulk2single",
)


def _iter_python_files():
    for dirname in TARGET_DIRS:
        yield from sorted((PACKAGE_ROOT / dirname).rglob("*.py"))


def _module_targets_external(module: str) -> bool:
    parts = [part for part in module.split(".") if part]
    return "external" in parts


def _is_external_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_module_targets_external(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if _module_targets_external(module):
            return True
        if node.level and module == "external":
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


def _find_top_level_external_imports_textually(source: str) -> list[tuple[int, int, str]]:
    violations: list[tuple[int, int, str]] = []
    lines = source.splitlines()
    collecting = False
    current_start = 0
    current_parts: list[str] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        is_top_level = len(line) == len(stripped)

        if collecting:
            current_parts.append(stripped)
            if not stripped.endswith(("(", "\\")):
                stmt = " ".join(part.rstrip("\\") for part in current_parts).strip()
                module_text = stmt.split(" import ", 1)[0].replace("from ", "").replace("import ", "").strip()
                if _module_targets_external(module_text):
                    violations.append((current_start, 0, stmt))
                collecting = False
                current_parts = []
            continue

        if not is_top_level:
            continue
        if stripped.startswith("from ") or stripped.startswith("import "):
            if stripped.endswith(("(", "\\")):
                collecting = True
                current_start = lineno
                current_parts = [stripped]
                continue
            module_text = stripped.split(" import ", 1)[0].replace("from ", "").replace("import ", "").strip()
            if _module_targets_external(module_text):
                violations.append((lineno, 0, stripped))

    return violations

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
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        violations = _find_top_level_external_imports_textually(source)
    else:
        visitor = _TopLevelExternalImportVisitor()
        visitor.visit(tree)
        violations = visitor.violations

    if violations:
        details = "\n".join(
            f"{path.relative_to(REPO_ROOT)}:{lineno}:{col} -> {stmt}"
            for lineno, col, stmt in violations
        )
        pytest.fail(
            "Top-level external imports are forbidden in alignment/single/bulk/io/space/pp/pl/llm/mcp/fm/bulk2single.\n"
            f"{details}"
        )
