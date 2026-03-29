import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "omicverse"
FORBIDDEN_MODULES = {
    "omicverse.utils._plot",
    "omicverse.utils._scatterplot",
    "omicverse.utils._venn",
}


def _iter_python_files():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT)
        if rel.parts[0] == "utils":
            continue
        yield path


def _resolve_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    package = list(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    up = node.level - 1
    if up:
        package = package[:-up]
    if node.module:
        package.extend(node.module.split("."))
    return ".".join(package)


@pytest.mark.parametrize("path", list(_iter_python_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_non_utils_modules_do_not_import_utils_plotting_backends(path: Path):
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        violations = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            for module in FORBIDDEN_MODULES:
                if module in stripped:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {stripped}")
        if violations:
            pytest.fail(
                "Plotting implementations should live under omicverse.pl; only utils compatibility modules may import "
                "omicverse.utils._plot/_scatterplot/_venn.\n" + "\n".join(violations)
            )
        return

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_MODULES:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_module(path, node)
            if module in FORBIDDEN_MODULES:
                imported = ", ".join(alias.name for alias in node.names)
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> from {module} import {imported}"
                )

    if violations:
        pytest.fail(
            "Plotting implementations should live under omicverse.pl; only utils compatibility modules may import "
            "omicverse.utils._plot/_scatterplot/_venn.\n" + "\n".join(violations)
        )
