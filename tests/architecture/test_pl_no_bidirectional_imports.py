import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "omicverse"
PL_ROOT = PACKAGE_ROOT / "pl"
UTILS_COMPAT_FILES = (
    PACKAGE_ROOT / "utils" / "_plot.py",
    PACKAGE_ROOT / "utils" / "_scatterplot.py",
    PACKAGE_ROOT / "utils" / "_venn.py",
)
FORBIDDEN_UTILS_PLOTTING_MODULES = {
    "omicverse.utils._plot",
    "omicverse.utils._scatterplot",
    "omicverse.utils._venn",
}


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


def _find_import_violations(path: Path, predicate) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        violations = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            if predicate(stripped):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {stripped}")
        return violations

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if predicate(alias.name):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> import {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_module(path, node)
            if predicate(module):
                imported = ", ".join(alias.name for alias in node.names)
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> from {module} import {imported}"
                )
    return violations


@pytest.mark.parametrize("path", sorted(PL_ROOT.rglob("*.py")), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_pl_does_not_import_utils_plotting_compat_modules(path: Path):
    violations = _find_import_violations(
        path,
        lambda module: module in FORBIDDEN_UTILS_PLOTTING_MODULES,
    )
    if violations:
        pytest.fail(
            "`omicverse.pl` must not depend on `omicverse.utils` plotting compatibility modules.\n"
            + "\n".join(violations)
        )


@pytest.mark.parametrize("path", UTILS_COMPAT_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_utils_plotting_compat_modules_do_not_directly_import_pl_package(path: Path):
    violations = _find_import_violations(
        path,
        lambda module: module == "omicverse.pl" or module.startswith("omicverse.pl."),
    )
    if violations:
        pytest.fail(
            "Utils plotting compatibility modules must not directly import `omicverse.pl` or its submodules; "
            "use runtime lazy import instead.\n" + "\n".join(violations)
        )
