import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PL_ROOT = REPO_ROOT / "omicverse" / "pl"


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


def _targets_scanpy(module: str) -> bool:
    return module == "scanpy" or module.startswith("scanpy.")


@pytest.mark.parametrize("path", sorted(PL_ROOT.rglob("*.py")), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_pl_has_no_scanpy_imports(path: Path):
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        violations = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("from ") or stripped.startswith("import ")):
                continue
            if "import scanpy" in stripped or "from scanpy" in stripped:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {stripped}")
    else:
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _targets_scanpy(alias.name):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = _resolve_module(path, node)
                if _targets_scanpy(module):
                    imported = ", ".join(alias.name for alias in node.names)
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> from {module} import {imported}"
                    )

    if violations:
        pytest.fail(
            "`omicverse.pl` should not import Scanpy directly; plotting compatibility should live in local modules.\n"
            + "\n".join(violations)
        )
