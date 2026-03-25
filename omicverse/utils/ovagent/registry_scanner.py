"""Static registry scanner — AST-based function discovery for code generation.

Extracted from ``smart_agent.py`` to reduce the facade to a thin composer.
Scans ``@register_function``-decorated definitions across the omicverse
package tree and produces registry-like records that the codegen pipeline
and tool runtime can use for function retrieval and scoring.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..._registry import _global_registry


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class RegistryScanner:
    """AST-based static registry scanner with caching and scoring.

    Provides all functionality previously located directly on
    ``OmicVerseAgent`` for static function discovery.
    """

    def __init__(self) -> None:
        self._static_registry_entries_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Aggregate collection
    # ------------------------------------------------------------------

    def collect_relevant_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return a compact set of registry entries relevant to *request*."""

        if max_entries <= 0:
            return []

        self.ensure_runtime_registry()

        runtime_entries = self.collect_runtime_entries(
            request, max_entries=max_entries * 3,
        )
        static_entries = self.collect_static_entries(
            request, max_entries=max_entries * 3,
        )

        merged: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
        for raw_entry in [*runtime_entries, *static_entries]:
            entry = self.normalize_entry(raw_entry)
            full_name = entry.get("full_name", "")
            if not full_name:
                continue
            score = self.score_entry(request, entry)
            if score <= 0:
                continue
            source_rank = 1 if entry.get("source") == "runtime" else 0
            current = merged.get(full_name)
            if current is None or (score, source_rank) > (current[0], current[1]):
                merged[full_name] = (score, source_rank, entry)

        ranked = sorted(
            merged.values(),
            key=lambda item: (item[0], item[1], item[2].get("full_name", "")),
            reverse=True,
        )
        return [entry for _, _, entry in ranked[:max_entries]]

    # ------------------------------------------------------------------
    # Runtime registry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_runtime_registry() -> None:
        """Hydrate the runtime registry when a partial lazy-import state is insufficient."""

        if getattr(_global_registry, "_registry", None):
            return

        try:
            from ...mcp.manifest import ensure_registry_populated
            ensure_registry_populated()
        except Exception:
            pass

    @staticmethod
    def collect_runtime_entries(
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Query the in-memory registry when it has been hydrated."""

        if not getattr(_global_registry, "_registry", None):
            return []

        seen: set = set()
        entries: List[Dict[str, Any]] = []

        def _add_matches(query: str) -> None:
            if not query or len(entries) >= max_entries:
                return
            for entry in _global_registry.find(query):
                full_name = entry.get("full_name", "")
                if not full_name or full_name in seen:
                    continue
                seen.add(full_name)
                entries.append(entry)
                if len(entries) >= max_entries:
                    return

        _add_matches(request)

        keywords = [
            token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\\.\\-]*", request or "")
            if len(token) >= 2
        ]
        for keyword in keywords[:12]:
            _add_matches(keyword)
            if len(entries) >= max_entries:
                break

        return entries

    # ------------------------------------------------------------------
    # Normalization / scoring
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert registry entries to public-facing ov.* names for code generation."""

        normalized = dict(entry)
        source = normalized.get("source")
        normalized["source"] = source if str(source).startswith("static_ast") else "runtime"

        original_full_name = str(normalized.get("full_name", "") or "")
        normalized["registry_full_name"] = original_full_name

        public_name = original_full_name
        short_name = str(normalized.get("short_name") or normalized.get("name") or "")

        if original_full_name.startswith("omicverse."):
            parts = original_full_name.split(".")
            if len(parts) >= 2:
                domain = parts[1]
                if domain == "_settings":
                    public_name = f"ov.core.{short_name or parts[-1]}"
                elif domain:
                    public_name = f"ov.{domain}.{short_name or parts[-1]}"

        normalized["full_name"] = public_name
        return normalized

    @staticmethod
    def score_entry(
        request: str,
        entry: Dict[str, Any],
    ) -> float:
        """Score a registry entry for lightweight code generation retrieval."""

        query = (request or "").strip().lower()
        if not query:
            return 0.0

        tokens = [
            token for token in re.findall(r"[a-z0-9_\\.\\-]+", query)
            if len(token) >= 2
        ]
        aliases = [str(alias).lower() for alias in (entry.get("aliases") or [])]
        haystack_parts = [
            entry.get("name", ""),
            entry.get("short_name", ""),
            entry.get("full_name", ""),
            entry.get("registry_full_name", ""),
            entry.get("category", ""),
            entry.get("description", ""),
            " ".join(aliases),
            " ".join(entry.get("examples", []) or []),
            " ".join(entry.get("imports", []) or []),
        ]
        haystack = " ".join(str(part) for part in haystack_parts).lower()

        score = 0.0
        if query == str(entry.get("full_name", "")).lower():
            score += 10.0
        if query == str(entry.get("short_name", "")).lower():
            score += 9.0
        if query in haystack:
            score += 4.0

        for alias in aliases:
            if alias == query:
                score += 8.0
            elif alias and alias in query:
                score += 2.0

        for token in tokens:
            if token in haystack:
                score += 1.25

        public_name = str(entry.get("full_name", ""))
        if public_name.startswith(("ov.pp.", "ov.single.", "ov.pl.", "ov.bulk.", "ov.space.")):
            score += 0.5

        if public_name.startswith("ov.datasets.") and not any(
            word in query for word in ("dataset", "download", "read", "load", "example", "demo")
        ):
            score -= 2.0

        if public_name.startswith("ov.core.") and not any(
            word in query for word in ("reference", "table", "gpu", "cpu", "settings")
        ):
            score -= 2.0

        return score

    # ------------------------------------------------------------------
    # Static AST scanning
    # ------------------------------------------------------------------

    def load_static_entries(self) -> List[Dict[str, Any]]:
        """Parse @register_function metadata plus nested method/branch capabilities."""

        if self._static_registry_entries_cache is not None:
            return self._static_registry_entries_cache

        package_root = Path(__file__).resolve().parents[2]
        search_roots = (
            "pp", "pl", "single", "bulk", "space", "utils",
            "io", "alignment", "external", "biocontext", "bulk2single", "datasets",
        )
        entries: List[Dict[str, Any]] = []
        seen: set = set()

        for root_name in search_roots:
            root = package_root / root_name
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*.py")):
                if file_path.name == "__init__.py":
                    continue
                if "__pycache__" in file_path.parts or ".ipynb_checkpoints" in file_path.parts:
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(file_path))
                except Exception:
                    continue

                for node in tree.body:
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    decorator = _find_register_function_decorator(node)
                    if decorator is None:
                        continue
                    for entry in _build_static_entries(file_path, node, decorator):
                        full_name = entry.get("full_name", "")
                        if not full_name or full_name in seen:
                            continue
                        seen.add(full_name)
                        entries.append(entry)

        self._static_registry_entries_cache = entries
        return entries

    def collect_static_entries(
        self,
        request: str,
        max_entries: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search the static AST-derived registry snapshot."""

        query = (request or "").strip().lower()
        if not query:
            return []

        tokens = [token for token in re.findall(r"[a-z0-9_\\.\\-]+", query) if len(token) >= 2]
        entries = self.load_static_entries()
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for entry in entries:
            aliases = entry.get("aliases", []) or []
            haystack = " ".join(
                [
                    entry.get("name", ""),
                    entry.get("short_name", ""),
                    entry.get("full_name", ""),
                    entry.get("category", ""),
                    entry.get("description", ""),
                    " ".join(aliases),
                    " ".join(entry.get("examples", []) or []),
                    " ".join(entry.get("imports", []) or []),
                ]
            ).lower()

            score = 0.0
            if query == entry.get("name", "").lower():
                score += 8.0
            if query == entry.get("short_name", "").lower():
                score += 8.0
            if query == entry.get("full_name", "").lower():
                score += 9.0
            if query in haystack:
                score += 4.0
            for alias in aliases:
                alias_lower = str(alias).lower()
                if query == alias_lower:
                    score += 8.0
                elif alias_lower and alias_lower in query:
                    score += 2.0
            for token in tokens:
                if token and token in haystack:
                    score += 1.0

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:max_entries]]


# ---------------------------------------------------------------------------
# Module-level helpers (stateless AST utilities)
# ---------------------------------------------------------------------------


def _find_register_function_decorator(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> Optional[ast.Call]:
    """Return the register_function decorator call when present."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Name) and func.id == "register_function":
            return decorator
        if isinstance(func, ast.Attribute) and func.attr == "register_function":
            return decorator
    return None


def _literal_eval_or_default(node: Optional[ast.AST], default: Any) -> Any:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except Exception:
        return default


def _derive_static_signature(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> str:
    """Build a lightweight signature string from AST."""
    if isinstance(node, ast.ClassDef):
        return f"{node.name}(...)"
    arg_names = [arg.arg for arg in node.args.args]
    if arg_names and arg_names[0] == "self":
        arg_names = arg_names[1:]
    return f"{node.name}({', '.join(arg_names)})"


def _collect_import_targets(statements: List[ast.stmt]) -> List[str]:
    """Collect import targets mentioned inside a function or branch body."""
    module = ast.Module(body=list(statements), type_ignores=[])
    imports: List[str] = []
    for child in ast.walk(module):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name:
                    imports.append(alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            if child.module:
                imports.append(child.module.split(".")[0])
            for alias in child.names:
                if alias.name and alias.name != "*":
                    imports.append(alias.name.split(".")[0])
    return list(dict.fromkeys(imports))


def _filter_examples_for_method(examples: List[str], method_name: str) -> List[str]:
    """Keep examples most relevant to a nested method entry."""
    if not examples:
        return []
    matches = [example for example in examples if method_name in str(example)]
    return matches[:3] if matches else list(examples[:2])


def _filter_examples_for_branch(
    examples: List[str], param_name: str, branch_value: str,
) -> List[str]:
    """Keep examples mentioning the relevant branch parameter/value when possible."""
    if not examples:
        return []
    value_lower = branch_value.lower()
    param_lower = param_name.lower()
    matches = [
        example for example in examples
        if value_lower in str(example).lower() or param_lower in str(example).lower()
    ]
    return matches[:3] if matches else list(examples[:2])


def _branch_subject_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _branch_string_values(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: List[str] = []
        for child in node.elts:
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                values.append(child.value)
        return values
    return []


def _extract_branch_variants(test: ast.AST) -> List[Tuple[str, str]]:
    """Extract simple string-dispatch branches from an ``if`` test."""
    variants: List[Tuple[str, str]] = []
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for value in test.values:
            variants.extend(_extract_branch_variants(value))
        return variants

    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return variants

    subject = _branch_subject_name(test.left)
    comparator = test.comparators[0]
    op = test.ops[0]
    values: List[str] = []

    if isinstance(op, ast.Eq):
        values = _branch_string_values(comparator)
    elif isinstance(op, ast.In):
        values = _branch_string_values(comparator)

    if subject and values:
        variants.extend((subject, value) for value in values)
    return variants


def _build_static_entry(
    file_path: Path,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    decorator: ast.Call,
) -> Optional[Dict[str, Any]]:
    """Convert one @register_function AST node into a registry-like record."""

    package_root = Path(__file__).resolve().parents[2]
    rel = file_path.relative_to(package_root)
    if not rel.parts:
        return None
    domain = rel.parts[0]
    if domain not in {"pp", "pl", "single", "bulk", "space", "utils"}:
        return None

    args = list(decorator.args)
    aliases = _literal_eval_or_default(args[0], []) if len(args) >= 1 else []
    category = _literal_eval_or_default(args[1], "") if len(args) >= 2 else ""
    description = _literal_eval_or_default(args[2], "") if len(args) >= 3 else ""
    examples = _literal_eval_or_default(args[3], []) if len(args) >= 4 else []

    kw = {item.arg: item.value for item in decorator.keywords if item.arg}
    aliases = _literal_eval_or_default(kw.get("aliases"), aliases)
    category = _literal_eval_or_default(kw.get("category"), category)
    description = _literal_eval_or_default(kw.get("description"), description)
    examples = _literal_eval_or_default(kw.get("examples"), examples)
    related = _literal_eval_or_default(kw.get("related"), [])
    prerequisites = _literal_eval_or_default(kw.get("prerequisites"), {})
    requires = _literal_eval_or_default(kw.get("requires"), {})
    produces = _literal_eval_or_default(kw.get("produces"), {})

    full_name = f"ov.{domain}.{node.name}"
    module_name = "omicverse." + ".".join(rel.with_suffix("").parts)

    return {
        "name": node.name,
        "short_name": node.name,
        "full_name": full_name,
        "module": module_name,
        "aliases": aliases or [],
        "category": category or domain,
        "description": description or (ast.get_docstring(node) or ""),
        "examples": examples or [],
        "related": related or [],
        "signature": _derive_static_signature(node),
        "docstring": ast.get_docstring(node) or "",
        "prerequisites": prerequisites or {},
        "requires": requires or {},
        "produces": produces or {},
        "source": "static_ast",
    }


def _build_static_method_entry(
    base_entry: Dict[str, Any],
    class_node: ast.ClassDef,
    method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Optional[Dict[str, Any]]:
    """Create a searchable entry for a public method on a registered class."""

    aliases: List[str] = [
        method_node.name,
        f"{class_node.name}.{method_node.name}",
        f"{base_entry.get('short_name', class_node.name)} {method_node.name}",
    ]
    for alias in (base_entry.get("aliases") or [])[:8]:
        alias = str(alias).strip()
        if alias:
            aliases.append(f"{alias} {method_node.name}")

    description = ast.get_docstring(method_node) or (
        f"Method `{method_node.name}` on {base_entry.get('full_name', class_node.name)}."
    )

    return {
        "name": method_node.name,
        "short_name": method_node.name,
        "full_name": f"{base_entry.get('full_name', class_node.name)}.{method_node.name}",
        "module": base_entry.get("module", ""),
        "aliases": list(dict.fromkeys(aliases)),
        "category": base_entry.get("category", ""),
        "description": description,
        "examples": _filter_examples_for_method(base_entry.get("examples", []), method_node.name),
        "related": [base_entry.get("full_name", "")],
        "signature": _derive_static_signature(method_node),
        "docstring": ast.get_docstring(method_node) or "",
        "prerequisites": base_entry.get("prerequisites", {}) or {},
        "requires": base_entry.get("requires", {}) or {},
        "produces": base_entry.get("produces", {}) or {},
        "source": "static_ast_method",
        "parent_full_name": base_entry.get("full_name", ""),
        "imports": _collect_import_targets(method_node.body),
    }


def _build_static_branch_entries(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    parent_entry: Dict[str, Any],
    owner_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create searchable entries for string-dispatched branches."""

    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        for param_name, branch_value in _extract_branch_variants(child.test):
            key = (param_name, branch_value)
            if key in seen:
                continue
            seen.add(key)
            imports = _collect_import_targets(child.body)
            aliases = [
                branch_value,
                f"{node.name} {branch_value}",
                f"{param_name} {branch_value}",
                f"{parent_entry.get('short_name', node.name)} {branch_value}",
            ]
            for alias in (owner_entry.get("aliases") or [])[:8]:
                alias = str(alias).strip()
                if alias:
                    aliases.append(f"{alias} {branch_value}")
            examples = _filter_examples_for_branch(
                parent_entry.get("examples", []),
                param_name,
                branch_value,
            )
            description = (
                f"Variant of `{parent_entry.get('full_name', node.name)}` when "
                f"`{param_name}='{branch_value}'`."
            )
            if imports:
                description += " Imports/uses: " + ", ".join(imports) + "."

            entries.append({
                "name": branch_value,
                "short_name": branch_value,
                "full_name": f"{parent_entry.get('full_name', node.name)}[{param_name}={branch_value}]",
                "module": parent_entry.get("module", ""),
                "aliases": list(dict.fromkeys(aliases)),
                "category": parent_entry.get("category", ""),
                "description": description,
                "examples": examples,
                "related": [parent_entry.get("full_name", "")],
                "signature": parent_entry.get("signature", _derive_static_signature(node)),
                "docstring": parent_entry.get("docstring", ""),
                "prerequisites": parent_entry.get("prerequisites", {}) or {},
                "requires": parent_entry.get("requires", {}) or {},
                "produces": parent_entry.get("produces", {}) or {},
                "source": "static_ast_branch",
                "parent_full_name": parent_entry.get("full_name", ""),
                "branch_parameter": param_name,
                "branch_value": branch_value,
                "imports": imports,
            })

    return entries


def _build_nested_entries(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    base_entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Expand registered classes/functions into searchable method and branch entries."""

    nested: List[Dict[str, Any]] = []
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name.startswith("_"):
                continue
            method_entry = _build_static_method_entry(base_entry, node, child)
            if not method_entry:
                continue
            nested.append(method_entry)
            nested.extend(_build_static_branch_entries(child, method_entry, base_entry))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nested.extend(_build_static_branch_entries(node, base_entry, base_entry))
    return nested


def _build_static_entries(
    file_path: Path,
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    decorator: ast.Call,
) -> List[Dict[str, Any]]:
    """Build the primary static entry plus nested method/branch entries."""
    base_entry = _build_static_entry(file_path, node, decorator)
    if not base_entry:
        return []
    entries = [base_entry]
    entries.extend(_build_nested_entries(node, base_entry))
    return entries
