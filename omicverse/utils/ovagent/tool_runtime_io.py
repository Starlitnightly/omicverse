"""IO tool handlers: filesystem read/edit/write, glob, grep, notebook.

Extracted from ``tool_runtime.py`` during Phase 3 decomposition.
These are stateless handler functions that receive the ``AgentContext``
(and optionally a plan-mode checker) as explicit parameters.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

if TYPE_CHECKING:
    from .protocol import AgentContext


def handle_read(
    ctx: "AgentContext",
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
    pages: str = "",
) -> str:
    """Read a file and return its content with line numbers."""
    path = ctx._resolve_local_path(file_path)
    if not path.exists():
        return f"File not found: {path}"
    if path.is_dir():
        return f"Path is a directory, not a file: {path}"

    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        try:
            import nbformat
        except ImportError:
            return "Notebook reading requires nbformat to be installed."
        nb = nbformat.read(path, as_version=4)
        cells = []
        for idx, cell in enumerate(nb.cells):
            if idx < offset:
                continue
            if len(cells) >= max(1, limit):
                break
            source = (
                cell.source
                if isinstance(cell.source, str)
                else "".join(cell.source)
            )
            preview = "\n".join(source.splitlines()[:20])
            cells.append(f"[{idx}] {cell.cell_type}\n{preview}")
        return (
            "\n\n".join(cells)
            if cells
            else f"No notebook cells in range for {path}"
        )

    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError:
            return "PDF reading requires pypdf to be installed."
        reader = pypdf.PdfReader(str(path))
        page_numbers: List[int] = []
        if pages:
            for part in pages.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-", 1)
                    page_numbers.extend(
                        range(
                            max(1, int(start)),
                            min(len(reader.pages), int(end)) + 1,
                        )
                    )
                elif part:
                    page_numbers.append(int(part))
        else:
            page_numbers = list(
                range(1, min(len(reader.pages), 5) + 1)
            )
        snippets = []
        for page_no in page_numbers[:20]:
            text = reader.pages[page_no - 1].extract_text() or ""
            snippets.append(f"## Page {page_no}\n{text[:4000]}")
        return (
            "\n\n".join(snippets)
            if snippets
            else f"No readable PDF text in {path}"
        )

    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        stat = path.stat()
        return json.dumps(
            {
                "type": "image",
                "path": str(path),
                "mime": mime,
                "size_bytes": stat.st_size,
            },
            ensure_ascii=False,
            indent=2,
        )

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    segment = lines[max(0, offset) : max(0, offset) + max(1, limit)]
    numbered = [
        f"{idx + offset + 1:6d}\t{line[:2000]}"
        for idx, line in enumerate(segment)
    ]
    return "\n".join(numbered) if numbered else f"(empty file) {path}"


def handle_edit(
    ctx: "AgentContext",
    plan_mode_checker: Callable[[str], bool],
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace text in a file."""
    ctx._ensure_server_tool_mode("Edit")
    if plan_mode_checker("Edit"):
        return "Edit is blocked while the session is in plan mode."
    path = ctx._resolve_local_path(file_path)
    if not path.exists():
        return f"File not found: {path}"
    ctx._request_tool_approval(
        "Edit",
        reason=f"Edit file {path}",
        payload={"file_path": str(path)},
    )
    content = path.read_text(encoding="utf-8", errors="replace")
    occurrences = content.count(old_string)
    if occurrences == 0:
        return f"Edit failed: old_string was not found in {path}"
    updated = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    path.write_text(updated, encoding="utf-8")
    return json.dumps(
        {
            "file_path": str(path),
            "replacements": occurrences if replace_all else 1,
            "status": "updated",
        },
        ensure_ascii=False,
        indent=2,
    )


def handle_write(
    ctx: "AgentContext",
    plan_mode_checker: Callable[[str], bool],
    file_path: str,
    content: str,
) -> str:
    """Write content to a file."""
    ctx._ensure_server_tool_mode("Write")
    if plan_mode_checker("Write"):
        return "Write is blocked while the session is in plan mode."
    path = ctx._resolve_local_path(file_path)
    ctx._request_tool_approval(
        "Write",
        reason=f"Write file {path}",
        payload={"file_path": str(path)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return json.dumps(
        {
            "file_path": str(path),
            "bytes_written": len(content.encode("utf-8")),
        },
        ensure_ascii=False,
        indent=2,
    )


def handle_glob(
    ctx: "AgentContext",
    pattern: str,
    root: str = "",
    max_results: int = 200,
) -> str:
    """Glob for files matching a pattern."""
    base = (
        ctx._resolve_local_path(root, allow_relative=True)
        if root
        else Path(ctx._refresh_runtime_working_directory())
    )
    matches = sorted(str(p) for p in base.glob(pattern))[
        : max(1, max_results)
    ]
    return json.dumps(
        {"root": str(base), "pattern": pattern, "matches": matches},
        ensure_ascii=False,
        indent=2,
    )


def handle_grep(
    ctx: "AgentContext",
    pattern: str,
    root: str = "",
    glob: str = "",
    max_results: int = 200,
) -> str:
    """Search file contents with grep/ripgrep."""
    base = (
        ctx._resolve_local_path(root, allow_relative=True)
        if root
        else Path(ctx._refresh_runtime_working_directory())
    )
    rg_path = shutil.which("rg")
    if rg_path:
        cmd = [rg_path, "-n", pattern, str(base)]
        if glob:
            cmd[1:1] = ["-g", glob]
    else:
        cmd = ["grep", "-R", "-n", pattern, str(base)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )
    if proc.returncode not in (0, 1):
        return (
            proc.stderr
            or proc.stdout
            or f"Grep failed with exit code {proc.returncode}"
        )
    lines = [
        line
        for line in proc.stdout.splitlines()
        if line.strip()
    ][: max(1, max_results)]
    return "\n".join(lines) if lines else f"No matches for {pattern}"


def handle_notebook_edit(
    ctx: "AgentContext",
    plan_mode_checker: Callable[[str], bool],
    file_path: str,
    cell_index: int,
    source: str,
    cell_type: str = "",
) -> str:
    """Edit a notebook cell."""
    ctx._ensure_server_tool_mode("NotebookEdit")
    if plan_mode_checker("NotebookEdit"):
        return (
            "NotebookEdit is blocked while the session is in plan mode."
        )
    path = ctx._resolve_local_path(file_path)
    ctx._request_tool_approval(
        "NotebookEdit",
        reason=f"Edit notebook {path}",
        payload={"file_path": str(path), "cell_index": cell_index},
    )
    try:
        import nbformat
    except ImportError:
        return "NotebookEdit requires nbformat to be installed."
    nb = nbformat.read(path, as_version=4)
    if cell_index < 0 or cell_index >= len(nb.cells):
        return f"Notebook cell index out of range: {cell_index}"
    nb.cells[cell_index].source = source
    if cell_type:
        nb.cells[cell_index].cell_type = cell_type
    nbformat.write(nb, path)
    return json.dumps(
        {
            "file_path": str(path),
            "cell_index": cell_index,
            "status": "updated",
        },
        ensure_ascii=False,
        indent=2,
    )
