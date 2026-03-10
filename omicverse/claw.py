"""CLI for code-only OmicVerse generation."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TextIO, Tuple


_DEFAULT_SOCKET_ENV = "OMICVERSE_CLAW_SOCKET"
_DEFAULT_SOCKET_FALLBACK = Path.home() / ".cache" / "omicverse" / "claw.sock"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omicverse claw",
        description="Generate OmicVerse Python code from a natural-language request.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Natural-language request to convert into OmicVerse Python code.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="LLM model to use for code generation.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Provider API key. Falls back to environment variables when omitted.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional custom model endpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file path to write the generated code.",
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=8,
        help="Maximum number of relevant registry functions to include in the lightweight prompt.",
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Skip the lightweight review pass and return first-pass code.",
    )
    parser.add_argument(
        "--debug-registry",
        action="store_true",
        help="Print matched registry entries and skills to stderr before code generation.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run a persistent local claw daemon that keeps OmicVerse imported.",
    )
    parser.add_argument(
        "--use-daemon",
        action="store_true",
        help="Send this request to a running local claw daemon instead of starting a fresh process.",
    )
    parser.add_argument(
        "--stop-daemon",
        action="store_true",
        help="Ask the running local claw daemon to stop.",
    )
    parser.add_argument(
        "--socket",
        default=None,
        help="Unix socket path for claw daemon communication.",
    )
    return parser


def _ensure_local_omicverse_package_path() -> None:
    """Mirror Jarvis' local package path fix-up for editable-like installs."""

    package_dir = Path(__file__).resolve().parent
    utils_dir = package_dir / "utils"
    repo_root = package_dir.parent

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    pkg = sys.modules.get("omicverse")
    if pkg is None:
        return

    package_paths = getattr(pkg, "__path__", None)
    if package_paths is None:
        importlib.invalidate_caches()
        return

    package_dir_str = str(package_dir)
    if package_dir_str not in package_paths:
        package_paths.append(package_dir_str)

    utils_pkg = sys.modules.get("omicverse.utils")
    if utils_pkg is not None:
        utils_paths = getattr(utils_pkg, "__path__", None)
        utils_dir_str = str(utils_dir)
        if utils_paths is not None and utils_dir_str not in utils_paths:
            utils_paths.append(utils_dir_str)

    importlib.invalidate_caches()


def _load_agent_factory():
    """Reuse Jarvis-style Agent loading so claw shares the same ecosystem path."""

    try:
        from .utils.smart_agent import Agent

        return Agent
    except ImportError:
        _ensure_local_omicverse_package_path()
        from omicverse.utils.smart_agent import Agent

        return Agent


def _build_agent(args: argparse.Namespace):
    Agent = _load_agent_factory()

    return Agent(
        model=args.model,
        api_key=args.api_key,
        endpoint=args.endpoint,
        enable_reflection=not args.no_reflection,
        enable_result_review=False,
        use_notebook_execution=False,
        enable_filesystem_context=False,
        verbose=False,
    )


def _print_debug_registry(
    agent,
    question: str,
    max_functions: int,
    stream: Optional[TextIO] = None,
) -> None:
    """Print matched runtime registry entries to stderr."""

    target = stream or sys.stderr
    agent._ensure_runtime_registry_for_codegen()
    entries = agent._collect_runtime_registry_entries(
        question,
        max_entries=max_functions,
    )

    print("== Claw Registry Matches ==", file=target)
    if not entries:
        print("(none)", file=target)
    else:
        for entry in entries:
            normalized = agent._normalize_registry_entry_for_codegen(entry)
            full_name = normalized.get("full_name", "")
            signature = entry.get("signature", "")
            source = normalized.get("source", "runtime")
            print(f"- {full_name} :: {signature} [{source}]", file=target)


def _forward_captured_output_to_stderr(buffer: io.StringIO) -> None:
    """Relay internal stdout diagnostics to stderr so generated code stays clean."""

    text = buffer.getvalue().strip()
    if text:
        print(text, file=sys.stderr)


class _DebugProgress:
    """Small stderr progress helper used only for `--debug-registry`."""

    def __init__(self, enabled: bool, total: int) -> None:
        self.enabled = enabled
        self._bar = None
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=total,
                file=sys.stderr,
                leave=False,
                dynamic_ncols=True,
                desc="claw",
            )
        except Exception:
            self.enabled = False

    def step(self, description: str) -> None:
        if self._bar is None:
            return
        if self._bar.total is not None and self._bar.n >= self._bar.total:
            self._bar.total = self._bar.n + 1
            self._bar.refresh()
        self._bar.set_description(description)
        self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _make_progress_callback(progress: _DebugProgress, prefix: str = ""):
    """Create a small callback that maps internal codegen phases onto the tqdm bar."""

    def _callback(message: str) -> None:
        label = f"{prefix}{message}" if prefix else message
        progress.step(label)

    return _callback


def _print_registry_loaded(agent, stream: Optional[TextIO] = None) -> None:
    """Always emit the registry-loaded summary to stderr for claw visibility."""

    target = stream or sys.stderr
    try:
        stats = agent._get_registry_stats()
    except Exception:
        return
    print(
        f"📚 Function registry loaded: {stats['total_functions']} functions in {stats['categories']} categories",
        file=target,
    )


def _default_socket_path() -> Path:
    """Return the default Unix socket path for the local claw daemon."""

    raw = os.environ.get(_DEFAULT_SOCKET_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_SOCKET_FALLBACK


def _resolve_socket_path(args: argparse.Namespace) -> Path:
    """Resolve the socket path used for daemon communication."""

    if args.socket:
        return Path(args.socket).expanduser()
    return _default_socket_path()


def _serialize_daemon_message(conn: socket.socket, payload: Dict[str, Any]) -> None:
    """Write one newline-delimited JSON message to the connection."""

    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _parse_daemon_request(conn: socket.socket) -> Dict[str, Any]:
    """Read one JSON request from a daemon client connection."""

    chunks: List[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _request_daemon_stream(
    socket_path: Path,
    payload: Dict[str, Any],
    on_event=None,
) -> Dict[str, Any]:
    """Send one request to the local claw daemon and consume streamed events."""

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps(payload).encode("utf-8"))
        client.shutdown(socket.SHUT_WR)

        buffer = ""
        final_payload: Optional[Dict[str, Any]] = None
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                if message.get("type") == "progress":
                    if on_event is not None:
                        on_event(message)
                elif message.get("type") == "final":
                    final_payload = message.get("payload") or {}
                else:
                    final_payload = message

    if final_payload is None:
        raise RuntimeError("claw daemon returned an empty response")
    return final_payload


def _request_daemon(socket_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send one request to the local claw daemon and return the final JSON reply."""

    return _request_daemon_stream(socket_path, payload)


def _build_daemon_payload(args: argparse.Namespace, question: str) -> Dict[str, Any]:
    """Convert CLI args into a daemon request payload."""

    return {
        "action": "generate",
        "question": question,
        "model": args.model,
        "api_key": args.api_key,
        "endpoint": args.endpoint,
        "max_functions": max(1, args.max_functions),
        "no_reflection": bool(args.no_reflection),
        "debug_registry": bool(args.debug_registry),
    }


def _daemon_defaults_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract daemon-side default agent configuration from CLI args."""

    return {
        "model": args.model,
        "api_key": args.api_key,
        "endpoint": args.endpoint,
        "no_reflection": bool(args.no_reflection),
    }


def _merge_daemon_payload(payload: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill request payload with daemon startup defaults when fields are omitted."""

    merged = dict(defaults)
    merged.update(payload)
    for key in ("model", "api_key", "endpoint"):
        if payload.get(key) is None:
            merged[key] = defaults.get(key)
    if "no_reflection" not in payload:
        merged["no_reflection"] = defaults.get("no_reflection", False)
    return merged


def _build_agent_from_payload(payload: Dict[str, Any]):
    """Create an agent using request-scoped configuration from daemon payload."""

    request_args = SimpleNamespace(
        model=payload.get("model") or "gpt-5.2",
        api_key=payload.get("api_key"),
        endpoint=payload.get("endpoint"),
        no_reflection=bool(payload.get("no_reflection")),
    )
    return _build_agent(request_args)


def _agent_cache_key(payload: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    """Build a stable key for daemon-side agent reuse."""

    return (
        str(payload.get("model") or "gpt-5.2"),
        str(payload.get("endpoint") or ""),
        str(payload.get("api_key") or ""),
        bool(payload.get("no_reflection")),
    )


def _handle_daemon_generate(
    payload: Dict[str, Any],
    daemon_defaults: Dict[str, Any],
    agent_cache: Dict[Tuple[str, str, str, bool], Any],
    emit_event=None,
) -> Dict[str, Any]:
    """Handle one generate request inside the daemon process."""

    captured_output = io.StringIO()
    debug_output = io.StringIO()
    try:
        effective_payload = _merge_daemon_payload(payload, daemon_defaults)
        cache_key = _agent_cache_key(effective_payload)
        agent = agent_cache.get(cache_key)
        agent_reused = agent is not None
        if agent is None:
            if emit_event is not None:
                emit_event({"type": "progress", "message": "daemon: init agent"})
            with contextlib.redirect_stdout(captured_output):
                agent = _build_agent_from_payload(effective_payload)
            agent_cache[cache_key] = agent
        elif emit_event is not None:
            emit_event({"type": "progress", "message": "daemon: reuse agent"})

        question = str(effective_payload.get("question") or "").strip()
        if not question:
            raise ValueError("question cannot be empty")

        max_functions = int(effective_payload.get("max_functions") or 8)
        with contextlib.redirect_stdout(captured_output):
            if agent_reused:
                _print_registry_loaded(agent, stream=debug_output)
            if effective_payload.get("debug_registry"):
                if emit_event is not None:
                    emit_event({"type": "progress", "message": "daemon: inspect registry"})
                _print_debug_registry(
                    agent,
                    question,
                    max_functions=max_functions,
                    stream=debug_output,
                )
            code = agent.generate_code(
                question,
                adata=None,
                max_functions=max_functions,
                progress_callback=(
                    None
                    if emit_event is None
                    else lambda message: emit_event(
                        {"type": "progress", "message": f"daemon: {message}"}
                    )
                ),
            )
        if emit_event is not None:
            emit_event({"type": "progress", "message": "daemon: finalize"})
        return {
            "ok": True,
            "code": code,
            "logs": captured_output.getvalue(),
            "debug": debug_output.getvalue(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "logs": captured_output.getvalue(),
            "debug": debug_output.getvalue(),
        }


def _prime_daemon_imports() -> str:
    """Import the Agent path once so the daemon keeps OmicVerse warm in memory."""

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        _load_agent_factory()
    return captured_output.getvalue()


def _run_daemon(args: argparse.Namespace) -> int:
    """Run the local claw daemon in the foreground."""

    socket_path = _resolve_socket_path(args)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    if socket_path.exists():
        try:
            _request_daemon(socket_path, {"action": "ping"})
        except Exception:
            socket_path.unlink()
        else:
            print(f"claw daemon already running at {socket_path}", file=sys.stderr)
            return 1

    warm_logs = _prime_daemon_imports().strip()
    if warm_logs:
        print(warm_logs, file=sys.stderr)

    daemon_defaults = _daemon_defaults_from_args(args)
    agent_cache: Dict[Tuple[str, str, str, bool], Any] = {}
    init_output = io.StringIO()
    eager_payload = _merge_daemon_payload({}, daemon_defaults)
    eager_key = _agent_cache_key(eager_payload)
    with contextlib.redirect_stdout(init_output):
        agent_cache[eager_key] = _build_agent_from_payload(eager_payload)
    _forward_captured_output_to_stderr(init_output)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen()
        print(f"claw daemon listening on {socket_path}", file=sys.stderr)

        should_stop = False
        while not should_stop:
            conn, _ = server.accept()
            with conn:
                payload = _parse_daemon_request(conn)
                action = str(payload.get("action") or "generate")
                if action == "ping":
                    response = {"ok": True, "message": "pong"}
                elif action == "shutdown":
                    response = {"ok": True, "message": "stopping"}
                    should_stop = True
                else:
                    response = _handle_daemon_generate(
                        payload,
                        daemon_defaults,
                        agent_cache,
                        emit_event=lambda event: _serialize_daemon_message(conn, event),
                    )
                _serialize_daemon_message(conn, {"type": "final", "payload": response})
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()

    return 0


def _run_via_daemon(args: argparse.Namespace, question: str) -> int:
    """Send one codegen request to the running claw daemon."""

    socket_path = _resolve_socket_path(args)
    if not socket_path.exists():
        print(
            f"claw daemon is not running at {socket_path}. Start it with: "
            f"omicverse claw --daemon --socket {socket_path}",
            file=sys.stderr,
        )
        return 1

    progress = _DebugProgress(enabled=bool(args.debug_registry), total=2)
    progress.step("daemon: send request")
    response = _request_daemon_stream(
        socket_path,
        _build_daemon_payload(args, question),
        on_event=lambda event: progress.step(str(event.get("message") or "daemon")),
    )
    if response.get("logs"):
        print(str(response["logs"]).strip(), file=sys.stderr)
    if response.get("debug"):
        print(str(response["debug"]).strip(), file=sys.stderr)
    progress.step("daemon: done")
    progress.close()
    if not response.get("ok"):
        print(str(response.get("error") or "claw daemon request failed"), file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(str(response["code"]), encoding="utf-8")
    print(str(response["code"]))
    return 0


def _stop_daemon(args: argparse.Namespace) -> int:
    """Ask the running claw daemon to stop."""

    socket_path = _resolve_socket_path(args)
    if not socket_path.exists():
        print(f"claw daemon is not running at {socket_path}", file=sys.stderr)
        return 1

    response = _request_daemon(socket_path, {"action": "shutdown"})
    print(str(response.get("message") or "stopped"), file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    init_output = io.StringIO()
    captured_output = io.StringIO()

    if args.daemon:
        return _run_daemon(args)

    if args.stop_daemon:
        return _stop_daemon(args)

    if not question:
        parser.error("question cannot be empty")

    if args.use_daemon:
        try:
            return _run_via_daemon(args, question)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    progress = _DebugProgress(enabled=bool(args.debug_registry), total=8)
    try:
        with contextlib.redirect_stdout(init_output):
            agent = _build_agent(args)
        _forward_captured_output_to_stderr(init_output)
        progress.step("init agent")
        with contextlib.redirect_stdout(captured_output):
            if args.debug_registry:
                _print_debug_registry(
                    agent,
                    question,
                    max_functions=max(1, args.max_functions),
                )
        progress.step("inspect registry")
        with contextlib.redirect_stdout(captured_output):
            code = agent.generate_code(
                question,
                adata=None,
                max_functions=max(1, args.max_functions),
                progress_callback=_make_progress_callback(progress),
            )
        progress.close()
        _forward_captured_output_to_stderr(captured_output)
    except Exception as exc:
        progress.close()
        _forward_captured_output_to_stderr(init_output)
        _forward_captured_output_to_stderr(captured_output)
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")

    print(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
