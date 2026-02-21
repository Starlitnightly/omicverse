"""
Terminal backend — PTY-based interactive shell sessions.

Each call to /api/terminal/create spawns a real shell (bash/zsh) in a
pseudo-terminal.  The frontend connects via SSE to /api/terminal/stream
and sends keystrokes via POST to /api/terminal/input.
"""

import os
import pty
import select
import struct
import fcntl
import termios
import signal
import queue
import threading
import uuid
import json
import base64
import logging
import shutil

from flask import Blueprint, request, jsonify, Response

terminal_bp = Blueprint('terminal', __name__)

# Global registry of live sessions
_sessions: dict = {}
_sessions_lock = threading.Lock()


# ── Session class ─────────────────────────────────────────────────────────────

class TerminalSession:
    """Wraps a real shell running in a PTY."""

    def __init__(self, session_id: str, cwd: str | None = None):
        self.session_id  = session_id
        self.output_q    = queue.Queue()
        self.alive       = False
        self.master_fd   = None
        self.pid         = None
        self._start(cwd or os.getcwd())

    # ── private ───────────────────────────────────────────────────────────────

    def _start(self, cwd: str) -> None:
        shell = (shutil.which('zsh') or shutil.which('bash') or '/bin/sh')
        env   = os.environ.copy()
        env.update({'TERM': 'xterm-256color', 'COLORTERM': 'truecolor'})

        self.pid, self.master_fd = pty.fork()

        if self.pid == 0:          # ── child ──
            try:
                os.chdir(cwd)
            except Exception:
                pass
            os.execve(shell, [shell], env)
            os._exit(1)           # should never reach here

        else:                      # ── parent ──
            self.alive = True
            t = threading.Thread(target=self._reader, daemon=True)
            t.start()

    def _reader(self) -> None:
        """Read bytes from PTY and push them to the output queue."""
        while self.alive:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.2)
                if r:
                    data = os.read(self.master_fd, 4096)
                    if not data:
                        break
                    self.output_q.put(data)
            except OSError:
                break
        self.alive = False
        self.output_q.put(None)   # sentinel

    # ── public ────────────────────────────────────────────────────────────────

    def write(self, data: bytes) -> None:
        if self.alive and self.master_fd is not None:
            try:
                os.write(self.master_fd, data)
            except OSError:
                self.alive = False

    def resize(self, rows: int, cols: int) -> None:
        if self.alive and self.master_fd is not None:
            try:
                winsize = struct.pack('HHHH', rows, cols, 0, 0)
                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
            except Exception:
                pass

    def kill(self) -> None:
        self.alive = False
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, os.WNOHANG)
            except OSError:
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None


# ── Routes ────────────────────────────────────────────────────────────────────

@terminal_bp.route('/api/terminal/create', methods=['POST'])
def terminal_create():
    """Spawn a new shell session, return its session_id."""
    data = request.json or {}
    cwd  = data.get('cwd') or os.getcwd()
    sid  = str(uuid.uuid4())
    sess = TerminalSession(sid, cwd)
    with _sessions_lock:
        # Reap stale sessions (limit to 8 concurrent)
        dead = [k for k, v in _sessions.items() if not v.alive]
        for k in dead:
            _sessions.pop(k, None)
        _sessions[sid] = sess
    return jsonify({'session_id': sid})


@terminal_bp.route('/api/terminal/stream/<session_id>')
def terminal_stream(session_id):
    """SSE stream of terminal output (base64-encoded raw bytes)."""
    with _sessions_lock:
        sess = _sessions.get(session_id)
    if not sess:
        return jsonify({'error': 'session not found'}), 404

    def generate():
        while True:
            try:
                data = sess.output_q.get(timeout=20)
                if data is None:             # session ended
                    yield f"data: {json.dumps({'type': 'exit'})}\n\n"
                    break
                payload = base64.b64encode(data).decode('ascii')
                yield f"data: {json.dumps({'type': 'output', 'data': payload})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except Exception as exc:
                logging.error(f'terminal stream error: {exc}')
                break

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':    'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@terminal_bp.route('/api/terminal/input', methods=['POST'])
def terminal_input():
    """Forward keystrokes from the frontend to the PTY."""
    data = request.json or {}
    sid  = data.get('session_id', '')
    raw  = data.get('data', '')
    with _sessions_lock:
        sess = _sessions.get(sid)
    if not sess:
        return jsonify({'error': 'session not found'}), 404
    sess.write(raw.encode('utf-8'))
    return jsonify({'ok': True})


@terminal_bp.route('/api/terminal/resize', methods=['POST'])
def terminal_resize():
    """Inform the PTY of a new window size."""
    data = request.json or {}
    sid  = data.get('session_id', '')
    rows = int(data.get('rows', 24))
    cols = int(data.get('cols', 80))
    with _sessions_lock:
        sess = _sessions.get(sid)
    if not sess:
        return jsonify({'error': 'session not found'}), 404
    sess.resize(rows, cols)
    return jsonify({'ok': True})


@terminal_bp.route('/api/terminal/kill', methods=['POST'])
def terminal_kill():
    """Destroy a terminal session."""
    data = request.json or {}
    sid  = data.get('session_id', '')
    with _sessions_lock:
        sess = _sessions.pop(sid, None)
    if sess:
        sess.kill()
    return jsonify({'ok': True})
