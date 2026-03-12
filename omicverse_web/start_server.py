#!/usr/bin/env python3
"""
OmicVerse Single Cell Analysis Platform Launcher.

Starts the web server with dependency checks and safer host resolution.
"""

import sys
import os
import subprocess
import importlib
import argparse
import socket
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('scanpy', 'Scanpy'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('werkzeug', 'Werkzeug')
    ]
    
    optional_packages = [
        ('omicverse', 'OmicVerse')
    ]
    
    missing_required = []
    missing_optional = []
    
    print("Checking dependencies...")
    
    # Check required packages
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"[ok] {name} installed")
        except ImportError:
            missing_required.append((package, name))
            print(f"[missing] {name} not installed")
    
    # Check optional packages
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"[ok] {name} installed (optional)")
        except ImportError:
            missing_optional.append((package, name))
            print(f"[warning] {name} not installed (optional)")

    if missing_required:
        print("\nMissing required dependencies:")
        for package, name in missing_required:
            print(f"   - {name} ({package})")
        print("\nInstall them with:")
        packages = " ".join([pkg for pkg, _ in missing_required])
        print(f"   pip install {packages}")
        return False

    if missing_optional:
        print("\nRecommended optional dependencies:")
        for package, name in missing_optional:
            print(f"   - {name} ({package})")
        packages = " ".join([pkg for pkg, _ in missing_optional])
        print(f"   pip install {packages}")
    
    return True

def check_files():
    """Check if required files exist."""
    current_dir = Path(__file__).parent
    required_files = [
        'app.py',
        'single_cell_analysis_standalone.html',
    ]
    
    print("\nChecking required files...")
    
    missing_files = []
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"[ok] {file_path}")
        else:
            missing_files.append(file_path)
            print(f"[missing] {file_path}")

    if missing_files:
        print("\nMissing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    return True

def get_available_port(start_port=5050):
    """Find an available port starting from ``start_port``."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return None


def _default_bind_host(remote_mode: bool) -> str:
    return "127.0.0.1" if remote_mode else "0.0.0.0"


def _can_bind_host(host: str) -> bool:
    candidate = (host or "").strip()
    if not candidate:
        return False
    try:
        socket.getaddrinfo(candidate, 0, type=socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False


def _resolve_bind_host(cli_host: str | None, remote_mode: bool) -> tuple[str, str | None]:
    explicit = (cli_host or "").strip()
    if explicit:
        if _can_bind_host(explicit):
            return explicit, None
        fallback = _default_bind_host(remote_mode)
        return fallback, f"Requested host {explicit!r} could not be resolved; falling back to {fallback}"

    env_host = (os.environ.get("OV_WEB_HOST") or "").strip()
    if env_host:
        if _can_bind_host(env_host):
            return env_host, None
        fallback = _default_bind_host(remote_mode)
        return fallback, f"Environment variable OV_WEB_HOST={env_host!r} could not be resolved; falling back to {fallback}"

    legacy_host = (os.environ.get("HOST") or "").strip()
    if legacy_host and _can_bind_host(legacy_host):
        return legacy_host, None
    return _default_bind_host(remote_mode), None


def _parse_args(argv=None):
    """Parse launcher CLI arguments."""
    remote_mode = os.environ.get("OV_WEB_REMOTE_MODE", "0") == "1"

    parser = argparse.ArgumentParser(
        description="Launch OmicVerse web server."
    )
    parser.add_argument(
        "--host",
        # Remote mode defaults to loopback for safety (require tunnel/proxy)
        default=None,
        help="Host to bind (default: 127.0.0.1 in remote mode, 0.0.0.0 otherwise).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "0") or 0),
        help="Port to bind. If omitted, auto-select from 5050.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable Flask debug mode.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=remote_mode,
        help="Enable remote mode (bind loopback, require tunnel/proxy).",
    )
    parser.set_defaults(debug=False)
    return parser.parse_args(argv)

def main(argv=None):
    """Run the launcher."""
    args = _parse_args(argv)
    resolved_host, host_note = _resolve_bind_host(args.host, args.remote)
    args.host = resolved_host

    print("OmicVerse Single Cell Analysis Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return 1

    print(f"Python {sys.version.split()[0]}")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check files
    if not check_files():
        return 1
    
    # Resolve port (explicit first, then auto-discovery)
    port = args.port if args.port > 0 else get_available_port()
    if port is None:
        print("Could not find an available port")
        return 1

    if host_note:
        print(f"Warning: {host_note}")

    print(f"\nServer will start on port {port}")

    # Set environment variables
    os.environ['PORT'] = str(port)
    os.environ['FLASK_ENV'] = 'development'

    # Propagate remote mode so app.py can read it
    if args.remote:
        os.environ['OV_WEB_REMOTE_MODE'] = '1'
        # Force loopback bind in remote mode unless explicitly overridden
        if args.host == '0.0.0.0':
            args.host = '127.0.0.1'

    # Start the server
    print("\nStarting server...")
    print("-" * 50)
    if args.remote:
        print(f"Remote mode: bound to {args.host}:{port} (loopback only)")
        print("   Access via SSH tunnel:")
        print(f"     ssh -L {port}:127.0.0.1:{port} user@server")
        print(f"   Then open: http://localhost:{port}")
    else:
        print(f"Local access:  http://localhost:{port}")
        print(f"Network bind:  http://{args.host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        sys.path.insert(0, str(Path(__file__).parent))
        from app import app
        app.run(debug=args.debug, host=args.host, port=port, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        return 0
    except Exception as e:
        print(f"\nStartup failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
