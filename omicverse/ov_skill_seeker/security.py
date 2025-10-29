from __future__ import annotations

"""
Security utilities for OmicVerse Skill Seeker.

Provides URL validation, hostname checks, and filesystem safety guards.
"""

import ipaddress
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


# Allowed URL schemes
ALLOWED_SCHEMES = {"http", "https"}

# Blocked hostnames (localhost, private IPs, etc.)
BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
}

# Blocked file extensions for downloads
BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".sh",
    ".scr", ".vbs", ".ps1", ".app", ".deb", ".rpm"
}


def is_private_ip(hostname: str) -> bool:
    """Check if a hostname resolves to a private IP address."""
    try:
        # Try to parse as IP address
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        # Not an IP address, hostname validation will be done separately
        return False


def validate_url(url: str, *, allow_private: bool = False) -> None:
    """
    Validate that a URL is safe to access.

    Parameters
    ----------
    url : str
        URL to validate
    allow_private : bool, optional
        If True, allow private/local IP addresses (default: False)

    Raises
    ------
    SecurityError
        If the URL fails security checks
    """
    if not url or not isinstance(url, str):
        raise SecurityError("URL must be a non-empty string")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SecurityError(f"Invalid URL format: {e}")

    # Check scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise SecurityError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Only {', '.join(ALLOWED_SCHEMES)} are permitted."
        )

    # Check hostname
    hostname = parsed.hostname
    if not hostname:
        raise SecurityError("URL must have a hostname")

    # Check for blocked hostnames
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise SecurityError(f"Access to '{hostname}' is not allowed")

    # Check for private IPs (unless explicitly allowed)
    if not allow_private and is_private_ip(hostname):
        raise SecurityError(
            f"Access to private IP address '{hostname}' is not allowed"
        )

    # Check for suspicious patterns
    if ".." in parsed.path:
        raise SecurityError("URL path contains suspicious '..' pattern")


def sanitize_filename(name: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.

    Parameters
    ----------
    name : str
        Original filename
    max_length : int, optional
        Maximum filename length (default: 255)

    Returns
    -------
    str
        Sanitized filename

    Raises
    ------
    SecurityError
        If the filename cannot be sanitized safely
    """
    if not name or not isinstance(name, str):
        raise SecurityError("Filename must be a non-empty string")

    # Remove any path separators
    name = name.replace("/", "-").replace("\\", "-")

    # Remove null bytes
    name = name.replace("\x00", "")

    # Remove leading/trailing dots and spaces
    name = name.strip(". ")

    # Replace multiple consecutive spaces/dashes with single ones
    name = re.sub(r"[ -]+", "-", name)

    # Check for blocked extensions
    name_lower = name.lower()
    for ext in BLOCKED_EXTENSIONS:
        if name_lower.endswith(ext):
            raise SecurityError(f"File extension '{ext}' is not allowed")

    # Truncate if too long
    if len(name) > max_length:
        # Try to preserve extension
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            base, ext = parts
            max_base = max_length - len(ext) - 1
            name = base[:max_base] + "." + ext
        else:
            name = name[:max_length]

    # Final check
    if not name or name in {".", "..", "~"}:
        raise SecurityError(f"Filename '{name}' is not safe")

    return name


def validate_output_path(
    path: Path, *, base_dir: Optional[Path] = None, create: bool = True
) -> Path:
    """
    Validate and resolve an output path to prevent directory traversal.

    Parameters
    ----------
    path : Path
        Path to validate
    base_dir : Path, optional
        Base directory to constrain paths to. If None, uses current directory.
    create : bool, optional
        If True, create the directory if it doesn't exist (default: True)

    Returns
    -------
    Path
        Resolved absolute path

    Raises
    ------
    SecurityError
        If the path attempts to escape the base directory
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Resolve to absolute path
    try:
        base_dir = base_dir.resolve()
        resolved = path.resolve()
    except Exception as e:
        raise SecurityError(f"Failed to resolve path: {e}")

    # Check if resolved path is within base_dir
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise SecurityError(
            f"Path '{path}' attempts to escape base directory '{base_dir}'"
        )

    # Create directory if requested
    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise SecurityError(f"Failed to create directory: {e}")

    return resolved


def sanitize_html_content(html: str) -> str:
    """
    Basic sanitization of HTML content before processing.

    Parameters
    ----------
    html : str
        HTML content to sanitize

    Returns
    -------
    str
        Sanitized HTML (basic cleaning only)

    Notes
    -----
    This is a minimal sanitizer. The main protection comes from converting
    HTML to markdown, which strips most dangerous content. This function
    just removes obvious script tags and similar.
    """
    if not html:
        return ""

    # Remove script tags and their content
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags and their content
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove event handlers (onclick, onerror, etc.)
    html = re.sub(r'\son\w+\s*=\s*["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)

    # Remove javascript: URLs
    html = re.sub(r'javascript:', '', html, flags=re.IGNORECASE)

    return html
