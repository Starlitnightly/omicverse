"""
Logging configuration for OmicVerse Agent debugging.

This module provides utilities to enable debug logging for the OVAgent system.
Set the OVAGENT_DEBUG environment variable to enable detailed logging:

    export OVAGENT_DEBUG=1
    python your_script.py

Or in Python:
    import os
    os.environ['OVAGENT_DEBUG'] = '1'
    import omicverse as ov
    # ... rest of your code
"""

import logging
import os
import sys


def setup_agent_logging(level: str = None) -> None:
    """
    Configure logging for OmicVerse Agent components.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
               If None, checks OVAGENT_DEBUG environment variable.
               If OVAGENT_DEBUG is set to '1', 'true', or 'yes', enables DEBUG logging.
    """
    # Determine logging level
    if level is None:
        debug_mode = os.getenv('OVAGENT_DEBUG', '').lower() in ('1', 'true', 'yes')
        level = 'DEBUG' if debug_mode else 'INFO'

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger for omicverse.utils
    logger = logging.getLogger('omicverse.utils')
    logger.setLevel(numeric_level)

    # Create console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    # Also configure specific agent loggers
    for logger_name in ['omicverse.utils.smart_agent', 'omicverse.utils.agent_backend']:
        specific_logger = logging.getLogger(logger_name)
        specific_logger.setLevel(numeric_level)

    if numeric_level == logging.DEBUG:
        print(f"✓ OVAgent debug logging enabled (level: {level})", file=sys.stderr)


def enable_debug_logging() -> None:
    """Enable DEBUG level logging for all OVAgent components."""
    setup_agent_logging('DEBUG')


def disable_debug_logging() -> None:
    """Disable debug logging (set to INFO level)."""
    setup_agent_logging('INFO')


# Auto-enable debug logging if OVAGENT_DEBUG environment variable is set
if os.getenv('OVAGENT_DEBUG', '').lower() in ('1', 'true', 'yes'):
    setup_agent_logging('DEBUG')
