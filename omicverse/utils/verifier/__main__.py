"""
Entry point for running the verifier as a module.

Usage:
    python -m omicverse.utils.verifier verify ./notebooks
    python -m omicverse.utils.verifier validate
    python -m omicverse.utils.verifier extract --directory ./notebooks
    python -m omicverse.utils.verifier test-selection --task "Analyze DEGs"
"""

from .cli import main

if __name__ == '__main__':
    main()
