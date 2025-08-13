#!/usr/bin/env python3
"""
Script to update import statements for OmicVerse integration.
Run this after migration to fix import paths.
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path):
    """Update import statements in a Python file."""
    if not file_path.suffix == ".py":
        return
    
    content = file_path.read_text()
    original_content = content
    
    # Update relative imports
    content = re.sub(
        r"from src\.api\.",
        "from omicverse.external.datacollect.api.",
        content
    )
    
    content = re.sub(
        r"from src\.collectors\.",
        "from omicverse.external.datacollect.collectors.",
        content
    )
    
    content = re.sub(
        r"from src\.models\.",
        "from omicverse.external.datacollect.models.",
        content
    )
    
    content = re.sub(
        r"from src\.utils\.",
        "from omicverse.external.datacollect.utils.",
        content
    )
    
    content = re.sub(
        r"from config\.",
        "from omicverse.external.datacollect.config.",
        content
    )
    
    # Update imports from config
    content = re.sub(
        r"from config\.config import",
        "from omicverse.external.datacollect.config.config import",
        content
    )
    
    if content != original_content:
        file_path.write_text(content)
        print(f"Updated imports in: {file_path}")


def main():
    """Update all import statements in the migrated module."""
    module_dir = Path("omicverse/external/datacollect")
    
    if not module_dir.exists():
        print("Error: Module directory not found")
        return
    
    print("Updating import statements...")
    
    for py_file in module_dir.rglob("*.py"):
        update_imports_in_file(py_file)
    
    print("Import update complete!")


if __name__ == "__main__":
    main()
