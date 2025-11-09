#!/usr/bin/env python3
"""
OmicVerse Skill Seeker CLI

Utilities to list, validate, and package OmicVerse Agent Skills bundled
under `.claude/skills`.

Examples:
    # Show discovered skills (title, slug)
    python -m omicverse.ov_skill_seeker --list

    # Validate all skills
    python -m omicverse.ov_skill_seeker --validate

    # Package a single skill by slug
    python -m omicverse.ov_skill_seeker --package bulk-combat-correction

    # Package all skills to output directory
    python -m omicverse.ov_skill_seeker --package-all

By default, the CLI assumes it is installed within the OmicVerse project
tree and resolves the project root relative to this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

from omicverse.utils.skill_registry import (
    SkillRegistry,
    SkillDefinition,
    build_skill_registry,
    build_multi_path_skill_registry,
)

logger = logging.getLogger("ov_skill_seeker")


def _resolve_repo_root() -> Path:
    """Resolve OmicVerse repository root (folder that contains .claude/skills)."""
    here = Path(__file__).resolve()
    # omicverse/omicverse/ov_skill_seeker/cli.py -> repo root is three parents up
    repo_root = here.parents[3]
    if not (repo_root / ".claude" / "skills").exists():
        # Fallback: current working directory
        cwd = Path.cwd()
        if (cwd / ".claude" / "skills").exists():
            return cwd
    return repo_root


def _load_registry(project_root: Path) -> SkillRegistry:
    """Load skills using dual-path discovery to match Agent behavior."""
    cwd = Path.cwd()
    registry = build_multi_path_skill_registry(project_root, cwd)
    return registry


def _print_skill_list(skills: List[SkillDefinition]) -> None:
    if not skills:
        print("No skills found.")
        return
    print(f"Discovered {len(skills)} skill(s):\n")
    for s in skills:
        print(f"- {s.name} (slug: {s.slug})")
        if s.description:
            print(f"  {s.description}")
        print(f"  path: {s.path}")


def _validate_skill(defn: SkillDefinition) -> Tuple[bool, List[str]]:
    """Validate required metadata/files for a single skill.

    Returns:
        (is_valid, messages)
    """
    msgs: List[str] = []
    ok = True

    # Required metadata already enforced by registry parsing; we double-check
    if not defn.name or not defn.slug or not defn.description:
        ok = False
        msgs.append("Missing required metadata (title/slug/description).")

    # Required files
    skill_md = defn.path / "SKILL.md"
    if not skill_md.exists():
        ok = False
        msgs.append("SKILL.md not found.")

    # Reference is strongly recommended (warning if missing)
    ref_md = defn.path / "reference.md"
    if not ref_md.exists():
        msgs.append("reference.md not found (warning).")

    return ok, msgs


def _zip_skill(defn: SkillDefinition, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{defn.slug}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(defn.path.rglob("*")):
            if path.is_file():
                # Skip macOS artifacts, Python caches, and existing zip files
                if path.name in {".DS_Store"} or "__pycache__" in path.parts or path.suffix == ".zip":
                    continue
                arcname = path.relative_to(defn.path)
                zf.write(path, arcname)
    return zip_path


def _package_skill_with_validation(defn: SkillDefinition, out_dir: Path) -> Path:
    """Package a skill after validating metadata and required files."""

    ok, msgs = _validate_skill(defn)
    if not ok:
        print(f"⚠️  {defn.slug} has validation issues:")
        for message in msgs:
            print(f"   - {message}")
    zip_path = _zip_skill(defn, out_dir)
    print(f"✅ Packaged {defn.slug} -> {zip_path}")
    return zip_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "OmicVerse Skill Seeker: list, validate, and package bundled skills. "
            "Discovers skills from both package installation (.claude/skills) and current working directory (.claude/skills). "
            "User skills override package skills when slugs collide."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing .claude/skills (defaults to auto-detected)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered skills",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate skills (frontmatter + required files)",
    )
    parser.add_argument(
        "--package",
        metavar="SLUG",
        help="Package a single skill by slug",
    )
    parser.add_argument(
        "--package-all",
        action="store_true",
        help="Package all skills",
    )
    parser.add_argument(
        "--create-from-link",
        metavar="URL",
        help="Create a new skill by scraping a single link (same-domain crawl)",
    )
    parser.add_argument(
        "--name",
        metavar="TITLE",
        default=None,
        help="Display title for a newly created skill (used with --create-from-link)",
    )
    parser.add_argument(
        "--description",
        metavar="TEXT",
        default=None,
        help="Description for a newly created skill (used with --create-from-link)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Max pages to crawl for link-based skill creation",
    )
    parser.add_argument(
        "--target",
        choices=["skills", "output"],
        default="skills",
        help="Where to write new skills (skills: .claude/skills, output: ./output)",
    )
    parser.add_argument(
        "--package-after",
        action="store_true",
        help="Package newly created skill after building",
    )
    parser.add_argument(
        "--build-config",
        type=Path,
        default=None,
        help="Path to unified build JSON config to create a new skill",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for packaged zips (defaults to <project_root>/output)",
    )

    args = parser.parse_args(argv)

    project_root = args.project_root or _resolve_repo_root()
    skills_root = project_root / ".claude" / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(project_root)
    skills = list(registry.skills.values())

    if args.create_from_link:
        # Determine base output dir for new skill
        if args.target == "skills":
            base_out = skills_root
        else:
            base_out = (args.out_dir or (project_root / "output"))
            base_out.mkdir(parents=True, exist_ok=True)

        from .link_builder import build_from_link
        built_dir = build_from_link(
            link=args.create_from_link,
            output_root=base_out,
            name=args.name,
            description=args.description,
            max_pages=args.max_pages,
        )
        print(f"\n✅ Created skill at: {built_dir}")

        if args.package_after:
            zip_path = _zip_skill(
                SkillDefinition(
                    name=args.name or built_dir.name,
                    slug=built_dir.name,
                    description=args.description or "",
                    path=built_dir,
                    body=(built_dir / "SKILL.md").read_text(encoding="utf-8"),
                    metadata={},
                ),
                args.out_dir or (project_root / "output"),
            )
            print(f"✅ Packaged -> {zip_path}")

    if args.list:
        _print_skill_list(skills)

    if args.validate:
        print("\nValidation results:\n")
        total = 0
        failures = 0
        for defn in skills:
            total += 1
            ok, msgs = _validate_skill(defn)
            status = "OK" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"- {defn.slug}: {status}")
            for m in msgs:
                print(f"  • {m}")
        print(f"\nSummary: {total - failures}/{total} valid")

    if args.build_config:
        from .unified_builder import build_from_config
        out_dir = args.out_dir or (project_root / "output")
        out_dir.mkdir(parents=True, exist_ok=True)
        built_dir = build_from_config(args.build_config, out_dir)
        print(f"\n✅ Built skill at: {built_dir}")

    if args.package or args.package_all:
        out_dir = args.out_dir or (project_root / "output")
        packaged: List[Path] = []

        if args.package:
            target = registry.skills.get(args.package)
            if not target:
                print(f"❌ Skill not found (by slug): {args.package}")
                return 1
            packaged.append(_package_skill_with_validation(target, out_dir))

        if args.package_all:
            for defn in skills:
                packaged.append(_package_skill_with_validation(defn, out_dir))

        if not packaged:
            print("No skills packaged.")

    # If no actionable flags provided, show help
    if not any([
        args.list,
        args.validate,
        args.package,
        args.package_all,
        args.create_from_link,
        args.build_config,
    ]):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
