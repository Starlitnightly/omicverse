import sys
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "omicverse" / "utils" / "skill_registry.py"

spec = importlib.util.spec_from_file_location("omicverse.utils.skill_registry", MODULE_PATH)
skill_registry = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = skill_registry
spec.loader.exec_module(skill_registry)

SkillDefinition = skill_registry.SkillDefinition
build_multi_path_skill_registry = skill_registry.build_multi_path_skill_registry


def test_prompt_instructions_truncation():
    body = "\n".join(["## Step 1", "Do something", "## Step 2", "Do something else"]) * 5
    skill = SkillDefinition(
        name="Test",
        slug="test",
        description="desc",
        path=Path("."),
        body=body,
        metadata={},
    )
    snippet = skill.prompt_instructions(max_chars=40, provider="openai")
    assert isinstance(snippet, str)
    assert len(snippet) <= 40


def test_multi_path_registry_returns_instance(tmp_path):
    pkg_root = tmp_path / "pkg"
    cwd_root = tmp_path / "cwd"
    (pkg_root / ".claude" / "skills").mkdir(parents=True)
    (cwd_root / ".claude" / "skills").mkdir(parents=True)

    registry = build_multi_path_skill_registry(pkg_root, cwd_root)
    assert hasattr(registry, "skills")
    assert registry.skills == {}
