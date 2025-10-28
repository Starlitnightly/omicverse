---
Updates (2025-10-28)
- Fixed CI AttributeError in tests due to monkeypatch parent-attribute resolution. Hardened module stubbing so dotted targets (omicverse, omicverse.utils, omicverse.ov_skill_seeker) exist during patching.
- Verified locally: all Skill Seeker tests pass (test_ov_skill_seeker.py and test_smart_agent.py). No functional changes to ov.Agent.seeker.
- What changed in tests: force-register stubs in sys.modules, set parent attributes explicitly, load real omicverse.agent for _zip_dir, and register builder submodule stubs.
- Next steps (tracked in progress.json): security hardening + documentation, CI smoke tests, notebook example for ov.Agent.seeker.
