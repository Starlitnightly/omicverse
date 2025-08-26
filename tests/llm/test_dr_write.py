import types, sys, pathlib, importlib


def load_components():
    repo = pathlib.Path(__file__).resolve().parents[2] / "omicverse"
    omv = types.ModuleType("omicverse"); omv.__path__=[str(repo)]; sys.modules.setdefault("omicverse", omv)
    llm = types.ModuleType("omicverse.llm"); llm.__path__=[str(repo/"llm")]; sys.modules.setdefault("omicverse.llm", llm)
    dr = types.ModuleType("omicverse.llm.dr"); dr.__path__=[str(repo/"llm"/"dr")]; sys.modules.setdefault("omicverse.llm.dr", dr)
    ov = types.ModuleType("OvIntelligence"); sys.modules.setdefault("OvIntelligence", ov)
    oq = types.ModuleType("OvIntelligence.query_manager")
    class QM:
        @staticmethod
        def validate_query(q):
            return True, ""
    oq.QueryManager = QM
    sys.modules.setdefault("OvIntelligence.query_manager", oq)
    mf = types.ModuleType("omicverse.llm.model_factory")
    class DummyFactory:
        @staticmethod
        def create_model(*a, **k):
            return object()
    mf.ModelFactory = DummyFactory
    sys.modules["omicverse.llm.model_factory"] = mf
    rm_module = importlib.import_module("omicverse.llm.dr.research_manager")
    scope_module = importlib.import_module("omicverse.llm.dr.scope.brief")
    agent_module = importlib.import_module("omicverse.llm.dr.research.agent")
    return rm_module.ResearchManager, scope_module.ProjectBrief, agent_module.Finding, agent_module.SourceCitation


ResearchManager, ProjectBrief, Finding, SourceCitation = load_components()


class DummyDoc:
    def __init__(self, text, id="1"):
        self.text = text
        self.id = id


class DummyStore:
    def search(self, query):
        return [DummyDoc(f"text about {query}")]


def make_rm():
    return ResearchManager(vector_store=DummyStore())


def test_write_composes_report():
    rm = make_rm()
    brief = ProjectBrief(title="Test Title", objectives=["topic"], constraints=[])
    findings = [Finding(topic="topic", text="summary", sources=[SourceCitation(source_id="1", content="ref")])]
    report = rm.write(brief, findings)
    assert "topic" in report
    assert "ref" in report


def test_run_pipeline(monkeypatch):
    rm = make_rm()
    monkeypatch.setattr(rm.scope_manager, "generate_brief", lambda: ProjectBrief(title="Test Title", objectives=["topic"], constraints=[]))
    report = rm.run("Test Query")
    assert "topic" in report
    assert "text about topic" in report
    assert "[1]" in report
