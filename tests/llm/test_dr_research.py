import types, sys, pathlib, importlib


def load_components():
    repo = pathlib.Path(__file__).resolve().parents[2] / "omicverse"
    managed = [
        "omicverse",
        "omicverse.llm",
        "omicverse.llm.dr",
        "OvIntelligence",
        "OvIntelligence.query_manager",
        "omicverse.llm.model_factory",
    ]
    original = {name: sys.modules.get(name) for name in managed}
    omv = types.ModuleType("omicverse"); omv.__path__=[str(repo)]; sys.modules["omicverse"] = omv
    llm = types.ModuleType("omicverse.llm"); llm.__path__=[str(repo/"llm")]; sys.modules["omicverse.llm"] = llm
    dr = types.ModuleType("omicverse.llm.dr"); dr.__path__=[str(repo/"llm"/"dr")]; sys.modules["omicverse.llm.dr"] = dr
    ov = types.ModuleType("OvIntelligence"); sys.modules["OvIntelligence"] = ov
    oq = types.ModuleType("OvIntelligence.query_manager")
    class QM:
        @staticmethod
        def validate_query(q):
            return True, ""
    oq.QueryManager = QM
    sys.modules["OvIntelligence.query_manager"] = oq
    mf = types.ModuleType("omicverse.llm.model_factory")
    class DummyFactory:
        @staticmethod
        def create_model(*a, **k):
            return object()
    mf.ModelFactory = DummyFactory
    sys.modules["omicverse.llm.model_factory"] = mf
    try:
        rm_module = importlib.import_module("omicverse.llm.dr.research_manager")
        scope_module = importlib.import_module("omicverse.llm.dr.scope.brief")
        agent_module = importlib.import_module("omicverse.llm.dr.research.agent")
        return rm_module.ResearchManager, scope_module.ProjectBrief, agent_module.Finding
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


ResearchManager, ProjectBrief, Finding = load_components()


class DummyDoc:
    def __init__(self, text):
        self.text = text
        self.id = "doc"


class DummyStore:
    def search(self, query):
        return [DummyDoc(f"about {query}")]


def test_research_generates_findings():
    rm = ResearchManager(vector_store=DummyStore())
    brief = ProjectBrief(title="t", objectives=["topic1"], constraints=[])
    findings = rm.research(brief)
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.topic == "topic1"
    assert f.text == "about topic1"
    assert f.sources and f.sources[0].content == "about topic1"
