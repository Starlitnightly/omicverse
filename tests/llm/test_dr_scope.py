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
        return rm_module.ResearchManager, scope_module.ProjectBrief
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


ResearchManager, ProjectBrief = load_components()


class DummyStore:
    def search(self, query):
        return []


def test_scope_returns_brief():
    rm = ResearchManager(vector_store=DummyStore())
    brief = rm.scope("Mitochondria study")
    assert isinstance(brief, ProjectBrief)
    assert brief.title == "Mitochondria study"
    assert brief.objectives == []
