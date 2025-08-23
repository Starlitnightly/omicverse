import types, sys, pathlib, importlib


def load_components():
    repo = pathlib.Path(__file__).resolve().parents[2] / "omicverse"
    omv = types.ModuleType("omicverse"); omv.__path__=[str(repo)]; sys.modules.setdefault("omicverse", omv)
    llm = types.ModuleType("omicverse.llm"); llm.__path__=[str(repo/"llm")]; sys.modules.setdefault("omicverse.llm", llm)
    dr = types.ModuleType("omicverse.llm.dr"); dr.__path__=[str(repo/"llm"/"dr")]; sys.modules.setdefault("omicverse.llm.dr", dr)
    # stub external dependency
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
    return rm_module.ResearchManager, scope_module.ProjectBrief


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
