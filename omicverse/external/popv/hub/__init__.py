import importlib

__all__ = [
    "HubMetadata",
    "HubModel",
    "HubModelCardHelper",
    "create_criticism_report",
]

_LAZY_ATTRS = {
    "HubMetadata": ("._metadata", "HubMetadata"),
    "HubModelCardHelper": ("._metadata", "HubModelCardHelper"),
    "HubModel": ("._model", "HubModel"),
    "create_criticism_report": ("._get_metrics", "create_criticism_report"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        from scvi.utils import error_on_missing_dependencies

        error_on_missing_dependencies("huggingface_hub")
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
