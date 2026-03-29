import importlib


def __getattr__(name):
    cnmf_module = importlib.import_module("..external.cnmf.cnmf", __package__)
    if hasattr(cnmf_module, name):
        value = getattr(cnmf_module, name)
        globals()[name] = value
        return value
    if name == "Hotspot":
        from ..external.hotspot import Hotspot

        globals()[name] = Hotspot
        return Hotspot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    cnmf_module = importlib.import_module("..external.cnmf.cnmf", __package__)
    return sorted(set(globals()) | set(dir(cnmf_module)) | {"Hotspot"})
