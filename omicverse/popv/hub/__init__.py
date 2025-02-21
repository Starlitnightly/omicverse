from scvi.utils import error_on_missing_dependencies

error_on_missing_dependencies("huggingface_hub")


from ._metadata import HubModelCardHelper, HubMetadata  # noqa
from ._model import HubModel  # noqa
from ._get_metrics import create_criticism_report  # noqa

__all__ = [
    "HubMetadata",
    "HubModel",
    "HubModelCardHelper",
    "create_criticism_report",
]
