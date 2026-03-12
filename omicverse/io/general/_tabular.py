import pandas as pd

from ..._registry import register_function


@register_function(
    aliases=['读取CSV', 'read_csv', 'csv reader'],
    category="utils",
    description="Thin wrapper around pandas.read_csv used across OmicVerse tutorials for reproducible tabular input handling.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.utils.read_csv("metadata.csv", index_col=0)'],
    related=['utils.save', 'utils.load']
)
def read_csv(**kwargs):
    """
    Read a CSV file via ``pandas.read_csv``.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments accepted by ``pandas.read_csv`` (for example ``filepath_or_buffer``,
        ``sep``, ``index_col``, ``dtype``).

    Returns
    -------
    pandas.DataFrame
        Parsed table.
    """
    return pd.read_csv(**kwargs)
