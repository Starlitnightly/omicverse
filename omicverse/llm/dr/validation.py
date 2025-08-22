"""Utilities for validating queries used in the research pipeline."""

from OvIntelligence.query_manager import QueryManager


def validate_query(query: str) -> None:
    """Validate a query string.

    Parameters
    ----------
    query:
        The query string to validate.

    Raises
    ------
    ValueError
        If the query fails validation checks provided by
        :class:`OvIntelligence.query_manager.QueryManager`.
    """
    valid, message = QueryManager.validate_query(query)
    if not valid:
        raise ValueError(message)


__all__ = ["validate_query"]
