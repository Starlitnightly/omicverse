"""
AnnData Helpers - Fast preprocessing state analysis
"""
import logging


def canonical_embedding_keys(adata):
    """Return the actual obsm keys to expose as embeddings, deduplicating by
    display name and preferring the ``X_``-prefixed variant when both exist.

    Examples
    --------
    obsm = {'X_umap', 'UMAP', 'X_pca'}  →  ['X_pca', 'X_umap']
      (UMAP is dropped because X_umap already covers that display name)

    obsm = {'umap', 'UMAP', 'X_pca'}  →  ['X_pca', 'umap', 'UMAP']
      (no X_ version exists for either, so both are kept as-is)

    Returns
    -------
    list[str]
        Actual obsm key strings in the order they were first encountered.
    """
    # First pass: collect keys, building display_name → [keys] mapping
    display_to_keys: dict = {}
    for key in adata.obsm.keys():
        display = key[2:].lower() if key.startswith('X_') else key.lower()
        display_to_keys.setdefault(display, []).append(key)

    result = []
    for keys in display_to_keys.values():
        if len(keys) == 1:
            result.append(keys[0])
        else:
            # Multiple keys share the same display name.
            # Prefer the X_-prefixed one; if none, keep the first.
            preferred = next((k for k in keys if k.startswith('X_')), keys[0])
            result.append(preferred)
    return result


def resolve_embedding_key(adata, name: str) -> str:
    """Resolve *name* to the actual key present in ``adata.obsm``.

    Lookup order:
    1. Exact match  (name  is in obsm)
    2. Prefixed     (X_{name} is in obsm)
    3. Stripped     (name without leading X_ is in obsm)
    4. Case-insensitive exact match

    Raises ``KeyError`` if no match found.
    """
    if name in adata.obsm:
        return name
    prefixed = f'X_{name}'
    if prefixed in adata.obsm:
        return prefixed
    stripped = name[2:] if name.startswith('X_') else name
    if stripped in adata.obsm:
        return stripped
    # Case-insensitive fallback
    name_lower = name.lower()
    for k in adata.obsm.keys():
        if k.lower() == name_lower or k.lower() == f'x_{name_lower}' or k[2:].lower() == name_lower:
            return k
    raise KeyError(f"Embedding '{name}' not found in obsm")


def analyze_data_state(adata):
    """Fast heuristic analysis of adata preprocessing state.

    All operations are O(nnz) or cheaper — no densification, no row sums.
    Returns: x_max, x_min, is_int, is_log1p, is_normalized, is_scaled,
             estimated_target_sum
    """
    import numpy as _np, math as _math
    import scipy.sparse as _sp

    try:
        X = adata.X

        # ── max / min — efficient on sparse or dense ──────────────────────────
        x_max_val = float(X.max())
        x_min_val = float(X.min())

        # ── sample ≤300 stored nonzero values to check dtype ─────────────────
        if _sp.issparse(X):
            nz = X.data[:300] if len(X.data) >= 300 else X.data
        else:
            flat = _np.asarray(X).ravel()
            nz   = flat[flat != 0][:300]

        # is_int: all sampled nonzero values are whole numbers?
        is_int = bool(len(nz) > 0 and _np.all(_np.abs(nz - _np.round(nz)) < 1e-4))

        # is_log1p: uns key (canonical), then x_max < 30 heuristic
        is_log1p = bool('log1p' in adata.uns or (not is_int and 0 < x_max_val < 30))

        has_negative = bool(x_min_val < 0)
        is_scaled    = bool(has_negative and x_max_val <= 50)

        # is_normalized: rule-based — no row sum needed
        if is_scaled:
            is_normalized = False
        elif is_log1p:
            is_normalized = True
        elif is_int:
            is_normalized = False
        else:
            is_normalized = True   # float, non-log1p, non-scaled → likely normalized

        # estimated_target_sum via expm1(x_max) — single float op, zero overhead
        # log1p(target_sum) ≈ x_max (upper bound for the most expressed gene)
        estimated_target_sum = None
        _COMMON_TS = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        if is_normalized and is_log1p and x_max_val > 1:
            approx = _math.expm1(x_max_val)
            estimated_target_sum = min(_COMMON_TS, key=lambda v: abs(v - approx))

        return {
            'x_max': round(x_max_val, 4),
            'x_min': round(x_min_val, 4),
            'is_int': is_int,
            'is_log1p': is_log1p,
            'is_normalized': is_normalized,
            'is_scaled': is_scaled,
            'estimated_target_sum': estimated_target_sum,
        }
    except Exception as _e:
        logging.warning(f"analyze_data_state failed: {_e}")
        return {}
