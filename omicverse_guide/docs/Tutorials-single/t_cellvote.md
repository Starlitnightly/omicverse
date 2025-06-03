# Consensus annotation with CellVote

`CellVote` combines results from multiple cell type annotation methods using a simple majority voting strategy. This helps resolve inconsistent predictions and choose the most plausible cell identity for each cluster.

## Basic usage

1. Perform cell type annotation with different tools (e.g. `scsa_anno`, `scMulan_anno` or custom predictions) and store the results in `adata.obs`.
2. Provide known marker genes for each cluster. `omicverse.single.get_celltype_marker` can automatically extract markers.
3. Run `CellVote.vote` to obtain a final consensus cell type label.

```python
import ov

# adata contains clustering results in `leiden`
cv = ov.single.CellVote(adata)
markers = ov.single.get_celltype_marker(adata)

cv.vote(
    clusters_key="leiden",
    cluster_markers=markers,
    celltype_keys=["scsa_annotation", "scMulan_anno"],
)

print(adata.obs["CellVote_celltype"].value_counts())
```

The consensus labels are stored in `adata.obs['CellVote_celltype']`.
