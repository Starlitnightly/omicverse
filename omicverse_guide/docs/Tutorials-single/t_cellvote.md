# Consensus annotation with CellVote

`CellVote` combines results from multiple cell type annotation approaches using a simple majority vote. By aggregating predictions, inconsistent labels can be resolved and the most plausible identity can be assigned to each cluster.

## Prerequisites

1. Clustering results should be available in `adata.obs` (e.g. the `leiden` field).
2. At least two independent cell type annotation results need to be stored in `adata.obs`. Typical methods include `scsa_anno`, `scMulan_anno` or GPT-based annotations such as `gpt_celltype`.
3. A dictionary of marker genes for each cluster is required. You can generate this with `ov.single.get_celltype_marker`.

## Basic usage

```python
import ov

# adata contains clustering results in "leiden"
cv = ov.single.CellVote(adata)
markers = ov.single.get_celltype_marker(adata)

cv.vote(
    clusters_key="leiden",
    cluster_markers=markers,
    celltype_keys=["scsa_annotation", "scMulan_anno"],
)

print(adata.obs["CellVote_celltype"].value_counts())
```

The final consensus label is stored in `adata.obs['CellVote_celltype']`.

## Advanced options

The `vote` method exposes a few additional arguments:

- `model`, `base_url` and `provider` allow you to specify a large language model when using GPT-based annotation as one of the voting sources.
- `result_key` changes the output column name.

```python
cv.vote(
    clusters_key="leiden",
    cluster_markers=markers,
    celltype_keys=["scsa_annotation", "gpt_celltype"],
    model="gpt-3.5-turbo",  # choose any model supported by your provider
    provider="openai",
    result_key="vote_label",
)
```

## Tips

- Ensure that the marker dictionary contains biologically meaningful genes to help resolve disagreements between annotation methods.
- You can inspect `adata.obs[['scsa_annotation','scMulan_anno','CellVote_celltype']]` to compare individual predictions with the final vote.
- Any number of annotation columns can be provided in `celltype_keys`.

