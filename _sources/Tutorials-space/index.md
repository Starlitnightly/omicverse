# Tutorials of Spatial Transcriptomics

This page mirrors the `Space` section in `mkdocs.yml` and provides a markdown entry point for the spatial tutorial notebooks.

## Preprocess

- [Crop and Rotation of spatial transcriptomic data](t_crop_rotate.ipynb)
- [Visium 10x HD Cellpose](t_cellpose.ipynb)
- [Analyze Nanostring data](t_nanostring_preprocess.ipynb)
- [Analyze Xenium data](t_xenium_preprocess.ipynb)
- [Analyze Visium HD data](t_visium_hd_preprocess.ipynb)

## Cluster

See [`cluster/index.md`](cluster/index.md) for the full overview, recommendations and references. One notebook per spatial embedder, all clustered with [`pymclustR`](https://pypi.org/project/pymclustR/) (no rpy2 required):

- [GraphST](cluster/t_cluster_graphst.ipynb) — Long et al., *Nat. Commun.* 2023
- [BINARY](cluster/t_cluster_binary.ipynb) — Lin et al., *Cell Genomics* 2024
- [STAGATE](cluster/t_cluster_stagate.ipynb) — Dong & Zhang, *Nat. Commun.* 2022
- [CAST](cluster/t_cluster_cast.ipynb) — Tang et al., *Nat. Methods* 2024
- [BANKSY](cluster/t_cluster_banksy.ipynb) — Singhal et al., *Nat. Genet.* 2024
- [All methods in one notebook (legacy)](t_cluster_space.ipynb)
- [Spatial integration and clustering](t_staligner.ipynb)

## Deconvolution

- [Identifying Pseudo-Spatial Map](t_spaceflow.ipynb)
- [Spatial deconvolution with reference scRNA-seq](t_decov.ipynb)
- [FlashDeconv (fast, GPU-free deconvolution)](t_flashdeconv.ipynb)
- [Spatial deconvolution without reference scRNA-seq](t_starfysh_new.ipynb)

## Downstream

- [Spatial transition tensor of single cells](t_stt.ipynb)
- [Spatial Communication](t_commot_flowsig.ipynb)
- [Spatial IsoDepth Calculation](t_gaston.ipynb)
- [Single cell spatial alignment tools](t_slat.ipynb)
