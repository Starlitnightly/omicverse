from __future__ import annotations

from abc import abstractmethod

from popv import settings


class BaseAlgorithm:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        seen_result_key: str | None = None,
        result_key: str | None = None,
        embedding_key: str | None = None,
        umap_key: str | None = None,
    ) -> None:
        """
        Class to define base algorithm for celltype annotation.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
            Default is "_batch_annotation".
        labels_key
            Key in obs field of adata for cell-type information.
            Default is "_labels_annotation".
        seen_result_key
            Key in obs in which celltype predictions are stored. Defaults to result_key.
        result_key
            Key in obs in which celltype predictions are stored.
        embedding_key
            Key in obsm in which integrated embedding is stored.
        umap_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        if seen_result_key is None:
            self.seen_result_key = result_key
        else:
            self.seen_result_key = seen_result_key
        self.result_key = result_key
        self.embedding_key = embedding_key
        self.umap_key = umap_key
        self.enable_cuml = settings.cuml
        self.return_probabilities = settings.return_probabilities
        self.compute_umap_embedding = settings.compute_umap_embedding

    @abstractmethod
    def compute_integration(self, adata):
        """Compute integration of adata inplace."""

    @abstractmethod
    def predict(self, adata):
        """Predicts cell type of adata inplace."""

    @abstractmethod
    def compute_umap(self, adata):
        """Compute UMAP embedding of adata inplace."""
