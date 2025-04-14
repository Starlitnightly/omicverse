from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import cellxgene_census
import rich
import scanpy as sc
from huggingface_hub import ModelCard, snapshot_download
from rich.markdown import Markdown
from scvi import settings
from scvi.utils import dependencies

from popv.annotation import AlgorithmsNT, annotate_data
from popv.hub._metadata import HubMetadata, HubModelCardHelper
from popv.preprocessing import Process_Query

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class HubModel:
    """Wrapper for :class:`~scvi.model.base.BaseModelClass` backed by HuggingFace Hub.

    Parameters
    ----------
    repo_name
        ID of the HuggingFace repo where this model is uploaded
    local_dir
        Local directory where the data and pre-trained model reside.
    metadata
        Dict or a path to a file on disk where this metadata can be read from.
    model_card
        The model card for this pre-trained model. Model card is a markdown file that describes the
        pre-trained model/data and is displayed on HuggingFace. This can be either an instance of
        :class:`~huggingface_hub.ModelCard` or an instance of :class:`~popv.hub.HubModelCardHelper`
        that wraps the model card or a path to a file on disk where the model card can be read
        from.
    """

    def __init__(
        self,
        local_dir: str,
        metadata: dict | str | None = None,
        repo_name: str | None = None,
        model_card: HubModelCardHelper | ModelCard | str | None = None,
        ontology_dir: str | None = None,
    ):
        self._local_dir = local_dir
        self._ontology_dir = ontology_dir
        self._repo_name = repo_name

        self._model_path = f"{self._local_dir}"
        self._adata_path = f"{self._local_dir}/ref_adata.h5ad"
        self._minified_adata_path = f"{self._local_dir}/minified_ref_adata.h5ad"

        # lazy load - these are not loaded until accessed
        self._model = None
        self._adata = None
        self._minfied_adata = None

        # get the metadata from the parameters or from the disk
        metadata_path = f"{self._local_dir}/metadata.json"
        if isinstance(metadata, HubMetadata):
            self._metadata = metadata
        elif isinstance(metadata, str) or os.path.isfile(metadata_path):
            path = metadata if isinstance(metadata, str) else metadata_path
            content = Path(path).read_text()
            content_dict = json.loads(content)
            self._metadata = HubMetadata(**content_dict)
        else:
            raise ValueError("No metadata found")

        # get the model card from the parameters or from the disk
        model_card_path = f"{self._local_dir}/README.md"
        if isinstance(model_card, HubModelCardHelper):
            self._model_card = model_card.model_card
        elif isinstance(model_card, ModelCard):
            self._model_card = model_card
        elif isinstance(model_card, str) or os.path.isfile(model_card_path):
            path = model_card if isinstance(model_card, str) else model_card_path
            content = Path(path).read_text()
            self._model_card = ModelCard(content)
        else:
            raise ValueError("No model card found")

    def save(self, overwrite: bool = False) -> None:
        """Save the model card and metadata to the model directory.

        Parameters
        ----------
        overwrite
            Whether to overwrite existing files.
        """
        card_path = os.path.join(self._local_dir, "README.md")
        if os.path.isfile(card_path) and not overwrite:
            raise FileExistsError(f"Model card already exists at {card_path}. To overwrite, pass `overwrite=True`.")
        self.model_card.save(card_path)

        metadata_path = os.path.join(self._local_dir, "_required_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def annotate_data(
        self,
        query_adata: AnnData,
        query_batch_key: str | None = None,
        save_path: str = "tmp",
        prediction_mode: str = "fast",
        methods: list | None = None,
        gene_symbols: str | None = None,
    ) -> AnnData:
        """Annotate the query data with the trained model.

        Parameters
        ----------
        query_adata
            The query data to annotate.
        query_batch_key
            The batch key in the query data.
        save_path
            Path to save the query models.
        prediction_mode
            The prediction mode to use. Either "fast" or "inference".
            "fast" will only predict on the query data,
            while "inference" will integrate query and reference data.
        gene_symbols
            Gene symbols given as query_adata.var_names.

        Returns
        -------
        AnnData
            The annotated data.
        """
        ref_adata = self.adata if prediction_mode == "retrain" else self.minified_adata
        setup_dict = self.metadata.setup_dict
        if gene_symbols is not None:
            print("SSSSSS")
            query_adata = self.map_genes(adata=query_adata, gene_symbols=gene_symbols)
        print("LLLLLL", self.local_dir, os.listdir(self.local_dir))

        concatenate_adata = Process_Query(
            query_adata,
            ref_adata,
            query_batch_key=query_batch_key,
            ref_labels_key=setup_dict["ref_labels_key"],
            ref_batch_key=setup_dict["ref_batch_key"],
            unknown_celltype_label=setup_dict["unknown_celltype_label"],
            save_path_trained_models=self._local_dir,
            cl_obo_folder=self.ontology_dir,
            prediction_mode=prediction_mode,
            n_samples_per_label=100,
            hvg=None,
        ).adata
        methods_ = self.metadata.methods
        if prediction_mode == "fast":
            methods_ = [method for method in methods_ if method in AlgorithmsNT.FAST_ALGORITHMS]
        if methods is not None:
            if not set(methods).issubset(methods_):
                ValueError(f"Method {set(methods) - set(methods_)} is not supported. Consider retraining models.")
            methods_ = methods
        methods_kwargs = self.metadata.method_kwargs
        annotate_data(
            concatenate_adata,
            save_path=f"{save_path}/popv_output",
            methods=methods,
            methods_kwargs=methods_kwargs,
        )

        return concatenate_adata

    @dependencies("huggingface_hub")
    def push_to_huggingface_hub(
        self,
        repo_name: str,
        repo_token: str | None = None,
        repo_create: bool = False,
        repo_create_kwargs: dict | None = None,
        collection_slug: str | None = None,
        **kwargs,
    ):
        """Push this model to HuggingFace.

        If the dataset is too large to upload to HuggingFace, this will raise an
        exception prompting the user to upload the data elsewhere. Otherwise, the
        data, model card, and metadata are all uploaded to the given model repo.

        Parameters
        ----------
        repo_name
            ID of the HuggingFace repo where this model needs to be uploaded
        repo_token
            HuggingFace API token with write permissions if None uses token in HfFolder.get_token()
        repo_create
            Whether to create the repo
        repo_create_kwargs
            Keyword arguments passed into :meth:`huggingface_hub.HfApi.create_repo` if
            ``repo_create=True``.
        collection_slug
            The internal name in HuggingFace for a dataset collection.
        **kwargs
            Additional keyword arguments passed into :meth:`huggingface_hub.HfApi.upload_file`.
        """
        from huggingface_hub import HfApi, HfFolder, add_collection_item, create_repo

        self._repo_name = repo_name

        if repo_token is None:
            repo_token = HfFolder.get_token()
        elif os.path.isfile(repo_token):
            repo_token = Path(repo_token).read_text()
        if repo_create:
            repo_create_kwargs = repo_create_kwargs or {}
            create_repo(repo_name, token=repo_token, **repo_create_kwargs)
        api = HfApi()
        # upload the model card
        self.model_card.push_to_hub(repo_name, token=repo_token)
        # upload the model
        api.upload_folder(
            folder_path=self._local_dir,
            repo_id=repo_name,
            token=repo_token,
            ignore_patterns="*h5ad",  # Ignore all h5ad files.
            **kwargs,
        )
        # upload the metadata
        api.upload_file(
            path_or_fileobj=json.dumps(asdict(self.metadata), indent=4).encode(),
            path_in_repo="metadata.json",
            repo_id=repo_name,
            token=repo_token,
            **kwargs,
        )
        if os.path.isfile(f"{self._local_dir}/minified_ref_adata.h5ad"):
            api.upload_file(
                path_or_fileobj=f"{self._local_dir}/minified_ref_adata.h5ad",
                path_in_repo="minified_ref_adata.h5ad",
                repo_id=repo_name,
                token=repo_token,
                **kwargs,
            )
        collection_slug = collection_slug

        if collection_slug is not None:
            add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_name,
                item_type="model",
                exists_ok=True,
            )

    @classmethod
    def pull_from_huggingface_hub(
        cls,
        repo_name: str,
        cache_dir: str | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        """Download the given model repo from HuggingFace.

        The model, its card, data, metadata are downloaded to a cached location on disk
        selected by HuggingFace and an instance of this class is created with that info
        and returned.

        Parameters
        ----------
        repo_name
            ID of the HuggingFace repo where this model needs to be uploaded
        cache_dir
            The directory where the downloaded model artifacts will be cached
        revision
            The revision to pull from the repo. This can be a branch name, a tag, or a full-length
            commit hash. If None, the default (latest) revision is pulled.
        kwargs
            Additional keyword arguments to pass to :func:`huggingface_hub.snapshot_download`.
        """
        if revision is None:
            warnings.warn(
                "No revision was passed, so the default (latest) revision will be used.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        snapshot_folder = snapshot_download(
            repo_id=repo_name,
            cache_dir=cache_dir,
            revision=revision,
            **kwargs,
        )
        ontology_snapshot = snapshot_download(
            repo_id="popV/ontology",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        model_card = ModelCard.load(repo_name)
        return cls(
            snapshot_folder,
            model_card=model_card,
            repo_name=repo_name,
            ontology_dir=ontology_snapshot,
        )

    def __repr__(self):
        def eval_obj(obj):
            return "No" if obj is None else "Yes"

        print(
            "HubModel with:\n"
            f"local_dir: {self._local_dir}\n"
            f"model loaded? {eval_obj(self._model)}\n"
            f"adata loaded? {eval_obj(self._adata)}\n"
            f"metadata:\n{self.metadata}\n"
            f"model_card:"
        )
        rich.print(Markdown(self.model_card.content.replace("\n", "\n\n")))
        return ""

    @property
    def local_dir(self) -> str:
        """The local directory where the data and pre-trained model reside."""
        return self._local_dir

    @property
    def ontology_dir(self) -> str:
        """The local directory where the models are downloaded."""
        return self._ontology_dir

    @property
    def repo_name(self) -> str:
        """The local directory where the data and pre-trained model reside."""
        return self._repo_name

    @property
    def metadata(self) -> dict:
        """The metadata for this model."""
        return self._metadata

    @property
    def model_card(self) -> ModelCard:
        """The model card for this model."""
        return self._model_card

    @property
    def adata(self) -> AnnData | None:
        """Returns the full training data for this model.

        If the data has not been loaded yet, this will call :func:`cellxgene_census.download_source_h5ad`.
        Otherwise, it will simply return the loaded data.
        """
        if self._adata is None:
            cellxgene_census.download_source_h5ad(
                dataset_id=self.metadata.cellxgene_url.rsplit("/", 2)[1].rsplit(".")[0],
                census_version="latest",
                to_path=self._adata_path,
            )
            self._adata = sc.read_h5ad(self._adata_path)
        return self._adata

    @property
    def minified_adata(self) -> AnnData | None:
        """Returns the minified data for this model.

        If the data has not been loaded yet, this will call :func:`scanpy.read_h5ad`.
        Otherwise, it will simply return the loaded data.
        """
        if self._minfied_adata is None:
            self._minfied_adata = sc.read_h5ad(self._minified_adata_path)
        return self._minfied_adata

    def map_genes(self, adata, gene_symbols) -> AnnData | None:
        """Map genes to CELLxGENE census gene IDs."""
        with cellxgene_census.open_soma() as census:
            var_df = cellxgene_census.get_var(
                census,
                organism="homo_sapiens",
            )
            feature_dict = dict(zip(var_df[gene_symbols], var_df["feature_id"], strict=True))
        adata.var["old_index"] = adata.var_names
        adata.var_names = adata.var_names.map(feature_dict)
        adata = adata[:, adata.var.index.notna()].copy()
        return adata
