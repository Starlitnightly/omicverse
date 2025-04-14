from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from anndata import AnnData
from huggingface_hub import ModelCard, ModelCardData

from ._template import (
    template,
)


@dataclass
class HubMetadata:
    """Encapsulates the required metadata for `popV` hub models.

    Parameters
    ----------
    popv_version
        The version of `popV` that the model was trained with.
    anndata_version
        The version of anndata used during model training.
    setup_dict
        The setup dictionary used to preprocess the data.
    prediction_keys
        The keys used to store the predictions in the AnnData object.
    methods_
        The methods used to predict celltypes
    method_kwargs
        The keyword arguments used in the methods.
    cellxgene_url
        Link to the data in the CELLxGENE portal (viewer).
    """

    popv_version: str
    anndata_version: str
    setup_dict: dict
    prediction_keys: list[str]
    method_kwargs: dict
    methods: list[str]
    method_kwargs: dict
    cellxgene_url: str | None = None

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData,
        anndata_version: str,
        popv_version: str,
        **kwargs,
    ):
        """Create a `HubMetadata` object from an AnnData file.

        Parameters
        ----------
        adata
            The AnnData object used to train the model.
        anndata_version
            The version of anndata used during model training.
        popv_version
            The version of `popV` that the model was trained with.
        kwargs
            Additional keyword arguments to pass to the HubMetadata initializer.
        """
        setup_dict = adata.uns["_setup_dict"]
        prediction_keys = adata.uns["prediction_keys"]
        methods = adata.uns["methods"]
        method_kwargs = adata.uns["method_kwargs"]

        return cls(
            popv_version=popv_version,
            anndata_version=anndata_version,
            setup_dict=setup_dict,
            prediction_keys=prediction_keys,
            methods=methods,
            method_kwargs=method_kwargs,
            **kwargs,
        )

    def save(self, save_path: str, overwrite: bool = False) -> None:
        """Save the metadata to a JSON file.

        Parameters
        ----------
        save_path
            The path to which to save the metadata as a JSON file.
        overwrite
            Whether to overwrite the file if it already exists.
        """
        if os.path.isfile(save_path) and not overwrite:
            raise FileExistsError(f"File already exists at {save_path}. To overwrite, pass `overwrite=True`.")
        with open(save_path, "w") as f:
            json.dump(asdict(self), f, indent=4)


@dataclass
class HubModelCardHelper:
    """A helper for creating a `ModelCard` for `popV` hub models.

    Parameters
    ----------
    license_info
        The license information for the model.
    scvi_version
        The version of `scvi-tools` that the model was trained with.
    anndata_version
        The version of anndata used during model training.
    tissues
        The tissues of the training data.
    cellxgene_url
        Link to the data in the CELLxGENE portal.
    description
        A description of the model.
    references_
        A list of references for the model.
    metrics_report
        A dictionary containing the metrics report for the model.

    Attributes
    ----------
    model_card : ModelCard
        Stores the model card.

    Notes
    -----
    It is not required to use this class to create a `ModelCard`. But this helps you do so in a way
    that is consistent with other `popV` hub models. You can think of this as a
    template. The resulting
    huggingface :class:`~huggingface_hub.ModelCard` can be accessed via the
    :attr:`~popv.hub.HubModelCardHelper.model_card` property.
    """

    license_info: str
    anndata_version: str
    tissues: list[str] = field(default_factory=list)
    cellxgene_url: str | None = None
    description: str = "To be added..."
    references: str = "To be added..."
    metrics_report: str | None = None
    training_code_url: str = "Not provided by uploader."

    def __post_init__(self):
        self.model_card = self._to_model_card()

    @classmethod
    def from_dir(
        cls,
        local_dir: str,
        license_info: str,
        anndata_version: str,
        metrics_report: str | None = None,
        **kwargs,
    ):
        """Create a `HubModelCardHelper` object from a local directory.

        Parameters
        ----------
        local_dir
            The local directory containing the model files.
        license_info
            The license information for the model.
        anndata_version
            The version of anndata used during model training.
        metrics_report
            Path to the json with stored metrics report.
        data_is_minified
            Whether the training data uploaded with the model has been minified.
        kwargs
            Additional keyword arguments to pass to the HubModelCardHelper initializer.
        """
        if metrics_report is None:
            if os.path.isfile(f"{local_dir}/accuracies.json"):
                with open(f"{local_dir}/accuracies.json") as f:
                    metrics_report = json.load(f)
            else:
                metrics_report = None
        else:
            with open(metrics_report) as f:
                metrics_report = json.load(f)

        return cls(
            license_info,
            anndata_version,
            metrics_report=metrics_report,
            **kwargs,
        )

    def _to_model_card(self) -> ModelCard:
        # define tags
        tags = [
            "biology",
            "genomics",
            "single-cell",
            f"anndata_version:{self.anndata_version}",
            "popV",
        ]
        for t in self.tissues:
            tags.append(f"tissue: {t}")

        # define the card data, which is the header
        card_data = ModelCardData(
            license=self.license_info,
            library_name="popV",
            tags=tags,
        )

        if self.metrics_report is not None:
            validation_accuracies = self.metrics_report.get("query_accuracy", "Not provided by uploader.")
            train_accuracies = self.metrics_report.get("ref_accuracy", "Not provided by uploader.")
        else:
            validation_accuracies = "Not provided by uploader."
            train_accuracies = "Not provided by uploader."

        # create the content from the template
        content = template.format(
            card_data=card_data.to_yaml(),
            description=self.description,
            cellxgene_url=self.cellxgene_url,
            references=self.references,
            validation_accuracies=validation_accuracies,
            train_accuracies=train_accuracies,
            training_code_url=self.training_code_url,
        )

        # finally create and return the actual card
        return ModelCard(content)
