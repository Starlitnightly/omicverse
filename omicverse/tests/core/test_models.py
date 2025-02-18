"""Test various algorithms implemented in PopV."""

import sys

import anndata
import numpy as np
import popv
import pytest
import scanpy as sc
from popv.preprocessing import Process_Query
from popv.reproducibility import _accuracy


# Enable cuml in popv.setting depending on pytest flag
@pytest.fixture(autouse=True)
def configure_popv(pytestconfig):
    """Configure popv.settings based on pytest flag."""
    if pytestconfig.getoption("--cuml"):
        popv.settings.cuml = True
    popv.settings.accelerator = pytestconfig.getoption("--accelerator")


def _get_test_anndata(
    cl_obo_folder="resources/ontology/",
    prediction_mode="retrain",
    ref_adata=None,
    output_folder="tests/tmp_testing/popv_test_results/",
):
    if ref_adata is None:
        ref_adata_path = "resources/dataset/test/ts_lung_subset.h5ad"
        ref_adata = sc.read(ref_adata_path)
        ref_layer_key = None
    else:
        ref_adata = ref_adata
        ref_layer_key = "scvi_counts"

    query_adata_path = "resources/dataset/test/lca_subset.h5ad"
    query_adata = sc.read(query_adata_path)
    assert query_adata.n_vars == query_adata.X.shape[1]

    ref_labels_key = "cell_ontology_class"
    ref_batch_key = "donor_assay"
    min_celltype_size = np.min(ref_adata.obs.groupby("cell_ontology_class").size())
    n_samples_per_label = np.max((min_celltype_size, 20))

    query_batch_key = None
    unknown_celltype_label = "unknown"
    hvg = 4000 if prediction_mode == "retrain" else None

    adata = Process_Query(
        query_adata,
        ref_adata,
        query_batch_key=query_batch_key,
        ref_labels_key=ref_labels_key,
        ref_batch_key=ref_batch_key,
        ref_layer_key=ref_layer_key,
        unknown_celltype_label=unknown_celltype_label,
        save_path_trained_models=output_folder,
        cl_obo_folder=cl_obo_folder,
        prediction_mode=prediction_mode,
        n_samples_per_label=n_samples_per_label,
        hvg=hvg,
    )

    return adata


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="Test does not run on Python 3.10")
def test_annotation_hub(private: bool):
    """Test Annotation and Plotting pipeline without ontology."""
    output_folder = "tests/tmp_testing/popv_test_results_hub/"
    adata = _get_test_anndata(output_folder=output_folder).adata
    popv.annotation.annotate_data(
        adata,
        save_path="tests/tmp_testing/popv_test_results/",
        methods_kwargs={
            "KNN_BBKNN": {"method_kwargs": {"use_annoy": True}},
            "KNN_SCVI": {"train_kwargs": {"max_epochs": 3}},
            "SCANVI_POPV": {"train_kwargs": {"max_epochs": 3, "max_epochs_unsupervised": 1}},
        },
    )
    minified_adata = popv._utils.get_minified_adata(adata)
    minified_adata.write(f"{output_folder}/minified_ref_adata.h5ad")
    popv.hub.create_criticism_report(
        adata,
        save_folder=output_folder,
    )
    model_json = {
        "description": "Tabula Sapiens is a benchmark, first-draft human cell atlas of over 1.1M cells from 28 organs of 24 normal human subjects. This work is the product of the Tabula Sapiens Consortium. Taking the organs from the same individual controls for genetic background, age, environment, and epigenetic effects, and allows detailed analysis and comparison of cell types that are shared between tissues.",
        "tissues": ["test"],
        "cellxgene_url": "test",
        "references": "Tabula Sapiens reveals transcription factor expression, senescence effects, and sex-specific features in cell types from 28 human organs and tissues, The Tabula Sapiens Consortium; bioRxiv, doi: https://doi.org/10.1101/2024.12.03.626516",
        "license_info": "cc-by-4.0",
    }
    hmch = popv.hub.HubModelCardHelper.from_dir(output_folder, anndata_version=anndata.__version__, **model_json)
    hm = popv.hub.HubMetadata.from_anndata(
        adata,
        popv_version=popv.__version__,
        anndata_version=anndata.__version__,
        cellxgene_url=model_json["cellxgene_url"],
    )
    hmo = popv.hub.HubModel(output_folder, model_card=hmch, metadata=hm)
    if private:
        hmo.push_to_huggingface_hub(
            repo_name="popV/test",
            repo_token=None,
            repo_create=True,
            repo_create_kwargs={"exist_ok": True},
        )
    hmo = popv.hub.HubModel.pull_from_huggingface_hub(
        "popV/test", cache_dir="tests/tmp_testing/popv_test_results_hub_pulled/"
    )
    query_adata_path = "resources/dataset/test/lca_subset.h5ad"
    query_adata = sc.read(query_adata_path)
    hmo.annotate_data(query_adata, prediction_mode="fast")


def test_bbknn():
    """Test BBKNN algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.KNN_BBKNN(method_kwargs={"use_annoy": True})

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_knn_bbknn_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_bbknn_prediction"].isnull().any()


def test_onclass():
    """Test Onclass algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.ONCLASS(
        max_iter=2,
    )
    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_onclass_prediction" in adata.obs.columns
    assert not adata.obs["popv_onclass_prediction"].isnull().any()


def test_xgboost():
    """Test Random Forest algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.XGboost()
    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_xgboost_prediction" in adata.obs.columns
    assert not adata.obs["popv_xgboost_prediction"].isnull().any()


def test_scanorama():
    """Test Scanorama algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.KNN_SCANORAMA()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_knn_scanorama_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_scanorama_prediction"].isnull().any()


def test_harmony():
    """Test Harmony algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.KNN_HARMONY()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_knn_harmony_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_harmony_prediction"].isnull().any()


def test_scanvi():
    """Test SCANVI algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.SCANVI_POPV(train_kwargs={"max_epochs": 2, "max_epochs_unsupervised": 1})

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_scanvi_prediction" in adata.obs.columns
    assert not adata.obs["popv_scanvi_prediction"].isnull().any()


def test_scvi():
    """Test SCVI algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.KNN_SCVI(train_kwargs={"max_epochs": 3})

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_knn_on_scvi_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_on_scvi_prediction"].isnull().any()


def test_svm():
    """Test Support Vector Machine algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.Support_Vector()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_svm_prediction" in adata.obs.columns
    assert not adata.obs["popv_svm_prediction"].isnull().any()


def test_celltypist():
    """Test Celltypist algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.CELLTYPIST()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_umap(adata)

    assert "popv_celltypist_prediction" in adata.obs.columns
    assert not adata.obs["popv_celltypist_prediction"].isnull().any()


def test_annotation():
    """Test Annotation and Plotting pipeline."""
    adata = _get_test_anndata().adata
    popv.annotation.annotate_data(
        adata,
        save_path="tests/tmp_testing/popv_test_results/",
        methods_kwargs={
            "knn_on_bbknn": {"method_kwargs": {"use_annoy": True}},
            "knn_on_scvi": {"train_kwargs": {"max_epochs": 3}},
            "scanvi": {"train_kwargs": {"max_epochs": 3, "max_epochs_unsupervised": 1}},
        },
    )
    popv.visualization.agreement_score_bar_plot(adata)
    popv.visualization.prediction_score_bar_plot(adata)
    popv.visualization.make_agreement_plots(adata, prediction_keys=adata.uns["prediction_keys"], show=False)
    popv.visualization.celltype_ratio_bar_plot(adata)
    obo_fn = "resources/ontology/cl_popv.json"
    _accuracy._ontology_accuracy(
        adata[adata.obs["_dataset"] == "ref"],
        obo_file=obo_fn,
        gt_key="cell_ontology_class",
        pred_key="popv_prediction",
    )
    _accuracy._fine_ontology_sibling_accuracy(
        adata[adata.obs["_dataset"] == "ref"],
        obo_file=obo_fn,
        gt_key="cell_ontology_class",
        pred_key="popv_prediction",
    )

    assert "popv_majority_vote_prediction" in adata.obs.columns
    assert not adata.obs["popv_majority_vote_prediction"].isnull().any()

    adata = _get_test_anndata(ref_adata=adata[adata.obs["_dataset"] == "ref"], prediction_mode="inference").adata
    popv.annotation.annotate_data(
        adata,
        save_path="tests/tmp_testing/popv_test_results/",
        methods_kwargs={
            "knn_on_bbknn": {"method_kwargs": {"use_annoy": True}},
            "knn_on_scvi": {"train_kwargs": {"max_epochs": 3}},
            "scanvi": {"train_kwargs": {"max_epochs": 3}},
        },
    )

    adata = _get_test_anndata(prediction_mode="inference").adata
    popv.annotation.annotate_data(
        adata,
        save_path="tests/tmp_testing/popv_test_results/",
        methods_kwargs={
            "knn_on_bbknn": {"method_kwargs": {"use_annoy": True}},
            "knn_on_scvi": {"train_kwargs": {"max_epochs": 3}},
            "scanvi": {"train_kwargs": {"max_epochs": 3}},
        },
    )

    adata = _get_test_anndata(prediction_mode="fast").adata
    popv.annotation.annotate_data(
        adata,
        save_path="tests/tmp_testing/popv_test_results/",
        methods_kwargs={
            "knn_on_bbknn": {"method_kwargs": {"use_annoy": True}},
            "knn_on_scvi": {"train_kwargs": {"max_epochs": 3}},
            "scanvi": {"train_kwargs": {"max_epochs": 3}},
        },
    )


def test_annotation_no_ontology():
    """Test Annotation and Plotting pipeline without ontology."""
    adata = _get_test_anndata(cl_obo_folder=False).adata
    popv.annotation.annotate_data(
        adata, methods=["Support_Vector", "Random_Forest", "ONCLASS"], save_path="tests/tmp_testing/popv_test_results/"
    )
    popv.visualization.agreement_score_bar_plot(adata)
    popv.visualization.prediction_score_bar_plot(adata)
    popv.visualization.make_agreement_plots(adata, prediction_keys=adata.uns["prediction_keys"])
    popv.visualization.celltype_ratio_bar_plot(adata, save_folder="tests/tmp_testing/popv_test_results/")
    popv.visualization.celltype_ratio_bar_plot(adata, normalize=False)
    adata.obs["empty_columns"] = "a"
    input_data = adata.obs[["empty_columns", "popv_rf_prediction"]].values.tolist()
    popv.reproducibility._alluvial.plot(input_data)

    assert "popv_majority_vote_prediction" in adata.obs.columns
    assert not adata.obs["popv_majority_vote_prediction"].isnull().any()

    adata = _get_test_anndata(cl_obo_folder=False, prediction_mode="inference").adata
    popv.annotation.annotate_data(adata, methods=["Support_Vector", "Random_Forest"], save_path=None)

    assert "popv_majority_vote_prediction" in adata.obs.columns
    assert not adata.obs["popv_majority_vote_prediction"].isnull().any()

    adata = _get_test_anndata(cl_obo_folder=False, prediction_mode="fast").adata
    popv.annotation.annotate_data(adata, methods=["Support_Vector", "Random_Forest"], save_path=None)

    assert "popv_majority_vote_prediction" in adata.obs.columns
    assert not adata.obs["popv_majority_vote_prediction"].isnull().any()
