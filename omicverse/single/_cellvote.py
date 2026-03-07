
import requests
from .._registry import register_function


@register_function(
    aliases=["细胞投票", "CellVote", "cellvote", "细胞类型投票", "集成注释"],
    category="single",
    description="Multi-method cell type annotation consensus using ensemble voting with LLM arbitration",
    examples=[
        "# Initialize CellVote",
        "cv = ov.single.CellVote(adata)",
        "# Get cluster markers first", 
        "markers = ov.single.get_celltype_marker(adata, clustertype='leiden')",
        "# Vote using multiple annotation methods",
        "result = cv.vote(clusters_key='leiden', cluster_markers=markers,",
        "                 celltype_keys=['scsa_annotation', 'gpt_celltype'])",
        "# Use SCSA annotation",
        "scsa_result = cv.scsa_anno()",
        "# Use GPT annotation", 
        "gpt_result = cv.gpt_anno()",
        "# Use GPTBioInsightor annotation",
        "gbi_result = cv.gbi_anno()",
        "# Use scMulan annotation",
        "scmulan_result = cv.scMulan_anno()",
        "# Use PopV annotation",
        "popv_result = cv.popv_anno(ref_adata, 'celltype', 'batch')"
    ],
    related=["single.get_celltype_marker", "single.gptcelltype", "single.pySCSA"]
)
class CellVote(object):
    """Ensemble cell-type annotation manager with multiple backends.

    Parameters
    ----------
    adata : AnnData
        Query single-cell AnnData to annotate.
    """

    def __init__(self, adata) -> None:
        self.adata = adata

    def popv_anno(
        self,
        ref_adata,
        ref_labels_key,
        ref_batch_key,
        query_batch_key=None,
        cl_obo_folder=None,
        save_path="tmp",
        prediction_mode="fast",
        methods=None,
        methods_kwargs=None,
    ):
        """Annotate cells using PopV.

        Parameters
        ----------
        ref_adata
            Reference :class:`anndata.AnnData` object.
        ref_labels_key
            Column in ``ref_adata.obs`` with cell-type labels.
        ref_batch_key
            Column in ``ref_adata.obs`` with batch labels.
        query_batch_key
            Batch key in query data. Defaults to ``None``.
        cl_obo_folder
            Path to ontology resources or ``None`` to disable ontology.
        save_path
            Directory to store intermediate models and predictions.
        prediction_mode
            Mode to use in PopV preprocessing. See :class:`popv.preprocessing.Process_Query`.
        methods
            Single algorithm name or list of algorithm names to run. ``None`` selects
            a default set of models based on ``prediction_mode``.
        methods_kwargs
            Dictionary of algorithm specific keyword arguments.
        """
        from ..popv.preprocessing import Process_Query
        from ..popv.annotation import annotate_data

        pq = Process_Query(
            self.adata,
            ref_adata,
            ref_labels_key=ref_labels_key,
            ref_batch_key=ref_batch_key,
            query_batch_key=query_batch_key,
            cl_obo_folder=cl_obo_folder,
            save_path_trained_models=save_path,
            prediction_mode=prediction_mode,
        ).adata

        annotate_data(
            pq,
            save_path=save_path,
            methods=methods,
            methods_kwargs=methods_kwargs,
        )
        self.adata = pq
        return pq

    def scsa_anno(self):
        """Annotate cells using the SCSA pipeline.

        This is a convenience wrapper around :class:`pySCSA` that runs the
        annotation and stores the result back into ``adata``.

        Returns
        -------
        pandas.DataFrame
            The table of annotation results produced by ``pySCSA.cell_anno``.
        """
        from ._anno import pySCSA

        scsa = pySCSA(self.adata)
        result = scsa.cell_anno()
        scsa.cell_auto_anno(self.adata)
        return result

    def gpt_anno(self):
        """Annotate cells using GPT based approach.

        The function extracts marker genes for each cluster and sends them to
        :func:`gptcelltype` for annotation. The resulting cell types are added
        to ``adata.obs['gpt_celltype']``.

        Returns
        -------
        dict
            Mapping of cluster id to predicted cell type.
        """
        from ._anno import get_celltype_marker
        from ._gptcelltype import gptcelltype

        markers = get_celltype_marker(self.adata)
        result = gptcelltype(markers)
        self.adata.obs["gpt_celltype"] = (
            self.adata.obs["leiden"].map(result).astype("category")
        )
        return result

    def gbi_anno(self):
        """Annotate clusters using GPTBioInsightor.

        The method sends cluster marker genes to
        :func:`gptbioinsightor.get_celltype` and stores the predicted cell types
        with highest score in ``adata.obs['gbi_celltype']``.

        Parameters are equivalent to those in
        :func:`gptbioinsightor.get_celltype`.

        Returns
        -------
        dict
            Score dictionary returned by GPTBioInsightor.
        """

        from gptbioinsightor import get_celltype, add_obs

        score_dic = get_celltype(self.adata)
        add_obs(self.adata, score_dic, add_key="gbi_celltype", cluster_key="leiden")
        return score_dic

    def scMulan_anno(self):
        """Annotate cells with the scMulan large language model.

        This is a thin wrapper around :mod:`omicverse.external.scMulan` that
        performs gene symbol unification, runs the pretrained model and stores
        predictions back into ``self.adata``.

        Returns
        -------
        :class:`anndata.AnnData`
            The annotated AnnData object.
        """
        from scipy.sparse import csc_matrix
        from ..external import scMulan
        import scanpy as sc

        # ensure CSC format for scMulan
        if not isinstance(self.adata.X, csc_matrix):
            self.adata.X = csc_matrix(self.adata.X)

        adata = scMulan.GeneSymbolUniform(
            input_adata=self.adata, output_dir="./", output_prefix="scmulan"
        )
        if adata.X.max() > 10:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        model = scMulan.model_inference("./ckpt/ckpt_scMulan.pt", adata)
        model.cuda_count()
        model.get_cell_types_and_embds_for_adata(parallel=True, n_process=1)
        self.adata = model.adata
        return self.adata

    def vote(
        self,
        clusters_key=None,
        cluster_markers=None,
        celltype_keys=[],
        model="gpt-3.5-turbo",
        base_url=None,
        species="human",
        organization="stomach",
        provider="openai",
        result_key="CellVote_celltype",
    ):
        r"""Generate consensus cell-type labels by LLM arbitration.

        Parameters
        ----------
        clusters_key : str or None, default=None
            Column in ``adata.obs`` containing cluster IDs.
        cluster_markers : dict or None, default=None
            Marker genes per cluster, typically from
            ``ov.single.get_celltype_marker``.
        celltype_keys : list, default=[]
            Candidate annotation columns in ``adata.obs`` used to build a
            candidate label set per cluster.
        model : str, default='gpt-3.5-turbo'
            Chat model used for final arbitration.
        base_url : str or None, default=None
            Optional custom API base URL.
        species : str, default='human'
            Species context string used in prompts.
        organization : str, default='stomach'
            Tissue/organ context string used in prompts.
        provider : str, default='openai'
            Provider preset for default endpoint selection.
        result_key : str, default='CellVote_celltype'
            Output column name written to ``adata.obs``.

        Returns
        -------
        dict
            Mapping from cluster ID to voted cell-type label.
        """

        cluster_celltypes = {}
        adata = self.adata
        adata.obs["best_clusters"] = adata.obs[clusters_key]
        adata.obs["best_clusters"] = adata.obs["best_clusters"].astype("category")
        for ct in adata.obs["best_clusters"].cat.categories:
            ct_li = []
            for celltype_key in celltype_keys:
                # selected the major cells as the present cells of cluster
                ct1 = (
                    adata.obs.loc[adata.obs["best_clusters"] == ct, celltype_key]
                    .value_counts()
                    .index[0]
                )
                ct_li.append(ct1)

            cluster_celltypes[ct] = ct_li

        result = get_cluster_celltype(
            cluster_celltypes,
            cluster_markers,
            species=species,
            organization=organization,
            model=model,
            base_url=base_url,
            provider=provider,
        )
        adata.obs[result_key] = (
            adata.obs["best_clusters"].map(result).astype("category")
        )
        adata.obs[result_key] = [i.capitalize() for i in adata.obs[result_key].tolist()]
        return result


@register_function(
    aliases=["获取集群细胞类型", "get_cluster_celltype", "cluster_celltype", "集群类型获取", "LLM细胞注释"],
    category="single",
    description="LLM-powered cluster cell type determination with retry mechanism and error handling",
    prerequisites={
        'functions': ['get_celltype_marker']
    },
    requires={},
    produces={},
    auto_fix='escalate',
    examples=[
        "# Basic cluster cell type determination",
        "cluster_celltypes = {'0': ['T cell', 'B cell'], '1': ['NK', 'T cell']}",
        "cluster_markers = {'0': ['CD3D', 'IL7R'], '1': ['NKG7', 'GNLY']}",
        "result = ov.single.get_cluster_celltype(cluster_celltypes, cluster_markers,",
        "                                        'human', 'PBMC', 'gpt-4', None, 'openai')",
        "# With custom API settings",
        "result = ov.single.get_cluster_celltype(cluster_celltypes, cluster_markers,",
        "                                        'mouse', 'Brain', 'qwen-plus',",
        "                                        'https://custom.api.com/v1', 'qwen')",
        "# With retry configuration",
        "result = ov.single.get_cluster_celltype(cluster_celltypes, cluster_markers,",
        "                                        'human', 'Liver', 'gpt-3.5-turbo',",
        "                                        None, 'openai', timeout=60, max_retries=5)"
    ],
    related=["single.CellVote", "single.gptcelltype", "single.get_celltype_marker"]
)
def get_cluster_celltype(
    cluster_celltypes,
    cluster_markers,
    species,
    organization,
    model,
    base_url,
    provider,
    api_key=None,
    timeout=30,
    max_retries=2,
    retry_backoff=1.5,
    verbose=True,
):
    """Resolve one final cell type for each cluster with LLM calls.

    Parameters
    ----------
    cluster_celltypes : dict
        Candidate labels per cluster, usually ``dict[cluster_id -> list[str]]``.
    cluster_markers : dict
        Marker genes per cluster, usually ``dict[cluster_id -> list[str]]``.
    species : str
        Species context used in prompt construction.
    organization : str
        Tissue/organ context used in prompt construction.
    model : str
        Chat completion model name.
    base_url : str or None
        API base URL. If ``None``, inferred from ``provider``.
    provider : str
        Provider preset used for base URL fallback.
    api_key : str or None, default=None
        Optional API key override. If ``None``, reads ``AGI_API_KEY``.
    timeout : int, default=30
        Request timeout in seconds for each API call.
    max_retries : int, default=2
        Maximum retry count for failed requests.
    retry_backoff : float, default=1.5
        Exponential backoff base for retry waiting time.
    verbose : bool, default=True
        Whether to print retry/fallback diagnostics.

    Returns
    -------
    dict
        Mapping from cluster ID to predicted (or fallback) cell-type label.
    """
    # from openai import OpenAI
    import os
    import time
    import numpy as np
    import pandas as pd
    import requests as requests

    if base_url is None:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "kimi":
            base_url = "https://api.moonshot.cn/v1"
        elif provider == "qwen":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if not api_key is None:
        QWEN_API_KEY = api_key
    else:
        QWEN_API_KEY = os.getenv("AGI_API_KEY")

    # 在这里配置您在本站的API_KEY
    api_key = QWEN_API_KEY

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    cluster_celltype = {}
    from tqdm import tqdm

    for cluster_id, celltypes in tqdm(cluster_celltypes.items()):
        markers = cluster_markers.get(cluster_id, [])
        question = (
            f"Given the species: {species} and organization: {organization}, "
            f"determine the most suitable cell type for cluster {cluster_id}. "
            f"The possible cell types are: {', '.join(celltypes)}. "
            f"The gene markers for this cluster are: {', '.join(markers)}. "
            f"Which cell type best represents this cluster? "
            f"Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types."
            f"Do not provide the plural form of celltype."
        )

        params = {
            "messages": [{"role": "user", "content": question}],
            # 如果需要切换模型，在这里修改
            "model": model,
        }
        url = f"{base_url}/chat/completions"
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=params, stream=False, timeout=timeout
                )
                if response.status_code >= 400:
                    try:
                        err_json = response.json()
                    except Exception:
                        err_json = {"error": response.text[:200]}
                    raise RuntimeError(
                        f"HTTP {response.status_code} from provider: {err_json}"
                    )

                try:
                    res = response.json()
                except Exception as e:
                    raise ValueError(f"Invalid JSON response: {str(e)}")

                try:
                    content = (
                        res.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                except Exception as e:
                    raise ValueError(f"Malformed completion format: {str(e)}")

                if not content:
                    raise ValueError("Empty completion content")

                first_line = content.split("\n", 1)[0].strip()
                label = first_line.lower() if first_line else None

                if not label:
                    raise ValueError("Empty label parsed from completion")

                cluster_celltype[cluster_id] = label
                last_error = None
                break

            except (requests.Timeout) as e:
                last_error = f"Timeout: {str(e)}"
            except requests.RequestException as e:
                last_error = f"Request error: {str(e)}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"

            if attempt < max_retries:
                sleep_s = retry_backoff ** attempt
                if verbose:
                    print(
                        f"[CellVote] cluster {cluster_id}: attempt {attempt+1} failed ({last_error}); retrying in {sleep_s:.1f}s"
                    )
                time.sleep(sleep_s)

        if last_error is not None and cluster_id not in cluster_celltype:
            if verbose:
                print(
                    f"[CellVote] cluster {cluster_id}: using fallback due to errors: {last_error}"
                )
            fallback = (celltypes[0].lower() if celltypes else "unknown")
            cluster_celltype[cluster_id] = fallback

    return cluster_celltype
