

class CellVote(object):

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

        This is a thin wrapper around :mod:`omicverse.externel.scMulan` that
        performs gene symbol unification, runs the pretrained model and stores
        predictions back into ``self.adata``.

        Returns
        -------
        :class:`anndata.AnnData`
            The annotated AnnData object.
        """
        from scipy.sparse import csc_matrix
        from ..externel import scMulan
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
        """
        Vote the Best celltype from scRNA-seq

        Arguments:
            clusters_key: str, the clusters key for annotation, such as leiden, louvain
            cluster_markers: dict, the markers of cluster, we can use `ov.single.get_celltype_marker` to obtain.
            celltype_keys: list, the celltype annotation columns stored in adata.obs, such as ['scsa_annotation','scMulan_anno']
            model: str, the LLM we used to identify the best matched cells in clusters.
            base_url: str, the LLM api url.
            species: str, the species of scRNA-seq,
            organization: str, the organization of scRNA-seq
            provider: str, if `base_url` is None, we can use default provider.

        Example:
        ```
        vote_obj=CellVote(adata)
        vote_obj.vote('leiden',marker_dict,
                        celltype_keys=['scsa_annotation','scMulan_anno'],
                        )
        ```
        You can found the result in adata.obs['CellVote_celltype']

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


def get_cluster_celltype(
    cluster_celltypes,
    cluster_markers,
    species,
    organization,
    model,
    base_url,
    provider,
    api_key=None,
):
    # from openai import OpenAI
    import os
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
        # print(question)

        params = {
            "messages": [{"role": "user", "content": question}],
            # 如果需要切换模型，在这里修改
            "model": model,
        }
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=params, stream=False
        )
        res = response.json()
        answer = res["choices"][0]["message"]["content"].split("\n")
        # 将回答加入结果字典
        cluster_celltype[cluster_id] = answer[0].lower()

    return cluster_celltype


