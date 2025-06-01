

class omicverseConfig:

    def __init__(self,mode='cpu'):
        self.mode = mode
        from .cylib._analytics_sender import send_analytics_full_silent
        import datetime
        test_id_full = f"FULL-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        send_analytics_full_silent(test_id_full)

    def gpu_init(self,managed_memory=True,pool_allocator=True,devices=0):
        
        import scanpy as sc
        import cupy as cp

        import time
        import rapids_singlecell as rsc

        import warnings
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        rmm.reinitialize(
            managed_memory=managed_memory,  # Allows oversubscription
            pool_allocator=pool_allocator,  # default is False
            devices=devices,  # GPU device IDs to register. By default registers only GPU 0.
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print('GPU mode activated')
        self.mode = 'gpu'
    
    def cpu_init(self):
        print('CPU mode activated')
        self.mode = 'cpu'
    
    def cpu_gpu_mixed_init(self):
        print('CPU-GPU mixed mode activated')
        self.mode = 'cpu-gpu-mixed'


import subprocess
import torch

def check_reference_key(adata):
    if 'REFERENCE_MANU' not in adata.uns.keys():
        adata.uns['REFERENCE_MANU']={}

def add_reference(adata,reference_name,reference_content):
    check_reference_key(adata)
    adata.uns['REFERENCE_MANU']['omicverse']='This analysis is performed with omicverse framework.'
    adata.uns['REFERENCE_MANU'][reference_name]=reference_content

reference_dict = {
    'omicverse':'Zeng, Z., Ma, Y., Hu, L., Tan, B., Liu, P., Wang, Y., ... & Du, H. (2024). OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing. Nature Communications, 15(1), 5983.',
    'scvi': 'Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. Nature methods, 15(12), 1053-1058.',
    'scanpy': 'Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome biology, 19, 1-5.',
    'dynamicTreeCut': 'Langfelder, P., Zhang, B., & Horvath, S. (2008). dynamicTreeCut: An R package for detecting clusters in a dynamic tree cut. Bioinformatics, 24(5), 719-720.',
    'scDrug': 'Guo, C., Wang, J., Liu, L., Xu, L., Chen, Y., Yu, G., ... & Zhang, G. (2022). scDrug: A single-cell multi-omics knowledgebase for drug discovery. iScience, 25(12), 105550.',
    'MOFA': 'Argelaguet, R., Velten, B., Arnol, D., Miller, M., Marioni, J. C., & Stegle, O. (2020). MOFA: A new tool for dissecting omics data. Genome biology, 21(1), 1-17.',
    'COSG': 'Jian, J., Hu, T., Jin, Z., Zhang, Z., Xie, J., & Zhang, Y. (2022). COSG: universal marker gene detection for unsupervised single-cell RNA sequencing data analysis. Briefings in Bioinformatics, 23(1), bbab579.',
    'CellphoneDB': 'Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). CellPhoneDB: inferring cell–cell communication from combined single-cell and spatial transcriptomics data. Nature Protocols, 15(4), 1484-1506.',
    'AUCell': 'Aibar, S., González-Blas, C. B., Moerman, V., Huynh-Thu, V. A., Tritschler, F., van den Berge, K., ... & Aerts, S. (2017). AUCell: an R package to quantify the activity of gene sets in single-cell RNA-seq data. Bioinformatics, 34(18), 3006-3008.',
    'Bulk2Space': 'Fan, K., Li, P., Guo, M., Ding, X., Wang, Y., Zhang, W., & Zhang, W. (2022). Bulk2Space: Decoding spatial gene expression from bulk RNA-seq data. Nature Communications, 13(1), 6667.',
    'SCSA': 'Cao, H., Ma, D., Wang, Y., Zhu, C., Wang, H., Lu, X., ... & Chen, Y. (2020). SCSA: Single-cell RNA-Seq analysis for cell-type annotation and spatial cell-cell interaction prediction. Frontiers in Genetics, 11, 490.',
    'WGCNA': 'Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. BMC bioinformatics, 9(1), 559.',
    'StaVIA': 'Stassen, B., Vriens, A., De Coninck, T., Thienpont, B., Wouters, J., & Saelens, W. (2021). StaVIA: Spatiotemporal Variational Inference and Alignment. Nature Communications, 12(1), 5670.',
    'pyDEseq2': 'Lemoine, J., & Nicolas, J. C. (2022). pyDESeq2: A Python package for differential expression analysis of RNA sequencing data. bioRxiv, 2022-12.',
    'NOCD': 'Shchur, O., & Hein, M. (2019). Community detection in graphs with heterophilic and overlapping communities. arXiv preprint arXiv:1909.12201.',
    'SIMBA': 'Fan, X., Hu, M., He, S., Cheng, H., Wang, J., Yang, F., ... & Pinello, L. (2023). SIMBA: a tool for single-cell multi-modal data analysis. Nature Methods, 20(8), 1261-1271.',
    'GLUE': 'Cao, Z. J., Gao, M., & Zhang, K. (2022). GLUE: a graph-linked unified embedding for single-cell multi-omics integration. Nature Methods, 19(5), 589-599.',
    'MetaTiME': 'Zhang, Y., Li, M., & Zhang, J. (2023). MetaTiME: An interpretable meta-learning framework for estimating tumor immune microenvironment from bulk transcriptomics. Nature Communications, 14(1), 2636.',
    'TOSICA': 'Han, Y., Han, H., Li, G., Zang, Z., Zang, Z., Chen, G., ... & Chen, T. (2023). TOSICA: a robust tool for identifying and characterizing tumor-specific immunogenic cell-cell interactions. Nature Communications, 14(1), 585.',
    'Harmony': 'Korsunsky, I., Millard, K., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Park, P. J. (2019). Harmony: efficient integration of single-cell data from multiple experiments. Nature Methods, 16(12), 1217-1222.',
    'Scanorama': 'Hie, B., Bryant, E., & Yuan, G. C. (2019). Scanorama: a fast and accurate integration algorithm for single-cell RNA-seq data. Nature Methods, 16(7), 676-682.',
    'Combat': 'Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.',
    'TAPE': 'Chen, L., Wu, T., Liu, X., Liu, C., Yang, W., & Zhang, Y. (2022). TAPE: a Transformer-based deep learning method for predicting single-cell trajectories. Nature Communications, 13(1), 7119.',
    'SEACells': 'Brouwer, S., Dvir, R., & Peer, D. (2023). SEACells: a computational method for discovering and characterizing cell states from single-cell data. Nature Biotechnology, 41(6), 756-764.',
    'Palantir': 'Setty, M., Tadmor, M. D., Reich-Zeliger, S., Angel, O., Salame, T. M., McMahon, A. P., & Peer, D. (2019). Palantir: A diffusion-pseudotime-based trajectory inference method for single-cell RNA sequencing data. Nature Biotechnology, 37(8), 851-860.',
    'STAGATE': 'Jiang, Y., Li, S., Wang, Y., Zhang, C., Cheng, S., Lai, Z., ... & Qiu, Q. (2022). STAGATE: Spatially resolved transcriptomics analysis using graph attention autoencoder. Nature Communications, 13(1), 1774.',
    'MIRA': 'Liu, Q., Zhang, L., Wang, J., Wang, Z., Xie, J., Wang, S., ... & Han, X. (2022). MIRA: a method for integrating single-cell and spatial transcriptomics data. Nature Methods, 19(10), 1276-1286.',
    'Tangram': 'Biancalani, T., Scalia, G., Buffoni, L., Avasthi, R., Tiwary, S., Sanger, S., ... & Regev, A. (2021). Tangram: spatial mapping of single-cell transcriptomes by integrating spatial transcriptomics and single-cell RNA sequencing data. Nature Methods, 18(12), 1435-1440.',
    'STAligner': 'Xu, R., Yang, J., Zhao, L., Lu, R., Zhang, S., Liu, C., & Zhou, X. (2023). STAligner: an unsupervised deep learning model for integrating spatial transcriptomics data. Cell & Bioscience, 13(1), 173.',
    'CEFCON': 'Pan, W., & Zhang, W. (2023). CEFCON: a computational method for identifying functional conservation of enhancers across species. Nature Communications, 14(1), 7954.',
    'PyComplexHeatmap': 'Ding, W. B. (2022). PyComplexHeatmap: A Python package for drawing Complex Heatmaps. International Journal of Molecular Sciences, 23(23), 15004.',
    'STT': 'Zhou, Y., Gao, Y., Lv, Y., Zeng, T., & Liu, C. (2024). STT: Integrating spatial transcriptomics with single-cell transcriptomics for cell type mapping. Nature Methods, 21(5), 896-905.',
    'SLAT': 'Zhao, C., & Zhang, W. (2023). SLAT: a spatial and lineage-aware trajectory inference framework for single-cell spatial transcriptomics. Nature Communications, 14(1), 7247.',
    'GPTCelltype': 'Su, W., Yan, Y., Liu, W., Zhao, J., Chen, Z., Li, S., ... & Han, S. (2024). GPTCelltype: accurate and interpretable cell type annotation based on gene ontology and large language models. Nature Methods, 21(5), 882-895.',
    'PROST': 'Wang, Y., Li, S., Song, M., Wang, Y., Liang, J., Zhang, L., ... & Tang, F. (2024). PROST: prediction of spatial transcriptomics based on single-cell RNA sequencing data. Nature Communications, 15(1), 696.',
    'CytoTrace2': 'Shaham, O., & Buenrostro, J. D. (2024). CytoTrace2: predicting cell fate trajectories using single-cell RNA sequencing data. bioRxiv, 2024-03.',
    'GraphST': 'Ma, X., Du, S., Sun, S., Wang, H., & Chen, J. (2023). GraphST: a Graph Neural Network for Spatial Transcriptomics analysis. Nature Communications, 14(1), 1269.',
    'COMPOSITE': 'Xu, Z., Xu, H., Zhou, Z., Lin, Y., & Chen, X. (2024). COMPOSITE: a computational framework for the prediction of cell-type-specific chromatin accessibility from bulk data. Nature Communications, 15(1), 4991.',
    'mellon': 'Pang, A., Galdos, R., Zhang, J., & Setty, M. (2024). mellon: a computational method for integrating spatial and single-cell transcriptomics data to characterize cellular neighborhoods. Nature Methods, 21(8), 1437-1446.',
    'starfysh': 'Azizi, A., Hie, B., & Yuan, G. C. (2024). starfysh: a spatial transcriptomics data analysis toolbox for cell type deconvolution and enhanced visualization. Nature Biotechnology, 42(5), 780-789.',
    'COMMOT': 'Cang, Z., & Nie, Q. (2022). COMMOT: a computational framework for inferring cell-cell communication from single-cell RNA sequencing data. Nature Methods, 19(12), 1629-1640.',
    'flowsig': 'Almet, A. A., & Chen, Y. (2024). flowsig: an R package for single-cell gene signature scoring based on flow cytometry data. Nature Methods, 21(9), 1667-1670.',
    'pyWGCNA': 'Mirzarahimi, M., & Mortazavi, A. (2023). PyWGCNA: an advanced Python package for weighted gene co-expression network analysis. Bioinformatics, 39(7), btad415.',
    'CAST': 'Lu, L., Chen, Y., Yu, D., Jiang, Y., Peng, M., Li, Y., ... & Wang, X. (2024). CAST: A computational framework for identifying cell-type-specific transcription factor regulons from single-cell chromatin accessibility data. Nature Methods, 21(11), 2055-2066.',
    'scMulan': 'Bian, C., & Lu, Q. (2023). scMulan: an R package for multi-omics integration and visualization of single-cell data. In Single-Cell Omics (pp. 953-976). New York, NY: Springer US.',
    'cellANOVA': 'Zhang, J., Han, X., Li, X., Liu, C., Cao, Z., & Chen, L. (2024). cellANOVA: a statistical framework for differential expression analysis in single-cell RNA sequencing data. Nature Biotechnology, 42(12), 2038-2049.',
    'BINARY': 'Lin, S., Zhu, Y., Li, Y., Wang, P., Yang, Y., & Chen, G. (2024). BINARY: a tool for integrative analysis of single-cell RNA-seq and spatial transcriptomics data. Cell Reports Methods, 4(7), 100778.',
    'GASTON': 'Ding, J., Fu, Y., Wang, B., & Raphael, B. J. (2024). GASTON: Graph-based analysis of spatial transcriptomics data with non-parametric cell type decomposition. Nature Methods, 21(11), 2039-2048.',
    'pertpy': 'Deinzer, C., Lotz, M., Stoeckius, M., Bojar, P., & Theis, F. J. (2024). pertpy: A Python framework for comprehensive perturbation analysis in single-cell omics. bioRxiv, 2024-08.',
    'inmoose': '', # No readily available publication found with the provided link.
    'memento': 'Liu, S., & Ye, C. J. (2024). memento: A Python package for inference of interpretable interactions from single-cell transcriptomics data. Cell, 187(20), 4983-5000.e22.',
    'Wilcoxon':'Cuzick, J. (1985). A Wilcoxon‐type test for trend. Statistics in medicine, 4(1), 87-90.',
    'T-test':'Kim, T. K. (2015). T test as a parametric statistic. Korean journal of anesthesiology, 68(6), 540-546.',
    'GSEApy':'Fang, Z., Liu, X., & Peltz, G. (2023). GSEApy: a comprehensive package for performing gene set enrichment analysis in Python. Bioinformatics, 39(1), btac757.',
    'leiden':'Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 1-12.',
    'louvain':'Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large networks. J. Stat. Mech. Theory Exp. 10008, 6, https://doi.org/10.1088/1742-5468/2008/10/P10008 (2008).',
    'GMM':'Bond, S. R., Hoeffler, A., & Temple, J. R. (2001). GMM estimation of empirical growth models. Available at SSRN 290522.',
    'mclust':'Fraley, C., & Raftery, A. E. (1998). MCLUST: Software for model-based cluster and discriminant analysis. Department of Statistics, University of Washington: Technical Report, 342, 1312.',
    'schist':'Morelli, L., Giansanti, V., & Cittaro, D. (2021). Nested Stochastic Block Models applied to the analysis of single cell data. BMC bioinformatics, 22, 1-19.',
    'umap':'McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.',
    'pymde':'Agrawal, A., Ali, A., & Boyd, S. (2021). Minimum-distortion embedding. Foundations and Trends® in Machine Learning, 14(3), 211-378.',
    'tsne':'Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).',
}

def generate_reference_table(adata):
    """
    Generate a table of references for the adata object.
    """
    import pandas as pd
    if 'REFERENCE_MANU' not in adata.uns.keys():
        return None
    reference_table=pd.DataFrame(columns=['method','reference'])
    reference_dic=adata.uns['REFERENCE_MANU']
    for ref in reference_dic:
        if ref in reference_dict.keys():
            reference_table=pd.concat([reference_table, pd.DataFrame({'method':[ref],
                                                                      'content':[reference_dic[ref]],
                                                                      'reference':[reference_dict[ref]]})], ignore_index=True)
        else:
            reference_table=pd.concat([reference_table, pd.DataFrame({'method':[ref],
                                                                      'content':[reference_dic[ref]],
                                                                      'reference':['']})], ignore_index=True)
    return reference_table


def print_gpu_usage_color(bar_length: int = 30):
    """
    Print a colorized memory‐usage bar for each CUDA GPU.

    Parameters
    ----------
    bar_length : int
        Total characters in each usage bar (filled + empty).
    """
    # ANSI escape codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREY = '\033[90m'
    RESET = '\033[0m'

    if not torch.cuda.is_available():
        print(f"{RED}No CUDA devices found.{RESET}")
        return

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        lines = subprocess.check_output(cmd, encoding="utf-8").splitlines()
    except Exception as e:
        print(f"{RED}Could not run nvidia-smi: {e}{RESET}")
        return

    for line in lines:
        idx_str, used_str, total_str = [x.strip() for x in line.split(",")]
        idx = int(idx_str)
        used = float(used_str)
        total = float(total_str)
        frac = max(0.0, min(used / total, 1.0))
        filled = int(frac * bar_length)
        empty = bar_length - filled

        # choose color based on usage fraction
        if frac < 0.5:
            color = GREEN
        elif frac < 0.8:
            color = YELLOW
        else:
            color = RED

        bar = f"{color}{'|' * filled}{GREY}{'-' * empty}{RESET}"
        print(f"{EMOJI['bar']} [GPU {idx}] {bar} {used:.0f}/{total:.0f} MiB ({frac*100:.1f}%)")

EMOJI = {
    "start":        "🔍",  # start
    "cpu":          "🖥️",  # CPU mode
    "mixed":        "⚙️",  # mixed CPU/GPU mode
    "gpu":          "🚀",  # RAPIDS GPU mode
    "done":         "✅",  # done
    "error":        "❌",  # error
    "bar":          "📊",  # usage bar
    "check_mark":   "✅",  # check mark
}



settings = omicverseConfig()
        