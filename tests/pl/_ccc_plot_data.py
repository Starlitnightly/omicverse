from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "omicverse"
    / "datasets"
    / "data_files"
    / "TF"
    / "cellchat_interactions_and_tfs_human.csv"
)

_INTERACTION_IDS = [
    "TGFB1_TGFBR1_TGFBR2",
    "CXCL12_CXCR4",
    "MIF_CD74_CXCR4",
    "MDK_SDC1",
    "NRG1_ERBB3",
    "FN1_ITGAV_ITGB3",
]

_CELL_TYPES = ["EVT_1", "dNK1", "VCT"]

_MEANS = np.array(
    [
        [0.8, 1.6, 0.5, 0.7, 0.4, 2.2],
        [3.4, 4.8, 2.7, 1.9, 0.6, 1.5],
        [2.1, 1.1, 0.8, 3.3, 2.6, 4.1],
        [1.2, 3.6, 2.2, 0.9, 1.7, 1.8],
        [0.5, 2.9, 4.3, 0.6, 0.7, 1.0],
        [1.7, 1.4, 1.9, 2.8, 3.4, 2.9],
        [2.6, 0.9, 0.7, 1.5, 4.0, 5.2],
        [1.4, 2.3, 3.1, 1.1, 1.2, 2.0],
        [0.9, 1.0, 1.3, 2.4, 2.8, 4.6],
    ],
    dtype=float,
)

_PVALUES = np.array(
    [
        [0.18, 0.04, 0.25, 0.09, 0.33, 0.02],
        [0.003, 0.001, 0.01, 0.04, 0.22, 0.08],
        [0.02, 0.16, 0.20, 0.006, 0.04, 0.002],
        [0.09, 0.01, 0.03, 0.18, 0.05, 0.12],
        [0.21, 0.02, 0.001, 0.27, 0.14, 0.07],
        [0.04, 0.08, 0.02, 0.01, 0.004, 0.03],
        [0.01, 0.25, 0.19, 0.05, 0.008, 0.001],
        [0.11, 0.03, 0.009, 0.14, 0.16, 0.05],
        [0.13, 0.10, 0.04, 0.02, 0.01, 0.003],
    ],
    dtype=float,
)


def _load_var_frame() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, index_col=0)
    var = df.loc[_INTERACTION_IDS, :].copy()
    var["interacting_pair"] = var["interaction_name"].astype(str)
    var["classification"] = var["pathway_name"].astype(str)
    var["gene_a"] = var["ligand"].astype(str)
    var["gene_b"] = var["receptor"].astype(str)
    return var.loc[
        :,
        [
            "interacting_pair",
            "interaction_name",
            "interaction_name_2",
            "classification",
            "pathway_name",
            "gene_a",
            "gene_b",
            "ligand",
            "receptor",
            "annotation",
            "evidence",
        ],
    ]


def _obs_frame(cell_types: list[str]) -> pd.DataFrame:
    pairs = []
    for sender in cell_types:
        for receiver in cell_types:
            pairs.append(
                {
                    "pair_id": f"{sender}|{receiver}",
                    "sender": sender,
                    "receiver": receiver,
                }
            )
    obs = pd.DataFrame(pairs).set_index("pair_id")
    return obs


def _build_comm_adata(obs: pd.DataFrame, var: pd.DataFrame, means: np.ndarray, pvalues: np.ndarray) -> AnnData:
    adata = AnnData(X=means.copy(), obs=obs.copy(), var=var.copy())
    adata.layers["means"] = means.copy()
    adata.layers["pvalues"] = pvalues.copy()
    adata.uns["node_positions"] = {
        "EVT_1": (1.2, 0.15),
        "VCT": (-0.2, 0.95),
        "dNK1": (-0.35, -0.85),
        "SCT": (0.75, -1.1),
    }
    return adata


def make_comm_adata() -> AnnData:
    return _build_comm_adata(_obs_frame(_CELL_TYPES), _load_var_frame(), _MEANS, _PVALUES)


def make_comm_adata_shifted() -> AnnData:
    adata = make_comm_adata()
    shifted = adata.layers["means"].copy()
    shifted[:, 0] = shifted[:, 0] * 1.25 + 0.1
    shifted[:, 1] = shifted[:, 1] * 0.85 + 0.3
    shifted[:, 2] = shifted[:, 2] * 1.10 + 0.2
    shifted[:, 3] = shifted[:, 3] * 0.90 + 0.4
    shifted[:, 4] = shifted[:, 4] * 1.35
    shifted[:, 5] = shifted[:, 5] * 0.95 + 0.5
    adata.layers["means"] = shifted
    adata.X = shifted.copy()
    return adata


def make_comm_adata_with_receiver_only_group() -> AnnData:
    obs = pd.DataFrame(
        {
            "sender": ["EVT_1", "dNK1"],
            "receiver": ["dNK1", "SCT"],
        },
        index=["EVT_1|dNK1", "dNK1|SCT"],
    )
    var = _load_var_frame().loc[["CXCL12_CXCR4"], :].copy()
    means = np.array([[2.4], [1.7]], dtype=float)
    pvalues = np.array([[0.02], [0.03]], dtype=float)
    return _build_comm_adata(obs, var, means, pvalues)
