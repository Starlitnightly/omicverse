import networkx as nx
from popv import _utils


def _absolute_accuracy(adata, pred_key, gt_key, save_key=None):
    pred = adata.obs[pred_key].str.lower()
    gt = adata.obs[gt_key].str.lower()

    acc = (pred == gt).astype(int)
    if save_key is not None:
        adata.obs[save_key] = acc
    return acc


def _ontology_accuracy(adata, pred_key, gt_key, obo_file, save_key=None):
    G = _utils.make_ontology_dag(obo_file, lowercase=False).reverse()
    if not save_key:
        save_key = "ontology_accuracy"
    adata.obs[save_key] = "na"

    def match_type(n1, n2):
        if n1 == n2:
            return "exact"
        elif not set(G.predecessors(n1)).isdisjoint(G.predecessors(n2)):
            return "sibling"
        elif n1 in set(G.predecessors(n2)):
            return "parent"
        elif n2 in set(G.predecessors(n1)):
            return "child"
        else:
            return "no match"

    adata.obs[save_key] = adata.obs.apply(lambda x: match_type(x[pred_key], x[gt_key]), axis=1)


def _fine_ontology_sibling_accuracy(adata, obo_file, pred_key, gt_key, save_key=None):
    """Calculate the fine ontology accuracy and also determines the distance to siblings."""
    if save_key is None:
        save_key = f"{pred_key}_ontology_distance"
    adata.obs[save_key] = None

    dag = _utils.make_ontology_dag(obo_file, lowercase=False).reverse()
    dag_undirected = dag.to_undirected()

    ontology_distance_dict = {}

    for name, pred_ct, gt_ct in zip(
        adata.obs_names,
        adata.obs[pred_key],
        adata.obs[gt_key],
        strict=True,
    ):
        score = None
        combination = f"{pred_ct}_{gt_ct}"
        if combination in ontology_distance_dict:
            score = ontology_distance_dict[combination]
        else:
            if pred_ct == gt_ct:
                score = 0
            elif nx.has_path(dag, source=pred_ct, target=gt_ct):
                score = nx.shortest_path_length(dag, source=pred_ct, target=gt_ct) - 1
            elif nx.has_path(dag, target=pred_ct, source=gt_ct):
                score = nx.shortest_path_length(dag, source=gt_ct, target=pred_ct) - 1
                score *= -1
            elif nx.has_path(dag_undirected, target=pred_ct, source=gt_ct):
                score_ = nx.shortest_path_length(dag_undirected, source=pred_ct, target=gt_ct) - 1
                score = f"{score_}_sib"
            else:
                score = 1000

        ontology_distance_dict[combination] = score
        adata.obs.loc[name, save_key] = score
    adata.obs[save_key] = adata.obs[save_key].astype(str).astype("category")
