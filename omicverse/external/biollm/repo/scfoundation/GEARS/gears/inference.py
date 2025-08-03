import torch
import numpy as np
import scanpy as sc

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def evaluate(loader, model, uncertainty, device):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):

        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results


def compute_metrics(results, gene_idx=None):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
            
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'ctrl':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))
                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert


def compute_synergy_loss(results, mean_control, high_umi_idx, subtype = 'POTENTIATION'):
    pred_res = get_test_set_results_seen2(results, subtype)
    all_perts = np.unique(results['pert_cat'])
    linear_params = get_linear_params(pred_res, high_umi_idx,
                                            mean_control,all_perts)
    synergy_loss = np.sum([np.abs(linear_params[k]['pred']['mag']
                               - linear_params[k]['truth']['mag']) for k in
                                linear_params])
    
    mag = np.sum([np.abs(linear_params[k]['pred']['mag']) for k in
                                linear_params])
    return synergy_loss, mag


def non_zero_analysis(adata, test_res):
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}
        
        pert_idx = np.where(test_res['pert_cat'] == pert)[0]    
        de_idx = [geneid2idx[i] for i in adata.uns['top_non_zero_de_20'][pert2pert_full_id[pert]]]

        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx)
        pert_metric[pert]['frac_correct_direction_top20_non_zero'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(de_idx)
        pert_metric[pert]['frac_opposite_direction_top20_non_zero'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(de_idx)
        pert_metric[pert]['frac_0/1_direction_top20_non_zero'] = frac_direction_opposite
        
        mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        std = np.std(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        min_ = np.min(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        max_ = np.max(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        q25 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.6, axis = 0)
        
        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:            
            pred_mean = np.mean(test_res['pred'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
            true_mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
           
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_non_zero'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55_non_zero'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60_non_zero'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75_non_zero'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma_non_zero'] = np.mean(sigma)
            pert_metric[pert]['std_sigma_non_zero'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1_non_zero'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2_non_zero'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)
        
        p_idx = np.where(test_res['pert_cat'] == pert)[0]
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de_non_zero'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de_non_zero'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de_non_zero'] = val
                
    return pert_metric

def non_dropout_analysis(adata, test_res):
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}
        
        pert_idx = np.where(test_res['pert_cat'] == pert)[0]    
        de_idx = [geneid2idx[i] for i in adata.uns['top_non_dropout_de_20'][pert2pert_full_id[pert]]]
        non_zero_idx = adata.uns['non_zeros_gene_idx'][pert2pert_full_id[pert]]
        non_dropout_gene_idx = adata.uns['non_dropout_gene_idx'][pert2pert_full_id[pert]]
             
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx)
        pert_metric[pert]['frac_correct_direction_top20_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(de_idx)
        pert_metric[pert]['frac_opposite_direction_top20_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(de_idx)
        pert_metric[pert]['frac_0/1_direction_top20_non_dropout'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_correct_direction_non_zero'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_opposite_direction_non_zero'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_0/1_direction_non_zero'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_correct_direction_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_opposite_direction_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_0/1_direction_non_dropout'] = frac_direction_opposite
        
        mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        std = np.std(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        min_ = np.min(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        max_ = np.max(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        q25 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.6, axis = 0)
        
        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:            
            pred_mean = np.mean(test_res['pred'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
            true_mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
           
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_non_dropout'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55_non_dropout'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60_non_dropout'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75_non_dropout'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma_non_dropout'] = np.mean(sigma)
            pert_metric[pert]['std_sigma_non_dropout'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1_non_dropout'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2_non_dropout'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)
        
        p_idx = np.where(test_res['pert_cat'] == pert)[0]
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de_non_dropout'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
                
    return pert_metric
    
def deeper_analysis(adata, test_res, de_column_prefix = 'rank_genes_groups_cov', most_variable_genes = None):
    
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}

    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis = 0))[-200:]
        
    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}
        de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
        de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
        de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
        de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]    
        pred_mean = np.mean(test_res['pred_de'][pert_idx], axis = 0).reshape(-1,)
        true_mean = np.mean(test_res['truth_de'][pert_idx], axis = 0).reshape(-1,)
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0) - ctrl[0]) - np.sign(test_res['truth'][pert_idx].mean(0) - ctrl[0]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(geneid2name)
        pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction

        de_idx_map = {20: de_idx,
                      50: de_idx_50,
                      100: de_idx_100,
                      200: de_idx_200
                     }
        
        for val in [20, 50, 100, 200]:
            
            direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/val
            pert_metric[pert]['frac_correct_direction_' + str(val)] = frac_correct_direction

        mean = np.mean(test_res['truth_de'][pert_idx], axis = 0)
        std = np.std(test_res['truth_de'][pert_idx], axis = 0)
        min_ = np.min(test_res['truth_de'][pert_idx], axis = 0)
        max_ = np.max(test_res['truth_de'][pert_idx], axis = 0)
        q25 = np.quantile(test_res['truth_de'][pert_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth_de'][pert_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth_de'][pert_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth_de'][pert_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth_de'][pert_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth_de'][pert_idx], 0.6, axis = 0)

        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:            
            
            direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) - np.sign(true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(nonzero_des)
            pert_metric[pert]['frac_correct_direction_20_nonzero'] = frac_correct_direction
            
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma'] = np.mean(sigma)
            pert_metric[pert]['std_sigma'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)

        ## correlation on delta
        p_idx = np.where(test_res['pert_cat'] == pert)[0]

        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)- ctrl[0], test_res['truth'][p_idx].mean(0)-ctrl[0])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta'] = val
                
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta_de'] = val

        ## up fold changes > 10?
        pert_mean = np.mean(test_res['truth'][p_idx], axis = 0).reshape(-1,)

        fold_change = pert_mean/ctrl
        fold_change[np.isnan(fold_change)] = 0
        fold_change[np.isinf(fold_change)] = 0
        ## this is to remove the ones that are super low and the fold change becomes unmeaningful
        fold_change[0][np.where(pert_mean < 0.5)[0]] = 0

        o =  np.where(fold_change[0] > 0)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_all'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.333)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.33'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.1)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.1'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 3)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_3'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 10)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_10'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        ## most variable genes
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes] - ctrl[0][most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes]-ctrl[0][most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_hvg'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_hvg'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes])
                pert_metric[pert][m + '_top200_hvg'] = val


        ## top 20/50/100/200 DEs
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de'] = val

        
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])
                pert_metric[pert][m + '_top200_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top100_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top100_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])
                pert_metric[pert][m + '_top100_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top50_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top50_de'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])
                pert_metric[pert][m + '_top50_de'] = val



    return pert_metric


def GI_subgroup(pert_metric):
    GI_type2Score = {}
    test_pert_list = list(pert_metric.keys())
    for GI_type, gi_list in GIs.items():
        intersect = np.intersect1d(gi_list, test_pert_list)
        if len(intersect) != 0:
            GI_type2Score[GI_type] = {}

            for m in list(list(pert_metric.values())[0].keys()):
                GI_type2Score[GI_type][m] = np.mean([pert_metric[i][m] for i in intersect if m in pert_metric[i]])
                
    return GI_type2Score

def node_specific_batch_out(models, batch):
    # Returns output for all node specific loader as a matrix of dimension batch_size x nodes
    outs = []
    for idx in range(len(models)):
        outs.append(models[idx](batch).detach().cpu().numpy()[:,idx])
    return np.vstack(outs).T

# Run prediction over all batches
def batch_predict(loader, loaded_models, args):
    # Prediction for node specific GNNs
    preds = []
    print("Loader size: ", len(loader))
    for itr, batch in enumerate(loader):
        print(itr)
        batch = batch.to(args['device'])
        preds.append(node_specific_batch_out(loaded_models, batch))

    preds = np.vstack(preds)
    return preds

def get_high_umi_idx(gene_list):
    # Genes used for linear model fitting
    try:
        high_umi = np.load('../genes_with_hi_mean.npy', allow_pickle=True)
    except:
        high_umi = np.load('./genes_with_hi_mean.npy', allow_pickle=True)
    high_umi_idx = np.where([g in high_umi for g in gene_list])[0]
    return high_umi_idx

def get_mean_ctrl(adata):
    return adata[adata.obs['condition'] == 'ctrl'].to_df().mean().reset_index(
        drop=True)

def get_single_name(g, all_perts):
    name = g+'+ctrl'
    if name in all_perts:
        return name
    else:
        return 'ctrl+'+g


def get_test_set_results_seen2(res, sel_GI_type):
    # Get relevant test set results
    test_pert_cats = [p for p in np.unique(res['pert_cat']) if
                      p in GIs[sel_GI_type] or 'ctrl' in p]
    pred_idx = np.where([t in test_pert_cats for t in res['pert_cat']])
    out = {}
    for key in res:
        out[key] = res[key][pred_idx]
    return out

## Synergy loss calculation functions
def get_all_vectors(all_res, mean_control, double,
                    single1, single2, high_umi_idx):
    # Pred
    pred_df = pd.DataFrame(all_res['pred'])
    pred_df['condition'] = all_res['pert_cat']
    subset_df = pred_df[pred_df['condition'] == double].iloc[:, :-1]
    delta_double_pred = subset_df.mean(0) - mean_control
    single_df_1_pred = pred_df[pred_df['condition'] == single1].iloc[:, :-1]
    single_df_2_pred = pred_df[pred_df['condition'] == single2].iloc[:, :-1]

    # True
    truth_df = pd.DataFrame(all_res['truth'])
    truth_df['condition'] = all_res['pert_cat']
    subset_df = truth_df[truth_df['condition'] == double].iloc[:, :-1]
    delta_double_truth = subset_df.mean(0) - mean_control
    single_df_1_truth = truth_df[truth_df['condition'] == single1].iloc[:, :-1]
    single_df_2_truth = truth_df[truth_df['condition'] == single2].iloc[:, :-1]

    delta_single_truth_1 = single_df_1_truth.mean(0) - mean_control
    delta_single_truth_2 = single_df_2_truth.mean(0) - mean_control
    delta_single_pred_1 = single_df_1_pred.mean(0) - mean_control
    delta_single_pred_2 = single_df_2_pred.mean(0) - mean_control

    return {'single_pred_1': delta_single_pred_1.values[high_umi_idx],
            'single_pred_2': delta_single_pred_2.values[high_umi_idx],
            'double_pred': delta_double_pred.values[high_umi_idx],
            'single_truth_1': delta_single_truth_1.values[high_umi_idx],
            'single_truth_2': delta_single_truth_2.values[high_umi_idx],
            'double_truth': delta_double_truth.values[high_umi_idx]}


def get_coeffs_synergy(singles_expr, double_expr):
    results = {}
    results['ts'] = TheilSenRegressor(fit_intercept=False,
                                      max_subpopulation=1e5,
                                      max_iter=1000,
                                      random_state=1000)
    X = singles_expr
    y = double_expr
    try:
        results['ts'].fit(X, y.ravel())
    except:
        print(X)
        print(y)
    results['c1'] = results['ts'].coef_[0]
    results['c2'] = results['ts'].coef_[1]
    results['mag'] = np.sqrt((results['c1'] ** 2 + results['c2'] ** 2))
    return results


def Fit(all_vectors, type_='pertnet'):
    if type_ == 'pertnet':
        singles_expr = np.array(
            [all_vectors['single_pred_1'], all_vectors['single_pred_2']]).T
        first_expr = np.array([all_vectors['single_pred_1']]).T
        second_expr = np.array([all_vectors['single_pred_2']]).T
        double_expr = np.array(all_vectors['double_pred']).T

    elif type_ == 'truth':
        singles_expr = np.array(
            [all_vectors['single_truth_1'], all_vectors['single_truth_2']]).T
        first_expr = np.array([all_vectors['single_truth_1']]).T
        second_expr = np.array([all_vectors['single_truth_2']]).T
        double_expr = np.array(all_vectors['double_truth']).T

    return get_coeffs_synergy(singles_expr, double_expr)


def get_linear_params(pred_res, high_umi_idx, mean_control, all_perts):
    results = {}
    for c in set(pred_res['pert_cat']):
        if 'ctrl' in c:
            continue
        double = c
        single1 = get_single_name(double.split('+')[0], all_perts)
        single2 = get_single_name(double.split('+')[1], all_perts)
        all_vectors = get_all_vectors(pred_res, mean_control, double,
                                      single1, single2, high_umi_idx)

        pertnet_results = Fit(all_vectors, type_='pertnet')
        truth_results = Fit(all_vectors, type_='truth')

        results[c] = {
            'truth': truth_results,
            'pred': pertnet_results}

    return results


# Read in model for each gene

GIs = {
    'NEOMORPHIC': ['CBL+TGFBR2',
                  'KLF1+TGFBR2',
                  'MAP2K6+SPI1',
                  'SAMD1+TGFBR2',
                  'TGFBR2+C19orf26',
                  'TGFBR2+ETS2',
                  'CBL+UBASH3A',
                  'CEBPE+KLF1',
                  'DUSP9+MAPK1',
                  'FOSB+PTPN12',
                  'PLK4+STIL',
                  'PTPN12+OSR2',
                  'ZC3HAV1+CEBPE'],
    'ADDITIVE': ['BPGM+SAMD1',
                'CEBPB+MAPK1',
                'CEBPB+OSR2',
                'DUSP9+PRTG',
                'FOSB+OSR2',
                'IRF1+SET',
                'MAP2K3+ELMSAN1',
                'MAP2K6+ELMSAN1',
                'POU3F2+FOXL2',
                'RHOXF2BB+SET',
                'SAMD1+PTPN12',
                'SAMD1+UBASH3B',
                'SAMD1+ZBTB1',
                'SGK1+TBX2',
                'TBX3+TBX2',
                'ZBTB10+SNAI1'],
    'EPISTASIS': ['AHR+KLF1',
                 'MAPK1+TGFBR2',
                 'TGFBR2+IGDCC3',
                 'TGFBR2+PRTG',
                 'UBASH3B+OSR2',
                 'DUSP9+ETS2',
                 'KLF1+CEBPA',
                 'MAP2K6+IKZF3',
                 'ZC3HAV1+CEBPA'],
    'REDUNDANT': ['CDKN1C+CDKN1A',
                 'MAP2K3+MAP2K6',
                 'CEBPB+CEBPA',
                 'CEBPE+CEBPA',
                 'CEBPE+SPI1',
                 'ETS2+MAPK1',
                 'FOSB+CEBPE',
                 'FOXA3+FOXA1'],
    'POTENTIATION': ['CNN1+UBASH3A',
                    'ETS2+MAP7D1',
                    'FEV+CBFA2T3',
                    'FEV+ISL2',
                    'FEV+MAP7D1',
                    'PTPN12+UBASH3A'],
    'SYNERGY_SIMILAR_PHENO':['CBL+CNN1',
                            'CBL+PTPN12',
                            'CBL+PTPN9',
                            'CBL+UBASH3B',
                            'FOXA3+FOXL2',
                            'FOXA3+HOXB9',
                            'FOXL2+HOXB9',
                            'UBASH3B+CNN1',
                            'UBASH3B+PTPN12',
                            'UBASH3B+PTPN9',
                            'UBASH3B+ZBTB25'],
    'SYNERGY_DISSIMILAR_PHENO': ['AHR+FEV',
                                'DUSP9+SNAI1',
                                'FOXA1+FOXF1',
                                'FOXA1+FOXL2',
                                'FOXA1+HOXB9',
                                'FOXF1+FOXL2',
                                'FOXF1+HOXB9',
                                'FOXL2+MEIS1',
                                'IGDCC3+ZBTB25',
                                'POU3F2+CBFA2T3',
                                'PTPN12+ZBTB25',
                                'SNAI1+DLX2',
                                'SNAI1+UBASH3B'],
    'SUPPRESSOR': ['CEBPB+PTPN12',
                  'CEBPE+CNN1',
                  'CEBPE+PTPN12',
                  'CNN1+MAPK1',
                  'ETS2+CNN1',
                  'ETS2+IGDCC3',
                  'ETS2+PRTG',
                  'FOSB+UBASH3B',
                  'IGDCC3+MAPK1',
                  'LYL1+CEBPB',
                  'MAPK1+PRTG',
                  'PTPN12+SNAI1']
}
