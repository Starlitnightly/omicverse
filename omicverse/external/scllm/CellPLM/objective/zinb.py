import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

class ZINBReconstructionLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, out_dict, x_dict, ridge_lambda = 0.0):
        """Forward propagation.
        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : torch.Tensor
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.
        Returns
        -------
        result : float
            ZINB loss.
        """
        eps = 1e-10
        x = x_dict['x_seq'].to_dense()[x_dict['input_mask']]
        # x = x_dict['x_seq'].index_select(0, x_dict['input_mask']).to_dense()
        mean = out_dict['mean'][x_dict['input_mask']]
        disp = out_dict['disp'][x_dict['input_mask']]
        pi = out_dict['pi'][x_dict['input_mask']]
        scale_factor = x_dict['scale_factor'][x_dict['input_mask']]
        scale_factor = scale_factor.unsqueeze(-1)
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result


class NBImputationLoss(nn.Module):
    """NB loss class."""

    def __init__(self, dae=True, **kwargs):
        super().__init__()
        self.dae = dae
        self.downstream = None

    def forward(self, out_dict, x_dict):
        eps = 1e-10
        if 'gene_mask' not in x_dict:
            x_dict['gene_mask'] = torch.arange(x_dict['x_seq'].shape[1]).to(x_dict['x_seq'].device)
        mean = out_dict['mean']
        disp = out_dict['disp']
        if 'input_gene_mask' in x_dict:
            mean = mean[:, x_dict['input_gene_mask']]
            disp = disp[:, x_dict['input_gene_mask']]
        out_dict['pred'] = mean
        mean = mean[:, x_dict['gene_mask']]
        disp = disp[:, x_dict['gene_mask']]
        truth = x_dict['x_seq'].to_dense()[:, x_dict['gene_mask']]
        size_factor = truth.sum(1, keepdim=True) / mean.sum(1, keepdim=True)
        mean *= size_factor
        out_dict['pred'] *= size_factor

        if False:
            return F.mse_loss(torch.log1p(out_dict['pred']), torch.log1p(truth))
        else:
            t1 = torch.lgamma(disp + eps) + torch.lgamma(truth + 1.0) - torch.lgamma(truth + disp + eps)
            t2 = (disp + truth) * torch.log(1.0 + (mean / (disp + eps))) + (
                        truth * (torch.log(disp + eps) - torch.log(mean + eps)))
            nb_final = t1 + t2
            return nb_final.sum(-1).mean()

class NBDenoisingLoss(nn.Module):
    """NB loss class."""

    def __init__(self, dae=True, **kwargs):
        super().__init__()
        self.dae = dae
        self.downstream = None

    def forward(self, out_dict, x_dict):
        eps = 1e-10

        truth = x_dict['label']
        mean = out_dict['mean'][:, x_dict['gene_mask']]
        disp = out_dict['disp'][:, x_dict['gene_mask']]
        mean = mean / mean.sum(1, keepdim=True) * truth.sum(1, keepdim=True)
        out_dict['pred'] = mean

        if False:
            return F.mse_loss(torch.log1p(out_dict['pred']), torch.log1p(truth))
        else:
            t1 = torch.lgamma(disp + eps) + torch.lgamma(truth + 1.0) - torch.lgamma(truth + disp + eps)
            t2 = (disp + truth) * torch.log(1.0 + (mean / (disp + eps))) + (
                        truth * (torch.log(disp + eps) - torch.log(mean + eps)))
            nb_final = t1 + t2
            return nb_final.sum(-1).mean()

class NBReconstructionLoss(nn.Module):
    """NB loss class."""

    def __init__(self, dae=True, **kwargs):
        super().__init__()
        self.dae = dae

    def forward(self, out_dict, x_dict):
        eps = 1e-10

        y = x_dict['x_seq'].to_dense()
        truth = y[:, x_dict['gene_mask']]
        mean = out_dict['mean'][:, x_dict['gene_mask']]
        disp = out_dict['disp'][:, x_dict['gene_mask']]
        masked_nodes = x_dict['input_mask'].sum(1)>0

        if self.dae and self.training:
            truth_masked = (truth * x_dict['input_mask'])[masked_nodes] #/ (x_dict['input_mask'][masked_nodes].mean())
            mean_masked = (out_dict['mean'] * x_dict['input_mask'])[masked_nodes]
            disp_masked = (out_dict['disp'] * x_dict['input_mask'])[masked_nodes]
            mean_masked = mean_masked / mean_masked.sum(1, keepdim=True) * truth_masked.sum(1, keepdim=True)
            t1 = torch.lgamma(disp_masked + eps) + torch.lgamma(truth_masked + 1.0) - torch.lgamma(truth_masked + disp_masked + eps)
            t2 = (disp_masked + truth_masked) * torch.log(1.0 + (mean_masked / (disp_masked + eps))) + (
                        truth_masked * (torch.log(disp_masked + eps) - torch.log(mean_masked + eps)))
            nb_final_masked = t1 + t2
        else:
            nb_final_masked = 0.

        truth = truth[masked_nodes]
        mean = mean[masked_nodes]
        disp = disp[masked_nodes]
        mean = mean / mean.sum(1, keepdim=True) * truth.sum(1, keepdim=True)

        t1 = torch.lgamma(disp + eps) + torch.lgamma(truth + 1.0) - torch.lgamma(truth + disp + eps)
        t2 = (disp + truth) * torch.log(1.0 + (mean / (disp + eps))) + (truth * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2 + nb_final_masked

        return nb_final.sum(-1).mean()