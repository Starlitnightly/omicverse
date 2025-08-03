import torch.nn.functional as F
import torch.nn as nn
import torch

class ECSLatentLayer(nn.Module):
    def __init__(self, ecs_threshold=0.6, **kwargs):
        super().__init__()
        self.ecs_threshold = ecs_threshold
        self.is_adversarial = False

    def forward(self, x_dict):
        if self.training and 'ecs' in x_dict:
            cell_emb = x_dict['h']
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)
            return cell_emb, torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)
        else:
            return x_dict['h'], 0

