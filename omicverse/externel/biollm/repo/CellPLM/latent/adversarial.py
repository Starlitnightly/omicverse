import torch
from torch import nn
import torch.nn.functional as F

class BatchDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, dropout, target_classes):
        super(BatchDiscriminator, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
            ))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, target_classes))
        self.layers.cuda()

    def forward(self, h):
        rp = h.shape[0]
        h = h.mean(dim=0)
        for layer in self.layers:
            h = layer(h)
        return h.repeat(rp, 1)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, dropout, target_classes):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, target_classes))

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h

class AdversarialLatentLayer(nn.Module):
    """Adversarial latent layer for CellBert

    Parameters
    ----------
    input_dims : Iterable[int]
        List of input dimensions
    label_key : str
        Key of the label in the input dictionary
    batch_wise : bool
        Whether to use batch-wise discriminator
    discriminator_hidden : int
        Hidden dimension of the discriminator
    discriminator_layers : int
        Number of layers in the discriminator
    discriminator_dropout : float
        Dropout rate of the discriminator
    target_classes : int
        Number of classes in the discriminator
    disc_lr : float
        Learning rate of the discriminator
    disc_wd : float
        Weight decay of the discriminator
    """
    def __init__(self, input_dims, label_key, batch_wise=False, discriminator_hidden=128, discriminator_layers=2, discriminator_dropout=0.1,
                 target_classes=2, disc_lr=1e-3, disc_wd=1e-6, **kwargs):
        super().__init__()
        self.source_dims = input_dims
        self.label_keys = label_key
        num_src_dim = len(input_dims)
        if batch_wise:
            self.discriminator = BatchDiscriminator(num_src_dim, discriminator_hidden, discriminator_layers,
                                               discriminator_dropout, target_classes)
        else:
            self.discriminator = Discriminator(num_src_dim, discriminator_hidden, discriminator_layers,
                                                    discriminator_dropout, target_classes)
        self.is_adversarial = True
        self.set_d_optimizer(disc_lr, disc_wd)
        self.d_loss = 0
        self.trained = False

    def forward(self, x_dict):
        if self.training and self.trained:
            h = x_dict['h']
            y = self.discriminator(h[:, self.source_dims])
            return h, -F.cross_entropy(y, x_dict[self.label_keys])
        else:
            return x_dict['h'], 0

    def set_d_optimizer(self, lr=1e-3, wd=1e-6):
        self.d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=lr, weight_decay=wd)

    def d_step(self):
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()

    def d_iter(self, x_dict):
        y_nograd = self.discriminator(x_dict['h'][:, self.source_dims].detach())
        d_loss = F.cross_entropy(y_nograd, x_dict[self.label_keys])

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()
        self.trained = True
        return d_loss.item()
