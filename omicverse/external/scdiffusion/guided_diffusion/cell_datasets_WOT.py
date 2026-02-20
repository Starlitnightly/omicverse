import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from ..VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)   # dataset already filter cells and genes

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata = adata[np.where(np.isin(adata.obs['period'], ['D0','D0.5','D1','D1.5','D2','D2.5','D3','D4.5','D5','D5.5','D6','D6.5','D7','D7.5','D8']))[0]]
    print(adata)

    label_encoder = LabelEncoder()
    label_encoder.fit(adata.obs['period'])
    label_encoder.classes_= np.array(['D0','D0.5','D1','D1.5','D2','D2.5','D3','D4.5','D5','D5.5','D6','D6.5','D7','D7.5','D8'])
    classes = label_encoder.transform(adata.obs['period'])
    print(label_encoder.classes_)

    cell_data = adata.X

    # if not train autoencoder
    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path,num_gene,hidden_dim)
        cell_data1 = autoencoder(torch.tensor(cell_data)[::2].cuda(),return_latent=True).cpu().detach().numpy()
        cell_data2 = autoencoder(torch.tensor(cell_data)[1::2].cuda(),return_latent=True).cpu().detach().numpy()
        cell_data = np.concatenate((cell_data1,cell_data2))

    classes = np.concatenate((classes[::2],classes[1::2]))
    print(cell_data.shape)

    dataset = CellDataset(
        cell_data,
        classes
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict
