import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from math import exp
import copy


class VAE(nn.Module):
    r"""
    Variational Autoencoder for bulk-to-single-cell generation.
    
    This VAE implementation is designed for learning the mapping between bulk and
    single-cell expression patterns. The model uses a beta-VAE framework with
    customizable encoder-decoder architecture for generating synthetic single cells.
    
    The architecture consists of:
    - Multi-layer encoder producing mean and variance parameters
    - Latent space sampling with reparameterization trick
    - Multi-layer decoder reconstructing single-cell expressions
    """
    
    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
        r"""
        Initialize VAE with specified architecture.

        Arguments:
            embedding_size: Input feature dimension (number of genes)
            hidden_size_list: List of hidden layer sizes for encoder/decoder
            mid_hidden: Latent space dimensionality
            
        Returns:
            None
        """
        super(VAE, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden = mid_hidden

        self.enc_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden * 2]
        self.dec_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden]

        self.encoder = nn.ModuleList(
            [nn.Linear(self.enc_feature_size_list[i], self.enc_feature_size_list[i + 1]) for i in
             range(len(self.enc_feature_size_list) - 1)])
        self.decoder = nn.ModuleList(
            [nn.Linear(self.dec_feature_size_list[i], self.dec_feature_size_list[i - 1]) for i in
             range(len(self.dec_feature_size_list) - 1, 0, -1)])

    def encode(self, x):
        r"""
        Encode input to latent parameters.
        
        Passes input through encoder layers to produce mean and log variance
        parameters for the latent distribution.

        Arguments:
            x: Input tensor of shape (batch_size, embedding_size)
            
        Returns:
            torch.Tensor: Concatenated mean and log variance parameters
        """
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        r"""
        Decode latent representation to output space.
        
        Passes latent samples through decoder layers to reconstruct
        single-cell expression profiles.

        Arguments:
            x: Latent tensor of shape (batch_size, mid_hidden)
            
        Returns:
            torch.Tensor: Reconstructed expression tensor
        """
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            x = F.relu(x)
        return x

    def forward(self, x, used_device):
        r"""
        Forward pass through VAE.
        
        Performs encoding, latent sampling, decoding, and KL divergence calculation.

        Arguments:
            x: Input expression tensor
            used_device: Device for computation (CPU/GPU)
            
        Returns:
            tuple: (reconstructed_x, kl_divergence)
        """
        x = x.to(used_device)
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        x_hat = self.decode(hidden)
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + torch.pow(mu, 2) - 1 - sigma) / (x.shape[0] * x.shape[1])
        return x_hat, kl_div

    def get_hidden(self, x):
        r"""
        Extract latent representation from input.
        
        Encodes input and samples from the latent distribution.

        Arguments:
            x: Input expression tensor
            
        Returns:
            torch.Tensor: Sampled latent representation
        """
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) * torch.exp(sigma) ** 0.5 + mu  # var => std
        return hidden


# bulk deconvolution
class BulkDataset(Dataset):
    r"""
    PyTorch Dataset for single-cell data with labels.
    
    Wraps single-cell expression data and corresponding labels for use
    with PyTorch DataLoader in VAE training.
    """
    
    def __init__(self, single_cell, label):
        r"""
        Initialize dataset with expression data and labels.

        Arguments:
            single_cell: Single-cell expression matrix
            label: Cell type labels
            
        Returns:
            None
        """
        self.sc = single_cell
        self.label = label

    def __getitem__(self, idx):
        r"""
        Get single item from dataset.

        Arguments:
            idx: Index of item to retrieve
            
        Returns:
            tuple: (expression_data, label)
        """
        tmp_x = self.sc[idx]
        tmp_y_tag = self.label[idx]

        return (tmp_x, tmp_y_tag) 

    def __len__(self):
        r"""
        Get dataset size.
        
        Returns:
            int: Number of samples in dataset
        """
        return self.label.shape[0]


def train_vae(single_cell, label, used_device, batch_size, feature_size, epoch_num, learning_rate, hidden_size, patience=10):
    r"""
    Train VAE model for bulk-to-single-cell generation.
    
    Trains a beta-VAE model using single-cell reference data with early stopping
    based on reconstruction loss. The model learns to generate realistic single-cell
    expression profiles.

    Arguments:
        single_cell: Single-cell expression matrix (cells x genes)
        label: Cell type labels for single cells
        used_device: Computation device (CPU/GPU)
        batch_size: Training batch size
        feature_size: Number of input features (genes)
        epoch_num: Maximum training epochs
        learning_rate: Optimizer learning rate
        hidden_size: Latent space dimensionality
        patience: Early stopping patience (10)

    Returns:
        tuple: (best_model, training_history)
    """
    batch_size = batch_size
    feature_size = feature_size
    epoch_num = epoch_num
    lr = learning_rate
    hidden_list = [2048, 1024, 512]

    mid_hidden_size = hidden_size
    weight_decay = 5e-4
    dataloader = DataLoader(BulkDataset(single_cell=single_cell, label=label), batch_size=batch_size, shuffle=True,
                            pin_memory=True)
    criterion = nn.MSELoss()
    beta = 4
    beta1 = 0.9
    beta2 = 0.999
    vae = VAE(feature_size, hidden_list, mid_hidden_size).to(used_device)
    optimizer = AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    pbar = tqdm(range(epoch_num))
    min_loss = 1000000000000000
    vae.train()
    early_stop = 0
    history = []

    for epoch in pbar:
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            cell_feature, label = data
            cell_feature = torch.tensor(cell_feature, dtype=torch.float32).to(used_device)
            x_recon, total_kld = vae(cell_feature, used_device)

            recon_loss = criterion(x_recon, cell_feature)
            beta_vae_loss = recon_loss + beta * total_kld
            loss = beta_vae_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if train_loss < min_loss:
            min_loss = train_loss
            best_vae = copy.deepcopy(vae)
            early_stop = 0
            epoch_final = epoch  
        else:
            early_stop += 1

        history.append(train_loss)
        if early_stop == patience:
            print('Early stopping at epoch {}...'.format(epoch+1))
            break
        pbar.set_description('Train Epoch: {}'.format(epoch))
        pbar.set_postfix(loss=f"{train_loss:.4f}", min_loss=f"{min_loss:.4f}")
    print(f"min loss = {min_loss}")
    
    return best_vae,history


def load_vae(feature_size, hidden_size, path, used_device):
    r"""
    Load pre-trained VAE model from file.
    
    Loads a previously trained VAE model with specified architecture.

    Arguments:
        feature_size: Number of input features (genes)
        hidden_size: Latent space dimensionality
        path: Path to saved model state dict
        used_device: Computation device (CPU/GPU)

    Returns:
        VAE: Loaded VAE model
    """
    hidden_list = [2048, 1024, 512]
    vae = VAE(feature_size, hidden_list, hidden_size).to(used_device)
    vae.load_state_dict(torch.load(path, map_location=used_device))
    
    return vae


def generate_vae(net, ratio, single_cell, label, breed_2_list, index_2_gene, cell_number_target_num=None,
                 used_device=None):
    r"""
    Generate synthetic single-cell data using trained VAE.
    
    Uses a trained VAE model to generate synthetic single-cell expression profiles
    matching specified cell-type target numbers.

    Arguments:
        net: Trained VAE model
        ratio: Generation ratio (usually 1)
        single_cell: Reference single-cell expression matrix
        label: Cell type labels for reference cells
        breed_2_list: List mapping label indices to cell type names
        index_2_gene: List of gene names
        cell_number_target_num: Target number of cells per type (None)
        used_device: Computation device (CPU/GPU) (None)

    Returns:
        tuple: (generated_metadata, generated_expression_data)
    """
    # net in cuda now
    for p in net.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation

    net.eval()
    net.to(used_device)
    cell_all_generate = []
    label_all_generate = []

    all_to_generate = 0
    for x in cell_number_target_num.values():
        all_to_generate += x

    if cell_number_target_num != None:
        epochs = 10000 
        ratio = 1
    else:
        epochs = 1

    cell_feature = torch.from_numpy(single_cell).float()
    label = torch.from_numpy(label)

    with torch.no_grad():
        with tqdm(total=all_to_generate, desc='generating') as pbar:
            for epoch in range(epochs):
                key_list = []  # list
                generate_num = 0

                label_list = label.tolist()
                for i in range(len(label_list)):
                    if cell_number_target_num[label_list[i]] <= 0:
                        continue
                    else:
                        cell_number_target_num[label_list[i]] -= 1
                        generate_num += 1
                        key_list.append(i)

                if cell_number_target_num == None or all_to_generate == 0 or len(key_list) == 0:
                    assert all_to_generate == 0 and len(key_list) == 0
                    break

                import random
                random.shuffle(key_list)

                label = label.index_select(0, torch.tensor(key_list))
                cell_feature = cell_feature.index_select(0, torch.tensor(key_list))

                dataloader = DataLoader(BulkDataset(single_cell=cell_feature, label=label), batch_size=300,
                                        shuffle=False,
                                        pin_memory=True, num_workers=0)
                for batch_idx, data in enumerate(dataloader):  
                    cell_feature_batch, label_batch = data
                    cell_feature_batch = cell_feature_batch.to(used_device)

                    label_batch = label_batch.cpu().numpy()

                    for j in range(ratio): 
                        ans_l, _ = net(cell_feature_batch, used_device)
                        ans_l = ans_l.cpu().data.numpy()
                        cell_all_generate.extend(ans_l)
                        label_all_generate.extend(label_batch)

                all_to_generate -= generate_num
                pbar.update(generate_num)

    print("generated done!")
    generate_sc_meta, generate_sc_data = prepare_data(cell_all_generate, label_all_generate, breed_2_list,
                                                      index_2_gene)
    return generate_sc_meta, generate_sc_data


def prepare_data(cell_all_generate, label_all_generate, breed_2_list, index_2_gene):
    r"""
    Format generated data into structured DataFrames.
    
    Converts generated expression arrays and labels into properly formatted
    DataFrames with cell and gene annotations.

    Arguments:
        cell_all_generate: Generated expression matrix
        label_all_generate: Generated cell type labels
        breed_2_list: List mapping indices to cell type names
        index_2_gene: List of gene names

    Returns:
        tuple: (metadata_dataframe, expression_dataframe)
    """
    cell_all_generate = np.array(cell_all_generate)
    label_all_generate = np.array(label_all_generate)

    cell_all_generate_csv = pd.DataFrame(cell_all_generate)
    label_all_generate_csv = pd.DataFrame(label_all_generate)

    ids = label_all_generate_csv[0].tolist()
    breeds = []
    for id in ids:
        breeds.append(breed_2_list[id])
    name = ["C_" + str(i + 1) for i in range(label_all_generate.shape[0])]

    label_all_generate_csv.insert(1, "Cell_type", np.array(breeds))
    label_all_generate_csv.insert(1, "Cell", np.array(name))
    label_all_generate_csv = label_all_generate_csv.drop([0], axis=1)

    cell_all_generate_csv = cell_all_generate_csv.T
    cell_all_generate_csv.columns = name
    cell_all_generate_csv.index = index_2_gene

    return label_all_generate_csv, cell_all_generate_csv
