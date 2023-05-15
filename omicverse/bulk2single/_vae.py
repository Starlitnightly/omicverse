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
    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
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
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            x = F.relu(x)
        return x

    def forward(self, x, used_device):
        x = x.to(used_device)
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        x_hat = self.decode(hidden)
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + torch.pow(mu, 2) - 1 - sigma) / (x.shape[0] * x.shape[1])
        return x_hat, kl_div

    def get_hidden(self, x):
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) * torch.exp(sigma) ** 0.5 + mu  # var => std
        return hidden


# bulk deconvolution
class BulkDataset(Dataset):  
    def __init__(self, single_cell, label):
        self.sc = single_cell
        self.label = label

    def __getitem__(self, idx):
        tmp_x = self.sc[idx]
        tmp_y_tag = self.label[idx]

        return (tmp_x, tmp_y_tag) 

    def __len__(self):
        return self.label.shape[0]


def train_vae(single_cell, label, used_device, batch_size, feature_size, epoch_num, learning_rate, hidden_size, patience=10):
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
    hidden_list = [2048, 1024, 512]
    vae = VAE(feature_size, hidden_list, hidden_size).to(used_device)
    vae.load_state_dict(torch.load(path, map_location=used_device))
    
    return vae


def generate_vae(net, ratio, single_cell, label, breed_2_list, index_2_gene, cell_number_target_num=None,
                 used_device=None):
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
