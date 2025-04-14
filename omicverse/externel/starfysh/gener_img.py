import os
import sys
import numpy as np
import scipy.stats as stats
import pandas as pd
import anndata
import scanpy as sc
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,ConcatDataset
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from tqdm import tqdm

sys.path.append('HE-Net')
#torch.manual_seed(0) 




 
class dataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        spot, 
        exp_spot, 
        barcode_spot, 
        #img_size,
        #histo_img,
        transform=None
    ):

        super(dataset, self).__init__()
        self.spot = spot
        self.exp_spot = exp_spot
        self.barcode_spot = barcode_spot
        self.transform = transform
        #self.img_size = img_size
        #self.histo_img = histo_img
    
    def __getitem__(self, idx):
        #print(idx)
        spot_x = self.spot[idx,0]
        spot_y = self.spot[idx,1]
        exp_spot = self.exp_spot[idx,:]
        barcode_spot = self.barcode_spot[idx]

        #x_l = int(spot_x-self.img_size/2)
        #x_r = int(spot_x+self.img_size/2)
        #y_l = int(spot_y-self.img_size/2)
        #y_r = int(spot_y+self.img_size/2)

        #img_spot = img[y_l:y_r,x_l:x_r,:]
        #img_spot = self.histo_img[y_l:y_r,x_l:x_r]
        #img_spot = img_spot-img_spot.min()
        #img_spot = img_spot/img_spot.max()
        
        #if self.transform is not None:
        #    img_spot = self.transform(img_spot)
        return exp_spot, barcode_spot#, img_spot
        #return exp_spot
        
    def __len__(self):

        return len(self.spot)

def prep_dataset(adata):

    #library_id = list(adata.uns.get("spatial",{}).keys())[0]
    #histo_img = adata.uns['spatial'][str(library_id)]['images']['hires']
    #spot_diameter_fullres = adata.uns['spatial'][str(library_id)]['scalefactors']['spot_diameter_fullres']
    #tissue_hires_scalef = adata.uns['spatial'][str(library_id)]['scalefactors']['tissue_hires_scalef']
    #circle_radius = spot_diameter_fullres *tissue_hires_scalef * 0.5 
    image_spot = adata.obsm['spatial'] #*tissue_hires_scalef

    #img_size=32
    N_spot = image_spot.shape[0]
    train_index = np.random.choice(image_spot.shape[0], size=int(N_spot*0.6), replace=False)
    #val_index=np.random.choice(image_spot.shape[0], size=int(N_spot*0.1), replace=False)
    test_index=np.random.choice(image_spot.shape[0], size=int(N_spot*0.4), replace=True)

    #train_transforms = transforms.Compose([transforms.ToTensor()])
    #val_transforms = transforms.Compose([transforms.ToTensor()])
    #test_transforms = transforms.Compose([transforms.ToTensor()])

    train_spot = image_spot[train_index]
    test_spot = image_spot[test_index]
    #val_spot = image_spot[val_index]

    adata_df = adata.to_df()

    train_exp_spot = np.array(adata_df.iloc[train_index])
    test_exp_spot = np.array(adata_df.iloc[test_index])
    #val_exp_spot = np.array(adata_df.iloc[val_index])


    train_barcode_spot = adata_df.index[train_index]
    test_barcode_spot = adata_df.index[test_index]
    #val_barcode_spot = adata_df.index[val_index]

    train_set = dataset(train_spot, train_exp_spot, train_barcode_spot, None)
    test_set = dataset(test_spot, test_exp_spot, test_barcode_spot,None)

    all_set = dataset( image_spot,np.array(adata_df), adata_df.index, None)

    return train_set,test_set,all_set


def generate_img(dat_path, train_flag=True):
    """
    input:
    dat_path: the path for csv file

    """
    from henet import  utils
    from henet import datasets
    from henet import model
    dat_folder = 'data'
    dat_name = 'CID44971'
    n_genes = 6000
    adata1,variable_gene = utils.load_adata(dat_folder=dat_folder,
                                             dat_name=dat_name,
                                             use_other_gene=False,
                                             other_gene_list=None,
                                             n_genes=n_genes)
    

    adata5 = sc.read_csv(dat_path)
    adata5.obs_names = adata5.to_df().index
    adata5.var_names = adata5.to_df().columns
    adata5.var_names_make_unique()
    
    adata5.var["mt"] = adata5.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata5, qc_vars=["mt"], inplace=True)
    print('The datasets have',adata5.n_obs,'spots, and',adata5.n_vars,'genes')
    adata5.var['rp'] = adata5.var_names.str.startswith('RPS') + adata5.var_names.str.startswith('RPL')
    sc.pp.calculate_qc_metrics(adata5, qc_vars=['rp'], percent_top=None, log1p=False, inplace=True)
    
    # Remove mitochondrial genes
    #print('removing mt/rp genes')
    adata5 = adata5[:,-adata5.var['mt']]
    adata5 = adata5[:,-adata5.var['rp']]
    
    adata5_new = pd.DataFrame(np.zeros([adata5.to_df().shape[0],6000]),index =adata5.obs_names, columns=variable_gene )
    
    inter_gene = np.intersect1d(variable_gene,adata5.var_names)
    
    for i in inter_gene:
        adata5_new.loc[:,i] = np.array(adata5[:,i].X)
        
    adata5_new_anndata = anndata.AnnData(np.log(
                                    #np.clip(adata5_new,0,2000)
                                        adata5_new
                                        +1))
    
    other_list_temp = variable_gene.intersection(adata5.var.index)
    adata5 = adata5[:,other_list_temp]
    sc.pp.normalize_total(adata5,inplace=True)
    sc.pp.log1p(adata5)
    
    map_info_list = []
    for i in range(2500):
        x,y = np.where(np.arange(2500).reshape(50, 50)==i)
        map_info_list.append([x[0]+5,y[0]+5])   
    map_info = pd.DataFrame(map_info_list,columns=['array_row','array_col'],index = adata5.obs_names)
    
    tissue_hires_scalef = 0.20729685 
    adata5_new_anndata.obsm['spatial']=np.array(map_info)*32
    adata5.obsm['spatial']=np.array(map_info)*32
    
    
    

    train_set1,test_set1,all_set1 = datasets.prep_dataset(adata1, 'CID44971')

    train_set = ConcatDataset([train_set1,
                               
                              ])
    test_set = ConcatDataset([test_set1,
                              
                             ])


    train_dataloader = DataLoader(train_set, shuffle=True, num_workers=1, batch_size=8)
    test_dataloader= DataLoader(test_set, shuffle=True, num_workers=1, batch_size=8)
    all_dataloader1 = DataLoader(all_set1, shuffle=True, num_workers=1, batch_size=8)

    
    train_set5,test_set5,all_set5 = prep_dataset(adata5_new_anndata)
    all_dataloader5 = DataLoader(all_set5, batch_size=8)

    # Model Hyperparameters
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = 20
    x_dim  =1024
    hidden_dim = 400
    latent_dim = 20
    lr = 1e-3
    epochs = 1000

    encoder = model.Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,DEVICE=DEVICE)
    decoder = model.Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    vae = model.Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    check_point_filename = 'HE-Net/trained_model_on_lab1A.pt'
    vae.load_state_dict(torch.load(check_point_filename,map_location=torch.device('cpu') ))
    

    train_net = model.ResNet.resnet18()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    train_net.to(device)
    lr = 0.001
    #optimizer = torch.optim.SGD(oct_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(train_net.parameters(), lr=lr, amsgrad=True)
    

    vae.eval()
    if train_flag:
        for epoch in range(10):
            losses = []
            for batch_idx, (exp_spot, barcode_spot, img_spot_i) in enumerate(train_dataloader):
                #print(batch_idx)
                img_spot_i = img_spot_i.transpose(3,2).transpose(2,1)
                img_spot_i = img_spot_i.reshape(-1, 1024)

                img_spot_i = img_spot_i.to(DEVICE)

                x_hat, mean, log_var = vae(img_spot_i) 

                mean = mean.view(exp_spot.shape[0], 3, 20)
                mean = mean.view(exp_spot.shape[0], 60)
                #mean = mean.cpu().detach().numpy()
                x_hat= x_hat.view(exp_spot.shape[0], 3, 32, 32)
                img_spot_i= img_spot_i.view(exp_spot.shape[0], 3, 32, 32)

                #print(exp_spot.shape)
                output1 = train_net(exp_spot[:,None,:].to(DEVICE))
                loss_model = nn.L1Loss()
                #print(output1.flatten())  
                loss = F.mse_loss(output1.reshape(-1), mean.reshape(-1))#+0.3*loss_model(output1, y1)
                if train_flag: 
                    loss.backward() 
                    optimizer.step()
                    optimizer.zero_grad()
                losses.append(loss.cpu().detach().numpy())
            #print(np.mean(losses))
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", np.mean(losses))
            
            
    adata_test = adata5_new_anndata#adata5_new_anndata#adata5_new_anndata
    all_dataloader_test = all_dataloader5#all_dataloader5


    train_net.eval()
    #library_id = list(adata_test.uns.get("spatial",{}).keys())[0]
    img_size=32
    #tissue_hires_scalef = adata_test.uns['spatial'][str(library_id)]['scalefactors']['tissue_hires_scalef']
    #recon = np.ones(histo_img.shape)
    recon = np.ones([2000,2000,3])
    for batch_idx, (exp_spot, barcode_spot) in enumerate(all_dataloader_test):

            #img_spot_i = img_spot_i.transpose(3,2).transpose(2,1)
            #img_spot_i = img_spot_i.reshape(-1, 1024)

            #img_spot_i = img_spot_i.to(DEVICE)

            #x_hat, mean, log_var = vae(img_spot_i) 

            #mean = mean.view(exp_spot.shape[0], 3, 20)
            #mean = mean.view(exp_spot.shape[0], 60)
            #mean = mean.cpu().detach().numpy()
            #x_hat= x_hat.view(exp_spot.shape[0], 3, 32, 32)
            #img_spot_i= img_spot_i.view(exp_spot.shape[0], 3, 32, 32)

            #print(exp_spot.shape)
            output1 = train_net(exp_spot[:,None,:].to(DEVICE))
            #plt.figure()
            #plt.imshow(mean.cpu().detach())
            #plt.figure()
            #plt.imshow(output1.cpu().detach())


            output1 = output1.view(exp_spot.shape[0],3,20)
            #print(mean.shape)
            #print(output1.shape)


            output1_1 = output1[:,0,:]
            output1_2 = output1[:,1,:]
            output1_3 = output1[:,2,:]

            x_predicted_1 = vae.Decoder(output1_1)
            x_predicted_1 = x_predicted_1.view(exp_spot.shape[0],32,32).cpu().detach().numpy()
            x_predicted_2 = vae.Decoder(output1_2)
            x_predicted_2 = x_predicted_2.view(exp_spot.shape[0],32,32).cpu().detach().numpy()
            x_predicted_3 = vae.Decoder(output1_3)
            x_predicted_3 = x_predicted_3.view(exp_spot.shape[0],32,32).cpu().detach().numpy()

            for j,i in zip(range(len(barcode_spot)),barcode_spot):
                spot_x = ((adata_test[i].obsm['spatial'] )[0])[1]
                spot_y = ((adata_test[i].obsm['spatial'] )[0])[0]

                x_l = int(spot_x-img_size/2)
                x_r = int(spot_x+img_size/2)
                y_l = int(spot_y-img_size/2)
                y_r = int(spot_y+img_size/2)

                recon[y_l:y_r,x_l:x_r,0]=x_predicted_1[j]
                recon[y_l:y_r,x_l:x_r,1]=x_predicted_2[j]
                recon[y_l:y_r,x_l:x_r,2]=x_predicted_3[j]
    
    save_path = dat_path.split('/')[0]+'/'+dat_path.split('/')[1]
    if not os.path.exists(save_path+'/spatial'):
        os.makedirs(save_path+'/spatial')
    from skimage import io
    io.imsave(save_path+'/spatial/tissue_hires_image.png',(recon*255).astype('uint8'))
    map_info[['img_col','img_row']]=adata5_new_anndata.obsm['spatial']
    map_info.to_csv(save_path+'/spatial/tissue_positions_list.csv',index=True)
    adata5_new_anndata.write(save_path+'/'+dat_path.split('/')[1]+'.h5ad')

    
    return recon

