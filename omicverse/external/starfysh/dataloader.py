import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ._starfysh import LOGGER


#---------------------------
# Single Sample dataloader
#---------------------------

class VisiumDataset(Dataset):
    """
    Loading a single preprocessed ST AnnData, gene signature & Anchor spots for Starfysh training
    """

    def __init__(
        self,
        adata,
        args,
    ):
        spots = adata.obs_names
        genes = adata.var_names

        x = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A
        self.expr_mat = pd.DataFrame(x, index=spots, columns=genes)
        self.gexp = args.sig_mean_norm
        self.anchor_idx = args.pure_idx
        self.library_n = args.win_loglib

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )

        return (sample,
                torch.Tensor(self.gexp.iloc[idx, :]),  # normalized signature exprs
                torch.Tensor(self.anchor_idx[idx, :]),  # anchors
                torch.Tensor(self.library_n[idx,None]),  # library size
               )


class VisiumPoEDataSet(VisiumDataset):
    
    def __init__(
        self,
        adata,
        args,
    ):
        super(VisiumPoEDataSet, self).__init__(adata, args)
        self.image = args.img.astype(np.float64)
        self.map_info = args.map_info
        self.r = args.params['patch_r']
        self.spot_img_stack = []

        self.density_std = args.img.std()

        assert self.image is not None,\
            "Empty paired H&E image," \
            "please use regular `Starfysh` without PoE integration" \
            "if your dataset doesn't contain histology image"

        # Retrieve image patch around each spot
        scalef = args.scalefactor['tissue_hires_scalef']  # High-res scale factor
        h, w = self.image.shape[:2]
        patch_dim = (self.r*2, self.r*2, 3) if self.image.ndim == 3 else (self.r*2, self.r*2)

        for i in range(len(self.expr_mat)):
            xc = int(np.round(self.map_info.iloc[i]['imagecol'] * scalef))
            yc = int(np.round(self.map_info.iloc[i]['imagerow'] * scalef))

            # boundary conditions: edge spots
            yl, yr = max(0, yc-self.r), min(self.image.shape[0], yc+self.r)
            xl, xr = max(0, xc-self.r), min(self.image.shape[1], xc+self.r)
            top = max(0, self.r-yc)
            bottom = h if h > (yc+self.r) else h-(yc+self.r)
            left = max(0, self.r-xc)
            right = w if w > (xc+self.r) else w-(xc+self.r)

            #try:
            patch = np.zeros(patch_dim)
            patch[top:bottom, left:right] = self.image[yl:yr, xl:xr]
            self.spot_img_stack.append(patch)
            #except ValueError:
            #    LOGGER.warning('Skipping the patch loading of an edge spot...')


    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )
        spot_img_stack = self.spot_img_stack[idx]
        return (sample,
                torch.Tensor(self.anchor_idx[idx, :]),
                torch.Tensor(self.library_n[idx, None]),
                spot_img_stack,
                self.map_info.index[idx],
                torch.Tensor(self.gexp.iloc[idx, :]),
               )
    
#---------------------------
# Integrative Dataloader
#--------------------------- 


class IntegrativeDataset(VisiumDataset):
    """
    Loading multiple preprocessed ST sample AnnDatas, gene signature & Anchor spots for Starfysh training
    """

    def __init__(
        self,
        adata,
        args,
    ):
        super(IntegrativeDataset, self).__init__(adata, args)
        self.image = args.img
        self.map_info = args.map_info
        self.r = args.params['patch_r']

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )
        return (sample,   
                torch.Tensor(self.gexp.iloc[idx, :]),
                torch.Tensor(self.anchor_idx[idx, :]),
                torch.Tensor(self.library_n[idx, None])
               )
    
    
class IntegrativePoEDataset(VisiumDataset):

    def __init__(
        self,
        adata,
        args,
    ):
        super(IntegrativePoEDataset, self).__init__(adata, args)
        self.image = args.img
        self.map_info = args.map_info
        self.r = args.params['patch_r']
        

        assert self.image is not None,\
            "Empty paired H&E image," \
            "please use regular `Starfysh` without PoE integration" \
            "if your dataset doesn't contain histology image"
        spot_img_all = []
        
        # Retrieve image patch around each spot
        for sample_id in args.img.keys():
            
            scalef_i = args.scalefactor[sample_id]['tissue_hires_scalef']  # High-res scale factor
            h, w = self.image[sample_id].shape[:2]
            patch_dim = (self.r*2, self.r*2, 3) if self.image[sample_id].ndim == 3 else (self.r*2, self.r*2)

            list_ = adata.obs['sample'] == sample_id
            for i in range(len(self.expr_mat.loc[list_,:])):
                xc = int(np.round(self.map_info.loc[list_,:].iloc[i]['imagecol'] * scalef_i))
                yc = int(np.round(self.map_info.loc[list_,:].iloc[i]['imagerow'] * scalef_i))

                # boundary conditions: edge spots
                yl, yr = max(0, yc-self.r), min(self.image[sample_id].shape[0], yc+self.r)
                xl, xr = max(0, xc-self.r), min(self.image[sample_id].shape[1], xc+self.r)
                top = max(0, self.r-yc)
                bottom = h if h > (yc+self.r) else h-(yc+self.r)
                left = max(0, self.r-xc)
                right = w if w > (xc+self.r) else w-(xc+self.r)
            
                try:
                    patch = np.zeros(patch_dim)
                    patch[top:bottom, left:right] = self.image[sample_id][yl:yr, xl:xr]
                    spot_img_all.append(patch)
                except ValueError:
                    LOGGER.warning('Skipping the patch loading of an edge spot...')                
                
        self.spot_img_stack = list(spot_img_all)
        #print(self.spot_img_stack.shape)

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )
        return (sample,   
                torch.Tensor(self.anchor_idx[idx, :]),
                torch.Tensor(self.library_n[idx, None]),
                self.spot_img_stack[idx],
                self.map_info.index[idx],
                torch.Tensor(self.gexp.iloc[idx, :]),
               )
    
