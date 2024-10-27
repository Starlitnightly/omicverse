import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def crop_space_visium(adata,crop_loc,crop_area,
               library_id,scale,spatial_key='spatial',res='hires'):
    import squidpy as sq
    adata1=adata.copy()
    img = sq.im.ImageContainer(
        adata1.uns["spatial"][library_id]["images"][res], library_id=library_id
    )
    crop_corner = img.crop_corner(crop_loc[0], crop_loc[1], size=crop_area,scale=scale,)
    adata1.obsm['spatial1']=adata1.obsm[spatial_key]*\
                adata1.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop = crop_corner.subset(adata1,spatial_key='spatial1')
    adata_crop.uns["spatial"][library_id]["images"][res]=np.squeeze(crop_corner['image'].data,axis=2)
    adata_crop.obsm[spatial_key][:,0]=(adata_crop.obsm['spatial1'][:,0]-crop_loc[1])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    adata_crop.obsm[spatial_key][:,1]=(adata_crop.obsm['spatial1'][:,1]-crop_loc[0])/adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    
    return adata_crop
