import numpy as np
from kneed import DataGenerator, KneeLocator
from .dp_related import dp_bucketized
import torch
import matplotlib.pyplot as plt

def plot_ll_curve(gaston_model, A, S, max_domain_num=8, start_from=2):
    ll_list=get_ll_list(gaston_model, A, S, num_buckets=150, kmax=max_domain_num)
    
    kneedle=KneeLocator(np.arange(start_from,len(ll_list)+1), ll_list[start_from-1:], curve="convex", direction="decreasing")
    kneedle_opt=kneedle.knee
    print(f'Kneedle number of domains: {kneedle_opt}')

    fig,ax=plt.subplots(figsize=(4,3))
    
    plt.plot(np.arange(start_from,len(ll_list)+1), ll_list[start_from-1:])
    plt.scatter(np.arange(start_from,len(ll_list)+1), ll_list[start_from-1:])
    plt.axvline(kneedle_opt,ls='--',color='grey')
    
    plt.xlabel('Number of domains',fontsize=17)
    plt.ylabel('Negative \nlog-likelihood',fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks(np.arange(start_from,len(ll_list)+1),fontsize=15)
    
    plt.tight_layout()

def get_ll_list(model, A, S, num_buckets=150, kmax=10):
    N=A.shape[0]
    S_torch=torch.Tensor(S)
    belayer_depth=model.spatial_embedding(torch.Tensor(S)).detach().numpy().flatten()
    
    bin_endpoints=np.linspace(np.min(belayer_depth),np.max(belayer_depth)+0.01,num_buckets+1)
    error_mat,seg_map=dp_bucketized(A.T, bin_endpoints, kmax, xcoords=belayer_depth)
    return error_mat[-1,:]