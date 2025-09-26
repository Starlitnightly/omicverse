import pdb,sys,os
import anndata
import scanpy as sc
import argparse
import copy
import numpy as np
import faiss
import scipy
from sklearn.decomposition import PCA

### evaluation functions

def faiss_knn(query:np.array, x:np.array, n_neighbors:int=1) -> np.array:
    
    
    n_samples = x.shape[0]
    n_features = x.shape[1]
    x = np.ascontiguousarray(x)
    index = faiss.IndexFlatL2(n_features)
    index.add(x)
    if n_neighbors < 2:
        neighbors = 2
    else: 
        neighbors = n_neighbors
    weights, targets = index.search(query, neighbors)
    weights = weights[:,:n_neighbors]
    if -1 in targets:
        raise InternalError("Not enough neighbors were found. Please consider "
                            "reducing the number of neighbors.")
    return weights



## active learning functions 

def pick_batch_eee(reduced_bulk=None,\
                representatives=None,\
                cluster_labels=None,\
                xdim=None,\
                pseudobulk=None,\
                semis=None,\
                discount_rate = 1,\
                semi_dis_rate = 1,\
                batch_size=8\
               ):
    # 
    lhet = []
    lmp = [] 
    for i in range(len(representatives)):
        cluster_heterogeneity,in_cluster_uncertainty,uncertain_patient=compute_cluster_heterogeneity(cluster_number=i,\
                            reduced_bulk=reduced_bulk,\
                           representatives=representatives,\
                            cluster_labels=cluster_labels,\
                            xdim=xdim,\
                            pseudobulk= pseudobulk,\
                            semis=semis,\
                            discount_rate = 1,\
                            semi_dis_rate = 1\
                           )
        lhet.append(cluster_heterogeneity)
        lmp.append(uncertain_patient)
    
    new_representatives = copy.deepcopy(representatives)
    new_cluster_labels = copy.deepcopy(cluster_labels)
    #print('heterogeneities: ',lhet)
    for i in range(batch_size):
        new_num = len(new_representatives)
        mp_index = np.array(lhet).argmax()
        #print(mp_index)
        lhet[mp_index] = -999
        bestp, new_cluster_labels, hets = best_patient(cluster_labels=new_cluster_labels,representatives=new_representatives,\
                 reduced_bulk=reduced_bulk,cluster_num=mp_index,new_num=new_num)
        
        new_representatives = new_representatives + [bestp]
    
    return new_representatives,new_cluster_labels

def best_patient(cluster_labels=None,representatives=None,\
                 reduced_bulk=None,cluster_num=0,new_num=None):
    if new_num == None:
        new_num = len(representatives)
    pindices = np.where(np.array(cluster_labels)==cluster_num)[0]
    representative = representatives[cluster_num]
    hets=[]
    potential_new_labels = []
    for i in range(len(pindices)):
        potential_new_label = copy.deepcopy(cluster_labels)
        newrepre = pindices[i]
        het = 0
        if newrepre in representatives:
            hets.append(9999)
            potential_new_labels.append(potential_new_label)
            continue
        for j in range(len(pindices)):
            brepre = reduced_bulk[representative]
            brepre2 = reduced_bulk[newrepre]
            bj = reduced_bulk[pindices[j]]
            bdist1 = (brepre - bj)**2
            bdist1 = bdist1.sum()
            bdist1 = bdist1**0.5
            bdist2 = (brepre2 - bj)**2
            bdist2 = bdist2.sum()
            bdist2 = bdist2**0.5
            
            if bdist1 > bdist2:
                het = het + bdist2
                potential_new_label[pindices[j]]=new_num
            else:
                het = het + bdist1
        hets.append(het)
        potential_new_labels.append(potential_new_label)
    hets = np.array(hets)
    bestp = pindices[np.argmin(hets)]
    new_cluster_labels = potential_new_labels[np.argmin(hets)]
    return bestp, new_cluster_labels, hets

def update_membership(reduced_bulk=None,\
                      representatives=None,\
                      
                     ):
    new_cluster_labels = []
    for i in range(len(reduced_bulk)):
        
        dists=[]
        #dist to repres
        for j in representatives:
            bdist = (reduced_bulk[j] - reduced_bulk[i])**2 
            bdist = bdist.sum()
            bdist = bdist**0.5
            dists.append(bdist)
        membership = np.array(dists).argmin()
        new_cluster_labels.append(membership)
    return new_cluster_labels

def compute_cluster_heterogeneity(cluster_number=0,\
                            reduced_bulk=None,\
                           representatives=None,\
                            cluster_labels=None,\
                            xdim=None,\
                            pseudobulk=None,\
                            semis=None,\
                            discount_rate = 1,\
                            semi_dis_rate = 1,\
                           ):
    semiflag=0
    representative = representatives[cluster_number]
    in_cluster_uncertainty = []
    cluster_labels = np.array(cluster_labels)
    cluster_patient_indices = np.where(cluster_labels==cluster_number)[0]
    
    for i in range(len(cluster_patient_indices)): # number of patients in this cluster except the representative
        
        patient_index = cluster_patient_indices[i]
        
        if patient_index in representatives:
            in_cluster_uncertainty.append(0)
            continue
            
        # distance between this patient and representative
        bdist = (reduced_bulk[representative] - reduced_bulk[patient_index])**2 
        bdist = bdist.sum()
        bdist = bdist**0.5
        
        ma = np.array(xdim[patient_index]).copy(order='C')
        mb = np.array(xdim[representative]).copy(order='C')
        sdist = (faiss_knn(ma,mb,n_neighbors=1).mean())
        

        semiloss = np.log(1+pseudobulk[patient_index]) - np.log(1+semis[patient_index].mean(axis=0))
        semiloss = semiloss**2
        semiloss = semiloss.sum()
        semiloss = semiloss**0.5
        
        uncertainty = bdist + sdist*discount_rate + semi_dis_rate * semiloss
        
        in_cluster_uncertainty.append(uncertainty)
        
    cluster_heterogeneity = np.array(in_cluster_uncertainty).sum()
    uncertain_patient = cluster_patient_indices[np.array(in_cluster_uncertainty).argmax()] 

    return cluster_heterogeneity,in_cluster_uncertainty,uncertain_patient



def activeselection(name:str, representatives:str,cluster:str,batch:int,lambdasc:float,lambdapb:float) -> None:
    """
    Use active learning to select the next batch of representatives
    
    Parameters
    ----------
    name 
        Project name.
    representatives
        Path to a `.txt` file specifying the representatives.
    cluster
        Path to a `.txt` file specifying the cluster labels.
    batch
        Representative selection batch size.
    lambdasc
        Scaling factor for the single-cell transformation difficulty from the representative to the target.
    lambdapb
        Scaling factor for the pseudobulk data.difference. 
    
    Returns
    -------
        None
    
    Example
    -------
    >>> name = 'project_name'
    >>> representatives = name + '/status/init_representatives.txt'
    >>> cluster = name + '/status/init_cluster_labels.txt'
    >>> semidev.activeselection(name, representatives,cluster,batch=2,lambdasc=1,lambdapb=1)
    
    """
    
    
    print('Running active learning to select new representatives')
    
    sids = []
    f = open(name + '/sids.txt', 'r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    if representatives[-3:]=='txt':
        rep = []
        f = open(representatives,'r')
        lines = f.readlines()
        for l in lines:
            rep.append(int(l.strip()))
        f.close()

    if cluster[-3:]=='txt':
        cl=[]
        f = open(cluster,'r')
        lines = f.readlines()
        for l in lines:
            cl.append(int(l.strip()))
        f.close()
    
    bulkdata = anndata.read_h5ad(name + '/processed_bulkdata.h5ad')
    reduced_bulk = bulkdata.obsm['X_pca']
    
    #acquire semi-profiled cohort

    hvgenes = np.load(name+'/hvgenes.npy',allow_pickle=True)
    
    genelen = len(hvgenes)
    
    
    xs = []
    datalen = []
    for i in range(len(sids)):
        if i not in rep:
            sid = sids[i]
            representative = rep[cl[i]]
            x = np.load(name + '/inferreddata/'+sids[representative]+'_to_'+sid+'.npy')
            xs.append(np.log(x+1))
            datalen.append(x.shape[0])
        else:
            sid = sids[i]
            adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad')
            x = np.array(adata.X[:,:genelen])
            xs.append(x)
            datalen.append(x.shape[0])
    
    xs = np.concatenate(xs, axis=0)
    
    
    pca = PCA(n_components=100)
    xpcas = pca.fit_transform(xs)
    
    xpca = []
    semis = []
    offset = 0
    for i in range(len(sids)):
        xpca.append(xpcas[offset:offset+datalen[i],:])
        semis.append(xs[offset:offset+datalen[i],:])
        offset = offset + datalen[i]
    
    bdata = anndata.read_h5ad(name+'/processed_bulkdata.h5ad')
    pseudobulk = np.exp(bdata.X) - 1
    
    nrep, nlabels = pick_batch_eee(reduced_bulk = reduced_bulk,\
                    representatives = rep,\
                    cluster_labels = cl,\
                    xdim=xpca,\
                    pseudobulk = pseudobulk,\
                    semis=semis,\
                    discount_rate = lambdasc,\
                    semi_dis_rate = lambdapb,\
                    batch_size=batch\
                   )
    
    new_representatives = nrep
    new_cluster_labels = nlabels
    
    rnd = len(os.listdir(name + '/status'))//2+1
    
    f=open(name + '/status/eer_cluster_labels_'+str(rnd)+'.txt','w')
    for i in range(len(new_cluster_labels)):
        f.write(str(new_cluster_labels[i])+'\n')
    f.close()
    f=open(name + '/status/eer_representatives_'+str(rnd)+'.txt','w')
    for i in range(len(new_representatives)):
        f.write(str(new_representatives[i])+'\n')
        
    print('selection finished')
    f.close()



def main():
    parser=argparse.ArgumentParser(description="Selecting new representatives using active learning")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--representatives',required=True,help="A txt file including all the IDs of the representatives used in the current round of semi-profiling.")
    
    required.add_argument('--cluster',required=True,help="A txt file specifying the cluster membership.")
    
    required.add_argument('--name',required=True,help="Project name.")
    
    optional.add_argument('--batch',required=False, default='4', help="The batch size of representative selection (Default: 4)")
    
    optional.add_argument('--lambdasc',required=False,default='1.0', help="Scaling factor for the single-cell transformation difficulty from the representative to the target (Default: 1.0)")
    
    optional.add_argument('--lambdapb',required=False, default='1.0', help="Scaling factor for the pseudobulk data difference (Default: 1.0)")
    
    args = parser.parse_args()
    representatives = args.representatives
    cluster = args.cluster
    name = args.name
    batch = int(args.batch)
    lambdasc = float(args.lambdasc)
    lambdapb = float(args.lambdapb)
    activeselection(name, representatives,cluster,batch,lambdasc,lambdapb)

if __name__=="__main__":
    main()
