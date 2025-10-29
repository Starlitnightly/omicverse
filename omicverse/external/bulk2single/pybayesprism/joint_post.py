import xarray as xr
import numpy as np
import pandas as pd

class JointPost:
    def __init__(self, Z, theta, theta_cv = None, constant = None): 
        self.Z = Z
        self.theta = theta
        self.theta_cv = theta_cv
        self.constant = constant


    def new(bulk_id, gene_id, cell_type, gibbs_list):
        n = len(bulk_id)
        h = len(gene_id)
        k = len(cell_type)
        assert len(gibbs_list) == n

        Z = np.zeros((n, h, k))
        theta = np.zeros((n, k))
        theta_cv = np.zeros((n, k))

        for i in range(n):
            Z[i,:,:] = gibbs_list[i]['Z_n']
            theta[i,:] = gibbs_list[i]['theta_n']

        if 'theta.cv_n' in gibbs_list[0] and gibbs_list[0]['theta.cv_n'] is not None:
            for i in range(n):
                theta_cv[i,:] = gibbs_list[i]['theta.cv_n']
        
        constant = sum(subdict["gibbs.constant"] for subdict in gibbs_list)

        Z = xr.DataArray(Z, coords=[bulk_id, gene_id, cell_type],
                         dims=['bulk_id', 'gene_id', 'cell_type'])
        theta = pd.DataFrame(theta, index = bulk_id, columns = cell_type)
        theta_cv = pd.DataFrame(theta_cv, index = bulk_id, columns = cell_type)

        return JointPost(Z, theta, theta_cv, constant)


    def merge_K(self, map_ : dict):
        bulk_id = self.Z.coords['bulk_id'].values
        gene_id = self.Z.coords['gene_id'].values
        cell_type = self.Z.coords['cell_type'].values
        cell_type_merged = list(map_.keys())

        n = len(bulk_id)
        g = len(gene_id)
        k = len(cell_type)
        k_merged = len(cell_type_merged)

        assert sum([len(v) for v in map_.values()]) == k

        Z = np.zeros((n, g, k_merged))
        theta = np.zeros((n, k_merged))


        Z = xr.DataArray(Z, coords=[('bulk_id', bulk_id), ('gene_id', gene_id), ('cell_type_merged', cell_type_merged)])
        theta = pd.DataFrame(theta, index = bulk_id, columns = cell_type_merged)

        for i in range(k_merged):
            cell_type_merged_k = cell_type_merged[i]
            cell_types_k = map_[cell_type_merged_k]
            if len(cell_types_k) == 1:
                Z.loc[:, :, cell_type_merged_k] = np.squeeze(self.Z.loc[:, :, cell_types_k])
                theta.loc[:, cell_type_merged_k] = np.squeeze(self.theta.loc[:, cell_types_k])
            else:
                Z.loc[:, :, cell_type_merged_k] = np.sum(self.Z.loc[:, :, cell_types_k], axis = 2)
                theta.loc[:, cell_type_merged_k] = np.sum(self.theta.loc[:, cell_types_k], axis = 1)

        return JointPost(Z, theta)