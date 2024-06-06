from __future__ import division
from warnings import warn
import numpy as np
import scipy as s
import pandas as pd
import numpy.ma as ma
import os
import h5py
from ..core.nodes import *

# To keep same order of views and groups in the hdf5 file
# h5py.get_config().track_order = True

def _make_unique(x):
    """Make values in the input list unique"""
    from collections import Counter
    xs = list(x)
    y = list()
    c = Counter()
    for item in xs:
        new_item = item
        if xs.count(item) > 1:
            if c[item] == 0:
                # First instance stays as it is
                y.append(new_item)
                c[item] += 1
                continue
            while new_item in x:
                new_item = '-'.join([item, str(c[item])])
                c[item] += 1
        y.append(new_item)
    return np.array(y)

class saveModel():
    def __init__(self, model, outfile, data, sample_cov, intercepts, samples_groups,
        train_opts, model_opts, features_names, views_names, samples_names, groups_names, covariates_names,
        samples_metadata, features_metadata, 
        sort_factors=True, compression_level=9):

        # Check that the model is trained
        # NOTE: it might be not trained if saving when training is interrupted
        if not model.trained:
            print("Note: the model to be saved is not trained.")
        self.model = model

        # Initialise hdf5 file
        self.hdf5 = h5py.File(outfile,'w')
        self.file = outfile
        self.compression_level = compression_level

        # Initialise training data
        self.data = data

        # Define masks
        self.mask = [ x.mask for x in model.getNodes()["Y"].getNodes() ]

        # Initialise samples groups
        assert len(samples_groups) == data[0].shape[0], "length of samples groups does not match the number of samples in the data"
        self.samples_groups = samples_groups
        
        # Initialise GP prior (note groups are concatenated here)
        if not sample_cov is None:
            assert sample_cov.shape[0] == data[0].shape[0], "length of samples covariates does not match the number of samples in the data"
        self.sample_cov = sample_cov

        # Initialise intercepts
        self.intercepts = intercepts

        # Initialise options
        self.train_opts = train_opts
        self.model_opts = model_opts

        # Initialise dimension names
        self.views_names = views_names
        self.samples_names = samples_names
        self.features_names = features_names
        self.groups_names = groups_names
        if not self.sample_cov is None:
            self.covariates_names = covariates_names
        else:
            self.covariates_names = None

        # Initialise metadata
        self.samples_metadata = samples_metadata
        self.features_metadata = features_metadata

        # calculate variance explained 
        self.r2 = self.model.calculate_variance_explained()

        self.order_factors = self.sort_factors(sort_factors)

    def sort_factors(self, sort_factors):
        if sort_factors:
            order_factors = np.argsort( np.array(self.r2).sum(axis=(0,1), where= ~np.isnan(np.array(self.r2))) )[::-1]
            self.r2 = [x[:,order_factors] for x in self.r2]
        else:
            order_factors = np.arange(self.r2[0].shape[1])
        return order_factors

    def saveNames(self):
        """ Method to save sample and feature names"""

        dt = h5py.string_dtype(encoding='utf-8')
        
        # Save group names
        groups_grp = self.hdf5.create_group("groups")
        groups_grp.create_dataset("groups", data=np.array(self.groups_names, dtype=dt))        

        # Save views names
        views_grp = self.hdf5.create_group("views")
        views_grp.create_dataset("views", data=np.array(self.views_names, dtype=dt))

        # Save samples names
        samples_grp = self.hdf5.create_group("samples")
        for g in range(len(self.groups_names)):
            samples_grp.create_dataset(self.groups_names[g], data=np.array(self.samples_names[g], dtype=dt))

        # Save feature names
        features_grp = self.hdf5.create_group("features")
        for m in range(len(self.data)):
            features_grp.create_dataset(self.views_names[m], data=np.array(self.features_names[m], dtype=dt))

        # Save covariate names
        if not self.covariates_names is None:
            covariates_grp = self.hdf5.create_group("covariates")
            covariates_grp.create_dataset("covariates", data=np.array(self.covariates_names, dtype=dt))

    def saveMetaData(self):
        """ Method to save samples and features metadata """

        # Save samples metadata
        if self.samples_metadata:
            samples_meta = self.hdf5.create_group("samples_metadata")
            for g in range(len(self.groups_names)):
                group_meta = samples_meta.create_group(self.groups_names[g])
                cols = self.samples_metadata[g].columns
                
                if len(set(cols)) != len(cols):
                    warn("There are duplicated columns in samples metadata, some will be renamed")
                    cols = _make_unique(cols)
                    self.samples_metadata[g].columns = cols
                
                cols_cat = cols[self.samples_metadata[g].dtypes == "category"]
                
                for col in cols:
                    # Convert categorical columns
                    if col in cols_cat:
                        orig_type = self.samples_metadata[g][col].cat.categories.values.dtype
                        self.samples_metadata[g][col] = self.samples_metadata[g][col].astype(orig_type)

                    ctype = self.samples_metadata[g][col].dtype
                    
                    if ctype == "object":
                        try:
                            # Try to encode as ASCII strings
                            group_meta.create_dataset(col, data=np.array(self.samples_metadata[g][col], dtype="|S"))
                        except (UnicodeEncodeError, SystemError):
                            # Encode strings as Unicode
                            group_meta.create_dataset(col, data=np.char.encode(self.samples_metadata[g][col].values.astype("U"), encoding='utf8'))
                    else:
                        group_meta.create_dataset(col, data=np.array(self.samples_metadata[g][col], dtype=ctype.type))
                # # Store objects as strings
                # for col in cols[self.samples_metadata[g].dtypes == "object"]:
                #     self.samples_metadata[g][col] = self.samples_metadata[g][col].astype("|S")
                # types = [(cols[i], self.samples_metadata[g][k].dtype.type) for (i, k) in enumerate(cols)]
                # samples_meta.create_dataset(self.groups_names[g], data=np.array(self.samples_metadata[g]).astype(types), dtype=types)

        # Save features metadata
        if self.features_metadata:
            features_meta = self.hdf5.create_group("features_metadata")
            for m in range(len(self.views_names)):
                view_meta = features_meta.create_group(self.views_names[m])
                cols = self.features_metadata[m].columns
            
                if len(set(cols)) != len(cols):
                    warn("There are duplicated columns in features metadata, some will be renamed")
                    cols = _make_unique(cols)
                    self.features_metadata[m].columns = cols
                    
                cols_cat = cols[self.features_metadata[m].dtypes == "category"]
                
                for col in cols:
                    # Convert categorical columns
                    if col in cols_cat:
                        orig_type = self.features_metadata[m][col].cat.categories.values.dtype
                        self.features_metadata[m][col] = self.features_metadata[m][col].astype(orig_type)

                    ctype = self.features_metadata[m][col].dtype
                    ctype = '|S' if ctype == "object" else ctype.type
                    view_meta.create_dataset(col, data=np.array(self.features_metadata[m][col], dtype=ctype))
                # types = [(cols[i], self.features_metadata[m][k].dtype) for (i, k) in enumerate(cols)]
                # # Store objects as strings
                # types = [(col, ctype.type) if ctype != "object" else (col, np.str) for col, ctype in types]
                # features_meta.create_dataset(self.views_names[m], data=np.array(self.features_metadata[m]).astype(types), dtype=types)

    def saveData(self):
        """ Method to save the training data"""
        
        # Create HDF5 groups
        data_grp = self.hdf5.create_group("data")
        intercept_grp = self.hdf5.create_group("intercepts")

        for m in range(len(self.data)):
            data_subgrp = data_grp.create_group(self.views_names[m])
            intercept_subgrp = intercept_grp.create_group(self.views_names[m])
            for g in range(len(self.groups_names)):

                # Subset group
                samples_idx = np.where(np.array(self.samples_groups) == self.groups_names[g])[0]
                tmp = self.data[m][samples_idx,:]

                # Mask missing values
                tmp[self.mask[m][samples_idx,:]] = np.nan
                
                # Create hdf5 data set for data
                data_subgrp.create_dataset(self.groups_names[g], data=tmp, compression="gzip", compression_opts=self.compression_level)
                
                # Create hdf5 data set for intercepts
                intercept_subgrp.create_dataset(self.groups_names[g], data=self.intercepts[m][g])
        
        # Save sample covariates for GP prior
        if not self.sample_cov is None:
            cov_samples_grp = self.hdf5.create_group("cov_samples")
            cov_samples_transformed_grp = self.hdf5.create_group("cov_samples_transformed")
            if 'Sigma' in self.model.getNodes():
                sample_cov_transformed = self.model.getNodes()['Sigma'].sample_cov_transformed
            else:
                sample_cov_transformed = None

            for g in range(len(self.groups_names)):
                samples_idx = np.where(np.array(self.samples_groups) == self.groups_names[g])[0]
                tmp = self.sample_cov[samples_idx, :]
                # Create hdf5 data set for data
                cov_samples_grp.create_dataset(self.groups_names[g], data=tmp, compression="gzip",
                                           compression_opts=self.compression_level)
                if not sample_cov_transformed is None:
                    tmp_transformed = sample_cov_transformed[samples_idx, :]
                    cov_samples_transformed_grp.create_dataset(self.groups_names[g], data=tmp_transformed, compression="gzip",
                                           compression_opts=self.compression_level)

    def saveImputedData(self, mean, variance):
        """ Method to save the training data"""
        
        # Create HDF5 groups
        data_grp = self.hdf5.create_group("imputed_data")

        # Save mean
        for m in range(len(mean)):
            view_subgrp = data_grp.create_group(self.views_names[m])
            for g in range(len(self.groups_names)):

                # Subset group
                samples_idx = np.where(np.array(self.samples_groups) == self.groups_names[g])[0]

                # Create HDF5 subgroup
                group_subgrp = view_subgrp.create_group(self.groups_names[g])

                # Create hdf5 data sets for the mean and the variance
                group_subgrp.create_dataset("mean", data=mean[m][samples_idx,:], compression="gzip", compression_opts=self.compression_level)
                if variance is not None:
                    group_subgrp.create_dataset("variance", data=variance[m][samples_idx,:], compression="gzip", compression_opts=self.compression_level)

    def saveZpredictions(self, mean, variance, values, groups):
        """ Method to save the training data"""

        # Create HDF5 groups
        data_grp = self.hdf5.create_group("Z_predictions")
        # save values
        data_grp.create_dataset("new_values", data=values, compression="gzip",
                                    compression_opts=self.compression_level)
        for g in groups:
            # Create HDF5 subgroup
            group_subgrp = data_grp.create_group(self.groups_names[g])
            sampleidx = np.arange(values.shape[0]) + values.shape[0] * g

            # Save mean
            group_subgrp.create_dataset("mean", data=mean[sampleidx,:][:, self.order_factors], compression="gzip",
                                        compression_opts=self.compression_level)


            # Save variance
            if variance is not None:
                group_subgrp.create_dataset("variance", data=variance[sampleidx,:][:, self.order_factors], compression="gzip",
                                            compression_opts=self.compression_level)

    def saveExpectations(self, nodes="all"):

        # Get nodes from the model
        nodes_dic = self.model.getNodes()
        if type(nodes) is str:
            nodes = list(nodes_dic.keys()) if nodes=="all" else [nodes]
        elif type(nodes) is list or type(nodes) is tuple:
            assert set(nodes).issubset(["Z","W","Y","Tau","AlphaW","AlphaZ","ThetaZ","ThetaW", "Sigma", "U"]), "Unrecognised nodes"
        nodes_dic = {x: nodes_dic[x] for x in nodes if x in nodes_dic}

        # Define nodes with special characteristics 
        # (note that this code is ugly and is not proper class-oriented programming)
        multigroup_nodes = ["Y", "Tau", "Z"]
        multigroup_factors_nodes = ["AlphaZ", "ThetaZ"]
        # multiview_nodes = ["Y","Tau","Alpha","W"]

        # Create HDF5 group
        grp = self.hdf5.create_group("expectations")

        # Iterate over nodes
        for n in nodes_dic:
            # Create subgroup for the node
            node_subgrp = grp.create_group(n)

            # Collect node expectation
            exp = nodes_dic[n].getExpectation()
            # for saving of higher moments:
            # exp = nodes_dic[n].getExpectations()

            # Multi-view nodes
            if isinstance(nodes_dic[n], Multiview_Node):
                for m in range(nodes_dic[n].M):

                    # Multi-groups nodes (Tau, Y, and Z)
                    if n in multigroup_nodes:

                        # Create subgroup for the view
                        view_subgrp = node_subgrp.create_group(self.views_names[m])
                        
                        for g in self.groups_names:

                            # Add missing values to Tau and Y nodes
                            exp[m][self.mask[m]] = np.nan

                            # create hdf5 data set for the expectation
                            samp_indices = np.where(np.array(self.samples_groups) == g)[0]

                            view_subgrp.create_dataset(g, data=exp[m][samp_indices,:], compression="gzip", compression_opts=self.compression_level)

                    # Single-groups nodes (W)
                    else:
                        foo = exp[m].T
                        node_subgrp.create_dataset(self.views_names[m], data=foo[self.order_factors], compression="gzip", compression_opts=self.compression_level)

            # Single-view nodes
            else:

                # Multi-group nodes (Z)
                if n in multigroup_nodes:
                    for g in self.groups_names:
                        samp_indices = np.where(np.array(self.samples_groups) == g)[0]
                        foo = exp[samp_indices,:].T
                        node_subgrp.create_dataset(g, data=foo[self.order_factors], compression="gzip", compression_opts=self.compression_level)

                # Multi-group nodes with no samples but only factors (AlphaZ, ThetaZ)
                elif n in multigroup_factors_nodes:
                    for gi, g in enumerate(self.groups_names):
                        foo = exp[gi].T
                        node_subgrp.create_dataset(g, data=foo[self.order_factors], compression="gzip", compression_opts=self.compression_level)

                # Multi-group nodes with no samples but only factors (AlphaZ, ThetaZ)
                elif n in multigroup_factors_nodes:
                    for gi, g in enumerate(self.groups_names):
                        foo = exp[gi].T
                        node_subgrp.create_dataset(g, data=foo[self.order_factors], compression="gzip", compression_opts=self.compression_level)

                # Single-group nodes (Sigma)
                else:
                    node_subgrp.create_dataset("E", data=exp.T[:,:,self.order_factors], compression="gzip", compression_opts=self.compression_level)
        pass

    def saveParameters(self, nodes="all"):
        print("saveParameters() is currently depreciated, TO-DO: sort factors"); exit()

        # Get nodes from the model
        nodes_dic = self.model.getNodes()
        if type(nodes) is str:
            nodes = list(nodes_dic.keys()) if nodes=="all" else [nodes]
        elif type(nodes) is list or type(nodes) is tuple:
            assert set(nodes).issubset(["Z","W","Tau","AlphaW","AlphaZ","ThetaZ","ThetaW", "Sigma", "U"]), "Unrecognised nodes"
        nodes_dic = {x: nodes_dic[x] for x in nodes if x in nodes_dic}

        # Define nodes which special characteristics 
        # (note that this is ugly and is not proper class-oriented programming)
        multigroup_nodes = ["Y","Tau","Z"]

        # Create HDF5 group
        grp = self.hdf5.create_group("parameters")

        # Iterate over nodes
        for n in nodes_dic:
            # Create subgroup for the node
            node_subgrp = grp.create_group(n)

            # Collect node parameters
            par = nodes_dic[n].getParameters()

            # Multi-view nodes
            if isinstance(nodes_dic[n],Multiview_Node):
                for m in range(nodes_dic[n].M):

                    # Create subgroup for the view
                    view_subgrp = node_subgrp.create_group(self.views_names[m])

                    # Multi-groups nodes
                    if n in multigroup_nodes:

                        for g in self.groups_names:
                            grp_subgrp = view_subgrp.create_group(g)

                            # create hdf5 data set for the parameter
                            samp_indices = np.where(np.array(self.samples_groups) == g)[0]

                            for k in par[m].keys():
                                tmp = par[m][k][samp_indices,:]
                                grp_subgrp.create_dataset(k, data=tmp, compression="gzip", compression_opts=self.compression_level)

                    # Single-groups nodes
                    else:
                        for k in par[m].keys():
                            if k not in ["mean_B0","var_B0"]:
                                tmp = par[m][k].T
                                view_subgrp.create_dataset(k, data=tmp, compression="gzip", compression_opts=self.compression_level)

            # Single-view nodes
            else:

                # Multi-group nodes
                if n in multigroup_nodes:
                    for g in self.groups_names:
                        grp_subgrp = node_subgrp.create_group(g)
                        samp_indices = np.where(np.array(self.samples_groups) == g)[0]

                        for k in par.keys():
                            tmp = par[k][samp_indices,:].T
                            grp_subgrp.create_dataset(k, data=tmp, compression="gzip", compression_opts=self.compression_level)

                # Single-group nodes
                else:
                    for k in par.keys():
                        node_subgrp.create_dataset(k, data=par[k].T, compression="gzip", compression_opts=self.compression_level)

        pass

    def saveModelOptions(self):

        # Subset model options
        options_to_save = ["likelihoods", "spikeslab_factors", "spikeslab_weights", "ard_factors", "ard_weights"]
        opts = dict((k, np.asarray(self.model_opts[k]).astype('S')) for k in options_to_save)

        # Sort values by alphabetical order of views
        # order = np.argsort(self.views_names)
        # opts["likelihoods"] = opts["likelihoods"][order]

        # Create HDF5 group
        grp = self.hdf5.create_group('model_options')

        # Create HDF5 data sets
        for k, v in opts.items():
            grp.create_dataset(k, data=v)
        grp[k].attrs['names'] = np.asarray(list(opts.keys())).astype('S')

    def saveTrainOptions(self):
        """ Method to save the training options """

        # Subset training options
        opts = dict((k, self.train_opts[k]) for k in ["maxiter", "freqELBO", "start_elbo", "gpu_mode", "stochastic", "seed"])

        # Replace dictionaries (not supported in hdf5) by lists 
        # opts = self.train_opts
        for k,v in opts.copy().items():
            if type(v)==dict:
                for k1,v1 in v.items():
                    opts[str(k)+"_"+str(k1)] = v1
                opts.pop(k)

        # Remove strings from training options
        # self.train_opts['schedule'] = '_'.join(self.train_opts['schedule'])
        # if 'schedule' in opts.keys():
        #     del opts['schedule']
        # if 'convergence_mode' in opts.keys():
        #     del opts['convergence_mode']

        # Remove some training options
        # del opts['quiet']; del opts['start_drop']; del opts['freq_drop']; del opts['forceiter']; del opts['start_sparsity']

        # Create data set: only numeric options 
        self.hdf5.create_dataset("training_opts", data=np.array(list(opts.values()), dtype=np.float))
        self.hdf5['training_opts'].attrs['names'] = np.asarray(list(opts.keys())).astype('S')

    def saveSmoothOptions(self, smooth_opts):
        """ Method to save the smooth options """

        # Subset options
        options_to_save = ["scale_cov", "start_opt", "n_grid", "opt_freq", "sparseGP", "warping", "warping_freq", "warping_ref", "warping_open_begin", "warping_open_end", "model_groups"]

        opts = dict((k, np.asarray(smooth_opts[k]).astype('S')) for k in options_to_save)

        # Create data set
        # self.hdf5.create_dataset("smooth_opts".encode('utf8'), data=np.array(list(opts.values()), dtype=np.float))
        # self.hdf5['smooth_opts'].attrs['names'] = np.asarray(list(opts.keys())).astype('S')

        # Create HDF5 group
        grp = self.hdf5.create_group('smooth_opts')

        # Create HDF5 data sets
        for k, v in opts.items():
            grp.create_dataset(k, data=v)
        grp[k].attrs['names'] = np.asarray(list(opts.keys())).astype('S')

    def saveVarianceExplained(self):

        # Sort values by alphabetical order of views
        # order = np.argsort(self.views_names)
        # # order = [ i[0] for i in sorted(enumerate(self.views_names), key=lambda x:x[1]) ]

        # Store variance explained per factor in each view and group
        grp = self.hdf5.create_group("variance_explained")

        subgrp = grp.create_group("r2_per_factor")
        for g in range(len(self.groups_names)):
            # subgrp.create_dataset(self.groups_names[g], data=r2[g][order], compression="gzip",
            subgrp.create_dataset(self.groups_names[g], data=self.r2[g]*100, compression="gzip",
                               compression_opts=self.compression_level)

        # Store total variance explained for each view and group (using all factors)
        subgrp = grp.create_group("r2_total")
        r2_total = self.model.calculate_variance_explained(total=True)
        for g in range(len(self.groups_names)):
            # subgrp.create_dataset(self.groups_names[g], data=r2[g][order], compression="gzip",
            subgrp.create_dataset(self.groups_names[g], data=r2_total[g]*100, compression="gzip",
                               compression_opts=self.compression_level)

    def saveTrainingStats(self):
        """ Method to save the training statistics """

        # Get training statistics
        stats = self.model.getTrainingStats()

        # Create HDF5 group
        stats_grp = self.hdf5.create_group("training_stats")

        stats_grp.create_dataset("number_factors", data=stats["number_factors"])
        stats_grp.create_dataset("time", data=stats["time"])
        stats_grp.create_dataset("elbo", data=stats["elbo"])
        # stats_grp.create_dataset("elbo_terms", data=stats["elbo_terms"].T)
        # stats_grp['elbo_terms'].attrs['colnames'] = [a.encode('utf8') for a in stats["elbo_terms"].columns.values]

        if "length_scales" in stats.keys():
            stats_grp.create_dataset("length_scales", data=stats["length_scales"][self.order_factors])
            stats_grp.create_dataset("scales", data=stats["scales"][self.order_factors])
            stats_grp.create_dataset("Kg", data=stats["Kg"][self.order_factors])
