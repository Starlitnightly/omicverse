import numpy as np
import pandas as pd
import scipy as s
import sys
import os
from time import sleep, time, strftime
from typing import List, Optional, Union
from itertools import chain
from functools import wraps

from ..core.BayesNet import *
from ..core import gpu_utils
from ..build_model.build_model import *
from ..build_model.save_model import *
from ..build_model.utils import guess_likelihoods
from ..build_model.train_model import train_model
from ..core import gp_utils

from .. import config


def keyboardinterrupt_saver(func):
    @wraps(func)
    def saver(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        # Internal methods will raise TypeError when interrupted
        except (KeyboardInterrupt, TypeError):
            if self.train_opts["save_interrupted"]:
                print("Attempting to save the model at the current iteration...")
                if self.train_opts["outfile"] is not None and self.train_opts != "":
                    tmp_file = self.train_opts["outfile"]
                    tmp_file = tmp_file.rstrip(".hdf5") + "_interrupted.hdf5"
                else:
                    tmp_file = os.path.join('/tmp', "mofa_{}_interrupted.hdf5".format(strftime('%Y%m%d-%H%M%S')))
                self.save(outfile=tmp_file)
                print("Saved partially trained model in {}. Exiting now.".format(tmp_file))
            else:
                print("Exiting now without saving the partially trained model. To save a partially trained model, set save_interrupted in the training options to true.")
            sys.stdout.flush()
            sys.exit()
    return saver


def keyboardinterrupt_saver(func):
    @wraps(func)
    def saver(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        # Internal methods will raise TypeError when interrupted
        except (KeyboardInterrupt, TypeError):
            if self.train_opts["save_interrupted"]:
                print("Attempting to save the model at the current iteration...")
                if self.train_opts["outfile"] is not None and self.train_opts != "":
                    tmp_file = self.train_opts["outfile"]
                    tmp_file = tmp_file.rstrip(".hdf5") + "_interrupted.hdf5"
                else:
                    tmp_file = os.path.join('/tmp', "mofa_{}_interrupted.hdf5".format(strftime('%Y%m%d-%H%M%S')))
                self.save(outfile=tmp_file)
                print("Saved partially trained model in {}. Exiting now.".format(tmp_file))
            else:
                print("Exiting now without saving the partially trained model. To save a partially trained model, set save_interrupted in the training options to true.")
            sys.stdout.flush()
            sys.exit()
    return saver

class entry_point(object):
    def __init__(self):
        self.print_banner()
        self.dimensionalities = {"C":0}
        self.model = None

        # flags
        self.imputed = False     # probabilistic imputation
        self.Zcompleted = False  # mefisto

    def print_banner(self):
        """ Method to print the mofapy2 banner """
        
        banner = """
        #########################################################
        ###           __  __  ____  ______                    ### 
        ###          |  \/  |/ __ \|  ____/\    _             ### 
        ###          | \  / | |  | | |__ /  \ _| |_           ### 
        ###          | |\/| | |  | |  __/ /\ \_   _|          ###
        ###          | |  | | |__| | | / ____ \|_|            ###
        ###          |_|  |_|\____/|_|/_/    \_\              ###
        ###                                                   ### 
        ######################################################### 
       \n 
        """

        print(banner)
        sys.stdout.flush()

    def set_covariates(self, sample_cov, covariates_names=None):
        """"
        sample_cov: This can be either
                - 	a list of matrices per group
                    The dimensions of each matrix must be (samples, covariates)
                    The order of list elements and rows in each matrix must match the structure of data
                - 	a character specifying a column present in the samples' metadata. Note that this requires the metadata
                    to have the specified column present as well as the samples names as index.
        covariates_names: String or list of strings containing the name(s) of the covariate(s) (optional)
        """

        assert self.data is not None, "the data must be specified with set_data before specifying covariates"

        self.smooth_opts = {}
        self.smooth_opts["GP_factors"] = True

        # Define covariates
        # Define smooth options
        # if not hasattr(self, 'smooth_opts'):
        #     print("Options for the usage of smooth factors not defined, using default values...\n")
        #     self.set_smooth_options()

        # Sanity check
        if isinstance(sample_cov, str):
            sample_cov = [sample_cov]

        if isinstance(sample_cov, list) and all([isinstance(c, str) for c in sample_cov]):
            assert all([c in self.data_opts["samples_metadata"][0].columns for c in
                        sample_cov]), "sample_cov provided as string but not found in samples_metadata"
            sample_cov_list = []
            for g in range(self.dimensionalities['G']):
                samples4group = self.data_opts['samples_names'][g]
                df = self.data_opts["samples_metadata"][g]
                sample_cov_list.append(np.vstack([df[c][samples4group] for c in sample_cov]).transpose())
            sample_cov = sample_cov_list

        if not isinstance(sample_cov, list):
            if isinstance(sample_cov, dict):
                sample_cov = list(sample_cov.values())
            # if providing a single matrix, treat it as G=1
            elif isinstance(sample_cov, pd.DataFrame):
                sample_cov = [sample_cov.values]
            elif isinstance(sample_cov, np.ndarray):
                sample_cov = [sample_cov]
            else:
                print("Error: sample_cov not recognised");
                sys.stdout.flush();
                sys.exit()

        assert len(sample_cov) == self.dimensionalities[
            "G"], "sample_cov needs to be a list of same length as data (same number of groups)"

        for g in range(self.dimensionalities["G"]):
            if not isinstance(sample_cov[g], np.ndarray):
                if isinstance(sample_cov[g], pd.DataFrame):
                    sample_cov[g] = sample_cov[g].values
                else:
                    print("Error, sample_cov is not a numpy.ndarray or a pandas dataframe");
                    sys.stdout.flush();
                    sys.exit()
            sample_cov[g] = sample_cov[g].astype(np.float64)

        N = [len(x) for x in self.data_opts['samples_names']]
        if not all([sample_cov[g].shape[0] == N[g] for g in range(self.dimensionalities["G"])]):
            for g in range(self.dimensionalities["G"]):
                print(
                    "Error, number of rows in sample covariates does not match number of samples in input data (N=%d vs. N=%d)" % (
                        sample_cov[g].shape[0], N[g]))
            sys.stdout.flush();
            sys.exit()

        # concatenate groups in sample_cov and standardize sample_cov to avoid scale differences
        sample_cov = np.concatenate(sample_cov, axis=0)

        # Define dimensionality
        self.dimensionalities["C"] = sample_cov.shape[1]

        # If sample_cov loaded successfully, print verbose message
        if not sample_cov is None:
            print("Loaded %d covariate(s) for each sample..." % (sample_cov.shape[1]))
            print("\n")

        # Define covariate names
        if covariates_names is None:
            print("Covariates names not provided, using default naming convention:")
            print("- covariate1, ..., covariateC\n")
            self.smooth_opts['covariates_names'] = ["covariate%d" % (c) for c in range(self.dimensionalities["C"])]
        else:
            if isinstance(covariates_names, str):
                covariates_names = [covariates_names]
            assert isinstance(covariates_names, list), "covariates_names must be a string or a list"
            assert len(covariates_names) == self.dimensionalities["C"], "covariates_names must be of length equivalent to the number of covariates"
            self.smooth_opts['covariates_names'] = covariates_names

        self.sample_cov = sample_cov

        return None

    def set_data_matrix(self, data, likelihoods=None, views_names=None, groups_names=None, samples_names=None, features_names=None):
        """ Method to input the data in a wide matrix

        PARAMETERS
        ----------
        data: a nested list, first dimension for views, second dimension for groups.
              The dimensions of each matrix must be (samples,features)
        """

        if not hasattr(self, 'data_opts'): 
            # print("Data options not defined, using default values...\n")
            self.set_data_options()

        # Sanity check
        if not isinstance(data, list):
            if isinstance(data, dict):
                data = list(data.values())
            # if providing a single matrix, treat it as G=1 and M=1
            elif isinstance(data, pd.DataFrame):
                data = [[data.values]]
            elif isinstance(data, np.ndarray):
                data = [[data]]
            else:
                print("Error: Data not recognised"); sys.stdout.flush(); sys.exit()
        if len(data)==0:
            print("Error: Data is empty"); sys.stdout.flush(); sys.exit()

        # Convert input data to numpy array float64 format
        for m in range(len(data)):
            if isinstance(data[m], dict):
                data[m] = list(data[m].values())
            for p in range(len(data[m])):
                if not isinstance(data[m][p], np.ndarray):
                    if isinstance(data[m][p], pd.DataFrame):
                        data[m][p] = data[m][p].values
                    else:
                        print("Error, input data is not a numpy.ndarray or a pandas dataframe"); sys.stdout.flush(); sys.exit()
                if config.use_float32:
                    data[m][p] = data[m][p].astype(np.float32)
                else:
                    data[m][p] = data[m][p].astype(np.float64)

        # Save dimensionalities
        M = self.dimensionalities["M"] = len(data)
        G = self.dimensionalities["G"] = len(data[0])
        N = self.dimensionalities["N"] = [data[0][p].shape[0] for p in range(len(data[0]))]
        D = self.dimensionalities["D"] = [data[m][0].shape[1] for m in range(len(data))]

        # Define views names
        if views_names is None:
            print("View names not provided, using default naming convention:")
            print("- view1, view2, ..., viewM\n")
            self.data_opts['views_names'] = [ "view" + str(m) for m in range(M) ]
        else:
            assert len(views_names)==self.dimensionalities["M"], "Length of views names is not the same as the number of views"
            self.data_opts['views_names']  = views_names

        # Define features names
        if features_names is None:
            print("Features names not provided, using default naming convention:")
            print("- feature1_view1, featureD_viewM\n")
            self.data_opts['features_names'] = [ ["feature%d_view%d" % (d,m) for d in range(D[m])] for m in range(M) ]
        else:
            assert len(features_names)==self.dimensionalities["M"], "views_names must a nested list with length equivalent to the number of views"
            self.data_opts['features_names'] = features_names

        # Define groups names
        if groups_names is None:
            print("Groups names not provided, using default naming convention:")
            print("- group1, group2, ..., groupG\n")
            self.data_opts['groups_names'] = [ "group" + str(g) for g in range(G) ]
        else:
            assert len(groups_names)==self.dimensionalities["G"], "Length of groups names is not the same as the number of groups"
            self.data_opts['groups_names'] = groups_names

        # Define samples names
        if samples_names is None:
            print("Samples names not provided, using default naming convention:")
            print("- sample1_group1, sample2_group1, sample1_group2, ..., sampleN_groupG\n")
            self.data_opts['samples_names'] = [ ["sample%d_group%d" % (n,g) for n in range(N[g])] for g in range(G) ]
        else:
            assert len(samples_names)==self.dimensionalities["G"], "samples_names must a nested list with length equivalent to the number of groups"
            self.data_opts['samples_names'] = samples_names

        # Check for duplicated entries
        assert len(self.data_opts['groups_names']) == len(set(self.data_opts['groups_names'])), "Duplicated groups names"
        assert len(self.data_opts['views_names']) == len(set(self.data_opts['views_names'])), "Duplicated views names"

        tmp = list(chain(*self.data_opts['samples_names']))
        assert len(tmp) == len(set(tmp)), "Duplicated entries found in samples_names. Make sure that samples names are not duplicated across different groups"
        
        tmp = list(chain(*self.data_opts['features_names']))
        assert len(tmp) == len(set(tmp)), "Duplicated entries found in features_names. Make sure that feature names are not duplicated across different views"

        # Set samples groups (list with dimensionality N where each row is the corresponding group name)
        # self.data_opts['samples_groups'] = [list(samples_names_dict.keys())[i] for i in range(len(self.data_opts['groups_names'])) for n in range(len(list(samples_names_dict.values())[i]))]

        # WHY THIS DOES NOT WORK?? asd = [ [self.data_opts['groups_names'][g]]*N[g] for g in range(G) ]
        self.data_opts['samples_groups'] = []
        for g in range(G):
            self.data_opts['samples_groups'].append( [self.data_opts['groups_names'][g]]*N[g] )
        self.data_opts['samples_groups'] = np.concatenate(self.data_opts['samples_groups'])

        # If everything successful, print verbose message
        for m in range(M):
            for g in range(G):
                print("Successfully loaded view='%s' group='%s' with N=%d samples and D=%d features..." % (self.data_opts['views_names'][m],self.data_opts['groups_names'][g], data[m][g].shape[0], data[m][g].shape[1]))
        print("\n")

        # Store intercepts
        self.intercepts = [ [ np.nanmean(data[m][g],axis=0) for g in range(G)] for m in range(M) ]

        # Concatenate groups in data
        for m in range(len(data)):
            data[m] = np.concatenate(data[m])
            if type(data[m]) != np.ndarray:
                data[m] = np.array(data[m])
        self.dimensionalities["N"] = np.sum(self.dimensionalities["N"])

        # Define likelihoods
        if likelihoods is None:
            likelihoods = guess_likelihoods(data)
        elif isinstance(likelihoods, str):
            likelihoods = [likelihoods]
        assert len(likelihoods)==self.dimensionalities["M"], "Please specify one likelihood for each view"
        assert set(likelihoods).issubset(set(["gaussian","bernoulli","poisson"])), "Available likelihoods are 'gaussian','bernoulli', 'poisson'"
        self.likelihoods = likelihoods

        # Process the data (center, scaling, etc.)
        self.data = process_data(data, likelihoods, self.data_opts, self.data_opts['samples_groups'])

    def set_data_df(self, data, likelihoods=None):
        """Method to input the data in a long data.frame format

        PARAMETERS
        ----------
        data: pd.DataFrame
            a pandas DataFrame with columns (sample,feature,view,group,value)
            the order is irrelevant
        """

        # TO-DO: 
        # - CHECK FOR DUPLICATED GROUPS NAMES, VIEWS, NAMES, SAMPLES NAMES, FEATURES NAMES
        
        # Sanity checks
        if not hasattr(self, 'data_opts'): 
            # print("Data options not defined before setting the data, using default values...")
            self.set_data_options()

        assert isinstance(data, pd.DataFrame), "'data' has to be an instance of pd.DataFrame"

        if 'group' not in data.columns:
            print('\nNo "group" column found in the data frame, we will assume a common group for all samples...')
            data["group"] = "single_group"

        if 'view' not in data.columns:
            print('\nNo "view" column found in the data frame, we will assume a common view for all features...')
            data["view"] = "single_view"

        assert 'sample' in data.columns, "'data' has to contain the column 'sample'"
        assert 'feature' in data.columns, "'data' has to contain the column 'feature'"
        assert 'value' in data.columns, "'data' has to contain the column 'value'"

        # Check for duplicated entries
        assert data.duplicated(subset=["group","view","feature","sample"]).sum() == 0, "Duplicated entries found in the data"

        # Define feature group names and sample group names
        self.data_opts['views_names'] = np.sort(data["view"].unique()).tolist()
        self.data_opts['groups_names'] = np.sort(data["group"].unique()).tolist()
        self.data_opts['features_names'] = data.groupby(["view"])["feature"].unique()[self.data_opts['views_names']].tolist()
        self.data_opts['samples_names'] = data.groupby(["group"])["sample"].unique()[self.data_opts['groups_names']].tolist()

        # Convert data frame to list of matrices
        data['feature'] = data['feature'].astype(str) + data['view'].astype(str) # make sure there are no duplicated feature names before pivoting
        data_matrix = data.pivot(index='sample', columns='feature', values='value')

        # Sort rows and columns of the matrix according to the sample and feature names
        features_names_tmp = data.groupby(["view"])["feature"].unique()[self.data_opts['views_names']].tolist()
        data_matrix = data_matrix.loc[np.concatenate(self.data_opts['samples_names'])]
        data_matrix = data_matrix[[y for x in features_names_tmp for y in x]]

        # Split into a list of views, each view being a matrix
        tmp_features = data[["feature","view"]].drop_duplicates().groupby("view")["feature"].nunique()
        nfeatures = tmp_features.loc[self.data_opts['views_names']]
        data_matrix = np.split(data_matrix, np.cumsum(nfeatures)[:-1], axis=1)

        # Define sample groups
        self.data_opts['samples_groups'] = data[['sample', 'group']].drop_duplicates() \
                                            .set_index('sample').loc[np.concatenate(self.data_opts['samples_names'])] \
                                            .group.tolist()

        # Define dimensionalities
        self.dimensionalities = {}
        self.dimensionalities["M"] = M = len(self.data_opts['views_names'])
        # self.dimensionalities["N"] = len(self.data_opts['samples_names'])
        self.dimensionalities["N"] = N = len(np.concatenate(self.data_opts['samples_names']))
        self.dimensionalities["G"] = G = len(self.data_opts['groups_names'])
        self.dimensionalities["D"] = D = [len(x) for x in self.data_opts['features_names']]

        # Count the number of features per view and the number of samples per group
        tmp_samples = data[["sample","group","view"]].drop_duplicates().groupby(["group","view"])["sample"].nunique()
        tmp_features = data[["feature","group","view"]].drop_duplicates().groupby(["group","view"])["feature"].nunique()

        # If everything successful, print verbose message
        print("\n")
        for g in self.data_opts['groups_names']:
            for m in self.data_opts['views_names']:
                try:
                    print("Loaded group='%s' view='%s' with N=%d samples and D=%d features..." % (g, m, tmp_samples[g][m], tmp_features[g][m]) )
                except:
                    print("No data found for group='%s' and view='%s'..." % (g, m))
        print("\n")

        # Convert from pandas dataframe to numpy array
        for m in range(M):
            if isinstance(data_matrix[m], pd.DataFrame): data_matrix[m] = data_matrix[m].values

        # Store intercepts
        self.intercepts = [None for m in range(M)]
        tmp = [ len(x) for x in self.data_opts['samples_names'] ]
        for m in range(M):
            self.intercepts[m] = [np.nanmean(x,axis=0) for x in np.split(data_matrix[m], np.cumsum(tmp)[:-1], axis=0)]

        # Define likelihoods
        if likelihoods is None:
            likelihoods = guess_likelihoods(data_matrix)
        elif isinstance(likelihoods, str):
            likelihoods = [likelihoods]
        assert len(likelihoods)==self.dimensionalities["M"], "Please specify one likelihood for each view"
        assert set(likelihoods).issubset(set(["gaussian","bernoulli","poisson"])), "Available likelihoods are 'gaussian','bernoulli', 'poisson'"
        self.likelihoods = likelihoods

        # Process the data (i.e center, scale, etc.)
        self.data = process_data(data_matrix, likelihoods, self.data_opts, self.data_opts['samples_groups'])

    def set_data_from_anndata(self, adata, groups_label=None, use_raw=False, use_layer=None, likelihoods=None, features_subset=None, save_metadata=None):
        """ Method to input the data in AnnData format

        PARAMETERS
        ----------
        adata: an AnnotationData object
        groups_label (optional): a column name in adata.obs for grouping the samples
        use_raw (optional): use raw slot of AnnData as input values
        use_layer (optional): use a specific layer of AnnData as input values (supersedes use_raw option)
        likelihoods (optional): likelihoods to use (guessed from the data if not provided)
        features_subset (optional): .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
        """

        # TO-DO: 
        # - CHECK FOR DUPLICATED NAMES.
        # - IF NAMES NOT PROVIDED, USE DEFAULTS
        # - 

        # Sanity checks
        if not hasattr(self, 'data_opts'): 
            # print("Data options not defined before setting the data, using default values...")
            self.set_data_options()

        # Check groups_label is defined properly
        n_groups = 1  # no grouping by default
        if groups_label is not None:
            if not isinstance(groups_label, str):
                print("Error: groups_label should be a string present in the observations column names"); sys.stdout.flush(); sys.exit()
            if groups_label not in adata.obs.columns:
                print("Error: {} is not in observations names".format(groups_label)); sys.stdout.flush(); sys.exit()
            n_groups = adata.obs[groups_label].unique().shape[0]


        # Get the respective data slot
        if use_layer:
            if use_layer in adata.layers.keys():
                if callable(getattr(adata.layers[use_layer], "todense", None)):
                    data = [np.array(adata.layers[use_layer].todense())]    
                else:
                    data = [adata.layers[use_layer]]
                # Subset features if required
                if features_subset is not None:
                    data[0] = data[0][:,adata.var[features_subset].values]
            else:
                print("Error: Layer {} does not exist".format(use_layer)); sys.stdout.flush(); sys.exit()
        elif use_raw:
            adata_raw_dense = np.array(adata.raw[:,adata.var_names].X.todense())
            data = [adata_raw_dense]
            # Subset features if required
            if features_subset is not None:
                data[0] = data[0][:,adata.var[features_subset].values]
        else:
            if callable(getattr(adata.X, "todense", None)):
                data = [np.array(adata.X.todense())]
            else:
                data = [adata.X]
            # Subset features if required
            if features_subset is not None:
                data[0] = data[0][:,adata.var[features_subset].values]

        # Save dimensionalities
        M = self.dimensionalities["M"] = 1
        G = self.dimensionalities["G"] = n_groups
        N = self.dimensionalities["N"] = adata.shape[0]
        D = self.dimensionalities["D"] = [data[0].shape[1]]  # Feature may have been filtered
        n_grouped = [adata.shape[0]] if n_groups == 1 else adata.obs.groupby(groups_label).size().values

        # Define views names and features names and metadata
        self.data_opts['views_names'] = ["rna"]
        
        if features_subset is not None:
            self.data_opts['features_names'] = [adata.var_names[adata.var[features_subset]]]
        else:
            self.data_opts['features_names'] = [adata.var_names]

        if save_metadata:
            if features_subset is not None:
                self.data_opts['features_metadata'] = [adata.var[adata.var[features_subset]]]
            else:
                self.data_opts['features_metadata'] = [adata.var]

        # Define groups and samples names and metadata
        if groups_label is None:
            self.data_opts['groups_names'] = ["group1"]
            self.data_opts['samples_names'] = [adata.obs.index.values.tolist()]
            self.data_opts['samples_groups'] = ["group1"] * N
            if save_metadata:
                self.data_opts['samples_metadata'] = [adata.obs]
        else:
            # While grouping the pandas.DataFrame, the group_label would be sorted.
            # Hence the naive implementation `adata.obs[groups_label].unique()` to get group names
            # wouldn't match samples_names if the samples are not ordered according to their group beforehand.

            # List of names of groups, i.e. [group1, group2, ...]
            self.data_opts['groups_names'] = [str(g) for g in adata.obs.reset_index(drop=False).groupby(groups_label)[groups_label].apply(list).index.values]
            # Nested list of names of samples, one inner list per group, i.e. [[group1_sample1, group1_sample2, ...], ...]
            self.data_opts['samples_names'] = adata.obs.reset_index(drop=False).rename(columns={adata.obs.index.name:'index'}).groupby(groups_label)["index"].apply(list).tolist()
            # List of names of groups for samples ordered as they are in the original data, i.e. [group2, group1, group1, ...]
            self.data_opts['samples_groups'] = adata.obs[groups_label].values.astype(str)
            if save_metadata:
                # List of metadata tables for each group of samples
                self.data_opts['samples_metadata'] = [g for _, g in adata.obs.groupby(groups_label)]


        # If everything successful, print verbose message
        for m in range(M):
            for g in range(G):
                print("Loaded view='%s' group='%s' with N=%d samples and D=%d features..." % (self.data_opts['views_names'][m], self.data_opts['groups_names'][g], n_grouped[g], D[m]))
        print("\n")


        # Store intercepts (it is for one view only)
        self.intercepts = [[]]

        # Define likelihoods
        if likelihoods is None:
            likelihoods = guess_likelihoods(data)
        elif isinstance(likelihoods, str):
            likelihoods = [likelihoods]
        assert len(likelihoods)==self.dimensionalities["M"], "Please specify one likelihood for each view"
        assert set(likelihoods).issubset(set(["gaussian","bernoulli","poisson"])), "Available likelihoods are 'gaussian','bernoulli', 'poisson'"
        self.likelihoods = likelihoods

        # Process the data (center, scaling, etc.)
        for g in self.data_opts['groups_names']:
            samples_idx = np.where(np.array(self.data_opts['samples_groups']) == g)[0]
            self.intercepts[0].append(np.nanmean(data[0][samples_idx,:], axis=0))
        self.data = process_data(data, likelihoods, self.data_opts, self.data_opts['samples_groups'])

    def set_data_from_loom(self, loom, groups_label=None, layer=None, cell_id="CellID"):
        """ Method to input the data in Loom format

        PARAMETERS
        ----------
        loom: connection to loom file (loompy.loompy.LoomConnection)
        groups_label (optional): a key in loom.ca for grouping the samples
        layer (optional): a layer to be used instead of the main matrix
        cell_id (optional): the name of the cell ID attribute (default is CellID)
        """

        # TO-DO: 
        # - CHECK FOR DUPLICATED NAMES.
        # - IF NAMES NOT PROVIDED, USE DEFAULTS
        # - 

        # Sanity checks
        if not hasattr(self, 'data_opts'): 
            # print("Data options not defined before setting the data, using default values...")
            self.set_data_options()

        # Check groups_label is defined properly
        n_groups = 1  # no grouping by default
        if groups_label is not None:
            if not isinstance(groups_label, str):
                print("Error: groups_label should be a string present in the observations column names"); sys.stdout.flush(); sys.exit()
            if groups_label not in loom.ca.keys():
                print("Error: {} is not in observations names".format(groups_label)); sys.stdout.flush(); sys.exit()
            n_groups = pd.unique(loom.ca[groups_label]).shape[0]

        # Save dimensionalities
        M = self.dimensionalities["M"] = 1
        G = self.dimensionalities["G"] = n_groups
        N = self.dimensionalities["N"] = loom.shape[1]
        D = self.dimensionalities["D"] = [loom.shape[0]]
        n_grouped = [loom.shape[1]] if n_groups == 1 else pd.DataFrame({'label': loom.ca[groups_label]}).groupby('label').size().values

        # Define views names and features names
        self.data_opts['views_names'] = ["rna"]
        self.data_opts['features_names'] = [loom.ra.Accession] if 'Accession' in loom.ra.keys() else [loom.ra.Gene]

        # Define groups and samples names
        if groups_label is None:
            self.data_opts['groups_names'] = ["group1"]
            self.data_opts['samples_names'] = [loom.ca[cell_id]]
            self.data_opts['samples_groups'] = ["group1"] * N
        else:
            loom_metadata = pd.DataFrame(loom.ca[cell_id, groups_label])
            loom_metadata.columns = [cell_id, groups_label]
            # List of names of groups, i.e. [group1, group2, ...]
            self.data_opts['groups_names'] = loom_metadata.groupby(groups_label)[groups_label].apply(list).index.values
            # Nested list of names of samples, one inner list per group, i.e. [[group1_sample1, group1_sample2, ...], ...]
            self.data_opts['samples_names'] = loom_metadata.groupby(groups_label)[cell_id].apply(list).tolist()
            # List of names of groups for samples ordered as they are in the oridinal data, i.e. [group2, group1, group1, ...]
            self.data_opts['samples_groups'] = loom_metadata[groups_label].values

        # If everything successful, print verbose message
        for m in range(M):
            for g in range(G):
                print("Loaded view='%s' group='%s' with N=%d samples and D=%d features..." % (self.data_opts['views_names'][m], self.data_opts['groups_names'][g], n_grouped[g], D[m]))
        print("\n")

        # Store intercepts
        self.intercepts = [None]
        tmp = [ len(x) for x in self.data_opts['samples_names'] ]

        # Get the respective data slot
        if layer is not None:
            data = [loom.layers[layer][:,:].T]
        else:
            data = [loom[:,:].T]

        # Define likelihoods
        if likelihoods is None:
            likelihoods = guess_likelihoods(data)
        elif isinstance(likelihoods, str):
            likelihoods = [likelihoods]
        assert len(likelihoods)==self.dimensionalities["M"], "Please specify one likelihood for each view"
        assert set(likelihoods).issubset(set(["gaussian","bernoulli","poisson"])), "Available likelihoods are 'gaussian','bernoulli', 'poisson'"
        self.likelihoods = likelihoods

        # Process the data (center, scaling, etc.)
        self.intercepts[0] = [np.nanmean(x, axis=0) for x in [data[0][np.where(np.array(self.data_opts['samples_groups']) == g)[0],:] for g in self.data_opts['groups_names']]]
        self.data = process_data(data, self.data_opts, self.data_opts['samples_groups'])

    def set_train_options(self,
        iter=1000, startELBO=1, freqELBO=1, startSparsity=50, tolerance=None, convergence_mode="fast",
        startDrop=1, freqDrop=1, dropR2=None, nostop=False, verbose=False, quiet=False, seed=None,
        schedule=None, gpu_mode=False, weight_views = False, 
        outfile=None, save_interrupted=False):
        """ Set training options """

        # Sanity checks
        assert hasattr(self, 'model_opts'), "Model options have to be defined before training options"

        self.train_opts = {}

        # Maximum and minimum number of iterations
        self.train_opts['maxiter'] = int(iter)
        self.train_opts['min_iter'] = 0

        # Lower bound computation frequency
        if freqELBO is None or freqELBO==0: freqELBO=iter+1
        self.train_opts['freqELBO'] = int(freqELBO)
        if startELBO==0: startELBO=1
        if startELBO==None: startELBO=iter+1
        self.train_opts['start_elbo'] = int(startELBO)

        # Verbosity
        self.train_opts['verbose'] = bool(verbose)
        self.train_opts['quiet'] = bool(quiet)

        # GPU mode
        if gpu_mode:
            try:
                import cupy as cp

                print("\nGPU mode is activated\n")
            except ImportError:
                print("\nGPU mode is activated, but GPU not found... switching to CPU mode")
                print('For GPU mode, you need:')
                print('1 - Make sure that you are running MOFA+ on a machine with an NVIDIA GPU')
                print('2 - Install CUPY following instructions on https://docs-cupy.chainer.org/en/stable/install.html\n')
                sys.stdout.flush()
                gpu_mode = False
        self.train_opts['gpu_mode'] = gpu_mode

        # Minimum Variance explained threshold to drop inactive factors
        if dropR2 is False: dropR2 = None
        if dropR2 is not None:
            dropR2 = float(dropR2)
            if dropR2 < 0: dropR2 = None
        self.train_opts['drop'] = { "min_r2":dropR2 }
        self.train_opts['start_drop'] = int(startDrop)
        self.train_opts['freq_drop'] = int(freqDrop)
        if ((dropR2 is not None) & (verbose is True)): print("\nDropping factors with minimum threshold of {0}% variance explained\n".format(dropR2))

        if ((dropR2 is not None) & (self.dimensionalities["N"]>1e4)):
            print("Warning: actively dropping factors during model training can be slow in large data sets.")
            print("Consider training the model with set drop_factor_threshold = -1 and prune them a posteriori")


        # Tolerance level for convergence
        if tolerance is not None:
            print("Warning: tolerance argument is depreciated, use the 'convergence_mode' argument instead")
            self.train_opts['tolerance'] = float(tolerance)

        # Convergence mode
        self.train_opts['convergence_mode'] = str(convergence_mode)
        if verbose: print("Convergence mode: %s\n" % convergence_mode)

        # Do no stop even when convergence criteria is met
        self.train_opts['forceiter'] = nostop

        # Iteration to activate spike and slab sparsity
        self.train_opts['start_sparsity'] = int(startSparsity)

        # By default, stochastic training is not activated
        self.train_opts['stochastic'] = False

        # Training schedule
        if schedule is None:
            schedule = ['Y', 'W', 'Z', 'Tau']

            # Insert ThetaW after W if Spike and Slab prior on W
            if self.model_opts['spikeslab_weights']:
                ix = schedule.index("W"); schedule.insert(ix+1, 'ThetaW')

            # Insert ThetaZ after Z if Spike and Slab prior on Z
            if self.model_opts['spikeslab_factors']:
                ix = schedule.index("Z"); schedule.insert(ix+1, 'ThetaZ')

            # Insert AlphaW after W if ARD prior on W
            if self.model_opts['ard_weights']:
                ix = schedule.index("W"); schedule.insert(ix+1, 'AlphaW')

            # Insert AlphaZ after Z if ARD prior on Z
            if self.model_opts['ard_factors']:
                ix = schedule.index("Z"); schedule.insert(ix+1, 'AlphaZ')

        else:
            assert set(["Y","W","Z","Tau"]) <= set(schedule)
            if self.model_opts['ard_factors']: assert "AlphaZ" in schedule
            if self.model_opts['ard_weights']: assert "AlphaW" in schedule
            if self.model_opts['spikeslab_factors']: assert "ThetaZ" in schedule
            if self.model_opts['spikeslab_weights']: assert "ThetaW" in schedule

        self.train_opts['schedule'] = schedule

        # Seed
        if seed is None:  # Seed for the random number generator
            seed = int(round(time()*1000)%1e6)
        self.train_opts['seed'] = int(seed)
        # np.random.seed(self.train_opts['seed'])

        # Weight the views to avoid imbalance problems?
        self.train_opts['weight_views'] = weight_views
        if weight_views: print("\nweight_views set to True. Weighting the ELBO (the objective function) based on the number of features per view\n")

        # Output file name
        self.train_opts['outfile'] = outfile

        # If to save the partially trained model when the training is interrupted
        self.train_opts['save_interrupted'] = save_interrupted

    def set_stochastic_options(self, learning_rate=1., forgetting_rate=0., batch_size=1., start_stochastic=1):

        # Sanity checks
        if hasattr(self, 'smooth_opts'):
            print("Stochastic inference is not possible when using covariates - train_opts['stochastic'] is set to False - consider using the option 'sparseGP' instead.")
            self.train_opts['stochastic'] = False
            return None
        assert hasattr(self, 'train_opts'), "Train options not defined"
        assert 0 < learning_rate <= 1, 'Learning rate must range from 0 and 1'
        # assert 0 < forgetting_rate <= 1, 'Forgetting rate must range from 0 and 1'
        assert 0 < batch_size <= 1, 'Batch size must range from 0 to 1'
        assert start_stochastic >= 1, 'start_stochastic must be >= 1'

        if self.train_opts['drop']["min_r2"] is not None:
            print("Dropping factors is currently disabled with stochastic inference...")
            self.train_opts['drop']["min_r2"] = None

        # Edit schedule: Z should come first (after Y) in the training schedule
        # (THIS IS DONE IN THE BAYESNET CLASS)
        # self.train_opts['schedule'].pop( self.train_opts['schedule'].index("Z") )
        # self.train_opts['schedule'].insert(1,"Z")

        self.train_opts['stochastic'] = True
        self.train_opts['learning_rate'] = learning_rate
        self.train_opts['forgetting_rate'] = forgetting_rate
        self.train_opts['start_stochastic'] = start_stochastic
        self.train_opts['batch_size'] = batch_size

        self.train_opts['drop']["min_r2"] = None

    def set_smooth_options(self, scale_cov = False, start_opt=20, n_grid=20, opt_freq=10, model_groups = True,
        warping = False, warping_freq = 20, warping_ref = 0, warping_open_begin = True, warping_open_end = True,
        sparseGP = False, frac_inducing = None, warping_groups = None):

        """ 
        Method to activate and set options for a functional MOFA model (MEFISTO).
        This requires to specify a covariate using set_covariates.

        PARAMETERS:
        - scale_cov: Scale covariates to unit variance (relevant if you have multiple covariates on different scales). Default is False.
        - stat_opt: Iteration where to start the optimization of the hyperparameters for the Gaussian process
        - n_grid: number of grid points to use for optimization of the lengthscale
        - opt_freq: Frequency of the optimization of the hyperparameters for the Gaussian process (every opt_freq-th iteration)
        - model_groups: Boolean  whether to include a group-group covariance into the model. 
            Default is True, if set to False the same underlying patterns are assumed in all groups. 
        - warping: Boolean, indicating whether to align covariates between groups (only relevant for more than 1 group and a single covariate)
        - warping_freq, warping_ref, warping_open_begin, warping_open_end: Warping-specifc parameters deremining the frequency of warping, the refernce group
            wether to allow open begin or open end for the alginement
        - sparseGP: Boolean, whether to use sparse Gaussian processes (only recommmended for very large data sets)
        - frac_inducing: Fraction of original samples to use as inducing points when using sparse GPs (between 0 and 1)
        """ 

        # Sanity checks
        assert hasattr(self, 'smooth_opts'), "Please run set_covariates() before set_smooth_options()"
        assert hasattr(self, 'model_opts'), "Model options not defined. Please run set_model_opts() before set_smooth_options()"
        assert hasattr(self, 'train_opts'), "Training options not defined. Please run set_train_opts() before set_smooth_options()"
        assert self.sample_cov is not None, "Before setting smooth options, you need to define the covariates with set_covariates"

        # activate GP prior on factors
        self.smooth_opts['GP_factors'] = True

        # Define whether to scale covariates to unit variance
        self.smooth_opts['scale_cov'] = scale_cov
        # if (scale_cov): print("Scaling covariates to unit variance...\n")
        if self.smooth_opts['scale_cov']:
            self.sample_cov = (self.sample_cov - self.sample_cov.mean(axis=0)) / self.sample_cov.std(axis=0)

        # Define at which iteration to start optimizing the lengthscales, how many grid points to use for the lengthscales and at which frequency to optimize
        start_opt = max(0, start_opt)
        self.smooth_opts['start_opt'] = int(start_opt)
        self.smooth_opts['n_grid'] = int(n_grid)
        self.smooth_opts['opt_freq'] = int(opt_freq)

        # inactivate group-wise ARD when using the MEFISTO framework
        if self.model_opts['ard_factors'] is True:
            print("Smooth covariate framework is activated. This is not compatible with ARD prior on factors. Setting ard_factors to False...\n")
            self.model_opts['ard_factors'] = False
            self.train_opts['schedule'].remove('AlphaZ')

        # inactivate spike-slab on the factors when using the MEFISTO framework
        if self.model_opts['spikeslab_factors'] is True:
            print("Smooth covariate framework is activated. This is not compatible with spike-and-slab prior on factors. Setting spikeslab_factors to False...\n")
            self.model_opts['spikeslab_factors'] = False
            self.train_opts['schedule'].remove('ThetaZ')

        # Insert Sigma into training schedule if GP prior on Z
        self.train_opts['schedule'].insert(len(self.train_opts['schedule']), "Sigma")

        # Warping
        self.smooth_opts['warping'] = bool(warping)
        self.smooth_opts['warping_freq'] = int(warping_freq)
        self.smooth_opts['warping_ref'] = int(warping_ref)
        self.smooth_opts['warping_open_begin'] = bool(warping_open_begin)
        self.smooth_opts['warping_open_end'] = bool(warping_open_end)
        if warping_groups is None:
            self.smooth_opts['warping_groups'] = self.data_opts['samples_groups']
        else:
            self.smooth_opts['warping_groups'] = np.array(warping_groups)

        if self.smooth_opts['warping'] is True:
            assert self.dimensionalities["G"] > 1, "The warping functionality is only relevant when having multi-group data"
            assert self.dimensionalities["C"] == 1, "Warping only implemented for one dimensional covariates"
            print("##")
            print("## Warping set to True: aligning the covariates across groups")
            print("##")

        # Set a minium number of training iterations based on start of optimization and warping
        self.train_opts['min_iter'] =  self.smooth_opts['start_opt']
        if self.smooth_opts['warping']:
            self.train_opts['min_iter'] = max(self.train_opts['min_iter'], self.smooth_opts['warping_freq']) + 1

       # Sparse GPs
        if sparseGP is True:
            assert not self.smooth_opts['warping'], "The warping functionality cannot be used in conjunction with the sparseGP option."
            assert np.all([l == 'gaussian' for l in self.model_opts['likelihoods']]), "Non-gaussian likelihoods cannot be used in conjunction with the sparseGP option."

            self.smooth_opts['sparseGP'] = True
            if self.dimensionalities["N"] < 1000:
                print("Warning: sparseGP should only be used when having a large sample size (>1e3)\n")

            # Sparse GPs: set the number of inducing points
            if frac_inducing is None:
                frac_inducing = 0.75
            assert frac_inducing > 0 and frac_inducing < 1, "Fraction of inducing points should be in (0,1)."
            n_inducing = int(max(frac_inducing * self.dimensionalities["N"], 100)) # note: groups are already concatenated, N is total number of samples

            # Sparse GPs: Set the identity of the inducing points
            idx_inducing = set_inducing_points(self.data, self.sample_cov, self.data_opts['samples_groups'], self.dimensionalities, n_inducing, random = False, seed_inducing = 0)

            self.smooth_opts['frac_inducing'] = frac_inducing
            self.smooth_opts['n_inducing'] = n_inducing
            self.smooth_opts['idx_inducing'] = idx_inducing

            # Sparse GPs: Insert U in schedule
            ix = self.train_opts['schedule'].index("Z")
            self.train_opts['schedule'].insert(ix, 'U')

            print("##")
            print("## sparseGP set to True: using sparse Gaussian Process to speed up the training procedure")
            print("##")
        else:

            self.smooth_opts['sparseGP'] = False



        # Define whether to model a group covariance structure
        self.smooth_opts['model_groups'] = model_groups
        if self.dimensionalities["G"] < 2:
            self.smooth_opts['model_groups'] = False
        # self.smooth_opts['use_gpytorch'] = False # experimental, this could be passed as a model_option but to keep options uncluttered set to False

    def set_model_options(self, factors=10, spikeslab_factors=False, spikeslab_weights=True, ard_factors=False, ard_weights=True):
        """ Set model options """

        self.model_opts = {}

        # Define whether to use sample-wise spike and slab prior for Z
        self.model_opts['spikeslab_factors'] = spikeslab_factors

        # Define whether to use feature-wise spike and slab prior for W
        self.model_opts['spikeslab_weights'] = spikeslab_weights

        # Define whether to use group and factor-wise ARD prior for Z
        if ((self.dimensionalities["G"]>1) & (ard_factors==False)): 
            print("WARNING: 'ard_factors' in model_options should be set to True if using multiple groups unless you are using MEFISTO\n")
        # if self.dimensionalities["G"]>1: ard_factors = True
        self.model_opts['ard_factors'] = ard_factors

        # Define whether to use view and factor-wise ARD prior for W
        if ((self.dimensionalities["M"]>1) & (ard_weights==False)): 
            print("WARNING: 'ard_weights' in model_options should be set to True if using multiple views\n")
        # if self.dimensionalities["M"]>1: ard_weights = True
        self.model_opts['ard_weights'] = ard_weights

        # Define initial number of latent factors
        self.dimensionalities["K"] = self.model_opts['factors'] = int(factors)


        # Define likelihoods
        self.model_opts['likelihoods'] = self.likelihoods

        print("Model options:")
        print("- Automatic Relevance Determination prior on the factors: %s" % str(ard_factors))
        print("- Automatic Relevance Determination prior on the weights: %s" % str(ard_weights))
        print("- Spike-and-slab prior on the factors: %s" % str(spikeslab_factors))
        print("- Spike-and-slab prior on the weights: %s" % str(spikeslab_weights))
        print("Likelihoods:")
        for m in range(self.dimensionalities["M"]):
          print("- View %d (%s): %s" % (m,self.data_opts["views_names"][m],self.likelihoods[m]) )
        print("\n")

    def set_data_options(self, scale_views=False, scale_groups = False, center_groups = True, use_float32 = False):
        """ Set data processing options """

        if not hasattr(self, 'data_opts'): self.data_opts = {}
        self.data_opts['center_groups'] = center_groups

        # change from float64 to float32 (to decrease memory usage and improve speed)
        config.use_float32 = use_float32
        if (use_float32): print("use_float32 set to True: replacing float64 arrays by float32 arrays to speed up computations...\n")

        # Scale views to unit variance
        self.data_opts['scale_views'] = scale_views
        if (scale_views): print("Scaling views to unit variance...\n")
        
        # Scale groups to unit variance
        self.data_opts['scale_groups'] = scale_groups
        if (scale_groups): print("Scaling groups to unit variance...\n")

    def build(self):
        """ Build the model """

        # Sanity checks
        assert hasattr(self, 'train_opts'), "Training options not defined"
        assert hasattr(self, 'model_opts'), "Model options not defined"
        assert hasattr(self, 'dimensionalities'), "Dimensionalities are not defined"
        if hasattr(self, 'smooth_opts'):
            assert len(self.smooth_opts)>2, "Smooth covariates applied but smooth options not defined. Please define set_smooth_options() before build()"
        if np.any(np.array(self.dimensionalities["D"])<15):
            print("\nWarning: some view(s) have less than 15 features, MOFA won't be able to learn meaningful factors for these view(s)...\n")
        _, counts = np.unique(self.data_opts["samples_groups"], axis=0, return_counts=True)
        if not hasattr(self, 'smooth_opts'):
            if np.any(counts<15):
                print("\nWarning: some group(s) have less than 15 samples, MOFA won't be able to learn meaningful factors for these group(s)...\n")

        # Build the nodes
        if hasattr(self, 'smooth_opts'):
            tmp = build_mofa_smooth(self.data, self.sample_cov, self.dimensionalities, self.data_opts, self.model_opts, self.train_opts, self.smooth_opts)
        else:
            tmp = buildBiofam(self.data, self.dimensionalities, self.data_opts, self.model_opts, self.train_opts)
        tmp.main()

        # Create BayesNet class
        if self.train_opts['stochastic']:
            self.model = StochasticBayesNet(self.dimensionalities, tmp.get_nodes())
        else:
            self.model = BayesNet(self.dimensionalities, tmp.get_nodes())

    @keyboardinterrupt_saver
    def run(self):
        """ Run the model """

        # Sanity checks
        assert hasattr(self, 'model_opts'), "Model options not defined"
        assert hasattr(self, 'train_opts'), "Train options not defined"
        assert hasattr(self, 'data_opts'), "Data options not defined"

        # Fetch training schedule (order of updates for the different nodes)
        if 'schedule' in self.train_opts:
            assert set(self.train_opts['schedule']) == set(list(self.model.getNodes().keys())), "Some nodes defined in the training schedule are not present in the model, or viceversa"
        else:
            self.train_opts['schedule'] = self.model_builder.schedule

        # Set training options
        self.model.setTrainOptions(self.train_opts)

        # Train the model
        train_model(self.model)

    def mask_outliers(self):

        Z = self.model.nodes['Z'].getExpectation()
        zscore_cutoff = 3 * 1.96   # z-score cutoff
        value_cutoff = 1       # max factor value

        for g in range(len(self.data_opts['groups_names'])):
            idx = np.where(np.array(self.data_opts["samples_groups"]) == self.data_opts['groups_names'][g])[0]
            Ztmp = Z[idx,:] # is this by reference? apparently not

            # calculate outlier score
            z_score = np.absolute((Ztmp-Ztmp.mean(axis=0))) / np.std(Ztmp, axis=0)

            # mask outliers with np.nan
            Ztmp[(z_score>zscore_cutoff) & (np.absolute(Z[idx,:])>value_cutoff)] = np.nan
            Z[idx,:] = Ztmp

    def predict_factor(self, new_covariates = None, uncertainty=True, groups = "all"):
        """
        Predict factor values at new covariate values or in missing groups
        new_covariates: new covariate values to predict at, should be of same format as covariates.
                        If None all present covariate values are used
        uncertainty: provide uncertainty measures for the prediction
        groups: Groups for which to predict, default is "all"
        """

        assert hasattr(self, 'smooth_opts'), "Using predict_factors requires the use of GP_factors," \
                                              " maybe you want to use impute instead?"

        if new_covariates is None:
            new_covariates = self.sample_cov

        new_covariates = np.unique(new_covariates, axis =0)

        # get group-covariate combinations included in the model
        old_groups = pd.factorize(self.model.nodes['Sigma'].group_labels)[0]
        old_covariates = self.model.nodes['Sigma'].sample_cov_transformed
        all_covariates = np.unique(np.vstack([new_covariates, old_covariates]), axis =0)
        N = all_covariates.shape[0]
        M = new_covariates.shape[0]

        # get the necessary expectations
        Z = self.model.nodes['Z'].getExpectations()['E']
        K = Z.shape[1]

        Sigma_terms = self.model.nodes['Sigma'].getExpectations()
        Sigma = Sigma_terms['cov']
        Sigma_inv = Sigma_terms['inv']
                
        GP_param = self.model.nodes['Sigma'].getParameters()
        if groups == "all":
            G = len(self.model.nodes['Sigma'].groups)
            groups = np.arange(G)
        else:
            G = len(groups)
        if not self.smooth_opts['model_groups']:
            Kg = np.ones([K, G, G])
        else:
            Kg = self.model.nodes['Sigma'].getParameters()['Kg']
            Kg = Kg[:, groups, :][:, :, groups]

        # which rows/columns in Sigma_new correspond to original Sigma and which to new test covariates?
        old = np.hstack([old_groups[:,None], old_covariates])
        all = np.hstack([np.repeat(groups, all_covariates.shape[0])[:,None], np.tile(all_covariates.transpose(), G).transpose()])
        new = np.hstack([np.repeat(groups, new_covariates.shape[0])[:,None], np.tile(new_covariates.transpose(), G).transpose()])
        oldix = np.concatenate([np.where(np.all(np.equal(old[j,:], all), axis = 1))[0] for j in range(old.shape[0])])
        newidx = np.concatenate([np.where(np.all(np.equal(new[j,:], all), axis = 1))[0] for j in range(new.shape[0])])

        Z_new_mean = np.zeros([M*G, K])
        if uncertainty:
            Z_new_var = np.zeros([M*G, K])
        for k in range(K):
            Kc_new = self.model.nodes['Sigma'].Kc.eval_at_newpoints_k(all_covariates,k)
            K_new_k = GP_param['scale'][k] * np.kron(Kg[k,:,:], Kc_new)
            Sigma_new_k = K_new_k + (1 - GP_param['scale'][k]) * np.eye(Kc_new.shape[0] * G)
            if not self.smooth_opts['sparseGP']:
                Z_new_mean[:,k] = gpu_utils.dot(K_new_k[newidx, :][:, oldix], gpu_utils.dot(Sigma_inv[k,:,:], Z[:,k]))
            else:
                U = self.model.nodes['U'].getExpectations()['E']
                old_sub = oldix[self.model.nodes['Sigma'].idx_inducing]
                Z_new_mean[:, k] = gpu_utils.dot(K_new_k[newidx, :][:, old_sub],
                                                 gpu_utils.dot(Sigma_inv[k, :, :], U[:, k]))
            if uncertainty:
                # marginal variances p(z, z*|c, c*, y) =  p(z*|z, y,c,c*) *p(z|y,c,c*)  = p(z*|z, c*) * p(z|y,c)
                # Var[z*] = Var([E(z*|z)]) + Var(E(z*|z))
                # if not self.model_opts['mv_Znode']:
                #     Z2 = np.diag(self.model.nodes['Z'].getExpectations()['E2'][:,k] - self.model.nodes['Z'].getExpectations()['E'][:,k]**2)
                # TODO: avoid calculating full matrix, only require diagonal
                if not self.smooth_opts['sparseGP']:
                    Z2 = self.model.nodes['Z'].getExpectations()['cov'][k,:,:] #A_k
                    Z_new_var[:, k] = np.diag(Sigma_new_k[newidx, :][:, newidx] - \
                                              gpu_utils.dot(K_new_k[newidx, :][:, oldix],
                                                            gpu_utils.dot(Sigma_inv[k, :, :],
                                                                          K_new_k[oldix, :][:, newidx])) + \
                                              gpu_utils.dot(K_new_k[newidx, :][:, oldix],
                                                            gpu_utils.dot(Sigma_inv[k, :, :],
                                                                          gpu_utils.dot(Z2, gpu_utils.dot(
                                                                              Sigma_inv[k, :, :],
                                                                              K_new_k[oldix, :][:, newidx])))))
                else:
                    Z2 = self.model.nodes['U'].getExpectations()['cov'][k,:,:]
                    Z_new_var[:, k] = np.diag(Sigma_new_k[newidx, :][:, newidx] - \
                                              gpu_utils.dot(K_new_k[newidx, :][:, old_sub],
                                                            gpu_utils.dot(Sigma_inv[k, :, :],
                                                                          K_new_k[old_sub, :][:, newidx])) + \
                                              gpu_utils.dot(K_new_k[newidx, :][:, old_sub],
                                                            gpu_utils.dot(Sigma_inv[k, :, :],
                                                                          gpu_utils.dot(Z2, gpu_utils.dot(
                                                                              Sigma_inv[k, :, :],
                                                                              K_new_k[old_sub, :][:, newidx])))))


        if uncertainty:
            assert np.all(Z_new_var >= 0), "Something went wrong in the prediction: variances of the predictive distribution are negative"
            self.Zpredictions = {"mean": Z_new_mean, "variance": Z_new_var, 'values' : new_covariates, 'groups' : groups}
        else:
            self.Zpredictions = {"mean": Z_new_mean, "variance": None, 'values' : new_covariates, 'groups' : groups}

        self.Zcompleted = True

    def impute(self, uncertainty=True, mask_outliers = True):
        """
        impute missing values with or without uncertainty estimates
        """

        # detect and mask outliers that could skew the results
        if mask_outliers:
            self.mask_outliers()

        # get the necessary expectations
        W = [w['E'] for w in self.model.nodes['W'].getExpectations()]
        Z = self.model.nodes['Z'].getExpectations()['E']

        # Predict the mean
        pred_mean = [Z.dot(w.T)for w in W]

        # for non-gaussian likelihoods, convert from pseudodata space to observation space
        for m in range(len(pred_mean)):
            if self.model_opts["likelihoods"][m]=="bernoulli":
                pred_mean[m] = np.round(np.exp(pred_mean[m])/(1+np.exp(pred_mean[m])))
            elif self.model_opts["likelihoods"][m]=="poisson":
                pred_mean[m] = np.round(np.log(1.+np.exp(pred_mean[m])))

        # Predict the variance
        if uncertainty:
            W2 = [w['E2'] for w in self.model.nodes['W'].getExpectations()]
            Z2 = self.model.nodes['Z'].getExpectations()['E2']
            Tau = [tau['E'] for tau in self.model.nodes['Tau'].getExpectations()]
            pred_var = [Z2.dot(W2[v].T) - (Z**2.).dot(W[v].T**2.) + 1./Tau[v] for v in range(len(W))]
            self.imputed_data = { "mean":pred_mean, "variance":pred_var }
        else:
            self.imputed_data = { "mean":pred_mean, "variance":None }

        # Only impute missing data
        for m in range(len(W)):
            mask = self.model.nodes['Y'].getNodes()[m].getMask()
            self.imputed_data["mean"][m][~mask] = self.data[m][~mask]
            self.imputed_data["variance"][m][~mask] = np.nan

        self.imputed = True # change flag

    def save(self, outfile=None, save_data=True, save_parameters=False, expectations=None):
        """ Save the model in an hdf5 file """

        # Sanity checks
        assert hasattr(self, 'data'), "Data has to be defined before training the model"
        assert hasattr(self, 'model'), "No trained model found"

        # Use outfile from training options if an outfile to override it is not provided
        if outfile is None or outfile == "":
            if self.train_opts["outfile"] is None or self.train_opts["outfile"] == "":
                outfile = os.path.join('/tmp', "mofa_{}.hdf5".format(strftime('%Y%m%d-%H%M%S')))
                print("No output file name provided as a training options or to the save method. Saving to {} .".format(
                    outfile
                    ))
            else:
                outfile = self.train_opts["outfile"]

        if os.path.isfile(outfile):
            print("Warning: Output file {} already exists, it will be replaced".format(outfile))

        # Create output directory
        if not os.path.isdir(os.path.dirname(outfile)) and (os.path.dirname(outfile) != ''):
            print("Output directory does not exist, creating it...")
            os.makedirs(os.path.dirname(outfile))
        print("Saving model in %s..." % outfile)
        sys.stdout.flush()

        if not hasattr(self, 'smooth_opts') or not hasattr(self, 'sample_cov'):
            sample_cov = None
            covariates_names = None
        else:
            sample_cov = self.sample_cov
            covariates_names = self.smooth_opts['covariates_names']

        # Save the model
        tmp = saveModel(
          model = self.model,
          outfile = outfile,
          data = self.data,
          intercepts = self.intercepts,
          samples_groups = self.data_opts['samples_groups'],
          sample_cov = sample_cov,
          train_opts = self.train_opts,
          model_opts = self.model_opts,
          samples_names = self.data_opts['samples_names'],
          features_names = self.data_opts['features_names'],
          views_names = self.data_opts['views_names'],
          groups_names = self.data_opts['groups_names'],
          covariates_names=covariates_names,
          samples_metadata = self.data_opts["samples_metadata"] if "samples_metadata" in self.data_opts else None,
          features_metadata = self.data_opts["features_metadata"] if "features_metadata" in self.data_opts else None,
          compression_level = 9
        )

        # Save sample and feature names
        tmp.saveNames()

        # Save metadata
        tmp.saveMetaData()

        # Save expectations

        # If all likelihoods are gaussian there is no need to save the expectations of Y, just saving the data is enough
        # TO-DO: THERE IS STH WRONG WITH THIS, CHECK WITH NON-GAUSS LIK
        # if all([i=="gaussian" for i in self.model_opts["likelihoods"]]):
        #     tmp.saveExpectations(nodes=["W","Z"])
        # else:
        #     tmp.saveExpectations(nodes=["W","Z"])

        if expectations is None:
            # Default is to save only W and Z nodes
            expectations = ["W", "Z", "Sigma"]
        
        tmp.saveExpectations(nodes=expectations)

        # Save parameters
        if save_parameters:
            tmp.saveParameters(nodes=["W","Z"])

        # Save model options
        tmp.saveModelOptions()

        # Save training options
        tmp.saveTrainOptions()

        # Smooth options
        if hasattr(self, 'smooth_opts'):
            tmp.saveSmoothOptions(self.smooth_opts)

        # Save training statistics
        tmp.saveTrainingStats()

        # Save variance explained values
        tmp.saveVarianceExplained()

        # Save data
        if save_data: 
            tmp.saveData()

        # Save imputed data
        if self.imputed:
            tmp.saveImputedData(self.imputed_data["mean"], self.imputed_data["variance"])

        # Save predictions
        if self.Zcompleted:
            tmp.saveZpredictions(self.Zpredictions["mean"], self.Zpredictions["variance"], self.Zpredictions['values'], self.Zpredictions['groups'])

def mofa(adata, groups_label: bool = None, use_raw: bool = False, use_layer: bool = None, 
         features_subset: Optional[str] = None,
         likelihood: Optional[Union[str, List[str]]] = None, n_factors: int = 10,
         scale_views: bool = False, scale_groups: bool = False,
         ard_weights: bool = True, ard_factors: bool = True,
         spikeslab_weights: bool = True, spikeslab_factors: bool = False,
         n_iterations: int = 1000, convergence_mode: str = "fast",
         gpu_mode: bool = False, use_float32: bool = False,
         save_parameters: bool = False, save_data: bool = True, save_metadata: bool = True,
         seed: int = 1, outfile: Optional[str] = None,
         expectations: Optional[List[str]] = None,
         save_interrupted: bool = False,
         verbose: bool = False, quiet: bool = True, copy: bool = False):
    """
    Helper function to init and build the model in a single call
    from annotation data object

    PARAMETERS
    ----------
    adata: an AnnotationData object
    groups_label (optional): a column name in adata.obs for grouping the samples
    use_raw (optional): use raw slot of AnnData as input values
    use_layer (optional): use a specific layer of AnnData as input values (supersedes use_raw option)
    features_subset (optional): .var column with a boolean value to select genes (e.g. "highly_variable"), None by default
    likelihood (optional): likelihood to use, default is guessed from the data
    n_factors (optional): number of factors to train the model with
    scale_views (optional): scale views to unit variance
    scale_groups (optional): scale groups to unit variance
    ard_weights (optional): use view-wise sparsity
    ard_factors (optional): use group-wise sparsity
    spikeslab_weights (optional): use feature-wise sparsity (e.g. gene-wise)
    spikeslab_factors (optional): use sample-wise sparsity (e.g. cell-wise)
    n_iterations (optional): upper limit on the number of iterations
    convergence_mode (optional): fast, medium, or slow convergence mode
    gpu_mode (optional): if to use GPU mode
    use_float32 (optional): if to use float32 precision
    save_parameters (optional): if to save training parameters
    save_data (optional): if to save training data
    save_metadata (optional): if to load metadata from the AnnData object (.obs and .var tables) and save it, False by default
    seed (optional): random seed
    outfile (optional): path to HDF5 file to store the model
    expectations (optional): which nodes should be used to save expectations for (will save only W and Z by default);
    possible expectations names include Y, W, Z, Tau, AlphaZ, AlphaW, ThetaW, ThetaZ
    outfile (optional): output file name
    save_interrupted (optional): if to save partially trained model when the training is interrupted
    verbose (optional): print verbose information during training
    quiet (optional): silence messages during training procedure
    copy (optional): return a copy of AnnData instead of writing to the provided object
    """

    ent = entry_point()

    lik = [likelihood] if likelihood is not None else None

    ent.set_data_options(scale_views=scale_views, scale_groups=scale_groups, use_float32=use_float32)
    ent.set_data_from_anndata(adata, groups_label=groups_label, use_raw=use_raw, use_layer=use_layer,
                              likelihoods=lik, features_subset=features_subset, save_metadata=save_metadata)
    ent.set_model_options(ard_factors=ard_factors, ard_weights=ard_weights, 
                          spikeslab_weights=spikeslab_weights, spikeslab_factors=spikeslab_factors, 
                          factors=n_factors)
    ent.set_train_options(iter=n_iterations, convergence_mode=convergence_mode, 
                          gpu_mode=gpu_mode, seed=seed, verbose=verbose, quiet=quiet, outfile=outfile, save_interrupted=save_interrupted)

    ent.build()
    ent.run()

    ent.save(outfile, save_data=save_data, save_parameters=save_parameters, expectations=expectations)

    try:
        import h5py
    except ImportError:
        h5py = None

    if h5py:
        if outfile is not None:
            f = h5py.File(outfile)
            if copy:
                adata = adata.copy()

            z = np.concatenate([v[:,:] for k, v in f['expectations']['Z'].items()], axis=1).T
            if groups_label:
                # Samples are grouped in sample groups
                # so the rows of the Z matrix have to be re-ordered
                zs = np.concatenate([v[:] for k, v in f["samples"].items()], axis=0).astype(str)
                z = pd.DataFrame(z, index=zs).loc[adata.obs_names.values].to_numpy()
            adata.obsm['X_mofa'] = z

            w = np.concatenate([v[:,:] for k, v in f['expectations']['W'].items()], axis=1).T
            if features_subset is None:
                # If all the features were used in training
                adata.varm['LFs'] = w
            else:
                # Set the weights of features that were not used to zero
                adata.varm["LFs"] = np.zeros(shape=(adata.n_vars, w.shape[1]))
                adata.varm["LFs"][adata.var[features_subset]] = w
            if copy:
                return adata
            else:
                if features_subset is None:
                    print("Saved MOFA embeddings in adata.obsm['X_mofa'] slot and their loadings in adata.varm['LFs'].")
                else:
                    print("Saved MOFA embeddings in adata.obsm['X_mofa'] slot.")
    else:
        print("Can not add embeddings and loadings to AnnData object since h5py is not innp.alled.")
