import numpy as np

from math import ceil
from sklearn.cluster import KMeans
from copy import deepcopy

# All of this functionaliy has been taken from https://github.com/willtownes/nsf-paper/blob/main/utils/preprocess.py
# and https://github.com/willtownes/nsf-paper/blob/main/utils/postprocess.py
def rescale_spatial_coords(X,box_side=4):
  """
  X is an NxD matrix of spatial coordinates
  Returns a rescaled version of X such that aspect ratio is preserved
  But data are centered at zero and area of equivalent bounding box set to
  box_side^D
  Goal is to rescale X to be similar to a N(0,1) distribution in all axes
  box_side=4 makes the data fit in range (-2,2)
  """
  xmin = X.min(axis=0)
  X -= xmin
  x_gmean = np.exp(np.mean(np.log(X.max(axis=0))))
  X *= box_side/x_gmean
  return X - X.mean(axis=0)

def scanpy_sizefactors(Y):
  sz = Y.sum(axis=1,keepdims=True)
  return sz/np.median(sz)

def anndata_to_train_val(ad, layer=None, nfeat=None, train_frac=0.95,
                         sz="constant", dtp="float32", flip_yaxis=True):
  """
  Convert anndata object ad to a training data dictionary
  and a validation data dictionary
  Requirements:
  * rows of ad are pre-shuffled to ensure random split of train/test
  * spatial coordinates in ad.obsm['spatial']
  * features (cols) of ad sorted in decreasing importance (eg with deviance)
  """
  from contextlib import suppress
  if nfeat is not None: ad = ad[:,:nfeat]
  N = ad.shape[0]
  Ntr = round(train_frac*N)
  X = ad.obsm["spatial"].copy().astype(dtp)
  if flip_yaxis: X[:,1] = -X[:,1]
  X = rescale_spatial_coords(X)
  if layer is None: Y = ad.X
  else: Y = ad.layers[layer]
  with suppress(AttributeError):
    Y = Y.toarray() #in case Y is a sparse matrix
  Y = Y.astype(dtp)
  Dtr = {"X":X[:Ntr,:], "Y":Y[:Ntr,:]}
  Dval = {"X":X[Ntr:,:], "Y":Y[Ntr:,:]}
  if sz=="constant":
    Dtr["sz"] = np.ones((Ntr,1),dtype=dtp)
    Dval["sz"] = np.ones((N-Ntr,1),dtype=dtp)
  elif sz=="mean":
    Dtr["sz"] = Dtr["Y"].mean(axis=1,keepdims=True)
    Dval["sz"] = Dval["Y"].mean(axis=1,keepdims=True)
  elif sz=="scanpy":
    Dtr["sz"] = scanpy_sizefactors(Dtr["Y"])
    Dval["sz"] = scanpy_sizefactors(Dval["Y"])
  else:
    raise ValueError("unrecognized size factors 'sz'")
  Dtr["idx"] = np.arange(Ntr)
  if Ntr>=N: Dval = None #avoid returning an empty array
  return Dtr,Dval

def minibatch_size_adjust(num_obs,batch_size):
  """
  Calculate adjusted minibatch size that divides
  num_obs as evenly as possible
  num_obs : number of observations in full data
  batch_size : maximum size of a minibatch
  """
  
  nbatch = ceil(num_obs/float(batch_size))
  return int(ceil(num_obs/nbatch))

def prepare_datasets_tf(Dtrain,Dval=None,shuffle=False,batch_size=None):
  """
  Dtrain and Dval are dicts containing numpy np.arrays of data.
  Dtrain must contain the key "Y"
  Returns a from_tensor_slices conversion of Dtrain and a dict of tensors for Dval
  """
  
  from tensorflow import constant
  from tensorflow.data import Dataset
  Ntr = Dtrain["Y"].shape[0]
  if batch_size is None:
    #ie one batch containing all observations by default
    batch_size = Ntr
  else:
    batch_size = minibatch_size_adjust(Ntr,batch_size)
  Dtrain = Dataset.from_tensor_slices(Dtrain)
  if shuffle:
    Dtrain = Dtrain.shuffle(Ntr)
  Dtrain = Dtrain.batch(batch_size)
  if Dval is not None:
    Dval = {i:constant(Dval[i]) for i in Dval}
  return Dtrain, Ntr, Dval

def kmeans_inducing_pts(X,M):
  M = int(M)
  Z = np.unique(X, axis=0)
  unique_locs = Z.shape[0]
  if M<unique_locs:
    Z=KMeans(n_clusters=M).fit(X).cluster_centers_
  return Z

def t2np(X):
  return X.numpy().mean(axis=0)

def normalize_cols(W):
  """
  Rescale the columns of a matrix to sum to one
  """
  wsum = W.sum(axis=0)
  return W/wsum, wsum

def normalize_rows(W):
  """
  Rescale the rows of a matrix to sum to one
  """
  wsum = W.sum(axis=1)
  return (W.T/wsum).T, wsum

def interpret_nonneg(factors,loadings,lda_mode=False,sort=True):
  """
  Rescale factors and loadings from a nonnegative factorization
  to improve interpretability. Two possible rescalings:

  1. Soft clustering of observations (lda_mode=True):
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "factor_sums"
  factor_sums is the "n" in the multinomial
  (ie the sum of the counts per observations)

  2. Soft clustering of features (lda_mode=False):
  Rows of loadings matrix sum to one, cols of factors matrix sum to one
  Returns a dict with keys: "factors", "loadings", and "feature_sums"
  feature_sums is similar to an intercept term for each feature
  """
  if lda_mode:
    W,eF,eFsum = rescale_as_lda(factors,loadings,sort=sort)
    return {"factors":eF,"loadings":W,"totals":eFsum}
  else: #spatialDE mode
    eF,W,Wsum = rescale_as_lda(loadings,factors,sort=sort)
    return {"factors":eF,"loadings":W,"totals":Wsum}

def interpret_nsf(fit,X,S=10,**kwargs):
  """
  fit: object of type SF with non-negative factors
  X: spatial coordinates to predict on
  returns: interpretable loadings W, factors eF, and total counts vector
  """
  Fhat = t2np(fit.sample_latent_GP_funcs(X,S=S,chol=False)).T #NxL
  return interpret_nonneg(np.exp(Fhat),fit.W.numpy(),**kwargs)

def rescale_as_lda(factors,loadings,sort=True):
  """
  Rescale nonnegative factors and loadings matrices to be
  comparable to LDA:
  Rows of factor matrix sum to one, cols of loadings matrix sum to one
  """
  W = deepcopy(loadings)
  eF = deepcopy(factors)
  W,wsum = normalize_cols(W)
  eF,eFsum = normalize_rows(eF*wsum)
  if sort:
    o = np.argsort(-eF.sum(axis=0))
    return W[:,o],eF[:,o],eFsum
  else:
    return W,eF,eFsum