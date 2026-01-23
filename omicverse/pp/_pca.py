from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload, get_args, get_origin
from typing import Union as LegacyUnionType
from warnings import warn
from functools import reduce
from operator import or_

import anndata as ad
import numpy as np
from anndata import AnnData
from packaging.version import Version
from sklearn.utils import check_random_state

from .._settings import EMOJI, Colors



from scanpy import logging as logg
from ._compat import DaskArray, pkg_version
from scanpy._settings import settings
from scanpy._utils import _doc_params, _empty, is_backed_type
from scanpy.get import  _get_obs_rep
from scanpy.preprocessing._docs import doc_mask_var_hvg
from scanpy.preprocessing._pca._compat import _pca_compat_sparse
from scipy import sparse

# Handle Union types for different Python versions
try:
    from types import UnionType
except ImportError:
    # Python < 3.10
    UnionType = type(None)

def get_literal_vals(typ):
    """Get all literal values from a Literal or Union of ... of Literal type.
    
    This is a custom implementation based on scanpy's original function.
    
    Parameters
    ----------
    typ : type
        A Literal type or Union of Literal types
        
    Returns
    -------
    KeysView
        Keys view of all literal values from the type
    """
    # Get the origin and args of the type
    origin = get_origin(typ)
    args = get_args(typ)
    
    # Handle Union types (both typing.Union and | syntax)
    if origin is LegacyUnionType:
        return reduce(
            or_, (dict.fromkeys(get_literal_vals(t)) for t in args)
        ).keys()
    
    # Handle new Python 3.10+ Union syntax (X | Y)
    # For | syntax, origin might be None but __class__ could be UnionType
    try:
        if hasattr(typ, '__class__') and typ.__class__.__name__ == 'UnionType':
            return reduce(
                or_, (dict.fromkeys(get_literal_vals(t)) for t in args)
            ).keys()
    except (NameError, AttributeError):
        pass
    
    # Additional check for Union-like structures
    if hasattr(typ, '__origin__') and typ.__origin__ is LegacyUnionType:
        return reduce(
            or_, (dict.fromkeys(get_literal_vals(t)) for t in get_args(typ))
        ).keys()
    
    # Handle | syntax by checking for args when origin is None
    if origin is None and args and len(args) > 1:
        # This might be a | union
        try:
            return reduce(
                or_, (dict.fromkeys(get_literal_vals(t)) for t in args)
            ).keys()
        except (TypeError, RecursionError):
            pass
    
    # Handle Literal types
    if origin is Literal:
        return dict.fromkeys(args).keys()
    
    # If it's not a Literal or Union, raise an error
    msg = f"{typ} is not a valid Literal"
    raise TypeError(msg)


if TYPE_CHECKING:
    from collections.abc import Container
    from collections.abc import Set as AbstractSet
    from typing import LiteralString, TypeVar

    import dask_ml.decomposition as dmld
    import sklearn.decomposition as skld
    from numpy.typing import DTypeLike, NDArray

    from scanpy._utils import Empty
    from scanpy._utils.random import _LegacyRandom

    MethodDaskML = type[dmld.PCA | dmld.IncrementalPCA | dmld.TruncatedSVD]
    MethodSklearn = type[skld.PCA | skld.TruncatedSVD]

    T = TypeVar("T", bound=LiteralString)
    M = TypeVar("M", bound=LiteralString)


SvdSolvPCADaskML = Literal["auto", "full", "tsqr", "randomized"]
SvdSolvTruncatedSVDDaskML = Literal["tsqr", "randomized"]
SvdSolvDaskML = SvdSolvPCADaskML | SvdSolvTruncatedSVDDaskML

if pkg_version("scikit-learn") >= Version("1.5") or TYPE_CHECKING:
    SvdSolvPCASparseSklearn = Literal["arpack", "covariance_eigh"]
else:
    SvdSolvPCASparseSklearn = Literal["arpack"]
SvdSolvPCADenseSklearn = Literal["auto", "full", "randomized"] | SvdSolvPCASparseSklearn
SvdSolvTruncatedSVDSklearn = Literal["arpack", "randomized"]
SvdSolvSkearn = (
    SvdSolvPCADenseSklearn | SvdSolvPCASparseSklearn | SvdSolvTruncatedSVDSklearn
)

SvdSolvPCACustom = Literal["covariance_eigh"]

SvdSolver = SvdSolvDaskML | SvdSolvSkearn | SvdSolvPCACustom

SpBase = sparse.spmatrix | sparse.sparray  # noqa: TID251
"""Only use when you directly convert it to a known subclass."""

_CSArray = sparse.csr_array | sparse.csc_array  # noqa: TID251
"""Only use if you want to specially handle arrays as opposed to matrices."""

_CSMatrix = sparse.csr_matrix | sparse.csc_matrix  # noqa: TID251
"""Only use if you want to specially handle matrices as opposed to arrays."""

CSRBase = sparse.csr_matrix | sparse.csr_array  # noqa: TID251
CSCBase = sparse.csc_matrix | sparse.csc_array  # noqa: TID251
CSBase = _CSArray | _CSMatrix


# Helper utilities for cross-backend dtype and array handling
def _normalize_to_numpy_dtype(dt):
    """Normalize a dtype-like (including torch.dtype) to a NumPy dtype.

    Accepts strings, NumPy dtypes, and torch.dtypes. Falls back to float32.
    """
    import numpy as _np
    try:  # Handle torch dtypes if torch is available
        import torch as _torch  # type: ignore
        torch_map = {
            _torch.float32: _np.float32,
            _torch.float64: _np.float64,
            _torch.float16: _np.float16,
            # Map bfloat16 to float32 for safety if NumPy bfloat16 not available
            getattr(_torch, "bfloat16", None): getattr(_np, "bfloat16", _np.float32),
            _torch.int64: _np.int64,
            _torch.int32: _np.int32,
            _torch.int16: _np.int16,
            _torch.int8: _np.int8,
            _torch.uint8: _np.uint8,
            _torch.bool: _np.bool_,
        }
        if isinstance(dt, _torch.dtype):
            mapped = torch_map.get(dt)
            return _np.dtype(mapped if mapped is not None else _np.float32)
    except Exception:
        pass
    # Fall back to NumPy's dtype constructor
    try:
        return _np.dtype(dt)
    except Exception:
        return _np.dtype(_np.float32)


def _ensure_numpy_array(x):
    """Convert known GPU/accelerator array types (torch, cupy) to NumPy.

    Leaves NumPy arrays, SciPy sparse, and Dask arrays untouched.
    """
    # SciPy sparse should be left as-is
    if sparse.issparse(x):
        return x
    # Try torch tensors
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    # Try CuPy arrays
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    # Default: return as-is
    return x


@_doc_params(
    mask_var_hvg=doc_mask_var_hvg,
)
def pca(  # noqa: PLR0912, PLR0913, PLR0915
    data: AnnData | np.ndarray | CSBase,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    zero_center: bool | None = True,
    svd_solver: SvdSolver | None = None,
    random_state: _LegacyRandom = 0,
    return_info: bool = False,
    mask_var: NDArray[np.bool_] | str | None | Empty = _empty,
    use_highly_variable: bool | None = None,
    dtype: DTypeLike = "float32",
    chunked: bool = False,
    chunk_size: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
    use_gpu: bool = False,
) -> AnnData | np.ndarray | CSBase | None:
    r"""Principal component analysis with GPU acceleration support.

    Compute PCA coordinates, loadings and variance decomposition for single-cell data.
    This implementation includes GPU acceleration options using PyTorch/TorchDR 
    for improved performance on large datasets.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    n_comps
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.
    layer
        If provided, which element of layers to use for PCA.
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables
        (uses *scikit-learn* :class:`~sklearn.decomposition.TruncatedSVD` or
        *dask-ml* :class:`~dask_ml.decomposition.TruncatedSVD`),
        which allows to handle sparse input efficiently.
        Passing `None` decides automatically based on sparseness of the data.
    svd_solver
        SVD solver to use:

        `None`
            See `chunked` and `zero_center` descriptions to determine which class will be used.
            Depending on the class and the type of X different values for default will be set.
            For sparse *dask* arrays, will use `'covariance_eigh'`.
            If *scikit-learn* :class:`~sklearn.decomposition.PCA` is used, will give `'arpack'`,
            if *scikit-learn* :class:`~sklearn.decomposition.TruncatedSVD` is used, will give `'randomized'`,
            if *dask-ml* :class:`~dask_ml.decomposition.PCA` or :class:`~dask_ml.decomposition.IncrementalPCA` is used, will give `'auto'`,
            if *dask-ml* :class:`~dask_ml.decomposition.TruncatedSVD` is used, will give `'tsqr'`
        `'arpack'`
            for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`)
            Not available with *dask* arrays.
        `'covariance_eigh'`
            Classic eigendecomposition of the covariance matrix, suited for tall-and-skinny matrices.
            With dask, array must be CSR or dense and chunked as (N, adata.shape[1]).
        `'randomized'`
            for the randomized algorithm due to Halko (2009). For *dask* arrays,
            this will use :func:`~dask.array.linalg.svd_compressed`.
        `'auto'`
            chooses automatically depending on the size of the problem.
        `'tsqr'`
            Only available with dense *dask* arrays. "tsqr"
            algorithm from Benson et. al. (2013).

        .. versionchanged:: 1.9.3
           Default value changed from `'arpack'` to None.
        .. versionchanged:: 1.4.5
           Default value changed from `'auto'` to `'arpack'`.

        Efficient computation of the principal components of a sparse matrix
        currently only works with the `'arpack`' or `'covariance_eigh`' solver.

        If X is a sparse *dask* array, a custom `'covariance_eigh'` solver will be used.
        If X is a dense *dask* array, *dask-ml* classes :class:`~dask_ml.decomposition.PCA`,
        :class:`~dask_ml.decomposition.IncrementalPCA`, or
        :class:`~dask_ml.decomposition.TruncatedSVD` will be used.
        Otherwise their *scikit-learn* counterparts :class:`~sklearn.decomposition.PCA`,
        :class:`~sklearn.decomposition.IncrementalPCA`, or
        :class:`~sklearn.decomposition.TruncatedSVD` will be used.
    random_state
        Change to use different initial states for the optimization.
    return_info
        Only relevant when not passing an :class:`~anndata.AnnData`:
        see “Returns”.
    {mask_var_hvg}
    layer
        Layer of `adata` to use as expression values.
    dtype
        Numpy data type string to which to convert the result.
    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `random_seed` and `svd_solver`. Uses sklearn :class:`~sklearn.decomposition.IncrementalPCA` or
        *dask-ml* :class:`~dask_ml.decomposition.IncrementalPCA`. If `False`, perform a full PCA and
        use sklearn :class:`~sklearn.decomposition.PCA` or
        *dask-ml* :class:`~dask_ml.decomposition.PCA`
    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.
    key_added
        If not specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\ `['X_pca']`, the loadings as
        :attr:`~anndata.AnnData.varm`\ `['PCs']`, and the the parameters in
        :attr:`~anndata.AnnData.uns`\ `['pca']`.
        If specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\ ``[key_added]``, the loadings as
        :attr:`~anndata.AnnData.varm`\ ``[key_added]``, and the the parameters in
        :attr:`~anndata.AnnData.uns`\ ``[key_added]``.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Is ignored otherwise.

    Returns
    -------
    If `data` is array-like and `return_info=False` was passed,
    this function returns the PCA representation of `data` as an
    array of the same type as the input array.

    Otherwise, it returns `None` if `copy=False`, else an updated `AnnData` object.
    Sets the following fields:

    `.obsm['X_pca' | key_added]` : :class:`~scipy.sparse.csr_matrix` | :class:`~scipy.sparse.csc_matrix` | :class:`~numpy.ndarray` (shape `(adata.n_obs, n_comps)`)
        PCA representation of data.
    `.varm['PCs' | key_added]` : :class:`~numpy.ndarray` (shape `(adata.n_vars, n_comps)`)
        The principal components containing the loadings.
    `.uns['pca' | key_added]['variance_ratio']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Ratio of explained variance.
    `.uns['pca' | key_added]['variance']` : :class:`~numpy.ndarray` (shape `(n_comps,)`)
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.

    """
    logg_start = logg.info(f"computing PCA{EMOJI['start']}")
    if layer is not None and chunked:
        # Current chunking implementation relies on pca being called on X
        msg = "Cannot use `layer` and `chunked` at the same time."
        raise NotImplementedError(msg)

    # chunked calculation is not randomized, anyways
    if svd_solver in {"auto", "randomized"} and not chunked:
        logg.info(
            f"{EMOJI['warning']} scikit-learn's randomized PCA might not be exactly "
            "reproducible across different computational platforms. For exact "
            "reproducibility, choose `svd_solver='arpack'`."
        )
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(data)

    if isinstance(data, AnnData):
        if layer is None and not chunked and is_backed_type(data.X):
            msg = f"PCA is not implemented for matrices of type {type(data.X)} with chunked as False"
            raise NotImplementedError(msg)
        adata = data.copy() if copy else data
        return_anndata = True
    elif is_rust:
        adata = data
        return_anndata = True
    elif pkg_version("anndata") < Version("0.8.0rc1"):
        adata = AnnData(data, dtype=data.dtype)
        return_anndata = False
    else:
        adata = AnnData(data)
        return_anndata = False

    # Unify new mask argument and deprecated use_highly_varible argument
    mask_var_param, mask_var = _handle_mask_var(adata, mask_var, use_highly_variable)
    del use_highly_variable
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    if not is_rust:
        adata_comp = adata[:, mask_var] if mask_var is not None else adata
    else:
        adata_comp = adata.subset(var_indices=np.array(adata.var_names)[mask_var],inplace=False)
        #print(np.array(adata.var_names)[mask_var])

    if n_comps is None:
        min_dim = min(adata_comp.n_vars, adata_comp.n_obs)
        n_comps = min_dim - 1 if min_dim <= settings.N_PCS else settings.N_PCS

    logg.info(f"    with {n_comps=}")

    X = _get_obs_rep(adata_comp, layer=layer)

    # Handle rust backend X data
    if is_rust:
        # For rust backend, X might be a special object that needs slicing to get actual data
        if hasattr(X, '__getitem__') and not isinstance(X, (np.ndarray, sparse.spmatrix, sparse.sparray)):
            try:
                X = X[:]
            except Exception:
                pass

    if is_backed_type(X) and layer is not None:
        msg = f"PCA is not implemented for matrices of type {type(X)} from layers"
        raise NotImplementedError(msg)
    # See: https://github.com/scverse/scanpy/pull/2816#issuecomment-1932650529
    if (
        Version(ad.__version__) < Version("0.9")
        and mask_var is not None
        and isinstance(X, np.ndarray)
    ):
        warnings.warn(
            "When using a mask parameter with anndata<0.9 on a dense array, the PCA"
            "can have slightly different results due the array being column major "
            "instead of row major.",
            UserWarning,
            stacklevel=2,
        )

    # check_random_state returns a numpy RandomState when passed an int but
    # dask needs an int for random state
    if not isinstance(X, DaskArray):
        random_state = random_state
    elif not isinstance(random_state, int):
        msg = f"random_state needs to be an int, not a {type(random_state).__name__} when passing a dask array"
        raise TypeError(msg)

    if chunked:
        if (
            not zero_center
            or random_state
            or (svd_solver is not None and svd_solver != "arpack")
        ):
            logg.debug("Ignoring zero_center, random_state, svd_solver")

        incremental_pca_kwargs = dict()
        if use_gpu and not isinstance(X, DaskArray):
            from numpy import zeros
            import torch
            import gc
            from .._settings import get_optimal_device, prepare_data_for_device
            device = get_optimal_device(prefer_gpu=True, verbose=True)
            
            
            # For MPS devices, use MLX PCA instead of TorchDR
            if device.type == 'mps':
                try:
                    from ._pca_mlx import MLXPCA, MockPCA
                    logg.info(f"   {EMOJI['gpu']} Using MLX PCA for MPS device (chunked)")
                    print(f"   {Colors.GREEN}{EMOJI['gpu']} MLX PCA backend: Apple Silicon GPU chunked computation{Colors.ENDC}")
                    
                    # For chunked computation, we need to fit on all data first
                    # Collect all chunks and fit MLX PCA
                    all_chunks = []
                    for chunk, _, _ in adata_comp.chunked_X(chunk_size):
                        chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                        all_chunks.append(chunk_dense)
                    
                    # Concatenate all chunks
                    X_full = np.vstack(all_chunks)
                    
                    # Fit MLX PCA
                    mlx_pca = MLXPCA(n_components=n_comps, device="metal")
                    mlx_pca.fit(X_full)
                    
                    # Transform each chunk
                    X_pca = zeros((X.shape[0], n_comps), X.dtype)
                    start_idx = 0
                    for chunk, _, _ in adata_comp.chunked_X(chunk_size):
                        chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                        end_idx = start_idx + chunk_dense.shape[0]
                        X_pca[start_idx:end_idx] = mlx_pca.transform(chunk_dense)
                        start_idx = end_idx
                    
                    # Create a mock PCA object with sklearn-compatible interface
                    pca_ = MockPCA(mlx_pca)
                    
                except (ImportError, Exception) as e:
                    return e, None
                    logg.info(f"   {EMOJI['warning']} MLX PCA failed ({str(e)}), falling back to sklearn for MPS device (chunked)")
                    print(f"   {EMOJI['warning']} {Colors.WARNING}MLX PCA failed, using sklearn IncrementalPCA backend for MPS device{Colors.ENDC}")
                    
                    from sklearn.decomposition import IncrementalPCA
                    X_pca = zeros((X.shape[0], n_comps), X.dtype)
                    
                    pca_ = IncrementalPCA(n_components=n_comps, **incremental_pca_kwargs)
                    
                    for chunk, _, _ in adata_comp.chunked_X(chunk_size):
                        chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                        pca_.partial_fit(chunk_dense)

                    for chunk, start, end in adata_comp.chunked_X(chunk_size):
                        chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                        X_pca[start:end] = pca_.transform(chunk_dense)
            else:
                # Use TorchDR for non-MPS GPU devices (CUDA, etc.)
                logg.info(f"   {EMOJI['gpu']} Using TorchDR IncrementalPCA for {device.type.upper()} GPU (chunked)")
                print(f"   {EMOJI['gpu']} TorchDR IncrementalPCA backend: {device.type.upper()} GPU chunked computation")
                
                from torchdr import IncrementalPCA
                
                # Prepare data for GPU compatibility (float32 requirement)
                X = prepare_data_for_device(X, device, verbose=True)
                X_pca = zeros((X.shape[0], n_comps), X.dtype)
                
                # Reset memory stats only for CUDA devices
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                
                pca_ = IncrementalPCA(n_components=n_comps, device=device, 
                                      batch_size=chunk_size,
                                      **incremental_pca_kwargs)
                pca_.fit(X, check_input=True)
                X_pca = pca_.transform(X)
                
                del pca_
                gc.collect()
                
                # Clear cache based on device type
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        else:
            if isinstance(X, DaskArray):
                from dask.array import zeros
                from dask_ml.decomposition import IncrementalPCA

                incremental_pca_kwargs["svd_solver"] = _handle_dask_ml_args(
                    svd_solver, IncrementalPCA
                )
            else:
                from numpy import zeros
                from sklearn.decomposition import IncrementalPCA

            X_pca = zeros((X.shape[0], n_comps), X.dtype)
            pca_ = IncrementalPCA(n_components=n_comps, **incremental_pca_kwargs)

            for chunk, _, _ in adata_comp.chunked_X(chunk_size):
                chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                pca_.partial_fit(chunk_dense)

            for chunk, start, end in adata_comp.chunked_X(chunk_size):
                chunk_dense = chunk.toarray() if isinstance(chunk, CSBase) else chunk
                X_pca[start:end] = pca_.transform(chunk_dense)
        
    elif zero_center:
        if isinstance(X, CSBase) and (
            pkg_version("scikit-learn") < Version("1.4") or svd_solver == "lobpcg"
        ):
            if svd_solver not in (
                {"lobpcg"} | get_literal_vals(SvdSolvPCASparseSklearn)
            ):
                if svd_solver is not None:
                    msg = (
                        f"Ignoring {svd_solver=} and using 'arpack', "
                        "sparse PCA with sklearn < 1.4 only supports 'lobpcg' and 'arpack'."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                svd_solver = "arpack"
            elif svd_solver == "lobpcg":
                msg = (
                    f"{svd_solver=} for sparse relies on legacy code and will not be supported in the future. "
                    "Also the lobpcg solver has been observed to be inaccurate. Please use 'arpack' instead."
                )
                warnings.warn(msg, FutureWarning, stacklevel=2)
            X_pca, pca_ = _pca_compat_sparse(
                X, n_comps, solver=svd_solver, random_state=random_state
            )
        else:
            if not isinstance(X, DaskArray):
                if use_gpu:
                    from .._settings import get_optimal_device, prepare_data_for_device
                    device = get_optimal_device(prefer_gpu=True, verbose=True)
                    
                    # Use MLX for MPS devices (Apple Silicon optimization)
                    if device.type == 'mps':
                        try:
                            from ._pca_mlx import MLXPCA, MockPCA
                            logg.info(f"   {EMOJI['gpu']} Using MLX PCA for Apple Silicon MPS acceleration")
                            print(f"   {Colors.GREEN}{EMOJI['gpu']} MLX PCA backend: Apple Silicon GPU acceleration{Colors.ENDC}")

                            # Create MLX PCA instance (use "metal" for MLX)
                            mlx_pca = MLXPCA(n_components=n_comps, device="metal")

                            from scipy import sparse
                            if sparse.issparse(X):
                                print(f'    {Colors.GREEN}Converting sparse matrix to dense for MLX PCA{Colors.ENDC}')
                                X = X.toarray()
                            else:
                                X = np.asarray(X)

                            # Fit and transform (MLX PCA handles sparse matrices internally)
                            X_pca = mlx_pca.fit_transform(X)
                            
                            # Create a mock PCA object with sklearn-compatible interface
                            pca_ = MockPCA(mlx_pca)
                            
                        except (ImportError, Exception) as e:
                            logg.info(f"   {EMOJI['warning']} MLX PCA failed ({str(e)}), falling back to sklearn for MPS device")
                            print(f"   {EMOJI['warning']} {Colors.WARNING}MLX PCA failed, using sklearn backend for MPS device{Colors.ENDC}")
                            # For MPS devices, fall back to sklearn instead of TorchDR
                            from sklearn.decomposition import PCA
                            
                            svd_solver = _handle_sklearn_args(
                                svd_solver, PCA, sparse=isinstance(X, CSBase)
                            )
                            pca_ = PCA(
                                n_components=n_comps,
                                svd_solver=svd_solver,
                                random_state=random_state,
                            )
                            X_pca = pca_.fit_transform(X)
                    else:
                        # Use TorchDR for non-MPS GPU devices (CUDA, etc.)
                        logg.info(f"   {EMOJI['gpu']} Using TorchDR PCA for {device.type.upper()} GPU acceleration")
                        print(f"   {Colors.GREEN}{EMOJI['gpu']} TorchDR PCA backend: {device.type.upper()} GPU acceleration{Colors.ENDC}")
                        
                        if svd_solver == "auto":
                            svd_solver = "gesvd"
                        if svd_solver not in ["gesvd", "gesvdj", "gesvda"]:
                            svd_solver = "gesvd"
                        from torchdr import  PCA
                        import torch
                        
                        # Prepare data for GPU compatibility (float32 requirement)
                        X = prepare_data_for_device(X, device, verbose=True)

                        # TorchDR PCA requires dense arrays, convert sparse to dense
                        if isinstance(X, CSBase):
                            X = X.toarray()

                        pca_ = PCA(
                            n_components=n_comps,
                            device=device,
                            svd_driver=svd_solver,
                            random_state=random_state,
                        )
                        X_pca = pca_.fit_transform(X)
                else:
                    logg.info(f"   {EMOJI['cpu']} Using sklearn PCA for CPU computation")
                    print(f"   {Colors.CYAN}{EMOJI['cpu']} sklearn PCA backend: CPU computation{Colors.ENDC}")
                    
                    from sklearn.decomposition import PCA

                    svd_solver = _handle_sklearn_args(
                        svd_solver, PCA, sparse=isinstance(X, CSBase)
                    )
                    pca_ = PCA(
                        n_components=n_comps,
                        svd_solver=svd_solver,
                        random_state=random_state,
                    )
                    X_pca = pca_.fit_transform(X)
            elif isinstance(X._meta, CSBase) or svd_solver == "covariance_eigh":
                from ._dask import PCAEighDask

                if random_state != 0:
                    msg = f"Ignoring {random_state=} when using a sparse dask array"
                    warnings.warn(msg, UserWarning, stacklevel=2)
                if svd_solver not in {None, "covariance_eigh"}:
                    msg = f"Ignoring {svd_solver=} when using a sparse dask array"
                    warnings.warn(msg, UserWarning, stacklevel=2)
                pca_ = PCAEighDask(n_components=n_comps)
            else:
                from dask_ml.decomposition import PCA

                svd_solver = _handle_dask_ml_args(svd_solver, PCA)
                pca_ = PCA(
                    n_components=n_comps,
                    svd_solver=svd_solver,
                    random_state=random_state,
                )
            X_pca = pca_.fit_transform(X)
    else:
        if isinstance(X, DaskArray):
            if isinstance(X._meta, CSBase):
                msg = "Dask sparse arrays do not support zero-centering (yet)"
                raise TypeError(msg)
            from dask_ml.decomposition import TruncatedSVD

            svd_solver = _handle_dask_ml_args(svd_solver, TruncatedSVD)
        else:
            from sklearn.decomposition import TruncatedSVD

            svd_solver = _handle_sklearn_args(svd_solver, TruncatedSVD)

        logg.debug(
            "    without zero-centering: \n"
            "    the explained variance does not correspond to the exact statistical definition\n"
            "    the first component, e.g., might be heavily influenced by different means\n"
            "    the following components often resemble the exact PCA very closely"
        )
        pca_ = TruncatedSVD(
            n_components=n_comps, random_state=random_state, algorithm=svd_solver
        )
        X_pca = pca_.fit_transform(X)

    # Ensure X_pca is a NumPy array (or sparse) before dtype checks
    X_pca = _ensure_numpy_array(X_pca)

    # Normalize target dtype and cast if needed
    target_dtype = _normalize_to_numpy_dtype(dtype)
    # Use np.asarray to handle cases where X_pca is array-like
    if np.asarray(X_pca).dtype != target_dtype:
        X_pca = np.asarray(X_pca).astype(target_dtype, copy=False)

    if return_anndata:
        key_obsm, key_varm, key_uns = (
            ("X_pca", "PCs", "pca") if key_added is None else [key_added] * 3
        )
        adata.obsm[key_obsm] = X_pca

        if mask_var is not None:
            adata.varm[key_varm] = np.zeros(shape=(adata.n_vars, n_comps))
            # Handle CUDA tensors by converting to CPU first
            components = pca_.components_
            if hasattr(components, 'cpu'):  # Check if it's a torch tensor
                components = components.cpu().numpy()
            adata.varm[key_varm][mask_var] = components.T
        else:
            # Handle CUDA tensors by converting to CPU first
            components = pca_.components_
            if hasattr(components, 'cpu'):  # Check if it's a torch tensor
                components = components.cpu().numpy()
            adata.varm[key_varm] = components.T

        params = dict(
            zero_center=zero_center,
            use_highly_variable=mask_var_param == "highly_variable",
            mask_var=mask_var_param,
        )
        if layer is not None:
            params["layer"] = layer
        # Handle CUDA tensors for variance and variance_ratio
        # Check if attributes exist (some PCA implementations like TorchDR may not have them)
        if hasattr(pca_, 'explained_variance_'):
            variance = pca_.explained_variance_
        elif hasattr(pca_, 'singular_values_'):
            # Calculate explained variance from singular values if available
            variance = (pca_.singular_values_ ** 2) / (pca_.n_samples_ - 1) if hasattr(pca_, 'n_samples_') else (pca_.singular_values_ ** 2) / (n_comps - 1)
        else:
            # Fallback: provide zeros if no variance information is available
            variance = np.zeros(n_comps)
            logg.warning(f"   {EMOJI['warning']} PCA object missing explained_variance_ attribute, using zeros")
        
        if hasattr(pca_, 'explained_variance_ratio_'):
            variance_ratio = pca_.explained_variance_ratio_
        elif hasattr(pca_, 'singular_values_'):
            # Calculate explained variance ratio from singular values if available
            total_var = np.sum(variance) if isinstance(variance, np.ndarray) else variance.sum()
            variance_ratio = variance / total_var if total_var > 0 else np.zeros(n_comps)
        else:
            # Fallback: provide zeros if no variance ratio information is available
            variance_ratio = np.zeros(n_comps)
            logg.warning(f"   {EMOJI['warning']} PCA object missing explained_variance_ratio_ attribute, using zeros")
        
        if hasattr(variance, 'cpu'):  # Check if it's a torch tensor
            variance = variance.cpu().numpy()
        if hasattr(variance_ratio, 'cpu'):  # Check if it's a torch tensor
            variance_ratio = variance_ratio.cpu().numpy()
        #print(adata)
        adata.uns[key_uns] = dict(
            params=params,
            variance=variance,
            variance_ratio=variance_ratio,
        )

        logg.info(f"    finished{EMOJI['done']}", time=logg_start)
        logg.debug(
            "and added\n"
            f"    {key_obsm!r}, the PCA coordinates (adata.obs)\n"
            f"    {key_varm!r}, the loadings (adata.varm)\n"
            f"    'pca_variance', the variance / eigenvalues (adata.uns[{key_uns!r}])\n"
            f"    'pca_variance_ratio', the variance ratio (adata.uns[{key_uns!r}])"
        )
        return adata if copy else None
    else:
        logg.info(f"    finished{EMOJI['done']}", time=logg_start)
        if return_info:
            # Handle CUDA tensors for return values
            components = pca_.components_
            
            # Check if attributes exist (some PCA implementations like TorchDR may not have them)
            if hasattr(pca_, 'explained_variance_'):
                variance = pca_.explained_variance_
            elif hasattr(pca_, 'singular_values_'):
                # Calculate explained variance from singular values if available
                variance = (pca_.singular_values_ ** 2) / (pca_.n_samples_ - 1) if hasattr(pca_, 'n_samples_') else (pca_.singular_values_ ** 2) / (n_comps - 1)
            else:
                # Fallback: provide zeros if no variance information is available
                variance = np.zeros(n_comps)
                logg.warning(f"   {EMOJI['warning']} PCA object missing explained_variance_ attribute, using zeros")
            
            if hasattr(pca_, 'explained_variance_ratio_'):
                variance_ratio = pca_.explained_variance_ratio_
            elif hasattr(pca_, 'singular_values_'):
                # Calculate explained variance ratio from singular values if available
                total_var = np.sum(variance) if isinstance(variance, np.ndarray) else variance.sum()
                variance_ratio = variance / total_var if total_var > 0 else np.zeros(n_comps)
            else:
                # Fallback: provide zeros if no variance ratio information is available
                variance_ratio = np.zeros(n_comps)
                logg.warning(f"   {EMOJI['warning']} PCA object missing explained_variance_ratio_ attribute, using zeros")
            
            if hasattr(components, 'cpu'):  # Check if it's a torch tensor
                components = components.cpu().numpy()
            if hasattr(variance_ratio, 'cpu'):  # Check if it's a torch tensor
                variance_ratio = variance_ratio.cpu().numpy()
            if hasattr(variance, 'cpu'):  # Check if it's a torch tensor
                variance = variance.cpu().numpy()
                
            return (
                X_pca,
                components,
                variance_ratio,
                variance,
            )
        else:
            return X_pca


def _handle_mask_var(
    adata: AnnData,
    mask_var: NDArray[np.bool_] | str | Empty | None,
    use_highly_variable: bool | None,
) -> tuple[np.ndarray | str | None, np.ndarray | None]:
    """Unify new mask argument and deprecated use_highly_varible argument.

    Returns both the normalized mask parameter and the validated mask array.
    """
    # First, verify and possibly warn
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    if use_highly_variable is not None:
        hint = (
            'Use_highly_variable=True can be called through mask_var="highly_variable". '
            "Use_highly_variable=False can be called through mask_var=None"
        )
        msg = f"Argument `use_highly_variable` is deprecated, consider using the mask argument. {hint}"
        warn(msg, FutureWarning, stacklevel=2)
        if mask_var is not _empty:
            msg = f"These arguments are incompatible. {hint}"
            raise ValueError(msg)

    # Handle default case and explicit use_highly_variable=True
    if not is_rust:
        if use_highly_variable or (
            use_highly_variable is None
            and mask_var is _empty
            and "highly_variable" in adata.var.columns
        ):
            mask_var = "highly_variable"
        
            # Handle default case and explicit use_highly_variable=True
        if use_highly_variable or (
            use_highly_variable is None
            and mask_var is _empty
            and "highly_variable_features" in adata.var.columns
        ):
            mask_var = "highly_variable_features"
    else:
        if use_highly_variable or (
            use_highly_variable is None
            and mask_var is _empty
            and "highly_variable" in adata.var
        ):
            mask_var = "highly_variable"

        # Handle default case and explicit use_highly_variable=True
        if use_highly_variable or (
            use_highly_variable is None
            and mask_var is _empty
            and "highly_variable_features" in adata.var
        ):
            mask_var = "highly_variable_features"

    # Without highly variable genes, we don’t use a mask by default
    if mask_var is _empty or mask_var is None:
        return None, None
    return mask_var, _check_mask(adata, mask_var, "var")

def _check_mask(
    data: AnnData | np.ndarray | CSBase | DaskArray,
    mask: str | M,
    dim: Literal["obs", "var"],
    *,
    allow_probabilities: bool = False,
) -> M:  # Could also be a series, but should be one or the other
    """Validate mask argument.

    Params
    ------
    data
        Annotated data matrix or numpy array.
    mask
        Mask (or probabilities if `allow_probabilities=True`).
        Either an appropriatley sized array, or name of a column.
    dim
        The dimension being masked.
    allow_probabilities
        Whether to allow probabilities as `mask`
    """
    if mask is None:
        return mask
    desc = "mask/probabilities" if allow_probabilities else "mask"
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(data)
    if isinstance(mask, str):
        if not isinstance(data, AnnData):
            msg = f"Cannot refer to {desc} with string without providing anndata object as argument"
            #raise ValueError(msg)
        elif is_rust:
            pass

        annot: pd.DataFrame = getattr(data, dim)

        if is_rust:
            if mask not in annot:
                msg = (
                    f"Did not find `adata.{dim}[{mask!r}]`. "
                    f"Either add the {desc} first to `adata.{dim}`"
                    f"or consider using the {desc} argument with an array."
                )
                raise ValueError(msg)
            mask_array = annot[mask].to_numpy()
        else:
            if mask not in annot.columns:
                msg = (
                    f"Did not find `adata.{dim}[{mask!r}]`. "
                    f"Either add the {desc} first to `adata.{dim}`"
                    f"or consider using the {desc} argument with an array."
                )
                raise ValueError(msg)
            mask_array = annot[mask].to_numpy()
    else:
        if len(mask) != data.shape[0 if dim == "obs" else 1]:
            msg = f"The shape of the {desc} do not match the data."
            raise ValueError(msg)
        mask_array = mask
    import pandas as pd
    is_bool = pd.api.types.is_bool_dtype(mask_array.dtype)
    if not allow_probabilities and not is_bool:
        msg = "Mask array must be boolean."
        raise ValueError(msg)
    elif allow_probabilities and not (
        is_bool or pd.api.types.is_float_dtype(mask_array.dtype)
    ):
        msg = f"{desc} array must be boolean or floating point."
        raise ValueError(msg)

    return mask_array



@overload
def _handle_dask_ml_args(
    svd_solver: str | None, method: type[dmld.PCA | dmld.IncrementalPCA]
) -> SvdSolvPCADaskML: ...
@overload
def _handle_dask_ml_args(
    svd_solver: str | None, method: type[dmld.TruncatedSVD]
) -> SvdSolvTruncatedSVDDaskML: ...
def _handle_dask_ml_args(svd_solver: str | None, method: MethodDaskML) -> SvdSolvDaskML:
    import dask_ml.decomposition as dmld

    args: AbstractSet[SvdSolvDaskML]
    default: SvdSolvDaskML
    
    if method in (dmld.PCA, dmld.IncrementalPCA):
        args = get_literal_vals(SvdSolvPCADaskML)
        default = "auto"
    elif method == dmld.TruncatedSVD:
        args = get_literal_vals(SvdSolvTruncatedSVDDaskML)
        default = "tsqr"
    else:
        msg = f"Unknown {method=} in _handle_dask_ml_args"
        raise ValueError(msg)
    return _handle_x_args(svd_solver, method, args, default)


@overload
def _handle_sklearn_args(
    svd_solver: str | None, method: type[skld.TruncatedSVD], *, sparse: None = None
) -> SvdSolvTruncatedSVDSklearn: ...
@overload
def _handle_sklearn_args(
    svd_solver: str | None, method: type[skld.PCA], *, sparse: Literal[False]
) -> SvdSolvPCADenseSklearn: ...
@overload
def _handle_sklearn_args(
    svd_solver: str | None, method: type[skld.PCA], *, sparse: Literal[True]
) -> SvdSolvPCASparseSklearn: ...
def _handle_sklearn_args(
    svd_solver: str | None, method: MethodSklearn, *, sparse: bool | None = None
) -> SvdSolvSkearn:
    import sklearn.decomposition as skld

    args: AbstractSet[SvdSolvSkearn]
    default: SvdSolvSkearn
    suffix = ""
    match (method, sparse):
        case (skld.TruncatedSVD, None):
            args = get_literal_vals(SvdSolvTruncatedSVDSklearn)
            default = "randomized"
        case (skld.PCA, False):
            args = get_literal_vals(SvdSolvPCADenseSklearn)
            default = "arpack"
        case (skld.PCA, True):
            args = get_literal_vals(SvdSolvPCASparseSklearn)
            default = "arpack"
            suffix = " (with sparse input)"
        case _:
            msg = f"Unknown {method=} ({sparse=}) in _handle_sklearn_args"
            raise ValueError(msg)

    return _handle_x_args(svd_solver, method, args, default, suffix=suffix)


def _handle_x_args(
    svd_solver: str | None,
    method: type,
    args: Container[T],
    default: T,
    *,
    suffix: str = "",
) -> T:
    if svd_solver in args:
        return svd_solver
    if svd_solver is not None:
        msg = (
            f"Ignoring {svd_solver=} and using {default}, "
            f"{method.__module__}.{method.__qualname__}{suffix} only supports {args}."
        )
        # (4: caller of `pca` -> 3: `pca` -> 2: `_handle_{sklearn,dask_ml}_args` -> 1: here)
        warnings.warn(msg, UserWarning, stacklevel=4)
    return default
