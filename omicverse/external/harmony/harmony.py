# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from functools import partial
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
from datetime import datetime
try:
    from ..._settings import EMOJI, Colors
except ImportError:
    # Fallback for when imported directly
    EMOJI = {'start': '🚀', 'done': '✅', 'warning': '⚠️', 'cpu': '🖥️', 'gpu': '🚀'}
    Colors = type('Colors', (), {
        'CYAN': '\033[96m',
        'BOLD': '\033[1m', 
        'ENDC': '\033[0m',
        'GREEN': '\033[92m'
    })()

# from IPython.core.debugger import set_trace

def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta = None,
    lamb = None,
    sigma = 0.1, 
    nclust = None,
    tau = 0,
    block_size = 0.05, 
    max_iter_harmony = 10,
    max_iter_kmeans = 20,
    epsilon_cluster = 1e-5,
    epsilon_harmony = 1e-4, 
    plot_convergence = False,
    verbose = True,
    reference_values = None,
    cluster_prior = None,
    random_state = 0,
    cluster_fn = 'kmeans',
    use_gpu = True,
    **kwargs
):
    """Run Harmony batch correction algorithm with GPU acceleration.
    
    Parameters
    ----------
    use_gpu : bool, optional
        Whether to use GPU acceleration when available. Default: True.
        When True, will try to use MLX for Apple Silicon or PyTorch for CUDA/CPU.
    """

    # theta = None
    # lamb = None
    # sigma = 0.1
    # nclust = None
    # tau = 0
    # block_size = 0.05
    # epsilon_cluster = 1e-5
    # epsilon_harmony = 1e-4
    # plot_convergence = False
    # verbose = True
    # reference_values = None
    # cluster_prior = None
    # random_state = 0
    # cluster_fn = 'kmeans'. Also accepts a callable object with data, num_clusters parameters

    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells" 

    if nclust is None:
        nclust = np.min([np.round(N / 30.0), 100]).astype(int)

    if type(sigma) is float and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    if theta is None:
        theta = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(theta, float) or isinstance(theta, int):
        theta = np.repeat([theta] * len(phi_n), phi_n)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n)
    elif isinstance(lamb, float) or isinstance(lamb, int):
        lamb = np.repeat([lamb] * len(phi_n), phi_n)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n)

    assert len(lamb) == np.sum(phi_n), \
        "each batch variable must have a lambda"

    # Number of items in each category.
    N_b = phi.sum(axis = 1)
    # Proportion of items in each category.
    Pr_b = N_b / N

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

    lamb_mat = np.diag(np.insert(lamb, 0, 0))

    phi_moe = np.vstack((np.repeat(1, N), phi))

    np.random.seed(random_state)

    ho = Harmony(
        data_mat, phi, phi_moe, Pr_b, sigma, theta, max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, lamb_mat, verbose,
        random_state, cluster_fn, use_gpu
    )

    return ho

class Harmony(object):
    def __init__(
            self, Z, Phi, Phi_moe, Pr_b, sigma,
            theta, max_iter_harmony, max_iter_kmeans, 
            epsilon_kmeans, epsilon_harmony, K, block_size,
            lamb, verbose, random_state=None, cluster_fn='kmeans', use_gpu=True
    ):
        self.Z_corr = np.array(Z)
        self.Z_orig = np.array(Z)

        self.Z_cos = self.Z_orig / self.Z_orig.max(axis=0)
        self.Z_cos = self.Z_cos / np.linalg.norm(self.Z_cos, ord=2, axis=0)

        self.Phi             = Phi
        self.Phi_moe         = Phi_moe
        self.N               = self.Z_corr.shape[1]
        self.Pr_b            = Pr_b
        self.B               = self.Phi.shape[0] # number of batch variables
        self.d               = self.Z_corr.shape[0]
        self.window_size     = 3
        self.epsilon_kmeans  = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self.lamb            = lamb
        self.sigma           = sigma
        self.sigma_prior     = sigma
        self.block_size      = block_size
        self.K               = K                # number of clusters
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose         = verbose
        self.theta           = theta

        self.objective_harmony        = []
        self.objective_kmeans         = []
        self.objective_kmeans_dist    = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross   = []
        self.kmeans_rounds  = []

        # GPU acceleration setup
        self.use_gpu = use_gpu
        self.device_info = self._detect_optimal_harmony_backend(use_gpu, verbose) if use_gpu else {'backend': 'cpu', 'device': 'cpu'}
        self.device = self.device_info.get('device', 'cpu')
        self.backend = self.device_info.get('backend', 'cpu')
        
        if verbose and use_gpu:
            if self.backend == 'torch' and self.device == 'cuda':
                print(f"{EMOJI['gpu']} Using PyTorch CUDA acceleration for Harmony")
            elif self.backend == 'mlx' and self.device == 'mps':
                print(f"{EMOJI['gpu']} Using MLX Apple Silicon acceleration for Harmony") 
            elif self.backend == 'torch' and self.device == 'cpu':
                print(f"{EMOJI['cpu']} Using PyTorch CPU acceleration for Harmony")
            else:
                print(f"{EMOJI['cpu']} Using CPU implementation for Harmony")
        elif verbose:
            print(f"{EMOJI['cpu']} Using CPU implementation for Harmony")
        
        # Initialize arrays with appropriate backend
        self._init_arrays_with_backend()
        
        self.allocate_buffers()
        if cluster_fn == 'kmeans':
            cluster_fn = partial(Harmony._cluster_kmeans, random_state=random_state)
        self.init_cluster(cluster_fn)
        self.harmonize(self.max_iter_harmony, self.verbose)

    def result(self):
        if self.backend == 'torch':
            result = self.Z_corr.cpu().numpy().T  # Transpose to (cells, features)
        elif self.backend == 'mlx':
            import mlx.core as mx
            # Convert MLX array to numpy array properly
            # Try different conversion methods
            try:
                # Method 1: Direct conversion (most reliable for MLX)
                result = np.array(self.Z_corr, dtype=np.float32, copy=True).T  # Transpose to (cells, features)
            except:
                try:
                    # Method 2: Convert to list first, then to numpy
                    result = np.array(self.Z_corr.tolist(), dtype=np.float32).T  # Transpose to (cells, features)
                except:
                    try:
                        # Method 3: Try if MLX array has a numpy() method
                        if hasattr(self.Z_corr, 'numpy'):
                            result = self.Z_corr.numpy().T  # Transpose to (cells, features)
                        else:
                            raise AttributeError("No numpy method")
                    except:
                        # Method 4: Fallback - try mx.eval() but ensure it's 2D
                        evaluated = mx.eval(self.Z_corr)
                        if evaluated.ndim == 0:
                            # If 0D, we need to reshape based on original shape
                            result = np.array([evaluated.item()], dtype=np.float32).reshape(1, 1)
                        else:
                            result = np.array(evaluated, dtype=np.float32).T
        else:
            result = self.Z_corr.T  # Transpose to (cells, features)
        
        # Ensure result is a 2D numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        # Ensure it's 2D
        if result.ndim == 1:
            result = result.reshape(1, -1)
        elif result.ndim > 2:
            result = result.reshape(result.shape[0], -1)
        
        return result

    def _detect_optimal_harmony_backend(self, use_gpu=True, verbose=False):
        """Detect the optimal backend for Harmony acceleration."""
        try:
            from ..._settings import get_optimal_device, settings
            device = get_optimal_device(prefer_gpu=use_gpu, verbose=verbose)
            
            omicverse_mode = getattr(settings, 'mode', 'cpu')
            
            if verbose:
                print(f"   Omicverse mode: {omicverse_mode}")
                print(f"   Detected device: {device}")
            
            device_type = device.type if hasattr(device, 'type') else str(device)
                
            if device_type == 'mps' and use_gpu and omicverse_mode != 'cpu':
                try:
                    import mlx.core as mx
                    if mx.metal.is_available():
                        return {'backend': 'mlx', 'device': 'mps', 'available': True}
                except ImportError:
                    pass
                    
            elif device_type == 'cuda' and use_gpu and omicverse_mode != 'cpu':
                try:
                    import torch
                    if torch.cuda.is_available():
                        return {'backend': 'torch', 'device': 'cuda', 'available': True}
                except ImportError:
                    pass
            
            try:
                import torch
                return {'backend': 'torch', 'device': 'cpu', 'available': True}
            except ImportError:
                pass
            
            return {'backend': 'cpu', 'device': 'cpu', 'available': True}
            
        except ImportError:
            try:
                import torch
                return {'backend': 'torch', 'device': 'cpu', 'available': True}
            except ImportError:
                return {'backend': 'cpu', 'device': 'cpu', 'available': True}
    
    def _init_arrays_with_backend(self):
        """Initialize arrays with the appropriate backend."""
        if self.backend == 'torch':
            try:
                import torch
                self._torch_device = torch.device(self.device)
                # Convert numpy arrays to torch tensors
                self.Z_corr = torch.tensor(self.Z_corr, dtype=torch.float32, device=self._torch_device)
                self.Z_orig = torch.tensor(self.Z_orig, dtype=torch.float32, device=self._torch_device)
                self.Z_cos = torch.tensor(self.Z_cos, dtype=torch.float32, device=self._torch_device)
                self.Phi = torch.tensor(self.Phi, dtype=torch.float32, device=self._torch_device)
                self.Phi_moe = torch.tensor(self.Phi_moe, dtype=torch.float32, device=self._torch_device)
                self.Pr_b = torch.tensor(self.Pr_b, dtype=torch.float32, device=self._torch_device)
                self.sigma = torch.tensor(self.sigma, dtype=torch.float32, device=self._torch_device)
                self.theta = torch.tensor(self.theta, dtype=torch.float32, device=self._torch_device)
                self.lamb = torch.tensor(self.lamb, dtype=torch.float32, device=self._torch_device)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to initialize torch backend: {e}, falling back to CPU")
                self.backend = 'cpu'
                self.device = 'cpu'
        elif self.backend == 'mlx':
            try:
                import mlx.core as mx
                # Convert numpy arrays to MLX arrays
                self.Z_corr = mx.array(self.Z_corr)
                self.Z_orig = mx.array(self.Z_orig)
                self.Z_cos = mx.array(self.Z_cos)
                self.Phi = mx.array(self.Phi)
                self.Phi_moe = mx.array(self.Phi_moe)
                self.Pr_b = mx.array(self.Pr_b)
                # Ensure sigma is at least 1D for MLX
                if np.isscalar(self.sigma):
                    self.sigma = mx.array([self.sigma])
                else:
                    self.sigma = mx.array(self.sigma)
                self.theta = mx.array(self.theta)
                self.lamb = mx.array(self.lamb)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to initialize MLX backend: {e}, falling back to CPU")
                self.backend = 'cpu'
                self.device = 'cpu'
    
    def allocate_buffers(self):
        if self.backend == 'torch':
            import torch
            self._scale_dist = torch.zeros((self.K, self.N), device=self._torch_device)
            self.dist_mat = torch.zeros((self.K, self.N), device=self._torch_device)
            self.O = torch.zeros((self.K, self.B), device=self._torch_device)
            self.E = torch.zeros((self.K, self.B), device=self._torch_device)
            self.W = torch.zeros((self.B + 1, self.d), device=self._torch_device)
            self.Phi_Rk = torch.zeros((self.B + 1, self.N), device=self._torch_device)
        elif self.backend == 'mlx':
            import mlx.core as mx
            self._scale_dist = mx.zeros((self.K, self.N))
            self.dist_mat = mx.zeros((self.K, self.N))
            self.O = mx.zeros((self.K, self.B))
            self.E = mx.zeros((self.K, self.B))
            self.W = mx.zeros((self.B + 1, self.d))
            self.Phi_Rk = mx.zeros((self.B + 1, self.N))
        else:
            self._scale_dist = np.zeros((self.K, self.N))
            self.dist_mat = np.zeros((self.K, self.N))
            self.O = np.zeros((self.K, self.B))
            self.E = np.zeros((self.K, self.B))
            self.W = np.zeros((self.B + 1, self.d))
            self.Phi_Rk = np.zeros((self.B + 1, self.N))

    @staticmethod
    def _cluster_kmeans(data, K, random_state):
        # Start with cluster centroids
        # Convert to numpy if needed for sklearn
        if hasattr(data, 'cpu'):
            data_np = data.cpu().numpy()
        elif hasattr(data, 'shape') and str(type(data).__module__).startswith('mlx'):
            import mlx.core as mx
            data_np = np.array(data)
        else:
            data_np = np.asarray(data)
        
        model = KMeans(n_clusters=K, init='k-means++',
                       n_init=10, max_iter=25, random_state=random_state)
        model.fit(data_np)
        km_centroids, km_labels = model.cluster_centers_, model.labels_
        return km_centroids

    def init_cluster(self, cluster_fn):
        if self.backend == 'torch':
            self._init_cluster_torch(cluster_fn)
        elif self.backend == 'mlx':
            self._init_cluster_mlx(cluster_fn)
        else:
            self._init_cluster_cpu(cluster_fn)
    
    def _init_cluster_cpu(self, cluster_fn):
        self.Y = cluster_fn(self.Z_cos.T, self.K).T
        # (1) Normalize
        self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma[:,None]
        self.R -= np.max(self.R, axis = 0)
        self.R = np.exp(self.R)
        self.R = self.R / np.sum(self.R, axis = 0)
        # (3) Batch diversity statistics
        self.E = np.outer(np.sum(self.R, axis=1), self.Pr_b)
        self.O = np.inner(self.R , self.Phi)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])
    
    def _init_cluster_torch(self, cluster_fn):
        import torch
        # Convert to numpy for clustering, then back to torch
        Z_cos_np = self.Z_cos.cpu().numpy()
        Y_np = cluster_fn(Z_cos_np.T, self.K).T
        self.Y = torch.tensor(Y_np, dtype=torch.float32, device=self._torch_device)
        
        # (1) Normalize
        self.Y = self.Y / torch.norm(self.Y, dim=0, keepdim=True)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - torch.mm(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / self.sigma.unsqueeze(1)
        self.R -= torch.max(self.R, dim=0, keepdim=True)[0]
        self.R = torch.exp(self.R)
        self.R = self.R / torch.sum(self.R, dim=0, keepdim=True)
        # (3) Batch diversity statistics
        self.E = torch.outer(torch.sum(self.R, dim=1), self.Pr_b)
        self.O = torch.mm(self.R, self.Phi.T)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])
    
    def _init_cluster_mlx(self, cluster_fn):
        import mlx.core as mx
        # Convert to numpy for clustering, then back to mlx
        Z_cos_np = np.array(self.Z_cos)
        Y_np = cluster_fn(Z_cos_np.T, self.K).T
        self.Y = mx.array(Y_np)
        
        # (1) Normalize
        self.Y = self.Y / mx.linalg.norm(self.Y, axis=0, keepdims=True)
        # (2) Assign cluster probabilities
        self.dist_mat = 2 * (1 - mx.matmul(self.Y.T, self.Z_cos))
        self.R = -self.dist_mat
        self.R = self.R / mx.expand_dims(self.sigma, 1)
        self.R -= mx.max(self.R, axis=0, keepdims=True)
        self.R = mx.exp(self.R)
        self.R = self.R / mx.sum(self.R, axis=0, keepdims=True)
        # (3) Batch diversity statistics
        self.E = mx.outer(mx.sum(self.R, axis=1), self.Pr_b)
        self.O = mx.matmul(self.R, self.Phi.T)
        self.compute_objective()
        # Save results
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        if self.backend == 'torch':
            self._compute_objective_torch()
        elif self.backend == 'mlx':
            self._compute_objective_mlx()
        else:
            self._compute_objective_cpu()
    
    def _compute_objective_cpu(self):
        kmeans_error = np.sum(np.multiply(self.R, self.dist_mat))
        # Entropy
        _entropy = np.sum(safe_entropy(self.R) * self.sigma[:,np.newaxis])
        # Cross Entropy
        x = (self.R * self.sigma[:,np.newaxis])
        y = np.tile(self.theta[:,np.newaxis], self.K).T
        z = np.log((self.O + 1) / (self.E + 1))
        w = np.dot(y * z, self.Phi)
        _cross_entropy = np.sum(x * w)
        # Save results
        self.objective_kmeans.append(kmeans_error + _entropy + _cross_entropy)
        self.objective_kmeans_dist.append(kmeans_error)
        self.objective_kmeans_entropy.append(_entropy)
        self.objective_kmeans_cross.append(_cross_entropy)
    
    def _compute_objective_torch(self):
        import torch
        kmeans_error = torch.sum(self.R * self.dist_mat)
        # Entropy
        _entropy = torch.sum(safe_entropy_torch(self.R) * self.sigma.unsqueeze(1))
        # Cross Entropy
        x = (self.R * self.sigma.unsqueeze(1))
        y = self.theta.unsqueeze(0).repeat(self.K, 1)
        z = torch.log((self.O + 1) / (self.E + 1))
        w = torch.mm(y * z, self.Phi)
        _cross_entropy = torch.sum(x * w)
        # Save results (convert to CPU float for storage)
        self.objective_kmeans.append(float(kmeans_error.cpu() + _entropy.cpu() + _cross_entropy.cpu()))
        self.objective_kmeans_dist.append(float(kmeans_error.cpu()))
        self.objective_kmeans_entropy.append(float(_entropy.cpu()))
        self.objective_kmeans_cross.append(float(_cross_entropy.cpu()))
    
    def _compute_objective_mlx(self):
        import mlx.core as mx
        kmeans_error = mx.sum(self.R * self.dist_mat)
        # Entropy
        _entropy = mx.sum(safe_entropy_mlx(self.R) * mx.expand_dims(self.sigma, 1))
        # Cross Entropy
        x = (self.R * mx.expand_dims(self.sigma, 1))
        y = mx.tile(mx.expand_dims(self.theta, 0), [self.K, 1])
        z = mx.log((self.O + 1) / (self.E + 1))
        w = mx.matmul(y * z, self.Phi)
        _cross_entropy = mx.sum(x * w)
        # Save results (convert to numpy float for storage)
        self.objective_kmeans.append(float(kmeans_error + _entropy + _cross_entropy))
        self.objective_kmeans_dist.append(float(kmeans_error))
        self.objective_kmeans_entropy.append(float(_entropy))
        self.objective_kmeans_cross.append(float(_cross_entropy))

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        if verbose:
            print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running Harmony integration...")
            print(f"{Colors.CYAN}    Max iterations: {Colors.BOLD}{iter_harmony}{Colors.ENDC}")
            print(f"{Colors.CYAN}    Convergence threshold: {Colors.BOLD}{self.epsilon_harmony}{Colors.ENDC}")
            
        # Create progress bar for iterations
        pbar = tqdm(range(1, iter_harmony + 1), desc="Harmony iterations", disable=not verbose)
        
        for i in pbar:
            # STEP 1: Clustering
            self.cluster()
            # STEP 2: Regress out covariates
            self.Z_cos, self.Z_corr, self.W, self.Phi_Rk = moe_correct_ridge(
                self.Z_orig, self.Z_cos, self.Z_corr, self.R, self.W, self.K,
                self.Phi_Rk, self.Phi_moe, self.lamb
            )
            # STEP 3: Check for convergence
            converged = self.check_convergence(1)
            
            # Update progress bar description
            if verbose:
                pbar.set_description(f"Harmony iteration {i}/{iter_harmony}")
                if converged:
                    pbar.set_description(f"Harmony converged after {i} iterations")
            
            if converged:
                if verbose:
                    print(f"{EMOJI['done']} Harmony converged after {i} iteration{'s' if i > 1 else ''}")
                break
                
        pbar.close()
        
        if verbose and not converged:
            print(f"{EMOJI['warning']} Harmony stopped before convergence after {iter_harmony} iterations")
        return 0

    def cluster(self):
        if self.backend == 'torch':
            self._cluster_torch()
        elif self.backend == 'mlx':
            self._cluster_mlx()
        else:
            self._cluster_cpu()
    
    def _cluster_cpu(self):
        # Z_cos has changed
        # R is assumed to not have changed
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
        
        # Create progress bar for k-means iterations
        pbar = tqdm(range(self.max_iter_kmeans), desc="K-means clustering", disable=True)
        
        for i in pbar:
            # STEP 1: Update Y
            self.Y = np.dot(self.Z_cos, self.R.T)
            self.Y = self.Y / np.linalg.norm(self.Y, ord=2, axis=0)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - np.dot(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        
        pbar.close()
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0
    
    def _cluster_torch(self):
        import torch
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - torch.mm(self.Y.T, self.Z_cos))
        
        # Create progress bar for k-means iterations
        pbar = tqdm(range(self.max_iter_kmeans), desc="K-means clustering", disable=True)
        
        for i in pbar:
            # STEP 1: Update Y
            self.Y = torch.mm(self.Z_cos, self.R.T)
            self.Y = self.Y / torch.norm(self.Y, dim=0, keepdim=True)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - torch.mm(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        
        pbar.close()
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0
    
    def _cluster_mlx(self):
        import mlx.core as mx
        # Update Y to match new integrated data
        self.dist_mat = 2 * (1 - mx.matmul(self.Y.T, self.Z_cos))
        
        # Create progress bar for k-means iterations
        pbar = tqdm(range(self.max_iter_kmeans), desc="K-means clustering", disable=True)
        
        for i in pbar:
            # STEP 1: Update Y
            self.Y = mx.matmul(self.Z_cos, self.R.T)
            self.Y = self.Y / mx.linalg.norm(self.Y, axis=0, keepdims=True)
            # STEP 2: Update dist_mat
            self.dist_mat = 2 * (1 - mx.matmul(self.Y.T, self.Z_cos))
            # STEP 3: Update R
            self.update_R()
            # STEP 4: Check for convergence
            self.compute_objective()
            if i > self.window_size:
                converged = self.check_convergence(0)
                if converged:
                    break
        
        pbar.close()
        self.kmeans_rounds.append(i)
        self.objective_harmony.append(self.objective_kmeans[-1])
        return 0

    def update_R(self):
        if self.backend == 'torch':
            self._update_R_torch()
        elif self.backend == 'mlx':
            self._update_R_mlx()
        else:
            self._update_R_cpu()
    
    def _update_R_cpu(self):
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma[:,None]
        self._scale_dist -= np.max(self._scale_dist, axis=0)
        self._scale_dist = np.exp(self._scale_dist)
        # Update cells in blocks - use fixed seed for reproducible block ordering  
        np.random.seed(42)  # Use fixed seed for consistent ordering
        update_order = np.arange(self.N)
        np.random.shuffle(update_order)
        n_blocks = np.ceil(1 / self.block_size).astype(int)
        blocks = np.array_split(update_order, n_blocks)
        for b in blocks:
            # STEP 1: Remove cells
            self.E -= np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= np.dot(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            self.R[:,b] = np.multiply(
                self.R[:,b],
                np.dot(
                    np.power((self.E + 1) / (self.O + 1), self.theta),
                    self.Phi[:,b]
                )
            )
            self.R[:,b] = self.R[:,b] / np.linalg.norm(self.R[:,b], ord=1, axis=0)
            # STEP 3: Put cells back
            self.E += np.outer(np.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += np.dot(self.R[:,b], self.Phi[:,b].T)
        return 0
    
    def _update_R_torch(self):
        import torch
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / self.sigma.unsqueeze(1)
        self._scale_dist -= torch.max(self._scale_dist, dim=0, keepdim=True)[0]
        self._scale_dist = torch.exp(self._scale_dist)
        # Update cells in blocks - use exactly same order as CPU version
        np.random.seed(42)  # Use same seed as CPU version
        update_order_np = np.arange(self.N)
        np.random.shuffle(update_order_np)
        update_order = torch.tensor(update_order_np, device=self._torch_device)
        n_blocks = int(np.ceil(1 / self.block_size))
        # Use same block splitting as numpy version
        blocks_np = np.array_split(update_order_np, n_blocks)
        blocks = [torch.tensor(block, device=self._torch_device) for block in blocks_np]
        for b in blocks:
            if len(b) == 0:
                continue
            # STEP 1: Remove cells
            self.E -= torch.outer(torch.sum(self.R[:,b], dim=1), self.Pr_b)
            self.O -= torch.mm(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            # Power term should use theta as row vector to broadcast properly
            power_term = torch.pow((self.E + 1) / (self.O + 1), self.theta.unsqueeze(0))
            self.R[:,b] = self.R[:,b] * torch.mm(power_term, self.Phi[:,b])
            self.R[:,b] = self.R[:,b] / torch.norm(self.R[:,b], p=1, dim=0, keepdim=True)
            # STEP 3: Put cells back
            self.E += torch.outer(torch.sum(self.R[:,b], dim=1), self.Pr_b)
            self.O += torch.mm(self.R[:,b], self.Phi[:,b].T)
        return 0
    
    def _update_R_mlx(self):
        import mlx.core as mx
        self._scale_dist = -self.dist_mat
        self._scale_dist = self._scale_dist / mx.expand_dims(self.sigma, 1)
        self._scale_dist -= mx.max(self._scale_dist, axis=0, keepdims=True)
        self._scale_dist = mx.exp(self._scale_dist)
        # Update cells in blocks - use consistent ordering
        np.random.seed(42)  # Use same seed as CPU version
        update_order_np = np.arange(self.N)
        np.random.shuffle(update_order_np)
        update_order = mx.array(update_order_np)
        n_blocks = int(np.ceil(1 / self.block_size))
        block_size_actual = int(np.ceil(self.N / n_blocks))
        for i in range(n_blocks):
            start_idx = i * block_size_actual
            end_idx = min((i + 1) * block_size_actual, self.N)
            if start_idx >= end_idx:
                continue
            b = update_order[start_idx:end_idx]
            # STEP 1: Remove cells
            self.E -= mx.outer(mx.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O -= mx.matmul(self.R[:,b], self.Phi[:,b].T)
            # STEP 2: Recompute R for removed cells
            self.R[:,b] = self._scale_dist[:,b]
            # Power term should use theta as row vector to broadcast properly  
            power_term = mx.power((self.E + 1) / (self.O + 1), mx.expand_dims(self.theta, 0))
            self.R[:,b] = self.R[:,b] * mx.matmul(power_term, self.Phi[:,b])
            self.R[:,b] = self.R[:,b] / mx.sum(mx.abs(self.R[:,b]), axis=0, keepdims=True)
            # STEP 3: Put cells back
            self.E += mx.outer(mx.sum(self.R[:,b], axis=1), self.Pr_b)
            self.O += mx.matmul(self.R[:,b], self.Phi[:,b].T)
        return 0

    def check_convergence(self, i_type):
        obj_old = 0.0
        obj_new = 0.0
        # Clustering, compute new window mean
        if i_type == 0:
            okl = len(self.objective_kmeans)
            for i in range(self.window_size):
                obj_old += self.objective_kmeans[okl - 2 - i]
                obj_new += self.objective_kmeans[okl - 1 - i]
            if abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans:
                return True
            return False
        # Harmony
        if i_type == 1:
            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            if (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony:
                return True
            return False
        return True


def safe_entropy(x: np.array):
    y = np.multiply(x, np.log(x))
    y[~np.isfinite(y)] = 0.0
    return y

def safe_entropy_torch(x):
    import torch
    y = x * torch.log(x)
    y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
    return y

def safe_entropy_mlx(x):
    import mlx.core as mx
    y = x * mx.log(x)
    finite_mask = mx.isfinite(y)
    y = mx.where(finite_mask, y, mx.zeros_like(y))
    return y

def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    # Detect the backend from the input arrays
    if hasattr(Z_orig, 'device'):  # torch tensor
        return moe_correct_ridge_torch(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb)
    elif hasattr(Z_orig, 'shape') and hasattr(Z_orig, 'dtype') and str(type(Z_orig).__module__).startswith('mlx'):
        return moe_correct_ridge_mlx(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb)
    else:
        return moe_correct_ridge_cpu(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb)

def moe_correct_ridge_cpu(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk

def moe_correct_ridge_torch(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    import torch
    Z_corr = Z_orig.clone()
    for i in range(K):
        Phi_Rk = Phi_moe * R[i,:].unsqueeze(0)
        x = torch.mm(Phi_Rk, Phi_moe.T) + lamb
        try:
            x_inv = torch.inverse(x)
        except:
            # Fallback to pinverse if inverse fails
            x_inv = torch.pinverse(x)
        W = torch.mm(torch.mm(x_inv, Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= torch.mm(W.T, Phi_Rk)
    Z_cos = Z_corr / torch.norm(Z_corr, dim=0, keepdim=True)
    return Z_cos, Z_corr, W, Phi_Rk

def moe_correct_ridge_mlx(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    import mlx.core as mx
    Z_corr = mx.array(Z_orig)
    for i in range(K):
        Phi_Rk = Phi_moe * mx.expand_dims(R[i,:], 0)
        x = mx.matmul(Phi_Rk, Phi_moe.T) + lamb
        try:
            x_inv = mx.linalg.inv(x)
        except:
            # Fallback to CPU for pinverse since MLX GPU doesn't support it yet
            x_cpu = np.array(x)
            x_inv_cpu = np.linalg.pinv(x_cpu)
            x_inv = mx.array(x_inv_cpu)
        W = mx.matmul(mx.matmul(x_inv, Phi_Rk), Z_orig.T)
        W = mx.where(mx.arange(W.shape[0])[:, None] == 0, 0, W)  # W[0,:] = 0
        Z_corr -= mx.matmul(W.T, Phi_Rk)
    Z_cos = Z_corr / mx.linalg.norm(Z_corr, axis=0, keepdims=True)
    return Z_cos, Z_corr, W, Phi_Rk
