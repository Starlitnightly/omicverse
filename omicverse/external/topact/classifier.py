"""Mehods and classes for training gene expression classifiers."""

from abc import ABC, abstractmethod
from typing import Sequence
import warnings

import scipy.sparse
import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
import sklearn

from .countdata import CountData
from .sparsetools import rescale_rows
from . import Colors, EMOJI

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def check_gpu_available():
    """Check if GPU is available for acceleration.

    Returns:
        tuple: (gpu_available, device_name, sklearn_version_ok)
    """
    if not TORCH_AVAILABLE:
        return False, "No torch", False

    # Check sklearn version (need >= 1.8 for torch tensor support)
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    sklearn_version_ok = sklearn_version >= (1, 8)

    # Check for CUDA
    if torch.cuda.is_available():
        return True, f"CUDA ({torch.cuda.get_device_name(0)})", sklearn_version_ok

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True, "MPS (Apple Silicon)", sklearn_version_ok

    # Check for ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return True, "ROCm (AMD)", sklearn_version_ok

    return False, "CPU only", sklearn_version_ok


def sparse_to_torch(matrix: scipy.sparse.spmatrix, device: str = 'cuda'):
    """Convert scipy sparse matrix to torch tensor.

    Args:
        matrix: Scipy sparse matrix
        device: Target device ('cuda', 'mps', or 'cpu')

    Returns:
        torch.Tensor: Dense tensor on the specified device
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is not available")

    # Convert sparse to dense numpy array first
    if scipy.sparse.isspmatrix(matrix):
        dense = matrix.toarray()
    else:
        dense = np.array(matrix)

    # Convert to torch tensor and move to device
    tensor = torch.from_numpy(dense).float()

    # Handle different device types
    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()
    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        tensor = tensor.to('mps')
    else:
        tensor = tensor.cpu()

    return tensor


def normalize_rows(matrix: scipy.sparse.spmatrix | np.matrix,
                   r: int = 5
                   ) -> scipy.sparse.spmatrix:
    factors = (10 ** r) / np.array(matrix.sum(axis=1)).flatten()
    return rescale_rows(matrix, factors)


class Classifier(ABC):
    """Abstract class for a gene expression classifier.

    Attributes:
        trained: True if and only if the classifier has been trained.
        classes: If trained, then an ordered list of all classes.
    """

    def __init__(self):
        self.trained = False
        self.classes = []

    @abstractmethod
    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix,
                 silent: bool = False
                 ) -> npt.NDArray:
        """Takes gene expression samples and returns a classification.

        Args:
            samples:
                A matrix of gene expressions. Each row represents a single
                sample.
            silent:
                If True, suppress output messages (useful for multiprocessing).

        Returns:
            An matrix of probabilities, of shape (samples, classes). Each row
            records the confidence that the respective sample came from each
            class.
        """
        if not self.trained:
            raise RuntimeError('Must provide training data before classifying')

    @abstractmethod
    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        """Trains the classifier on annotated samples.

        Args:
            X_train:
                A matrix of gene expressions. Each row represents a single
                sample.
            y_train:
                A label for each sample.
        """
        self.trained = True


class SVCClassifier(Classifier):
    """Based on the implementation of Abdelaal et al. 2019

    Supports GPU acceleration with sklearn >= 1.8 and torch tensors.
    """

    def __init__(self, r_value: int = 5, use_gpu: bool = True):
        """Initialize SVC Classifier.

        Args:
            r_value: Normalization factor (default: 5)
            use_gpu: Whether to use GPU acceleration if available (default: True)
        """
        super().__init__()
        scaler = StandardScaler(with_mean=False)
        clf = Pipeline([('scaler', scaler), ('clf', LinearSVC(dual=False))])
        self.clf = CalibratedClassifierCV(clf)
        self.r_value = r_value
        self.use_gpu = use_gpu

        # Check GPU availability
        self.gpu_available, self.device_name, self.sklearn_version_ok = check_gpu_available()
        self.device = None

        if use_gpu and self.gpu_available and self.sklearn_version_ok:
            # Determine device type
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            print(f"{Colors.GREEN}🚀 GPU acceleration enabled: {self.device_name}{Colors.ENDC}")
        elif use_gpu and self.gpu_available and not self.sklearn_version_ok:
            print(f"{Colors.WARNING}⚠️ GPU available ({self.device_name}) but sklearn version < 1.8 (current: {sklearn.__version__}){Colors.ENDC}")
            print(f"{Colors.WARNING}   Upgrade sklearn to use GPU acceleration: pip install -U scikit-learn{Colors.ENDC}")
        elif use_gpu and not self.gpu_available:
            print(f"{Colors.BLUE}ℹ️ No GPU available, using CPU mode{Colors.ENDC}")

    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        print(f"{Colors.CYAN}{EMOJI['train']} Training SVC classifier...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Normalizing {X_train.shape[0]} samples with r_value={self.r_value}{Colors.ENDC}")
        X_train = normalize_rows(X_train, self.r_value)
        X_train = X_train.log1p()

        # Convert to torch tensor if GPU is available and sklearn supports it
        if self.device is not None:
            print(f"{Colors.CYAN}  → Converting to torch tensor for {self.device.upper()} acceleration...{Colors.ENDC}")
            try:
                X_train_torch = sparse_to_torch(X_train, device=self.device)
                print(f"{Colors.GREEN}  → Data moved to {self.device.upper()}: {X_train_torch.shape}{Colors.ENDC}")
                X_train = X_train_torch
            except Exception as e:
                print(f"{Colors.WARNING}⚠️ Failed to use GPU, falling back to CPU: {str(e)}{Colors.ENDC}")
                self.device = None

        print(f"{Colors.BLUE}  → Fitting classifier on {len(set(y_train))} cell types{Colors.ENDC}")
        self.clf.fit(X_train, y_train)
        self.classes = self.clf.classes_
        super().train(X_train, y_train)
        print(f"{Colors.GREEN}{EMOJI['done']} Training completed! Classes: {list(self.classes)}{Colors.ENDC}")

    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix,
                 silent: bool = False
                 ):  # -> npt.NDArray:
        super().classify(samples)
        if not silent:
            print(f"{Colors.CYAN}{EMOJI['classify']} Classifying {samples.shape[0]} samples...{Colors.ENDC}")
        test = normalize_rows(samples, self.r_value)
        if scipy.sparse.isspmatrix(test):
            test = test.log1p()
        elif isinstance(test, (np.matrix, np.ndarray)):
            test = np.log1p(test)
        else:
            raise ValueError(f"{samples} is not a matrix")

        # Convert to torch tensor if GPU is available
        if self.device is not None:
            try:
                test = sparse_to_torch(test, device=self.device)
            except Exception as e:
                if not silent:
                    print(f"{Colors.WARNING}⚠️ GPU classification failed, using CPU: {str(e)}{Colors.ENDC}")

        probs = self.clf.predict_proba(test)

        # Convert back to numpy if it's a torch tensor
        if self.device is not None and TORCH_AVAILABLE and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        if not silent:
            print(f"{Colors.GREEN}{EMOJI['done']} Classification completed!{Colors.ENDC}")
        # predicted = np.array([self.clf.classes_[i] for i in np.argmax(probs, axis=1)])
        # predicted_prob = np.max(probs, axis=1)
        return probs
        # return list(map(lambda x: x or None, list(predicted)))


def train_from_countmatrix(classifier: Classifier,
                           countmatrix: CountData,
                           label: str
                           ):
    print(f"{Colors.HEADER}{EMOJI['start']} Training classifier from count matrix...{Colors.ENDC}")
    print(f"{Colors.BLUE}  → Using label: {label}{Colors.ENDC}")
    classifier.train(countmatrix.matrix, list(countmatrix.metadata[label]))
