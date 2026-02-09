"""Mehods and classes for training gene expression classifiers."""

from abc import ABC, abstractmethod
from typing import Sequence

import scipy.sparse
import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from .countdata import CountData
from .sparsetools import rescale_rows
from . import Colors, EMOJI


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
                 samples: scipy.sparse.spmatrix | np.matrix
                 ) -> npt.NDArray:
        """Takes gene expression samples and returns a classification.

        Args:
            samples:
                A matrix of gene expressions. Each row represents a single
                sample.

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
    """Based on the implementation of Abdelaal et al. 2019"""

    def __init__(self, r_value: int = 5):
        super().__init__()
        scaler = StandardScaler(with_mean=False)
        clf = Pipeline([('scaler', scaler), ('clf', LinearSVC(dual=False))])
        self.clf = CalibratedClassifierCV(clf)
        self.r_value = r_value

    def train(self, X_train: scipy.sparse.spmatrix, y_train: Sequence[str]):
        print(f"{Colors.CYAN}{EMOJI['train']} Training SVC classifier...{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Normalizing {X_train.shape[0]} samples with r_value={self.r_value}{Colors.ENDC}")
        X_train = normalize_rows(X_train, self.r_value)
        X_train = X_train.log1p()
        print(f"{Colors.BLUE}  → Fitting classifier on {len(set(y_train))} cell types{Colors.ENDC}")
        self.clf.fit(X_train, y_train)
        self.classes = self.clf.classes_
        super().train(X_train, y_train)
        print(f"{Colors.GREEN}{EMOJI['done']} Training completed! Classes: {list(self.classes)}{Colors.ENDC}")

    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix
                 ):  # -> npt.NDArray:
        super().classify(samples)
        print(f"{Colors.CYAN}{EMOJI['classify']} Classifying {samples.shape[0]} samples...{Colors.ENDC}")
        test = normalize_rows(samples, self.r_value)
        if scipy.sparse.isspmatrix(test):
            test = test.log1p()
        elif isinstance(test, (np.matrix, np.ndarray)):
            test = np.log1p(test)
        else:
            raise ValueError(f"{samples} is not a matrix")

        probs = self.clf.predict_proba(test)
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
