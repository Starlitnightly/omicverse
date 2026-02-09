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

from .countdata import CountData
from .sparsetools import rescale_rows


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
        X_train = normalize_rows(X_train, self.r_value)
        X_train = X_train.log1p()
        self.clf.fit(X_train, y_train)
        self.classes = self.clf.classes_
        super().train(X_train, y_train)

    def classify(self,
                 samples: scipy.sparse.spmatrix | np.matrix
                 ):  # -> npt.NDArray:
        super().classify(samples)
        test = normalize_rows(samples, self.r_value)
        if scipy.sparse.isspmatrix(test):
            test = test.log1p()
        elif isinstance(test, (np.matrix, np.ndarray)):
            test = np.log1p(test)
        else:
            raise ValueError(f"{samples} is not a matrix")

        probs = self.clf.predict_proba(test)
        # predicted = np.array([self.clf.classes_[i] for i in np.argmax(probs, axis=1)])
        # predicted_prob = np.max(probs, axis=1)
        return probs
        # return list(map(lambda x: x or None, list(predicted)))


def train_from_countmatrix(classifier: Classifier,
                           countmatrix: CountData,
                           label: str
                           ):
    classifier.train(countmatrix.matrix, list(countmatrix.metadata[label]))
