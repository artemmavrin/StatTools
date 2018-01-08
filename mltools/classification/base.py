"""Defines the Classifier abstract base class."""

import abc

import numpy as np


class Classifier(metaclass=abc.ABCMeta):
    """Abstract base class for classifiers."""

    # Distinct classes---usually to be determined during model fitting
    _classes = None

    def _preprocess_classes(self, target):
        """Extract distinct classes from a target vector.

        This also converts the target vector to numeric indices pointing to the
        class in the `_classes` attribute.
        """
        self._classes, target = np.unique(target, return_inverse=True)
        return target

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the classifier."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Predict class labels from input feature data."""
        pass


class BinaryClassifier(Classifier):
    """Abstract base class for binary classifiers.

    In this case, the _classes attribute will be a list of the form [C0, C1].
    """

    def _preprocess_classes(self, target):
        """Extract distinct classes from a target vector, ensuring that there
        are at most 2 classes."""
        target = super(BinaryClassifier, self)._preprocess_classes(target)
        if len(self._classes) > 2:
            raise ValueError(f"This model is a binary classifier;"
                             f"found {len(self._classes)} distinct classes")
        return target

    @abc.abstractmethod
    def predict_prob(self, *args, **kwargs):
        """Return probability that data belongs to the first class."""
        pass

    def predict(self, x, cutoff=0.5):
        """Classify input samples according to their probability estimates.

        Parameters
        ----------
        x: array-like
            Feature matrix.
        cutoff: float between 0 and 1
            If P(y=C1|x)>cutoff, then x is classified as class C1, otherwise C0.

        Returns
        -------
        Vector of class labels
        """
        return self._classes[1 * (self.predict_prob(x) > cutoff)]
