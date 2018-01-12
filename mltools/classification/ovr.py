"""Defines the class for OVR (one-versus-rest) classification."""

import functools

import numpy as np

from .base import Classifier
from .base import BinaryClassifier


class OVRClassifier(Classifier):
    """Multiclass classification by solving a binary problem for each class.

    "OVR" stands for "one-versus-rest", meaning that for each class label, the
    binary classification problem of whether or not data belongs to that label
    is solved. The label with the highest estimated probability of being the
    true label is the one predicted for each sample.
    """

    # List of BinaryClassifier estimators corresponding to each class label.
    # For a particular class, the corresponding estimator estimates the
    # probability that an input belongs to that class.
    _estimators = None

    def __init__(self, base: type, *args, **kwargs):
        """Initialize an OVR classifier by specifying how to create the
        underlying binary classifier.

        Parameters
        ----------
        base: type
            A subclass of BinaryClassifier. Used to create binary classifiers
            for each class label.
        args: sequence
            Positional arguments for the binary classifier constructor.
        kwargs: dict
            Keyword arguments for the binary classifier constructor.
        """
        if not issubclass(base, BinaryClassifier):
            raise TypeError("Parameter 'base' must be a binary classifier type")

        self.base = functools.partial(base, *args, **kwargs)

    def fit(self, x, y, *args, **kwargs):
        """Fit the OVR classifier.

        Parameters
        ----------
        x: array-like
            Feature data matrix.
        y: array-like
            Target vector of class labels.
        args: sequence
            Positional arguments to pass to the underlying binary classifier's
            `fit` method.
        kwargs: dict
            Keyword arguments to pass to the underlying binary classifier's
            `fit` method.

        Returns
        -------
        This OVRClassifier instance.
        """
        y = self._preprocess_classes(y)

        self._estimators = []
        for i in range(len(self._classes)):
            clf = self.base()
            clf.fit(x, (y == i), *args, **kwargs)
            self._estimators.append(clf)
        return self

    def predict(self, x, *args, **kwargs):
        """Classify input samples according to their probability estimates.

        Parameters
        ----------
        x: array-like
            Feature matrix.
        args: sequence
            Positional arguments to pass to each class label estimator's
            `predict_prob` method.
        kwargs: dict
            Keyword arguments to pass to each class label estimator's
            `predict_prob` method.

        Returns
        -------
        Vector of class labels.
        """
        est = [clf.predict_prob(x, *args, **kwargs) for clf in self._estimators]
        return self._classes[np.argmax(est, axis=0)]
