"""Defines the class for OVR (one-versus-rest) classification."""

import numpy as np

from .base import Classifier
from .base import BinaryClassifier


class OVRClassifier(Classifier):
    """Classification by solving a binary problem for each class.

    OVR stands for "one-versus-rest", meaning that for each class label, a
    binary classification problem of whether or not data belongs to that label
    is solved. The label with the highest probability of being the true label is
    the one predicted for each sample.
    """

    # List of estimators corresponding to each class. For a particular class,
    # the corresponding estimator estimates the probability that an input
    # belongs to that class.
    _class_estimators = None

    def __init__(self, base: type, *args, **kwargs):
        """Initialize an OVR

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

        self.base = base
        self.args = args
        self.kwargs = kwargs

    def fit(self, x, y, *args, **kwargs):
        """Fit the OVR classifier.

        Parameters
        ----------
        x: array-like
            Feature data matrix.
        y: array-like
            Target vector of multiclass labels.
        args: sequence
            Positional arguments to pass to the underlying binary classifier's
            fit method.
        kwargs: dict
            Keyword arguments to pass to the underlying binary classifier's fit
            method.

        Returns
        -------
        This OVRClassifier instance.
        """
        y = self._preprocess_classes(y)

        self._class_estimators = []
        for label in self._classes:
            clf = self.base(*self.args, **self.kwargs)
            clf.fit(x=x, y=(y == label), *args, **kwargs)
            self._class_estimators.append(clf)
        return self

    def predict(self, x):
        """Classify input samples according to their probability estimates.

        Parameters
        ----------
        x: array-like
            Feature matrix.

        Returns
        -------
        Vector of class labels.
        """
        est = [clf.predict_prob(x) for clf in self._class_estimators]
        return self._classes[np.argmax(est, axis=0)]
