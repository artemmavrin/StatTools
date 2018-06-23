"""Defines the class for OVR (one-versus-rest) classification."""

import functools

import numpy as np

from .predictors import Classifier


class OVRClassifier(Classifier):
    """Multiclass classification by solving a binary problem for each class.

    "OVR" stands for "one-versus-rest", meaning that for each class label, the
    binary classification problem of whether or not data belongs to that label
    is solved. The label with the highest estimated probability of being the
    true label is the one predicted for each sample.
    """

    # List of Classifier estimators corresponding to each class label.
    # For a particular class, the corresponding estimator estimates the
    # probability that an input belongs to that class.
    _estimators = None

    def __init__(self, base: type, *args, **kwargs):
        """Initialize an OVR classifier by specifying how to create the
        underlying binary classifier.

        Parameters
        ----------
        base : type
            A subclass of Classifier. Used to create binary classifiers for each
            class label.
        args : sequence, optional
            Positional arguments for the binary classifier constructor.
        kwargs : dict, optional
            Keyword arguments for the binary classifier constructor.
        """
        if not issubclass(base, Classifier):
            raise TypeError(
                "Parameter 'base' must be a classifier type.")

        self.base = functools.partial(base, *args, **kwargs)

    def fit(self, x, y, *args, **kwargs):
        """Fit the OVR classifier.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        y : array-like
            Categorical response variable vector.
        args : sequence, optional
            Positional arguments to pass to the underlying binary classifier's
            fit() method.
        kwargs : dict, optional
            Keyword arguments to pass to the underlying binary classifier's
            fit() method.

        Returns
        -------
        This OVRClassifier instance.
        """
        y = self._preprocess_classes(y, max_classes=None)

        self._estimators = []
        for i in range(len(self.classes)):
            clf = self.base()
            clf.fit(x, (y == i), *args, **kwargs)
            self._estimators.append(clf)
        return self

    def predict_prob(self, x, *args, **kwargs):
        """Predict probability of each class for each input.

        These probabilities themselves are useless, because they are always
        0 or 1.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        args : sequence, optional
            Positional arguments to pass to each class label estimator's
            `predict_prob` method.
        kwargs : dict, optional
            Keyword arguments to pass to each class label estimator's
            `predict_prob` method.
        """
        q = self.predict(x, *args, **kwargs)
        p = np.zeros((len(x), len(self.classes)))
        for i in range(len(x)):
            j = np.where(self.classes == q[i])[0]
            p[i, j] = 1
        return p

    def predict(self, x, *args, **kwargs):
        """Classify input samples according to their probability estimates.
        Parameters
        ----------
        x : array-like
            Explanatory variable.
        args : sequence, optional
            Positional arguments to pass to each class label estimator's
            `predict_prob` method.
        kwargs : dict, optional
            Keyword arguments to pass to each class label estimator's
            `predict_prob` method.
        Returns
        -------
        Vector of predicted class labels.
        """
        p = [c.predict_prob(x, *args, **kwargs)[:, 1] for c in self._estimators]
        return self.classes[np.argmax(p, axis=0)]
