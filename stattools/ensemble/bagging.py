"""Bootstrap aggregating estimators for regression and classification."""

import abc
import functools

import numpy as np

from ..generic import Predictor, Classifier, Regressor
from ..resampling import Bootstrap


class BaggingEstimator(Predictor, metaclass=abc.ABCMeta):
    """Abstract base class for bagging estimators.

    Properties
    ----------
    base : callable
        Function to create the underlying estimators.
    _estimators : numpy.ndarray
        List of bootstrap estimators.
    """

    base = None
    _estimators: np.ndarray = None

    def __init__(self, base: type, *args, **kwargs):
        """Initialize a BaggingEstimator.

        Parameters
        ----------
        base : estimator subclass
            The underlying estimator type.
        args : sequence, optional
            Positional arguments for the estimator constructor.
        kwargs : dict, optional
            Keyword arguments for the estimator constructor.
        """
        if isinstance(self, BaggingClassifier):
            if not issubclass(base, Classifier):
                raise TypeError("Parameter 'base' must be a classifier type.")
        elif isinstance(self, BaggingRegressor):
            if not issubclass(base, Regressor):
                raise TypeError("Parameter 'base' must be a regressor type.")

        self.base = functools.partial(base, *args, **kwargs)

    def fit(self, x, y, n_boot=100, random_state=None, **kwargs):
        """Fit the bagging estimator.

        Parameters
        ----------
        x : array-like
            Explanatory variables.
        y : array-like
            Response variable.
        n_boot : int, optional
            Number of bootstrap estimators to create.
        random_state : int, optional
            Seed for the random number gnerator used to create bootstrap
            estimators.
        kwargs : dict, optional
            Additional keyword arguments to pass to the underlying estimator's
            fit() method.

        Returns
        -------
        This BaggingEstimator instance.
        """
        # Preprocess class labels in the case of classification
        if isinstance(self, BaggingClassifier):
            y = self._preprocess_classes(y, max_classes=None)

        # The statistic being bootstrapped is the underlying fitted estimator
        def stat(x_, y_):
            return self.base().fit(x_, y_, **kwargs)

        # Generate bootstrap estimators
        boot = Bootstrap(x, y, stat=stat, n_boot=n_boot,
                         random_state=random_state)
        self._estimators = boot.dist

        self.fitted = True
        return self


class BaggingClassifier(BaggingEstimator, Classifier):
    """Bagging classifier."""

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
        """Predict class using a majority vote over each bootstrap estimator's
        prediction.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        args : sequence, optional
            Additional positional arguments to pass to each bootstrap
            estimator's predict() function.
        kwargs : dict, optional
            Additional keyword arguments to pass to each bootstrap estimator's
            predict() function.

        Returns
        -------
        The model's class label predictions.
        """
        # Ensure the model is fitted
        if not self.fitted:
            raise self.unfitted_exception

        p = np.asarray([model.predict(x, *args, **kwargs)
                        for model in self._estimators])
        indices = np.empty(len(x), dtype=int)
        for i in range(len(x)):
            idx, counts = np.unique(p[:, i], return_counts=True)
            indices[i] = idx[np.argmax(counts)]
        return self.classes[indices]


class BaggingRegressor(BaggingEstimator, Regressor):
    """Bagging regressor."""

    def predict(self, x, *args, **kwargs):
        """Predict response using the average of each bootstrap estimator's
        prediction.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        args : sequence, optional
            Additional positional arguments to pass to each bootstrap
            estimator's predict() function.
        kwargs : dict, optional
            Additional keyword arguments to pass to each bootstrap estimator's
            predict() function.

        Returns
        -------
        The model's predictions.
        """
        # Ensure the model is fitted
        if not self.fitted:
            raise self.unfitted_exception

        p = [model.predict(x, *args, **kwargs) for model in self._estimators]
        return np.mean(p, axis=0)
