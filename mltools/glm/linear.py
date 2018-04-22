"""Variants of the linear model."""

import abc

import numpy as np

from .glm import GLM
from ..generic import Regressor
from ..optimization import Optimizer


class MSELoss(object):
    """Mean squared error loss function for linear regression.

    Minimizing this loss function is equivalent to maximizing the likelihood
    function of the linear regression model.
    """

    def __init__(self, x, y):
        """Initialize with the training data.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(self.x)

    def __call__(self, coef):
        """Compute the mean squared error loss for the training data."""
        residuals = self.x.dot(coef) - self.y
        return residuals.dot(residuals) / self.n

    def grad(self, coef):
        """Compute the gradient of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x.dot(coef) - self.y) / self.n

    def hess(self, _):
        """Compute the Hessian of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x) / self.n


class LinearModel(GLM, Regressor, metaclass=abc.ABCMeta):
    """Abstract base class for linear models."""

    # The link function for linear models is the identity function (which is
    # of course its own inverse)
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, standardize=True, fit_intercept=True):
        """Initialize a LinearModel object.

        Parameters
        ----------
        standardize : bool, optional
            Indicates whether the explanatory and response variables should be
            centered to have mean 0 and scaled to have variance 1.
        fit_intercept : bool, optional
            Indicates whether the model should fit an intercept term.
        """
        self.standardize = standardize
        self.fit_intercept = fit_intercept

    def predict(self, x):
        """Predict the response variable corresponding to the explanatory
        variable.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable.
        """
        return self.estimate(x)


class LinearRegression(LinearModel):
    """Classical linear regression via least squares/maximum likelihood
    estimation.
    """

    def fit(self, x, y, solver=None, **kwargs):
        """Fit the linear regression model via least squares.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        solver : None or Optimizer, optional
            Specify how to estimate the linear regression model coefficients.
            None:
                Ordinary least squares estimation. This is basically a wrapper
                for numpy.linalg.lstsq().
            Optimizer instance:
                Specify an Optimizer to minimize the MSE loss function.
        kwargs : dict, optional
            If `solver` is an Optimizer, these are additional keyword arguments
            for its optimize() method. Otherwise, these are ignored.

        Returns
        -------
        This LinearRegression instance.
        """
        # Validate input
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        if solver is None:
            # Fit the model by least squares
            self._coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        elif isinstance(solver, Optimizer):
            # Minimize the mean squared error loss function
            loss = MSELoss(x, y)
            coef0 = np.zeros(x.shape[1])
            self._coef = solver.optimize(x0=coef0, func=loss, **kwargs)
        else:
            raise ValueError(f"Unknown value for parameter 'solver': {solver}")

        self.fitted = True
        return self


def _forward_stepwise_selection(x, y, f_threshold, max_features, solver,
                                **kwargs):
    """Forward stepwise selection algorithm for feature selection in linear
    regression models.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Explanatory variables (already standardized).
    y : array-like, shape (n,)
        Response variable (already standardized).
    f_threshold : float, optional
        F-statistic threshold.
    max_features : int, optional
        Maximum number of features to include in the model.
    solver : None or Optimizer, optional
        Specify how to estimate the linear regression model coefficients.
    kwargs : dict, optional
        If `solver` is an Optimizer, these are additional keyword arguments
        for its optimize() method. Otherwise, these are ignored.

    Returns
    -------
    indices : list
        List of indices of features to include in the model
    """
    # Number of observations and features
    n, p = x.shape
    if max_features is None or max_features > p:
        max_features = p

    # Initialize the list of indices
    indices = []

    for k in range(max_features):
        # Compute the residual sum of squares for the current model
        if len(indices) == 0:
            residuals = y
        else:
            model = LinearRegression(standardize=False, fit_intercept=False)
            model.fit(x[:, indices], y, solver=solver, **kwargs)
            residuals = y - model.predict(x[:, indices])
        rss = np.sum(residuals ** 2)

        # Remaining indices
        indices_rem = [i for i in range(p) if i not in indices]

        # Initialize array of F statistics and estimators
        f_stat = np.empty(len(indices_rem))

        # Perform forward stepwise selection
        for i, l in enumerate(indices_rem):
            model_ = LinearRegression(standardize=False, fit_intercept=False)
            x_ = x[:, np.sort(indices + [l])]
            model_.fit(x_, y, solver=solver, **kwargs)
            residuals = y - model_.predict(x_)
            rss_new = np.sum(residuals ** 2)
            f_stat[i] = (n - k - 2) * (rss - rss_new) / rss_new

        # Check for early stopping
        if f_threshold is not None and np.max(f_stat) < f_threshold:
            break

        indices.append(indices_rem[f_stat.argmax()])
        indices.sort()

    return indices


class FSSLinearRegression(LinearRegression):
    """Linear regression with forward stepwise selection (FSS) for features.

    Properties
    ----------
    indices : numpy.ndarray
        Indices of included features
    """
    indices: np.ndarray = None

    def __init__(self):
        """Initialize an FSSLinearRegression instance."""
        super(FSSLinearRegression, self).__init__(standardize=True,
                                                  fit_intercept=True)

    def fit(self, x, y, f_threshold=None, max_features=None, solver=None,
            **kwargs):
        """Fit the linear model using forward stepwise selection.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        f_threshold : float, optional
            F-statistic threshold.
        max_features : int, optional
            Maximum number of features to include in the model.
        solver : None or Optimizer, optional
            Specify how to estimate the linear regression model coefficients.
            None:
                Ordinary least squares estimation. This is basically a wrapper
                for numpy.linalg.lstsq().
            Optimizer instance:
                Specify an Optimizer to minimize the MSE loss function.
        kwargs : dict, optional
            If `solver` is an Optimizer, these are additional keyword arguments
            for its optimize() method. Otherwise, these are ignored.

        Returns
        -------
        This FSSLinearRegression instance.
        """
        # Validate input
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Get feature indices from the forward stepwise selection algorithm
        indices = _forward_stepwise_selection(x, y, f_threshold, max_features,
                                              solver, **kwargs)

        # Initialize coefficients of the linear model
        self._coef = np.zeros(self._p)

        # Get coefficients for the selected features
        if len(indices) > 0:
            # Train a linear model using only the selected features
            x_ = x[:, indices]
            model = LinearRegression(standardize=False, fit_intercept=False)
            model.fit(x_, y)
            self._coef[indices] = model._coef

        self.indices = np.asarray(indices)
        self.fitted = True
        return self
