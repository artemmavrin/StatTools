"""Defines the LinearRegression class."""

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from ..generic import Regressor
from ..optimization import Optimizer
from ..regularization import lasso, ridge


class MSELoss(object):
    """Mean squared error loss function for linear regression."""

    def __init__(self, x, y):
        """Initialize with the training data.

        Parameters
        ----------
        x : array-like
            Training data feature matrix.
        y : array-like
            Training data target vector.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n_samples = self.x.shape[0]

        if self.n_samples != self.y.shape[0]:
            raise ValueError(
                f"Unequal number of samples: "
                f"{self.n_samples} feature samples, "
                f"{self.y.shape[0]} target samples")

    def __call__(self, w):
        """Compute the mean squared error loss for the training data."""
        residuals = self.x.dot(w) - self.y
        return residuals.dot(residuals) / self.n_samples

    def grad(self, w):
        """Compute the gradient of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x.dot(w) - self.y) / self.n_samples

    def hess(self, _):
        """Compute the Hessian of the mean squared error loss."""
        return 2 * self.x.T.dot(self.x) / self.n_samples


class LinearRegression(GeneralizedLinearModel, Regressor):
    """Linear regression via least squares estimation."""

    # The link function for linear regression is the identity function (which is
    # of course its own inverse).
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, penalty=None, lam=0.1, intercept=True, mle=False):
        """Initialize a LinearRegression object.

        Parameters
        ----------
        penalty: str, optional
            Type of regularization to impose (none if None).
            Currently supported:
            None
                No regularization
            "l1"
                L1 regularization (AKA LASSO regression)
            "l2"
                L2 regularization (AKA ridge regression)
        lam: float, optional
            Regularization parameter.
        intercept: bool, optional
            Indicates whether the module should fit an intercept term.
        mle: bool, optional
            Indicate whether to fit using maximum likelihood estimation instead
            of least squares if `penalty` is None. If `penalty` is not None,
            this parameter is ignored.
        """
        self.penalty = penalty
        self.lam = lam
        self.intercept = intercept
        self.mle = mle

    def fit(self, x, y, optimizer=None, *args, **kwargs):
        """Train the linear regression model.

        Parameters
        ----------
        x: array-like
            Feature matrix
        y: array-like
            Target vector
        optimizer: Optimizer
            Specifies the optimization algorithm used.
        args: sequence
            Additional positional arguments to pass to the optimizer's
            `optimize` method.
        kwargs: dict
            Additional keyword arguments to pass to the optimizer's `optimize`
            method.

        Returns
        -------
        This LinearRegression instance is returned.
        """
        x = self._preprocess_features(x, fitting=True)
        y = self._preprocess_target(y)

        if self.penalty is None and not self.mle:
            # Least squares regression
            self._weights, *_ = np.linalg.lstsq(x, y, rcond=None)
        else:
            # Maximum likelihood estimation
            self.loss = MSELoss(x, y)

            if self.penalty == "l1":
                self.loss = lasso(self.lam, self.loss)
            elif self.penalty == "l2":
                self.loss = ridge(self.lam, self.loss)
            elif self.penalty is not None:
                raise ValueError(f"Unknown penalty type: {self.penalty}")

            if not isinstance(optimizer, Optimizer):
                raise ValueError(f"Unknown minimization method: {optimizer}")

            w0 = np.zeros(x.shape[1])

            self._weights = optimizer.optimize(x0=w0, func=self.loss,
                                               *args, **kwargs)

        self._fitted = True
        return self

    def predict(self, x):
        """Predict numeric values of the targets corresponding to the data."""
        return self.estimate(x)
