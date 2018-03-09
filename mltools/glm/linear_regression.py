"""Linear regression model."""

import numbers

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from ..generic import Regressor
from ..optimization import Optimizer
from ..regularization import lasso, ridge


class MSELoss(object):
    """Mean squared error loss function for linear regression.

    Minimizing this loss function is equivalent to maximizing the likelihood
    function in the linear regression model.
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

        if len(x) != len(y):
            raise ValueError(
                f"Unequal number of observations: {len(x)} != {len(y)}")

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


class LinearRegression(GeneralizedLinearModel, Regressor):
    """Linear regression via least squares/maximum likelihood estimation."""

    # Mean squared error loss function
    loss: MSELoss = None

    # The link function for linear regression is the identity function (which is
    # of course its own inverse).
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, penalty=None, lam=0.1, fit_intercept=True):
        """Initialize a LinearRegression object.

        Parameters
        ----------
        penalty : None, "l1", or "l2", optional
            Type of regularization to impose on the loss function (if any).
            If None:
                No regularization.
            If "l1":
                L^1 regularization (LASSO - least absolute shrinkage and
                selection operator)
            If "l2":
                L^2 regularization (ridge regression)
        lam : positive float, optional
            Regularization parameter. Ignored if `penalty` is None.
        fit_intercept : bool, optional
            Indicates whether the module should fit an intercept term.
        """
        self.penalty = penalty
        self.fit_intercept = fit_intercept

        # Validate `lam`
        if penalty is not None:
            if not isinstance(lam, numbers.Real) or lam <= 0:
                raise ValueError("Parameter 'lam' must be a positive float.")
            else:
                self.lam = float(lam)

    def fit(self, x, y, optimizer=None, *args, **kwargs):
        """Fit the linear regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        optimizer : Optimizer, optional
            Specifies the optimization algorithm used. If the model's `penalty`
            is None, this is ignored. Otherwise, this is required because it
            specifies how to minimize the penalized loss function.
        args : sequence, optional
            Additional positional arguments to pass to `optimizer`'s optimize().
        kwargs : dict, optional
            Additional keyword arguments to pass to `optimizer`'s optimize().

        Returns
        -------
        This LinearRegression instance is returned.
        """
        # Validate input
        x = self._preprocess_x(x, fitting=True)
        y = self._preprocess_y(y)
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same length")

        if self.penalty is None:
            # Ordinary least squares estimation
            self.coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        else:
            # Maximum likelihood estimation by minimizing the mean squared error
            self.loss = MSELoss(x, y)

            if self.penalty == "l1":
                self.loss = lasso(self.lam, self.loss)
            elif self.penalty == "l2":
                self.loss = ridge(self.lam, self.loss)
            elif self.penalty is not None:
                raise ValueError(f"Unknown penalty type: {self.penalty}")

            if not isinstance(optimizer, Optimizer):
                raise ValueError(f"Unknown minimization method: {optimizer}")

            self.coef = optimizer.optimize(x0=np.zeros(x.shape[1]),
                                           func=self.loss, *args, **kwargs)

        self._fitted = True
        return self

    def predict(self, x):
        """Predict the response variable."""
        return self.estimate(x)
