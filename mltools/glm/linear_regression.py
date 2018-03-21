"""Linear regression models."""

import abc
import numbers

import numpy as np

from .generalized_linear_model import GeneralizedLinearModel
from ..generic import Regressor
from ..optimization import Optimizer
from ..regularization import lasso, ridge
from ..visualization import func_plot
from ..utils import validate_data


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


class LinearRegression(GeneralizedLinearModel, Regressor,
                       metaclass=abc.ABCMeta):
    """Abstract base class for linear regression models."""

    # The link function for linear regression is the identity function (which is
    # of course its own inverse)
    _inv_link = staticmethod(lambda x: x)

    def __init__(self, fit_intercept=True):
        """Initialize a LinearRegression object.

        Parameters
        ----------
        fit_intercept : bool, optional
            Indicates whether the module should fit an intercept term.
        """
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


class LinearModel(LinearRegression):
    """Linear regression via least squares/maximum likelihood estimation."""

    def fit(self, x, y):
        """Fit the linear model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.

        Returns
        -------
        This LinearModel instance.
        """
        # Validate input
        x = self._preprocess_x(x=x, fitting=True)
        y = self._preprocess_y(y=y, x=x)

        self.loss = MSELoss(x, y)
        self.coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.fitted = True
        return self


class LASSO(LinearRegression):
    """LASSO (least absolute shrinkage and selection operator) regression.
    This is just adding an L^1 regularization term to the MSE loss for linear
    regression.
    """

    # Regularization parameter
    penalty: float = None

    def __init__(self, penalty=0.1, fit_intercept=True):
        """Initialize a LASSO object.

        Parameters
        ----------
        penalty : positive float, optional
            Regularization parameter.
        fit_intercept : bool, optional
            Indicates whether the module should fit an intercept term.
        """
        super(LASSO, self).__init__(fit_intercept=fit_intercept)

        # Validate `lam`
        if not isinstance(penalty, numbers.Real) or penalty <= 0:
            raise ValueError("Parameter 'lam' must be a positive float.")
        self.penalty = float(penalty)

    def fit(self, x, y, optimizer, *args, **kwargs):
        """Fit the LASSO model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        optimizer : Optimizer, optional
            Specifies how to minimize the penalized loss function.
        args : sequence, optional
            Additional positional arguments to pass to `optimizer`'s optimize().
        kwargs : dict, optional
            Additional keyword arguments to pass to `optimizer`'s optimize().

        Returns
        -------
        This LASSO instance.
        """
        # Validate input
        x = self._preprocess_x(x=x, fitting=True)
        y = self._preprocess_y(y=y, x=x)

        self.loss = lasso(penalty=self.penalty, loss=MSELoss(x, y))

        if not isinstance(optimizer, Optimizer):
            raise ValueError(f"Unknown minimization method: {optimizer}")

        self.coef = optimizer.optimize(x0=np.zeros(x.shape[1]), func=self.loss,
                                       *args, **kwargs)

        self.fitted = True
        return self


class Ridge(LinearRegression):
    """Ridge regression. This is just adding an L^2 regularization term to the
    MSE loss for linear regression.
    """

    # Regularization parameter
    penalty: float = None

    def __init__(self, penalty=0.1, fit_intercept=True):
        """Initialize a RidgeRegression object.

        Parameters
        ----------
        penalty : positive float, optional
            Regularization parameter.
        fit_intercept : bool, optional
            Indicates whether the module should fit an intercept term.
        """
        super(Ridge, self).__init__(fit_intercept=fit_intercept)

        # Validate `lam`
        if not isinstance(penalty, numbers.Real) or penalty <= 0:
            raise ValueError("Parameter 'lam' must be a positive float.")
        self.penalty = float(penalty)

    def fit(self, x, y, optimizer, *args, **kwargs):
        """Fit the ridge regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        optimizer : Optimizer, optional
            Specifies how to minimize the penalized loss function.
        args : sequence, optional
            Additional positional arguments to pass to `optimizer`'s optimize().
        kwargs : dict, optional
            Additional keyword arguments to pass to `optimizer`'s optimize().

        Returns
        -------
        This Ridge instance.
        """
        # Validate input
        x = self._preprocess_x(x=x, fitting=True)
        y = self._preprocess_y(y=y, x=x)

        self.loss = ridge(penalty=self.penalty, loss=MSELoss(x, y))

        if not isinstance(optimizer, Optimizer):
            raise ValueError(f"Unknown minimization method: {optimizer}")

        self.coef = optimizer.optimize(x0=np.zeros(x.shape[1]), func=self.loss,
                                       *args, **kwargs)

        self.fitted = True
        return self
