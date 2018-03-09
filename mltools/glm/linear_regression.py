"""Linear regression models."""

import numbers

import matplotlib.pyplot as plt
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


class PolynomialRegression(LinearRegression):
    """Polynomial regression. This is a special case of linear regression, but
    we just use numpy.polyfit to avoid dealing with Vandermonde matrices.
    """

    # Degree of the polynomial model.
    deg: int = None

    # Polynomial function corresponding to the coefficients of the model
    poly: np.poly1d = None

    def _preprocess_x(self, x, fitting=False) -> np.ndarray:
        """Apply necessary validation and preprocessing to the explanatory
        variable of a generalized linear model.

        Parameters
        ----------
        x : array-like, shape (n, )
            Explanatory variable.
        fitting : bool, optional
            Indicates whether preprocessing is being done during fitting

        Returns
        -------
        x : numpy.ndarray, shape (n, )
            Updated explanatory variable.
        """
        # Coerce to NumPy array
        if np.ndim(x) == 1:
            x = np.asarray(x)
        else:
            raise ValueError("Explanatory variable must be 1-dimensional.")

        return x

    def __init__(self, deg=2):
        """Initialize a PolynomialRegression instance.

        Parameters
        ----------
        deg : int
            Degree of the polynomial model.
        """
        # Initialize the model
        super(PolynomialRegression, self).__init__()

        # Validate the degree
        if not isinstance(deg, numbers.Integral) or deg < 1:
            raise ValueError("'deg' must be a positive integer.")
        self.deg = int(deg)

    def fit(self, x, y, optimizer=None, *args, **kwargs):
        """Fit the linear regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        y : array-like, shape (n, )
            Response variable.
        optimizer : Optimizer, optional
            Ignored
        args : sequence, optional
            Ignored
        kwargs : dict, optional
            Ignored

        Returns
        -------
        This PolynomialRegression instance is returned.
        """
        # Validate input
        x = self._preprocess_x(x, fitting=True)
        y = self._preprocess_y(y)
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same length")

        # Compute the least squares coefficients
        coef = np.polyfit(x=x, y=y, deg=self.deg)
        self.poly = np.poly1d(coef)
        self.coef = np.flipud(coef)

        self._fitted = True
        return self

    def estimate(self, x):
        """Return the model's estimate for the given input data.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.

        Returns
        -------
        The polynomial model estimate.
        """
        # Check whether the model is fitted
        if not self.is_fitted():
            raise self.unfitted_exception()

        # Validate input
        x = self._preprocess_x(x, fitting=False)

        return self.poly(x)

    def fit_plot(self, x_min=None, x_max=None, num=500, ax=None, **kwargs):
        """Plot the polynomial regression curve.

        Parameters
        ----------
        x_min : float, optional
            Smallest explanatory variable observation. If not provided, grabs
            the smallest x value from the given axes.
        x_max : float, optional
            Biggest explanatory variable observation. If not provided, grabs the
            biggest x value from the given axes.
        num : int, optional
            Number of points to plot.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot.
        kwargs : dict, optional
            Additional keyword arguments to pass to plot()

        Returns
        -------
        The matplotlib.axes.Axes object on which the plot was drawn.
        """
        # Get axes if not provided
        if ax is None:
            ax = plt.gca()

        # Get bounds if not provided
        y_min, y_max = ax.get_ylim()
        reset_y = False
        if x_min is None or x_max is None:
            x_min, x_max = ax.get_xlim()
            reset_y = True

        x = np.linspace(x_min, x_max, num=num)
        y = self.predict(x)

        ax.plot(x, y, **kwargs)
        ax.set(xlim=(x_min, x_max))
        if reset_y:
            ax.set(ylim=(y_min, y_max))

        return ax

    def poly_str(self, precision=3):
        """Get a string representation of the estimated polynomial model.

        Parameters
        ----------
        precision : int
            Number of decimal places of the coefficients to print.
        """
        s = "y ="
        i = 0
        for c in self.coef:
            if i == 0:
                s += f" {c:.{precision}f}"
            elif i == 1:
                s += f" {'+' if c >= 0 else ''} {c:.{precision}f}x"
            else:
                s += f" {'+' if c >= 0 else ''} {c:.{precision}f}x^{i}"
            i += 1

        return s
