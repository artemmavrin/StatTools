"""Ridge regression (AKA Tychonoff regularization, AKA L2 penalization)"""

import numbers

import numpy as np

from .linear import LinearRegression


class Ridge(LinearRegression):
    """Ridge regression: linear regression with an L2 penalty."""

    def __init__(self, lam=0.1):
        """Initialize a Ridge object.

        Parameters
        ----------
        lam : float (>0)
            Regularization constant.
        """
        # Validate `lam`
        if not isinstance(lam, numbers.Real) or float(lam) <= 0:
            raise ValueError("Parameter 'lam' must be a positive float.")

        self.lam = float(lam)
        super(Ridge, self).__init__(standardize=True)

    def fit(self, x, y):
        """Fit the ridge regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.

        Returns
        -------
        This Ridge instance.
        """
        # Validate input
        x = self._preprocess_x(x=x)
        y = self._preprocess_y(y=y, x=x)

        # Fit the model by least squares
        a = x.T.dot(x) + self.lam * np.identity(self._p)
        b = x.T.dot(y)

        self._coef, *_ = np.linalg.lstsq(a=a, b=b, rcond=None)
        self.fitted = True
        return self
