"""Defines the GLM (generalized linear model) abstract base class."""

import abc
import warnings

import numpy as np

from ..generic import Fittable, Classifier
from ..utils import validate_samples


class GLM(Fittable, metaclass=abc.ABCMeta):
    """Abstract base class for generalized linear models.

    Properties
    ----------
    standardize : bool
        Indicates whether the explanatory variables (and, in the case of
        regression, the response variable also) should be centered to have mean
        0 and scaled to have variance 1.
    fit_intercept : bool
        Indicates whether the model should include an intercept term. This is
        ignored if `standardize` is True.
    intercept : float
        The intercept of the GLM.
    coef : numpy.ndarray
        Array of coefficients of each explanatory variable in the GLM.
    """

    standardize: bool = True
    fit_intercept: bool = True

    # Internal representation of the model coefficients (including intercept)
    _coef: np.ndarray = None

    # Vector of means of the explanatory variables
    _x_mean: np.ndarray = None

    # Vector of standard deviations of the explanatory variables
    _x_std: np.ndarray = None

    # Number of explanatory variables (not including the intercept if any)
    _p: int = None

    # Mean of the response variable (for regression only)
    _y_mean: float = 0.0

    # Standard deviation of the response variable (for regression only)
    _y_std: float = 1.0

    @property
    def intercept(self):
        """The intercept of the GLM."""
        if not self.fitted:
            raise self.unfitted_exception

        if self.standardize:
            return (self._y_mean
                    - self._y_std * np.sum(self._coef
                                           * self._x_mean / self._x_std))
        elif self.fit_intercept:
            return self._coef[0]
        else:
            return 0.0

    @property
    def coef(self):
        """Array of coefficients of each explanatory variable in the GLM."""
        if not self.fitted:
            raise self.unfitted_exception

        if self.standardize:
            return self._y_std * self._coef / self._x_std
        elif self.fit_intercept:
            return self._coef[1:]
        else:
            return self._coef

    @staticmethod
    @abc.abstractmethod
    def _inv_link(*args, **kwargs):
        """Inverse link function for the given generalized linear model."""
        pass

    def estimate(self, x):
        """Return the GLM estimate f(a + x*b), where `f` is the inverse of the
        GLM's link function, `a` is the intercept, `x` is the matrix of
        explanatory variables, and `b` is the vector of coefficients.

        Parameters
        ----------
        x : array-like
            Explanatory variable.
        """
        if not self.fitted:
            raise self.unfitted_exception
        x = validate_samples(x, n_dim=2)
        if x.shape[1] != self._p:
            raise ValueError("Wrong number of explanatory variables.")

        return self._inv_link(self.intercept + x.dot(self.coef))

    def _preprocess_x(self, x):
        """Apply necessary validation and preprocessing to the explanatory
        variable of a generalized linear model to prepare for fitting.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.

        Returns
        -------
        x : numpy.ndarray, shape (n, p) or (n, p + 1)
            Updated explanatory variable.
        """
        x = validate_samples(x, n_dim=2)
        n, self._p = x.shape

        # Standardize or add an intercept column as needed
        if self.standardize:
            self._x_mean = x.mean(axis=0)
            self._x_std = x.std(axis=0, ddof=0)
            if np.any(self._x_std == 0):
                warnings.warn("Some explanatory variables are constant.")
                self._x_std[self._x_std == 0] = 1.0
            x = (x - self._x_mean) / self._x_std
        elif self.fit_intercept:
            x = np.concatenate((np.ones((n, 1)), x), axis=1)

        return x

    def _preprocess_y(self, y, x=None):
        """Apply necessary validation and preprocessing to the response variable
        of a generalized linear model to prepare for fitting.

        Parameters
        ----------
        y : array-like, shape (n, )
            Response variable.
        x : array-like, shape (n, p)
            Explanatory variable. If provided, it is checked whether `x` and `y`
            have the same length (i.e., number of observations).

        Returns
        -------
        y : numpy.ndarray, shape (n, )
            Updated response variable.
        """
        if x is None:
            y = validate_samples(y, n_dim=1)
        else:
            y, _ = validate_samples(y, x, n_dim=(1, None), equal_lengths=True)

        # Standardize if necessary
        if not isinstance(self, Classifier) and self.standardize:
            self._y_mean = y.mean()
            self._y_std = y.std(ddof=0)
            if self._y_std == 0:
                warnings.warn("Response variable is constant.")
                self._y_std = 1.0
            y = (y - self._y_mean) / self._y_std

        return y
