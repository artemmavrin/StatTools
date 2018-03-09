"""Defines the abstract base class for generalized linear models."""

import abc

import numpy as np

from ..generic import Fittable


class GeneralizedLinearModel(Fittable, metaclass=abc.ABCMeta):
    """Generalized linear model abstract base class."""

    # Indicates whether the module should fit an intercept term
    fit_intercept: bool = True

    # Coefficients of the model
    coef: np.ndarray = None

    # Number of columns of compatible feature matrices---to be determined during
    # model fitting
    _num_features: int = None

    def _preprocess_x(self, x, fitting=False) -> np.ndarray:
        """Apply necessary validation and preprocessing to the explanatory
        variable of a generalized linear model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable.
        fitting : bool, optional
            Indicates whether preprocessing is being done during fitting

        Returns
        -------
        x : numpy.ndarray, shape (n, p) or (n, p + 1)
            Updated explanatory variable. If `intercept` is True, then a column
            of 1's is prepended to `x`.
        """
        # Coerce to NumPy array
        if np.ndim(x) <= 1:
            x = np.atleast_2d(x).T
        elif np.ndim(x) == 2:
            x = np.asarray(x)
        else:
            raise ValueError("Explanatory variable must be 2-dimensional.")

        # Prepend intercept column if necessary
        if self.fit_intercept:
            x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        if fitting:
            self._num_features = x.shape[1]
        else:
            if x.shape[1] != self._num_features:
                raise ValueError(f"Expected {self._num_features} columns, "
                                 f"but found {np.shape(x)[1]}")

        return x

    @staticmethod
    def _preprocess_y(y) -> np.ndarray:
        """Apply necessary validation and preprocessing to the response variable
        of a generalized linear model.

        Parameters
        ----------
        y : array-like, shape (n, )
            Response variable.

        Returns
        -------
        y : numpy.ndarray, shape (n, )
            Updated response variable.
        """
        # Coerce to NumPy array
        if np.ndim(y) <= 1:
            y = np.atleast_1d(y)
        else:
            raise ValueError("Response variable must be 1-dimensional.")

        return y

    @staticmethod
    @abc.abstractmethod
    def _inv_link(*args, **kwargs):
        """Inverse link function for the given generalized linear model."""
        pass

    def estimate(self, x):
        """Return the model's estimate for the given input data.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variable

        Returns
        -------
        link^{-1}(x * coef), where f is the model's inverse link function and
        coef is the model's coefficient vector.
        """
        # Check whether the model is fitted
        if not self.is_fitted():
            raise self.unfitted_exception()

        # Validate input
        x = self._preprocess_x(x, fitting=False)

        return self._inv_link(x.dot(self.coef))
