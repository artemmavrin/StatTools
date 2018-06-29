"""Linear models with regularization"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .linear import LinearModel
from ..utils import validate_float
from ..utils import validate_int


class Ridge(LinearModel):
    """Ridge regression: linear regression with an L2 penalty."""

    def __init__(self, lam=0.1):
        """Initialize a Ridge object.

        Parameters
        ----------
        lam : float (>0)
            Regularization constant.
        """
        # Validate parameters
        self.lam = validate_float(lam, "lam", positive=True)

        super(Ridge, self).__init__(standardize=True)

    def fit(self, x, y, names=None):
        """Fit the ridge regression model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Explanatory variables.
        y : array-like, shape (n,)
            Response variable.
        names : list, optional
            List of feature names corresponding to the columns of `x`.

        Returns
        -------
        This Ridge instance.
        """
        # Validate input
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_response(y=y, x=x)

        # Fit the model by least squares
        a = x.T.dot(x) + self.lam * np.identity(self._p)
        b = x.T.dot(y)

        self._coef, *_ = np.linalg.lstsq(a=a, b=b, rcond=None)
        self.fitted = True
        return self


def _soft_threshold(a, b):
    """Soft-threshold operator for the LASSO and elastic net."""
    return np.sign(a) * np.clip(np.abs(a) - b, a_min=0, a_max=None)


def _enet_loss(x, y, coef, lam, alpha):
    """Loss function for the elastic net."""
    mse = np.mean((y - x.dot(coef)) ** 2)
    l1 = np.sum(np.abs(coef))
    l2 = np.sum(coef ** 2)
    return 0.5 * mse + lam * (alpha * l1 + (1 - alpha) * 0.5 * l2)


def _enet_cd(x, y, coef0, lam, alpha, tol, max_iter, random, seed, callback):
    """Approximate the elastic net estimator for the linear model coefficients
    by coordinate descent.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Explanatory variables (AKA features/predictors/regressors).
    y : array-like, shape (n,)
        Response variable (AKA targets).
    coef0 : array-like, shape (p,)
        Initial guess for the elastic net coefficient estimator.
    lam : float (>0)
        Regularization constant.
    alpha : float (in [0, 1])
        L1/L2 mixing parameter. alpha=1 means L1 penalty only, alpha=0 means L2
        penalty only. 0<alpha<1 means weighted sum of L1 and L2 penalties.
    tol : float (>0)
        Convergence tolerance. The coordinate descent algorithm stops early if
        the largest coefficient update in that iteration is less than
        tol * (largest coefficient in that iteration).
    max_iter : int (>0)
        Maximum number of iterations of coordinate descent to perform.
    random : bool
        If True, coefficient selection is random during each iteration of the
        coordinate descent algorithm. If False, coefficient selection is cyclic
        (i.e., 0, 1, 2, ..., p-1, 0, 1, ...).
    seed : int
        Seed for a NumPy RandomState object. Used if `random` is True; otherwise
        ignored.
    callback : callable
        Optional function of the standardized coefficients to call during every
        iteration of the coordinate descent algorithm.

    Returns
    -------
    coef : numpy.ndarray, shape (p,)
        Approximate elastic net estimator for the linear model coefficients.
    n_iter : int (>0)
        Number of iterations of coordinate descent that were performed.

    References
    ----------
    Jerome Friedman, Trevor Hastie, and Robert Tibshirani. "Regularization Paths
        for Generalized Linear Models via Coordinate Descent". Journal of
        Statistical Software Vol. 33, No. 1, 2010, pp. 1--22. PMCID: PMC2929880
    """
    # Validate `tol` and `max_iter`
    tol = validate_float(tol, "tol", positive=True)
    max_iter = validate_int(max_iter, "max_iter", minimum=1)

    # Number of explanatory variables
    p = x.shape[1]

    # Initialize a random number generator
    rng = np.random.RandomState(seed)

    # Initialize the coefficient vector
    coef = np.copy(coef0)
    if callback is not None:
        callback(coef)

    # Coordinate descent algorithm
    n_iter = 0
    while True:
        n_iter += 1
        coef_max = 0.0
        coef_update_max = 0.0

        # Cycle through coefficients OR choose coefficients at random
        for j in range(p):
            if random:
                j = rng.randint(p)

            # Previous coefficient at index j
            coef_j_prev = coef[j]

            # Compute the update
            temp = coef[j] + np.mean((y - x.dot(coef)) * x[:, j])
            coef[j] = _soft_threshold(temp, lam * alpha) / 1 + lam * (1 - alpha)
            if callback is not None:
                callback(coef)

            if np.abs(coef[j]) > coef_max:
                coef_max = np.abs(coef[j])
            if np.abs(coef[j] - coef_j_prev) > coef_update_max:
                coef_update_max = np.abs(coef[j] - coef_j_prev)

        # Check for stopping criteria
        if (coef_max == 0 or coef_update_max / coef_max < tol or
                n_iter >= max_iter):
            break

    return coef, n_iter


def _enet_path(x, y, coef0, lambdas, alpha, tol, max_iter, random, seed):
    """Compute an elastic net regularization path.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Explanatory variables (AKA features/predictors/regressors).
    y : array-like, shape (n,)
        Response variable (AKA targets).
    coef0 : array-like, shape (p,)
        Initial guess for the elastic net coefficient estimator.
    lambdas : array-like, shape (k,)
        List of positive floats representing different regularization constants.
    alpha : float (in [0, 1])
        L1/L2 mixing parameter. alpha=1 means L1 penalty only, alpha=0 means L2
        penalty only. 0<alpha<1 means weighted sum of L1 and L2 penalties.
    tol : float (>0)
        Convergence tolerance. The coordinate descent algorithm stops early if
        the largest coefficient update in that iteration is less than
        tol * (largest coefficient in that iteration).
    max_iter : int (>0)
        Maximum number of iterations of coordinate descent to perform.
    random : bool
        If True, coefficient selection is random during each iteration of the
        coordinate descent algorithm. If False, coefficient selection is cyclic
        (i.e., 0, 1, 2, ..., p-1, 0, 1, ...).
    seed : int
        Seed for a NumPy RandomState object. Used if `random` is True; otherwise
        ignored.

    Returns
    -------
    path : numpy.ndarray, shape (k, p)
        Matrix in which the ith row is the coefficient vector for the ith
        lambda in `lambdas` (after sorting in reversed order). Moreover, the jth
        column is the regularization path of the jth coefficient.
    lambdas : array-like, shape (k,)
        List of positive floats representing different regularization constants.
    """
    # Ensure `lambdas` is sorted in reverse order
    lambdas = np.flipud(np.sort(lambdas))

    # Compute the path
    path = np.empty((len(lambdas), x.shape[1]))
    for i, lam in enumerate(lambdas):
        coef0, *_ = _enet_cd(x=x, y=y, coef0=coef0, lam=lam, alpha=alpha,
                             tol=tol, max_iter=max_iter, random=random,
                             seed=seed, callback=None)
        path[i, :] = coef0

    return path, lambdas


class ElasticNet(LinearModel):
    """Linear regression with the elastic net penalty (i.e., a linear
    combination of L1 (LASSO) and L2 (ridge) penalties).
    """

    def __init__(self, lam=0.1, alpha=1):
        """Initialize an elastic net model.

        Parameters
        ----------
        lam : float (>0)
            Regularization constant.
        alpha : float (in [0, 1])
            L1/L2 mixing parameter. alpha=1 means L1 penalty only, alpha=0 means
            L2 penalty only. 0<alpha<1 means weighted sum of L1 and L2
            penalties.
        """
        # Validate parameters
        self.lam = validate_float(lam, "lam", positive=True)
        self.alpha = validate_float(alpha, "alpha", minimum=0.0, maximum=1.0)

        super(ElasticNet, self).__init__(standardize=True, fit_intercept=True)

    def fit(self, x, y, names=None, tol=1e-4, max_iter=1000, random=False,
            seed=None, callback=None, warm_start=True, verbose=False):
        """Fit the elastic net model using coordinate descent.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable matrix (AKA feature matrix or design
            matrix). Columns of `x` correspond to different explanatory
            variables (i.e., regressors/predictors); rows of `x` correspond to
            different observations of the explanatory variables (i.e., n=number
            of observations, p=number of explanatory variables). If `x` is a
            scalar or one-dimensional array, then it is interpreted as a single
            explanatory variable (i.e., a matrix of shape (n, 1)).
        y : array-like, shape (n,)
            The response variable vector (AKA target vector).
        names : list, optional
            List of feature names corresponding to the columns of `x`.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Number of iterations of coordinate descent to perform.
        random : bool, optional
            If True, the coordinate along which to maximize is selected randomly
            in each iteration. Otherwise, the coordinates are cycled through in
            order.
        seed : int, optional
            Seed for a NumPy RandomState object. Used if `random` is True;
            otherwise ignored.
        callback : callable, optional
            Optional function of the standardized coefficients to call during
            every iteration of the coordinate descent algorithm.
        warm_start : bool, optional
            If True, initialize the coefficient with a previously computed
            coefficient vector (if available).
        verbose : bool, optional
            If True, print some fitting convergence results to stdout.

        Returns
        -------
        self : ElasticNet
            This ElasticNet instance.
        """
        # Validate explanatory and response variables
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_response(y=y, x=x)

        # Initialize the coefficients
        if warm_start and self.fitted:
            if len(self._coef) == self._p:
                coef0 = self._coef
            else:
                warnings.warn("Cannot perform warm start: existing coefficient "
                              "estimate has an incompatible length.")
                coef0 = np.zeros(self._p)
        else:
            coef0 = np.zeros(self._p)

        # Coordinate descent
        self._coef, n_iter = _enet_cd(x=x, y=y, coef0=coef0, lam=self.lam,
                                      alpha=self.alpha, tol=tol,
                                      max_iter=max_iter, random=random,
                                      seed=seed, callback=callback)

        if verbose:
            # Final value of the loss function of the standardized data
            loss = _enet_loss(x=x, y=y, coef=self._coef,
                              lam=self.lam, alpha=self.alpha)

            # Summarize fit
            print(f"Fitted elastic net (λ: {self.lam}, α: {self.alpha}) with "
                  f"{len(y)} observations, {self._p} predictors")
            print(f"\tNumber of iterations: {n_iter} (tolerance: {tol})")
            print(f"\tFinal (standardized) elastic net loss: {loss:.5f}")

        self.fitted = True
        return self

    def path(self, x, y, lam_min, lam_max, n_lam=50, names=None, tol=1e-4,
             max_iter=1000, random=False, seed=None):
        """Return a regularization path for the coefficients accross a grid of
        lambda values.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable matrix.
        y : array-like, shape (n,)
            The response variable vector.
        lam_min : float
            The minimum lambda.
        lam_max : float
            The maximum lambda.
        n_lam : int, optional
            Number of lambdas to consider.
        names : list, optional
            List of feature names corresponding to the columns of `x`.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Number of iterations of coordinate descent to perform.
        random : bool, optional
            If True, the coordinate along which to maximize is selected randomly
            in each iteration. Otherwise, the coordinates are cycled through in
            order.
        seed : int, optional
            Seed for a NumPy RandomState object. Used if `random` is True;
            otherwise ignored.

        Returns
        -------
        coefs : numpy.ndarray
            Matrix of coefficient paths. Each column is the path of one of the
            coefficients from the biggest to the smallest lambda. Note: these
            are coefficients for the standardized explanatory and response
            variables.
        lambdas : numpy.ndarray
            Discrete lambda values from the biggest to the smallest, spaced
            logarithmically.
        """
        # Validate explanatory and response variables
        x = self._preprocess_features(x=x, names=names)
        y = self._preprocess_response(y=y, x=x)

        # Initialize the coefficients
        coef0 = np.zeros(self._p)

        # Return the path
        lambdas = np.geomspace(lam_min, lam_max, n_lam)
        return _enet_path(x=x, y=y, coef0=coef0, lambdas=lambdas,
                          alpha=self.alpha, tol=tol, max_iter=max_iter,
                          random=random, seed=seed)

    def path_plot(self, x, y, lam_min, lam_max, n_lam=50, tol=1e-4,
                  max_iter=1000, random=False, seed=None, ax=None, **kwargs):
        """Draw the regularization path for the elastic net model.

        Parameters
        ----------
        x : array-like, shape (n, p)
            The explanatory variable matrix.
        y : array-like, shape (n,)
            The response variable vector.
        lam_min : float
            The minimum lambda.
        lam_max : float
            The maximum lambda.
        n_lam : int, optional
            Number of lambdas to consider.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Number of iterations of coordinate descent to perform.
        random : bool, optional
            If True, the coordinate along which to maximize is selected randomly
            in each iteration. Otherwise, the coordinates are cycled through in
            order.
        seed : int, optional
            Seed for a NumPy RandomState object. Used if `random` is True;
            otherwise ignored.
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the plot.
        kwargs : dict, optional
            Additional keyword arguments to provide to the plot() function of
            the axes.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes on which the plot was drawn.
        """
        path, lambdas = self.path(x=x, y=y, lam_min=lam_min, lam_max=lam_max,
                                  n_lam=n_lam, tol=tol, max_iter=max_iter,
                                  random=random, seed=seed)
        if ax is None:
            ax = plt.gca()

        for j in range(self._p):
            ax.plot(-np.log(lambdas), path[:, j], **kwargs)

        ax.set(title="Regularization Path")
        ax.set(xlabel="$-\log(\lambda)$", ylabel="Standardized Coefficient")
        ax.set(xlim=(-np.log(lam_max), -np.log(lam_min)))

        return ax
