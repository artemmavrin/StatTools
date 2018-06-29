"""Estimation for Gaussian mixture models using the EM algorithm.

See Section 9.2 in Bishop (2006).

References
----------
Christopher Bishop. Pattern Recognition and Machine Learning. Springer-Verlag
    New York (2006), pp. xx+738.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from ..cluster import KMeansCluster
from ..generic import Fittable
from ..utils.validation import validate_float
from ..utils.validation import validate_int
from ..utils.validation import validate_samples


class GaussianMixture(Fittable):
    """Estimate the parameters of a Gaussian mixture model (GMM).

    Parameters are estimated by numerical MLE (maximum likelihood estimation)
    using the EM algorithm (Dempster, Laird, & Rubin 1977).

    Properties
    ----------
    k : int
        Number of components in the Gaussian mixture model.
    p : int
        Number of dimensions of the feature space.
    means : numpy.ndarray
        Array of shape (k, p) consisting of the mean vectors of the k Gaussian
        components.
    covs : numpy.ndarray
        Array of shape (k, p, p) consisting of the covariance matrices of the k
        Gaussian components.
    weights : numpy.ndarray
        Array of shape (k,) consisting of the mixture weights for each Gaussian
        component.

    References
    ----------
    A. P. Dempster, N. M. Laird, and D. B. Rubin. "Maximum Likelihood from
        Incomplete Data via the EM Algorithm". Journal of the Royal Statistical
        Society. Series B (Methodological). Vol. 39, No. 1 (1977), pp. 1--38.
        JSTOR: https://www.jstor.org/stable/2984875
    Christopher Bishop. Pattern Recognition and Machine Learning.
        Springer-Verlag New York (2006), pp. xx+738.
    """
    k: int = None
    p: int = None
    means: np.ndarray = None
    covs: np.ndarray = None
    weights: np.ndarray = None

    def __init__(self, k):
        """Initialize a GMM by specifying the number of Gaussian components.

        Parameters
        ----------
        k : int
            Number of components in the Gaussian mixture model.
        """
        # Validate parameters
        self.k = validate_int(k, "k", minimum=1)

    def fit(self, x, tol=1e-5, iterations=None, repeats=5, random_state=None,
            init_iterations=10, init_repeats=30, cluster_kwargs=None):
        """Fit the Gaussian mixture model (i.e., estimate the parameters).

        Fitting is done using (potentially several runs of) the EM algorithm to
        maximize the Gaussian mixture log-likelihood. Several runs could be
        needed since the EM algorithm is not guaranteed to converge to a global
        maximum of the log-likelihood. The different runs vary in their choices
        of initial parameter values.

        The first run is always done by using k-means clustering to partition
        the data into k clusters, and using each cluster's sample mean, sample
        covariance, and proportion of the total data as the initial estimates.

        Subsequent runs are done by randomly assigning the observations into
        k roughly-equally-sized clusters and running the EM algorithm for a
        small number of iterations. This is repeated a small number of times,
        and the initial configuration yielding the largest log-likelihood
        becomes the initial configuration for the full EM algorithm. This method
        is adapted from Shireman, Steinley, & Brusco (2017).

        Parameters
        ----------
        x : array-like
            Sample matrix of shape (n, p) (n=number of observations, p=number of
            features).
        tol : float, optional
            Positive tolerance for algorithm convergence. If an iteration of the
            EM algorithm increases the log-likelihood by less than `tol`, then
            the algorithm is terminated.
        iterations : int or None, optional
            Maximum number of iterations to perform. If None, no maximum number
            is imposed and the algorithm will continue running until the
            stopping criterion determined by `tol` is satisfied.
        repeats : int, optional
            Number of times to repeat the algorithm with different initial
            parameter values. This can decrease the chance of finding only
            non-global maxima of the log-likelihood function.
        random_state : int or numpy.random.RandomState object, optional
            A valid initializer for a numpy.random.RandomState object.
        init_iterations : int, optional
            Number of iterations of the EM algorithm when initializing the
            parameter values using random cluster assignments (described above).
        init_repeats : int, optional
            Number of times to repeat the EM algorithm when initializing the
            parameter values using random cluster assignments (described above).
        cluster_kwargs : dict, optional
            Dictionary of keyword arguments to pass to KMeansCluster.fit(). To
            be used when determining initial parameters for the EM algorithm by
            k-means clustering.

        Returns
        -------
        This GaussianMixture instance.

        References
        ----------
        Emilie Shireman, Douglas Steinley, and Michael J. Brusco. "Examining the
            effect of initialization strategies on the performance of Gaussian
            mixture modeling". Behavior Research Methods. Vol. 49, No. 1 (2017),
            pp. 282--293. DOI: https://doi.org/10.3758/s13428-015-0697-6
        """
        # Validate parameters
        x, tol, iterations, repeats, init_iterations, init_repeats \
            = _validate_fit_params(x, self.k, tol, iterations, repeats,
                                   init_iterations, init_repeats)

        # n = number of observations, p = number of features
        n, self.p = x.shape

        # Seed the RNG
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        # Initialize arrays for the model parameters
        self.means = np.empty(shape=(self.k, self.p), dtype=np.float_)
        self.covs = np.empty(shape=(self.k, self.p, self.p), dtype=np.float_)
        self.weights = np.empty(shape=(self.k,), dtype=np.float_)

        # Repeat the EM algorithm with different initial parameter values. At
        # the end, the parameter estimates yielding the highest log-likelihood
        # will be selected.
        log_likelihood = -np.Inf
        for r in range(repeats):
            # Initialize the parameters of the model
            if r == 0:
                # First iteration: use k-means clustering
                if cluster_kwargs is None:
                    cluster_kwargs = dict()
                elif not isinstance(cluster_kwargs, dict):
                    raise TypeError(
                        "Parameter 'cluster_kwargs' must be a dict.")
                params = _init_param_cluster(x, k=self.k,
                                             random_state=random_state,
                                             **cluster_kwargs)
            else:
                # Subsequent iterations: use random cluster assignments
                params = _init_param_random(x, k=self.k,
                                            random_state=random_state,
                                            repeats=init_repeats, tol=tol,
                                            iterations=init_iterations)

            # Fit the model
            *params, ll_new = _fit_gmm(x, self.k, *params, tol=tol,
                                       iterations=iterations)

            # Compare the new log-likelihood with the best log-likelihood so far
            # and update the parameters if necessary
            if ll_new > log_likelihood:
                log_likelihood = ll_new
                self.means = params[0]
                self.covs = params[1]
                self.weights = params[2]

        self.fitted = True
        return self

    def pdf(self, x):
        """Compute the resulting Gaussian mixture model's density function.

        Parameters
        ----------
        x : array-like
            Sample matrix of shape (n, p) (n=number of observations, p=number of
            features). The number of features must be the same as in the sample
            matrix used to fit the model.
        """
        if not self.fitted:
            raise self.unfitted_exception

        # Validate the data matrix
        x = validate_samples(x, n_dim=2)
        if x.shape[1] != self.p:
            raise ValueError(f"Data has wrong number of columns: expected "
                             f"{self.p}, found {x.shape[1]}")

        val = _densities(x, self.k, self.means, self.covs).dot(self.weights)
        return val.reshape(-1)

    def plot(self, num=200, ax=None, **kwargs):
        """Plot the Gaussian mixture density in the 1D and 2D cases.

        In the 1D case, the density curve is plotted. In the 2D case, a contour
        plot of the density is drawn.

        Parameters
        ----------
        num : int
            Number of points to sample.
        ax : matplotlib.axes.Axes, optional
            The matplotlib.axes.Axes on which to plot.
        kwargs : dict
            Additional keyword arguments to pass to either ax.plot() (in the 1D
            case) or ax.contourf() (in the 2D case).

        Returns
        -------
        The matplotlib.axes.Axes on which the density was plotted.
        """
        num = validate_int(num, "num", minimum=1)

        if ax is None:
            ax = plt.gca()

        if self.p == 1:
            x_min, x_max = ax.get_xlim()
            x = np.linspace(x_min, x_max, num)
            ax.plot(x, self.pdf(x), **kwargs)
            ax.set_xlim(x_min, x_max)
        elif self.p == 2:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x = np.linspace(x_min, x_max, num)
            y = np.linspace(y_min, y_max, num)
            xx, yy = np.meshgrid(x, y)
            data = np.column_stack((xx.ravel(), yy.ravel()))
            density = self.pdf(data).reshape(-1, num)
            ax.contourf(xx, yy, density, **kwargs)
        else:
            raise AttributeError(
                "GMM plotting is only supported for 1D and 2D models.")

        return ax


def _init_param_cluster(x: np.ndarray, k: int,
                        random_state: np.random.RandomState, **kwargs):
    """Initialize the parameters for Gaussian mixture model fitting using
    k-means clustering.

    Parameters
    ----------
    x : numpy.ndarray
        Sample matrix of shape (n, p) (n=number of observations, p=number of
        features).
    k : int
        Number of components in the Gaussian mixture model.
    random_state : numpy.random.RandomState
        Random number generator.
    kwargs : dict
        Additional keyword arguments to pass to the KMeansCluster model used to
        partition the data.

    Returns
    -------
    means : numpy.ndarray of shape (k, p)
        Initial mean vectors for the EM algorithm.
    covs : numpy.ndarray of shape (k, p, p)
        Initial covariance matrices for the EM algorithm.
    weights : numpy.ndarray of shape (k,)
        Initial weights for the EM algorithm.
    """
    # n = number of observations, p = number of features
    n, p = x.shape

    # Initialize arrays for the parameters
    means = np.empty(shape=(k, p), dtype=np.float_)
    covs = np.empty(shape=(k, p, p), dtype=np.float_)
    weights = np.empty(shape=(k,), dtype=np.float_)

    # Partition the data using k-means clustering
    kmc = KMeansCluster(k=k, standardize=True)
    fit_params = dict(random_state=random_state)
    fit_params.update(kwargs)
    kmc.fit(x, **fit_params)
    clusters = kmc.predict(x=x)

    # Initialize the parameters using per-cluster estimates
    for i in range(k):
        means[i] = np.mean(x[clusters == i], axis=0)
        covs[i] = np.cov(x[clusters == i], rowvar=False)
        weights[i] = np.sum(clusters == i) / n

    return means, covs, weights


def _init_param_random(x: np.ndarray, k: int,
                       random_state: np.random.RandomState, repeats: int,
                       tol: float, iterations: int):
    """Initialize the parameters for Gaussian mixture model fitting using random
    cluster assignment and constrained EM iterations.

    Parameters
    ----------
    x : numpy.ndarray
        Sample matrix of shape (n, p) (n=number of observations, p=number of
        features).
    k : int
        Number of components in the Gaussian mixture model.
    random_state : numpy.random.RandomState
        Random number generator.
    repeats : int
        Number of times to repeat the constrained EM iterations.
    tol : float
        Tolerance to determine early stopping of the EM algorithm (based on
        convergence of the log-likelihood).
    iterations: int
        Number of iterations of the EM algorithm.

    Returns
    -------
    means : numpy.ndarray of shape (k, p)
        Initial mean vectors for the EM algorithm.
    covs : numpy.ndarray of shape (k, p, p)
        Initial covariance matrices for the EM algorithm.
    weights : numpy.ndarray of shape (k,)
        Initial weights for the EM algorithm.
    """
    # n = number of observations, p = number of features
    n, p = x.shape

    # Initialize arrays for the parameters
    means = np.empty(shape=(k, p), dtype=np.float_)
    covs = np.empty(shape=(k, p, p), dtype=np.float_)
    weights = np.empty(shape=(k,), dtype=np.float_)

    # Repeat the EM algorithm with different initial parameter values. At the
    # end, the parameter estimates yielding the highest log-likelihood will be
    # selected.
    log_likelihood = -np.Inf
    for _ in range(repeats):
        # Randomly divide the data into k clusters
        clusters = np.tile(np.arange(k), reps=(int(n / k) + 1))[:n]
        random_state.shuffle(clusters)

        # Initialize the parameters using per-cluster estimates
        means_ = np.empty(shape=(k, p), dtype=np.float_)
        covs_ = np.empty(shape=(k, p, p), dtype=np.float_)
        weights_ = np.empty(shape=(k,), dtype=np.float_)
        for i in range(k):
            means_[i] = np.mean(x[clusters == i], axis=0)
            covs_[i] = np.cov(x[clusters == i], rowvar=False)
            weights_[i] = np.sum(clusters == i) / n

        # Run the EM algorithm a restricted number of times and see which gives
        # the best log-likelihood at the end.
        means_, covs_, weights_, ll_new = _fit_gmm(x, k, means_, covs_,
                                                   weights_, tol=tol,
                                                   iterations=iterations)

        # Compare the new log-likelihood with the best log-likelihood so far and
        # update the parameters if necessary
        if ll_new > log_likelihood:
            log_likelihood = ll_new
            means = means_
            covs = covs_
            weights = weights_

    return means, covs, weights


def _densities(x, k, means, covs):
    """Compute the Gaussian density for each observation and each component.

    Parameters
    ----------
    x : numpy.ndarray
        Sample matrix of shape (n, p) (n=number of observations, p=number of
        features).
    k : int
        Number of Gaussian components.
    means : numpy.ndarray of shape (k, p)
        Mean vectors of the k Gaussian components.
    covs : numpy.ndarray of shape (k, p, p)
        Covariance matrices of the k Gaussian components.

    Returns
    -------
    densities : numpy.ndarray of shape (n, k)
        The entry in the i-th row and j-th column is the density of the j-th
        component at the i-th observation.
    """
    # Compute matrix of normal density values
    densities = np.empty(shape=(x.shape[0], k), dtype=np.float_)
    for j in range(k):
        densities[:, j] = st.multivariate_normal.pdf(x=x, mean=means[j],
                                                     cov=covs[j])
    return densities


def _responsibility(densities: np.ndarray, weights: np.ndarray):
    """Compute the responsibilities and densities of a Gaussian mixture model.

    This is the E step of the EM algorithm for fitting Gaussian mixture models.

    Parameters
    ----------
    densities : numpy.ndarray of shape (n, k)
        The entry in the i-th row and j-th column is the density of the j-th
        component at the i-th observation.
    weights : numpy.ndarray of shape (k,)
        Mixture weights for each Gaussian component.

    Returns
    -------
    gamma : numpy.ndarray of shape (n, k)
        Matrix of responsibilities. Each column represents the responsibility
        of that Gaussian component for the data.
    """
    # Compute matrix of responsibilities (Bishop (2006) calls it gamma)
    gamma = weights * densities
    gamma /= np.sum(gamma, axis=1, keepdims=True)

    return gamma


def _update_param(x: np.ndarray, k: int, gamma: np.ndarray):
    """Update the Gaussian mixture model parameters using the data and the
    responsibilities.

    This is the M step of the EM algorithm for fitting Gaussian mixture models.

    Parameters
    ----------
    x : numpy.ndarray
        Sample matrix of shape (n, p) (n=number of observations, p=number of
        features).
    k : int
        Number of Gaussian components.
    gamma : numpy.ndarray of shape (n, k)
        Matrix of responsibilities. Each column represents the responsibility
        of that Gaussian component for the data.

    Returns
    -------
    means : numpy.ndarray of shape (k, p)
        Updated mean vectors.
    covs : numpy.ndarray of shape (k, p, p)
        Updated covariance matrices.
    weights : numpy.ndarray of shape (k,)
        Updated weights.
    """
    p = x.shape[1]

    means = np.empty(shape=(k, p), dtype=np.float_)
    covs = np.zeros(shape=(k, p, p), dtype=np.float_)
    for j in range(k):
        responsibility = gamma.T[j].reshape(-1)
        means[j] = np.average(x, axis=0, weights=responsibility)
        covs[j] = np.cov(x, rowvar=False, aweights=responsibility)

    weights = np.mean(gamma, axis=0)

    return means, covs, weights


def _fit_gmm(x: np.ndarray, k: int, means: np.ndarray, covs: np.ndarray,
             weights: np.ndarray, tol: float, iterations: int):
    """Perform the Gaussian mixture model EM algorithm given initial parameters.

    Parameters
    ----------
    x : numpy.ndarray
        Sample matrix of shape (n, p) (n=number of observations, p=number of
        features).
    k : int
        Number of Gaussian components.
    means : numpy.ndarray of shape (k, p)
        Initial mean vectors for the EM algorithm.
    covs : numpy.ndarray of shape (k, p, p)
        Initial covariance matrices for the EM algorithm.
    weights : numpy.ndarray of shape (k,)
        Initial weights for the EM algorithm.
    tol : float
        Positive tolerance for algorithm convergence. If an iteration of the EM
        algorithm increases the log-likelihood by less than `tol`, then the
        algorithm is terminated.
    iterations : int or None, optional
        Maximum number of iterations to perform. If None, no maximum number is
        imposed and the algorithm will continue running until the stopping
        criterion determined by `tol` is satisfied.

    Returns
    -------
    means : numpy.ndarray of shape (k, p)
        Final mean vectors.
    covs : numpy.ndarray of shape (k, p, p)
        Final covariance matrices.
    weights : numpy.ndarray of shape (k,)
        Final weights.
    log_likelihood : float
        Final log-likelihood.
    """
    # Compute initial log likelihood
    densities = _densities(x, k, means, covs)
    log_likelihood = _log_likelihood(weights, densities)

    # EM algorithm
    counter = itertools.repeat(0) if iterations is None else range(iterations)
    for _ in counter:
        old_log_likelihood = log_likelihood

        # E step
        gamma = _responsibility(densities, weights)

        # M step
        means, covs, weights = _update_param(x, k, gamma)

        # Compute the new log-likelihood
        densities = _densities(x, k, means, covs)
        log_likelihood = _log_likelihood(weights, densities)

        # Check for convergence
        if old_log_likelihood <= log_likelihood < old_log_likelihood + tol:
            break

    return means, covs, weights, log_likelihood


def _log_likelihood(weights, densities):
    """Compute the Gaussian mixture model log-likelihood.

    Parameters
    ----------
    weights : numpy.ndarray of shape (k,)
        Mixture weights for each Gaussian component.
    densities : numpy.ndarray of shape (n, k)
        The entry in the i-th row and j-th column is the density of the j-th
        component at the i-th observation.

    Returns
    -------
    The log-likelihood.
    """
    return np.sum(np.log(densities.dot(weights)))


def _validate_fit_params(x, k, tol, iterations, repeats, init_iterations,
                         init_repeats):
    """Validate the parameters for GaussianMixture.fit().

    Parameters
    ----------
    See the parameter descriptions for GaussianMixture.fit().

    Returns
    -------
    The updated parameters.
    """
    x = validate_samples(x, n_dim=2)
    if len(x) <= k:
        print("There must be more observations than Gaussian components.")

    tol = validate_float(tol, "tol", positive=True)

    if iterations is not None:
        iterations = validate_int(iterations, "iterations", minimum=1)

    repeats = validate_int(repeats, "repeats", minimum=1)

    init_iterations = validate_int(init_iterations, "init_iterations",
                                   minimum=1)

    init_repeats = validate_int(init_repeats, "init_repeats", minimum=1)

    return x, tol, iterations, repeats, init_iterations, init_repeats
