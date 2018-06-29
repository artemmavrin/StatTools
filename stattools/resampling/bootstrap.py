"""Implementations of the non-parametric bootstrap and the Bayesian bootstrap
for sampling distribution estimation.

References
----------
Bradley Efron. "Bootstrap Methods: Another Look at the Jackknife". The Annals of
    Statistics, Volume 7, Number 1 (1979), 1--26. doi:10.1214/aos/1176344552
Donald B. Rubin. "The Bayesian bootstrap". The Annals of Statistics, Volume 9,
    Number 1 (1981), 130--134. doi:10.1214/aos/1176345338
"""

import numpy as np
import scipy.stats as st

from ..utils.validation import validate_float
from ..utils.validation import validate_func
from ..utils.validation import validate_int
from ..utils.validation import validate_samples


# Non-parametric bootstrap

def bootstrap(*samples, n_boot, random_state=None, ret_list=False):
    """Generate bootstrap samples by drawing with replacement.

    Parameters
    ----------
    samples : sequence of arrays
        Sequence of samples from which to draw bootstrap samples.
    n_boot : int
        Number of bootstrap samples to draw. This is necessarily a keyword
        argument.
    random_state : numpy.random.RandomState, int, or array-like, optional
        If a numpy.random.RandomState object, this is used as the random number
        generator for sampling with replacement. Otherwise, this is the seed
        for a numpy.random.RandomState object to be used as the random number
        generator. This is necessarily a keyword argument.
    ret_list : bool, optional
        Indicates whether the bootstrap samples should be returned as a list
        even if there is only one original sample. This is necessarily a keyword
        argument.

    Returns
    -------
    boots : list
        List of bootstrap samples drawn from each original sample.
    """
    # Validate the samples
    samples = validate_samples(*samples, equal_lengths=True, ret_list=True)
    n_samples = len(samples)
    n_obs = len(samples[0])

    # Ensure `n_boot` is a positive integer
    n_boot = validate_int(n_boot, "n_boot", minimum=1)

    # Initialize the random number generator if necessary
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # Initialize arrays for the bootstrap samples
    boots = [np.empty(((n_boot,) + samples[i].shape)) for i in range(n_samples)]

    # Generate the bootstrap samples
    for b in range(n_boot):
        indices = random_state.randint(0, n_obs, n_obs)
        boot = [sample.take(indices, axis=0) for sample in samples]
        for i in range(n_samples):
            boots[i][b] = boot[i]

    # Return the bootstrap samples
    if ret_list or n_samples > 1:
        return boots
    else:
        return boots[0]


class Bootstrap(object):
    """Class for generic sampling distribution estimation using the bootstrap.

    Properties
    ----------
    n_boot : int
        Number of bootstrap samples.
    samples_boot : list
        A list in which the ith element consists of bootstrap samples drawn from
        the ith original sample.
    dist : numpy.ndarray
        Bootstrap sampling distribution of the statistic.
    observed : object
        Observed value of the statistic.
    """
    n_boot: int = None
    samples_boot: list = None
    dist: np.ndarray = None
    observed = None

    def __init__(self, *samples, stat, n_boot, random_state=None, **kwargs):
        """Generate bootstrap estimates by sampling with replacement from a
        sample and re-computing the statistic each time.

        Parameters
        ----------
        samples: sequence of arrays
            Samples on which to perform the bootstrap resampling procedure. Each
            array in this sequence should have the same length (i.e., sample
            size), but there is no restriction on the shape otherwise.
        stat: callable or str
            The statistic to compute from the data. If this parameter is a
            string, then it should be the name of a NumPy array method (e.g.,
            "mean" or "std"). If this parameter is a function, then it should
            accept as many arrays (and of the same shape) as are in `samples`.
            The statistic is not assumed to be scalar-valued. This parameter is
            necessarily a keyword argument.
        n_boot : int
            Number of bootstrap samples to draw. This is necessarily a keyword
            argument.
        random_state : numpy.random.RandomState, int, or array-like, optional
            If a numpy.random.RandomState object, this is used as the random
            number generator for sampling with replacement. Otherwise, this is
            the seed for a numpy.random.RandomState object to be used as the
            random number generator. This is necessarily a keyword argument.
        kwargs: dict, optional
            Additional keyword arguments to pass to the function represented by
            the parameter `stat`.
        """
        # Validate the statistic
        stat = validate_func(stat, **kwargs)

        # Generate bootstrap samples
        self.samples_boot = bootstrap(*samples, n_boot=n_boot,
                                      random_state=random_state, ret_list=True)

        # Compute the bootstrap sampling distribution
        dist = [stat(*samples) for samples in zip(*self.samples_boot)]

        # Store bootstrap sampling distribution and the observed statistic
        self.dist = np.asarray(dist)
        self.observed = stat(*samples)

        # Store the number of bootstrap samples
        self.n_boot = n_boot

    def var(self):
        """Bootstrap estimate for the variance of the statistic."""
        return self.dist.var(axis=0, ddof=0)

    def se(self):
        """Bootstrap standard error estimate."""
        return self.dist.std(axis=0, ddof=0)

    def ci(self, alpha=0.05, kind="normal"):
        """Two-sided bootstrap confidence interval.

        Parameter
        ---------
        alpha : float in (0, 1), optional
            1 - alpha is the coverage probability of the interval.
        kind : "normal" or "pivotal", optional
            Specifies the type of bootstrap confidence interval to compute.

        Returns
        -------
        lower : float or numpy.ndarray
            Lower endpoint of the confidence interval.
        upper : float or numpy.ndarray
            Upper endpoint of the confidence interval.
        """
        alpha = validate_float(alpha, "alpha", minimum=0.0, maximum=1.0)

        if kind == "normal":
            z = st.norm(0, 1).ppf(alpha / 2)
            se = self.se()
            lower = self.observed + z * se
            upper = self.observed - z * se
        elif kind == "pivotal":
            q_lower = np.percentile(self.dist, (100 * (1 - alpha / 2)), axis=0)
            q_upper = np.percentile(self.dist, (100 * alpha / 2), axis=0)
            lower = 2 * self.observed - q_lower
            upper = 2 * self.observed - q_upper
        else:
            raise ValueError(f"Invalid parameter 'kind': {kind}")

        return lower, upper


# Bayesian bootstrap

def bayesian_bootstrap(*samples, n_boot, random_state=None):
    """Generate Bayesian bootstrap posterior distribution estimates.

    Parameters
    ----------
    samples : sequence of arrays
        Sequence of samples.
    n_boot : int
        Number of bootstrap samples to draw. This is necessarily a keyword
        argument.
    random_state : numpy.random.RandomState, int, or array-like, optional
        If a numpy.random.RandomState object, this is used as the random number
        generator. Otherwise, this is the seed for a numpy.random.RandomState
        object to be used as the random number generator. This is necessarily a
        keyword argument.

    Returns
    -------
    weights : numpy.ndarray of shape (n_boot, n_observations)
        List of posterior distribution estimates.
    """
    # Validate the samples
    samples = validate_samples(*samples, equal_lengths=True, ret_list=True)
    n_obs = len(samples[0])

    # Ensure `n_boot` is a positive integer
    n_boot = validate_int(n_boot, "n_boot", minimum=1)

    # Initialize the random number generator if necessary
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # Generate Bayesian bootstrap posterior distribution weights
    return random_state.dirichlet(np.repeat(1.0, n_obs), size=n_boot)


class BayesianBootstrap(object):
    """Class for generic sampling distribution estimation using the Bayesian
    bootstrap.

    Properties
    ----------
    n_boot : int
        Number of bootstrap samples.
    weights : numpy.ndarray
        List of Bayesian bootstrap posterior distribution estimates.
    dist : numpy.ndarray
        Bayesian bootstrap sampling distribution of the statistic.
    observed : object
        Observed value of the statistic.
    """
    n_boot: int
    weights: np.ndarray
    dist: np.ndarray
    observed: object

    def __init__(self, *samples, stat, n_boot, random_state=None, **kwargs):
        """Generate a Bayesian bootstrap sampling distribution.

        Parameters
        ----------
        samples: sequence of arrays
            Samples on which to perform the bootstrap resampling procedure. Each
            array in this sequence should have the same length (i.e., sample
            size), but there is no restriction on the shape otherwise.
        stat: callable or str
            The statistic to compute from the data. This must be a function with
            signature
                stat(*samples, weights, **kwargs)
            where the `weights` keyword argument is used to specify the
            posterior distribution, and is interpreted as the uniform
            distribution if not specified. This is necessarily a keyword
            argument.
        n_boot : int
            Number of bootstrap samples to draw. This is necessarily a keyword
            argument.
        random_state : numpy.random.RandomState, int, or array-like, optional
            If a numpy.random.RandomState object, this is used as the random
            number generator. Otherwise, this is the seed for a
            numpy.random.RandomState object. This is necessarily a keyword
            argument.
        kwargs: dict, optional
            Additional keyword arguments to pass to the function represented by
            the parameter `stat`.
        """
        # Validate the statistic
        stat = validate_func(stat, **kwargs)

        # Generate posterior distributions
        self.weights = bayesian_bootstrap(*samples, n_boot=n_boot,
                                          random_state=random_state)

        # Compute the bootstrap sampling distribution
        dist = [stat(*samples, weights=weight) for weight in self.weights]

        # Store bootstrap sampling distribution and the observed statistic
        self.dist = np.asarray(dist)
        self.observed = stat(*samples)

        # Store the number of bootstrap samples
        self.n_boot = n_boot

    def var(self):
        """Bayesian bootstrap estimate for the variance of the statistic."""
        return self.dist.var(axis=0, ddof=0)

    def se(self):
        """Bayesian bootstrap standard error estimate."""
        return self.dist.std(axis=0, ddof=0)
