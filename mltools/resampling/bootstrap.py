"""The non-parametric bootstrap for sampling distribution estimation."""

import numbers
from functools import partial
from operator import methodcaller

import numpy as np


class Bootstrap(object):
    """Class for sampling distribution estimation using the bootstrap."""

    # Bootstrap sampling distribution of the statistic
    dist: np.ndarray

    # Number of bootstrap samples
    n_boot: int

    # Observed value of the statistic
    observed: object

    def __init__(self, *data, stat, n_boot=1000, seed=None, **kwargs):
        """Generate bootstrap estimates by sampling with replacement from a
        sample and re-computing the statistic each time.

        Parameters
        ----------
        data: sequence of arrays
            Data on which to perform the bootstrap resampling procedure. Each
            array in this sequence should have the same length (i.e., sample
            size), but there is no restriction on the shape otherwise.
        stat: callable or str
            The statistic to compute from the data. If this parameter is a
            string, then it should be the name of a NumPy array method (e.g.,
            "mean" or "std"). In this case, there should be exactly one array in
            the sequence `data`. If this parameter is a function, then it should
            accept as many arrays (and of the same shape) as are in `data`. The
            statistic is not assumed to be scalar-valued. This parameter is
            necessarily a keyword argument.
        n_boot: int, optional
            Number of bootstrap samples to generate. This is necessarily a
            keyword argument.
        seed: int, optional
            Seed for a NumPy RandomState object. This is necessarily a keyword
            argument.
        kwargs: dict, optional
            Additional keyword arguments to pass to the function represented by
            the parameter `stat`.
        """
        # Ensure that there is some data
        if len(data) == 0:
            raise ValueError("No data provided")

        # Coerce each data sample to a NumPy array
        data = list(map(np.atleast_1d, data))

        # Ensure every data array has the same length `n` (i.e., sample size)
        n_sample = len(data[0])
        if n_sample == 0 or any(len(x) != n_sample for x in data):
            raise ValueError(
                "Data arrays must all have the same length (at least 1)")

        # Ensure `stat` is either callable or the name of a NumPy array method
        if callable(stat):
            stat = partial(stat, **kwargs)
        elif isinstance(stat, str):
            if len(data) > 1:
                raise ValueError("Parameter 'stat' cannot be a string if "
                                 "multiple data arrays are given")
            if (hasattr(np.ndarray, stat) and
                    callable(getattr(np.ndarray, stat))):
                stat = methodcaller(stat, **kwargs)
            else:
                raise AttributeError(f"NumPy arrays have no method {stat}")
        else:
            raise TypeError("Parameter 'stat' must be callable")

        # Ensure `n_boot` is a positive integer
        if not isinstance(n_boot, numbers.Integral) or n_boot <= 0:
            raise TypeError("Parameter 'n_boot' must be a positive integer.")
        n_boot = int(n_boot)

        # We do not pre-allocate an array for the bootstrap distribution of the
        # statistic because we do not know the dimension of `stat`'s output
        # beforehand
        dist_boot = []

        # Perform the non-parametric bootstrap by repeatedly drawing bootstrap
        # samples from the original data (with replacement) and computing the
        # statistic for each bootstrap sample
        rs = np.random.RandomState(seed)
        for _ in range(n_boot):
            indices = rs.randint(0, n_sample, n_sample)
            dist_boot.append(stat(*(x.take(indices, axis=0) for x in data)))

        # Store bootstrap empirical distribution and the observed statistic
        self.dist = np.asarray(dist_boot)
        self.observed = stat(*data)

        # Store the number of bootstrap samples
        self.n_boot = n_boot

    def var(self):
        """Bootstrap estimate for the variance of the statistic."""
        return self.dist.var(axis=0, ddof=0)

    def se(self):
        """Bootstrap standard error estimate."""
        return self.dist.std(axis=0, ddof=0)
