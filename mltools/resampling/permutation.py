"""Implements permutations tests."""

import numbers
from itertools import accumulate, permutations

import numpy as np

from ..utils import validate_data, validate_stat

# Number of permutations to randomly sample unless otherwise specified
_DEFAULT_MONTE_CARLO_SIZE = 1000

# Maximum data sample size n to perform an exact test using all n! permutations
# of the data
_MAX_EXACT_SIZE = 10


class PermutationTest(object):
    """General-purpose permutation test.

    In the sample size is small enough, the exact sampling distribution of the
    test statistic under the null hypothesis (that the labels of the data are
    exchangeable) is calculated by computing the values of the test statistic
    under all possible permutations of the labels on the sample. If the sample
    size is too large, the sampling distribution is approximated by Monte Carlo
    sampling of a specified number of random permutations.
    """

    # Original data samples
    data: list

    # Empirical distribution of test statistics of permuted data
    dist: np.ndarray

    # Number of permutations
    n_perm: int

    # Observed value of the test statistic
    observed: object

    def __init__(self, *data, stat, n_perm=None, seed=None, **kwargs):
        """Initialize a PermutationTest object.

        Parameters
        ----------
        data: sequence of arrays
            Each array in this sequence represents a differently labelled
            sample. The arrays can have different length, but they should
            otherwise have the same shape.
        stat: callable or str
            The statistic to compute from the data. If this parameter is a
            string, then it should be the name of a NumPy array method (e.g.,
            "mean" or "std"). In this case, the method will be called on the
            first array in the sequence of arrays only. Otherwise, this function
            should accept as many arrays (and of the same shape) as are in
            `data`. The statistic is not assumed to be scalar-valued. This
            parameter is necessarily a keyword argument.
        n_perm: int, optional
            Number of permutations to sample. If this parameter is not provided
            and the data are small enough, then all possible permutations will
            be sampled exactly once. Otherwise, permutations will be sampled
            randomly with replacement. This is necessarily a keyword argument.
        seed: int, optional
            Seed for a NumPy RandomState object. Only used if using Monte Carlo
            sampling to approximate the test statistic distribution. This is
            necessarily a keyword argument.
        kwargs: dict, optional
            Additional keyword arguments to pass to the function represented by
            the parameter `stat`.
        """
        # Ensure that there are at least two samples
        if len(data) < 2:
            raise ValueError("Not enough data provided")

        # Validate parameters
        self.data = validate_data(*data, equal_shapes=True, return_list=True)
        stat = validate_stat(stat)

        # Get indices corresponding to each data sample
        temp = [0] + list(accumulate(map(len, self.data)))
        indices = [np.arange(i, j) for i, j in zip(temp, temp[1:])]

        # Combine the data samples into one sample
        data = np.concatenate(self.data)

        # Determine the method of generating the test statistic distribution
        exact_test = False
        if n_perm is None:
            if len(data) <= _MAX_EXACT_SIZE:
                exact_test = True
                n_perm = np.math.factorial(len(data))
            else:
                n_perm = _DEFAULT_MONTE_CARLO_SIZE
        elif not isinstance(n_perm, numbers.Integral) or n_perm <= 0:
            raise TypeError("Parameter 'n_perm' must be a positive integer")
        else:
            n_perm = int(n_perm)

        # We do not pre-allocate an array for the sampling distribution of the
        # statistic on the permuted samples because we do not know the dimension
        # of `stat`'s output beforehand
        dist_perm = []

        # Generate the test statistic sampling distribution
        if exact_test:
            # Compute the distribution of the test statistic exactly
            for i, data_ in enumerate(map(np.asarray, permutations(data))):
                data_perm = (data_.take(idx, axis=0) for idx in indices)
                dist_perm.append(stat(*data_perm))
        else:
            # Approximate the distribution of the test statistic by Monte Carlo
            # sampling
            rs = np.random.RandomState(seed)
            for i in range(n_perm):
                data_ = rs.permutation(data)
                data_perm = (data_.take(idx, axis=0) for idx in indices)
                dist_perm.append(stat(*data_perm))

        # Store bootstrap empirical distribution and the observed statistic
        self.dist = np.asarray(dist_perm)
        self.observed = stat(*self.data)

        # Store the number of permutations
        self.n_perm = n_perm

    def p_value(self, tail="two-sided"):
        """Estimate a p-value for the permutation test.

        Parameters
        ----------
        tail: "two-sided" (default), "left", or "right"
            Specifies the kind of p-value to report (i.e., one-tailed or
            two-tailed).

        Returns
        -------
        p_value: float
            The p-value for the test.
        """
        dist = self.dist
        observed = self.observed
        n = self.n_perm

        # Compute the p-value
        if tail == "two-sided":
            p_value = np.sum(np.abs(dist) >= np.abs(observed)) / n
        elif tail == "left":
            p_value = np.sum(dist <= observed) / n
        elif tail == "right":
            p_value = np.sum(dist >= observed) / n
        else:
            raise ValueError(f"Unsupported value for parameter 'tail': {tail}")

        return p_value
