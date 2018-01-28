"""Defines the PermutationTest class."""

import itertools
import math
import numbers

import numpy as np

from .base import HypothesisTest, HypothesisTestResult

# Number of permutations to randomly sample unless otherwise specified
_DEFAULT_MONTE_CARLO_SIZE = 10000

# Maximum data size n to perform a true permutation test using all n!
# permutations of the data
_MAX_EXACT_SIZE = 10


class PermutationTest(HypothesisTest):
    """General-purpose permutation test."""

    # Empirical distribution of test statistics of permuted data
    dist = None

    def __init__(self, *data, statistic):
        """Initialize a PermutationTest object.

        Parameters
        ----------
        data: sequence
            Sequence of numerical data samples.
        statistic: callable
            The test statistic, a function of the data. This is necessarily a
            keyword argument.
        """
        # Validate input
        if not callable(statistic):
            raise TypeError("Parameter 'statistic' must be callable")
        if len(data) == 0:
            raise ValueError("No data provided.")
        elif any(np.ndim(x) != 1 for x in data):
            raise ValueError("Data must be 1D arrays.")

        self.data = list(map(np.asarray, data))
        self.statistic = statistic

    def test(self, n=None, seed=None, tail="two-sided"):
        """Perform the permutation test.

        The samples in the data are permuted n times and the test statistics of
        the permuted samples are recorded as an empirical distribution.

        Parameters
        ----------
        n: int, optional
            Number of permutations to sample.
            If this parameter is not provided and the data are small enough,
            then all possible permutations will be sampled exactly once.
            Otherwise, permutations will be sampled randomly with replacement
        seed: int, optional
            Seed for NumPy's random number generator. Only used if using Monte
            Carlo sampling to approximate the test statistic distribution.
        tail: "left", "right", or "two-sided" (default)
            Specifies the kind of test to perform (i.e., one-tailed or
            two-tailed).

        Returns
        -------
        res: HypothesisTestResult
            A named tuple with a "statistic" and "p_value" field. The
            "statistic" field stores the observed test statistic and the
            "p_value" field stores the test's two-sided p-value.
        """
        # Get slices corresponding to each data sample
        indices = list(itertools.accumulate(map(len, self.data)))
        slices = [slice(i, j) for i, j in zip([0] + indices, indices)]

        # Combine the data samples into one sample
        data = np.concatenate(self.data)

        # Determine the method of generating the test statistic distribution
        monte_carlo = True
        if n is None:
            if len(data) <= _MAX_EXACT_SIZE:
                monte_carlo = False
                n = math.factorial(len(data))
            else:
                n = _DEFAULT_MONTE_CARLO_SIZE
        elif not isinstance(n, numbers.Integral) or n <= 0:
            raise TypeError("Parameter 'n' must be a positive integer")

        # Generate the test statistic distribution
        dist = []
        if monte_carlo:
            # Approximate the distribution of the test statistic by Monte Carlo
            if seed is not None:
                np.random.seed(seed)
            for _ in range(int(n)):
                data_ = np.random.permutation(data)
                ts = self.statistic(*(data_[i] for i in slices))
                dist.append(ts)
        else:
            # Compute the distribution of the test statistic exactly
            for data_ in map(np.asarray, itertools.permutations(data)):
                ts = self.statistic(*(data_[i] for i in slices))
                dist.append(ts)
        dist = np.asarray(dist)

        # Compute the observed value of the test statistic and the p-value
        statistic = self.statistic(*self.data)
        if tail == "two-sided":
            p_value = np.sum(np.abs(dist) >= np.abs(statistic)) / n
        elif tail == "left":
            p_value = np.sum(dist <= statistic) / n
        elif tail == "right":
            p_value = np.sum(dist >= statistic) / n
        else:
            raise ValueError(f"Unsupported value for parameter 'tail': {tail}")

        # Store the test statistic distribution and return the test result
        self.dist = np.sort(dist)
        return HypothesisTestResult(statistic=statistic, p_value=p_value)
