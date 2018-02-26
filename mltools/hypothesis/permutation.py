"""Defines the PermutationTest class."""

from itertools import accumulate, permutations
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

    def test(self, n=None, alpha=0.05, tail="two-sided", seed=None):
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
        alpha: float, optional
            Significance level (i.e., probability of a Type I error) for the
            test.
        tail: "left", "right", or "two-sided" (default)
            Specifies the kind of test to perform (i.e., one-tailed or
            two-tailed).
        seed: int, optional
            Seed for NumPy's random number generator. Only used if using Monte
            Carlo sampling to approximate the test statistic distribution.

        Returns
        -------
        res: HypothesisTestResult
            A named tuple with "statistic", "p_value", "lower", and "upper"
            fields. The "statistic" field stores the observed test statistic.
            The "p_value" field stores the test's p-value for the given
            significance level alpha. The "lower" and "upper" fields store the
            test's upper and lower alpha-percentile interval bounds.
        """
        # Get slices corresponding to each data sample
        indices = list(accumulate(map(len, self.data)))
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
        self.dist = np.zeros(n)
        if monte_carlo:
            # Approximate the distribution of the test statistic by Monte Carlo
            if seed is not None:
                np.random.seed(seed)
            for i in range(int(n)):
                data_ = np.random.permutation(data)
                self.dist[i] = self.statistic(*(data_[i] for i in slices))
        else:
            # Compute the distribution of the test statistic exactly
            for i, data_ in enumerate(map(np.asarray, permutations(data))):
                self.dist[i] = self.statistic(*(data_[i] for i in slices))

        # Observed value of the test statistic
        statistic = self.statistic(*self.data)

        # Compute the p-value and the confidence interval bounds
        if tail == "two-sided":
            p_value = np.sum(np.abs(self.dist) >= np.abs(statistic)) / n
            lower = np.percentile(self.dist, q=100 * alpha / 2)
            upper = np.percentile(self.dist, q=100 * (1 - alpha / 2))
        elif tail == "left":
            p_value = np.sum(self.dist <= statistic) / n
            lower = -np.inf
            upper = np.percentile(self.dist, q=100 * alpha)
        elif tail == "right":
            p_value = np.sum(self.dist >= statistic) / n
            lower = np.percentile(self.dist, q=100 * (1 - alpha))
            upper = np.inf
        else:
            raise ValueError(f"Unsupported value for parameter 'tail': {tail}")

        return HypothesisTestResult(statistic=statistic, p_value=p_value,
                                    lower=lower, upper=upper)
