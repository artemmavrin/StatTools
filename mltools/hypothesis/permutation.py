"""Defines the PermutationTest class."""

import itertools

import numpy as np

from .base import HypothesisTest, HypothesisTestResult


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

        # Get slices corresponding to each data sample
        indices = list(itertools.accumulate(map(len, data)))
        self.slices = [slice(i, j) for i, j in zip([0] + indices, indices)]

        self.data = list(map(np.asarray, data))
        self.statistic = statistic

    def test(self, n=10000, seed=None):
        """Perform the permutation test.

        The samples in the data are permuted n times and the test statistics of
        the permuted samples are recorded as an empirical distribution.

        Parameters
        ----------
        n: int, optional
            Number of permutations to sample.
        seed: int, optional
            Seed for NumPy's random number generator.

        Returns
        -------
        res: HypothesisTestResult
            A named tuple with a "statistic" and "p_value" field. The
            "statistic" field stores the observed test statistic and the
            "p_value" field stores the test's two-sided p-value.
        """
        # Generate the empirical distribution
        data = np.concatenate(self.data)
        dist = []
        if seed is not None:
            np.random.seed(seed)
        for _ in range(n):
            data_ = np.random.permutation(data)
            test_statistic = self.statistic(*(data_[i] for i in self.slices))
            dist.append(test_statistic)
        dist = np.asarray(dist)

        # Compute the observed value of the test statistic and the p-value
        statistic = self.statistic(*self.data)
        p_value = np.sum(np.abs(dist) >= np.abs(statistic)) / len(dist)

        # Save the empirical distribution and return the test result
        self.dist = np.sort(dist)
        return HypothesisTestResult(statistic=statistic, p_value=p_value)
