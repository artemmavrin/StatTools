"""Defines the PermutationTest class."""

import itertools

import numpy as np


class PermutationTest(object):
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

        # Convert each sample to a Numpy array
        self.data = list(map(np.asarray, data))

        # Store the test statistic function and compute the true test statistic
        self.statistic = statistic
        self.true_statistic = statistic(*data)

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
        """
        data = np.concatenate(self.data)
        dist = []
        if seed is not None:
            np.random.seed(seed)
        for _ in range(n):
            data_ = np.random.permutation(data)
            test_statistic = self.statistic(*(data_[i] for i in self.slices))
            dist.append(test_statistic)

        self.dist = np.sort(dist)

    def p_value(self, kind="greater"):
        """Compute a p-value for the test.

        Parameters
        ----------
        kind: str, optional
            Specifies the kind of p-value to report (i.e., specifies the
            alternate hypothesis).
            Use "greater" to return P(X>=t|H0).
            Use "less" to return P(X<=t|H0).
            Use "two-sided" to return P(abs(X)>=abs(t)|H0).

        Returns
        -------
        The p-value.
        """
        dist = self.dist
        true = self.true_statistic
        if kind == "greater":
            return np.sum(dist >= true) / len(dist)
        elif kind == "less":
            return np.sum(dist <= true) / len(dist)
        elif kind == "two-sided":
            return np.sum(np.abs(dist) >= np.abs(true)) / len(dist)
        else:
            raise ValueError(f"Unknown p-value kind: {kind}")
