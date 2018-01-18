"""Defines the PermutationTest class."""

import numpy as np


class PermutationTest(object):
    """Permutation test for difference in means of two samples."""

    # Empirical re-sampling distribution of differences in means
    dist = None

    def __init__(self, x, y):
        """Initialize a PermutationTest object.

        Parameters
        ----------
        x: array-like
            One-dimensional numerical sample.
        y: array-like
            One-dimensional numerical sample.
        """
        if np.ndim(x) != 1 or np.ndim(y) != 1:
            raise ValueError("Data must be 1D arrays.")
        self.x = x
        self.y = y
        self.true_diff = np.mean(y) - np.mean(x)

    def test(self, n=10000, seed=None):
        """Perform the permutation test.
        The samples in x and y are permuted n times and the differences in means
        of the permuted samples are recorded as an empirical distribution.

        Parameters
        ----------
        n: int, optional
            Number of permutations to sample.
        seed: int, optional
            Seed for NumPy's random number generator.
        """
        data = np.concatenate((self.x, self.y))
        dist = []
        if seed is not None:
            np.random.seed(seed)
        for _ in range(n):
            data_ = np.random.permutation(data)
            x = data_[:self.x.size]
            y = data_[self.x.size:]
            dist.append(np.mean(y) - np.mean(x))

        self.dist = np.sort(dist)

    def p_value(self, kind):
        """Compute a one-sided or two-sided p-value for the test.

        Parameters
        ----------
        kind: str
            Specify the kind of p-value to report. Accepted values:
            "greater": right-sided p-value
            "less": left-sided p-value
            "two-sided": two-sided p-value

        Returns
        -------
        The p-value.
        """
        dist = self.dist
        if kind == "greater":
            return np.sum(dist >= self.true_diff) / len(dist)
        elif kind == "less":
            return np.sum(dist <= self.true_diff) / len(dist)
        elif kind == "two-sided":
            return np.sum(np.abs(dist) >= np.abs(self.true_diff)) / len(dist)
        else:
            raise ValueError(f"Unknown p-value type '{type}'.")
