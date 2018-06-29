"""Unit tests for the KMeansCluster class."""

import unittest
from itertools import product

import numpy as np

from stattools.cluster import KMeansCluster


class TestKMeansCluster(unittest.TestCase):
    def test_perfect_separation(self):
        """Check that perfectly separated data are clustered correctly."""
        # Number of clusters
        ks = (2, 3, 5)
        # Number of samples per cluster
        ns = (10, 20, 100)
        # Number of features
        ps = (1, 3, 5)

        rs = np.random.RandomState(0)

        for k, n, p in product(ks, ns, ps):
            xs = []
            for i in range(k):
                low = 3 * i
                high = 3 * i + 1
                xs.append(rs.uniform(low=low, high=high, size=(n, p)))
            x = np.row_stack(xs)

            model = KMeansCluster(k=k, random_state=rs)
            model.fit(x)
            clusters = model.predict(x)

            for i in range(k):
                ind = slice(i * n, (i + 1) * n)
                assert all(
                    c1 == c2 for c1, c2 in zip(clusters[ind], clusters[ind]))


if __name__ == "__main__":
    unittest.main()
