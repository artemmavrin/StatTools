"""Unit tests for the bootstrap function and Bootstrap class."""
import unittest

import numpy as np

from mltools.resampling import bootstrap, Bootstrap


class TestBootstrap(unittest.TestCase):
    def test_one_sample(self):
        n = 20
        p = 5
        x = np.arange(n * p).reshape((n, p))
        n_boot = 100
        random_state = 0
        boots = bootstrap(x, n_boot=n_boot, random_state=random_state)

        assert (len(boots) == n_boot)
        for x_boot in boots:
            assert (x_boot.shape == x.shape)
            assert (all(row in x.tolist() for row in x_boot.tolist()))

    def test_two_samples(self):
        n = 20
        p = 5
        x = np.arange(n * p).reshape((n, p))
        y = np.arange(n)

        n_boot = 100
        random_state = 0
        x_boots, y_boots = bootstrap(x, y, n_boot=n_boot,
                                     random_state=random_state)

        assert (len(x_boots) == n_boot)
        assert (len(y_boots) == n_boot)
        for x_boot, y_boot in zip(x_boots, y_boots):
            assert (x_boot.shape == x.shape)
            assert (y_boot.shape == y.shape)
            for row, i in zip(x_boot, y_boot):
                np.testing.assert_equal(x[int(i)], row)

    def test_two_samples_unequal_lengths(self):
        x = np.arange(100)
        y = np.arange(30)

        with self.assertRaises(ValueError):
            bootstrap(x, y, n_boot=100, random_state=0)

    def test_constant_statistic(self):
        x = np.repeat(0, repeats=100)
        y = np.repeat(1, repeats=100)
        n_boot = 100

        def stat(x_, y_):
            return [np.mean(x_), np.mean(y_)]

        boot = Bootstrap(x, y, stat=stat, n_boot=n_boot)
        np.testing.assert_equal(boot.dist, np.tile([0, 1], (n_boot, 1)))


if __name__ == "__main__":
    unittest.main()
