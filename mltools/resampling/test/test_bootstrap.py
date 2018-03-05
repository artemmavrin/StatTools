"""Unit tests for the bootstrap function."""

import unittest

import numpy as np

from mltools.resampling import Bootstrap


class TestBootstrap(unittest.TestCase):
    def test_constant(self):
        x = np.repeat(0, repeats=100)
        y = np.repeat(1, repeats=100)
        n_boot = 100

        def stat(x_, y_):
            return [np.mean(x_), np.mean(y_)]

        boot = Bootstrap(x, y, stat=stat, n_boot=n_boot)
        np.testing.assert_equal(boot.dist, np.tile([0, 1], (n_boot, 1)))


if __name__ == "__main__":
    unittest.main()
