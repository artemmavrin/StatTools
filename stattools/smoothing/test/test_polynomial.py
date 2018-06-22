"""Unit tests for the PolynomialSmoother class."""

import unittest
from itertools import product

import numpy as np

from stattools.smoothing import PolynomialSmoother


class TestPolynomialSmoother(unittest.TestCase):
    def test_perfect_polynomial_interpolation(self):
        """Fit polynomial models with no error (should be perfect fits)."""
        rs = np.random.RandomState(0)
        for deg, _ in product(range(1, 10), range(5)):
            coefficients = rs.normal(scale=10, size=(deg + 1))
            poly = np.poly1d(np.flipud(coefficients))
            x = rs.uniform(low=-4, high=4, size=10)
            y = poly(x)

            model = PolynomialSmoother(deg=deg, standardize=False)
            model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, coefficients[0])
            np.testing.assert_almost_equal(model.coef, coefficients[1:])

            model = PolynomialSmoother(deg=deg, standardize=True)
            model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, coefficients[0])
            np.testing.assert_almost_equal(model.coef, coefficients[1:])


if __name__ == "__main__":
    unittest.main()
