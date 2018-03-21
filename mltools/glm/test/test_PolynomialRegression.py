"""Unit tests for the PolynomialRegression class."""

import unittest

import numpy as np

from mltools.glm import PolynomialRegression


class TestPolynomialRegression(unittest.TestCase):
    def test_perfect_polynomial_interpolation(self):
        rs = np.random.RandomState(0)
        for deg in np.arange(1, 10):
            for _ in range(5):
                coefficients = rs.normal(scale=10, size=(deg + 1))
                poly = np.poly1d(np.flipud(coefficients))
                x = rs.uniform(low=-4, high=4, size=10)
                y = poly(x)

                model = PolynomialRegression(deg=deg, standardize=False)
                model.fit(x, y)
                np.testing.assert_almost_equal(model.predict(x), y)
                np.testing.assert_almost_equal(model.intercept, coefficients[0])
                np.testing.assert_almost_equal(model.coef, coefficients[1:])

                model = PolynomialRegression(deg=deg, standardize=True)
                model.fit(x, y)
                np.testing.assert_almost_equal(model.predict(x), y)
                np.testing.assert_almost_equal(model.intercept, coefficients[0])
                np.testing.assert_almost_equal(model.coef, coefficients[1:])


if __name__ == "__main__":
    unittest.main()
