"""Unit tests for the LinearRegression class."""

import itertools
import unittest
import warnings

import numpy as np

from mltools.glm import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_horizontal_line(self):
        intercepts = np.arange(-5, 5)
        x = np.arange(5)
        for intercept in intercepts:
            y = np.repeat(intercept, repeats=len(x))

            model = LinearRegression(standardize=False, fit_intercept=True)
            model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [0])

            model = LinearRegression(standardize=True)
            with warnings.catch_warnings(record=True):
                model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [0])

    def test_perfect_line_thru_origin(self):
        slopes = np.arange(-5, 5)
        x = np.arange(5)
        for slope in slopes:
            if slope == 0:
                continue
            y = slope * x
            for kwargs in ({"standardize": False, "fit_intercept": False},
                           {"standardize": False, "fit_intercept": True},
                           {"standardize": True}):
                model = LinearRegression(**kwargs)
                model.fit(x, y)
                np.testing.assert_almost_equal(model.predict(x), y)
                np.testing.assert_almost_equal(model.intercept, 0)
                np.testing.assert_almost_equal(model.coef, [slope])

    def test_perfect_line(self):
        intercepts = np.arange(-5, 5)
        slopes = np.arange(-5, 5)
        x = np.arange(5)
        for intercept, slope in itertools.product(intercepts, slopes):
            if slope == 0:
                continue
            y = intercept + slope * x

            model = LinearRegression(standardize=False, fit_intercept=True)
            model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [slope])

            model = LinearRegression(standardize=True)
            if slope == 0:
                with warnings.catch_warnings(record=True):
                    model.fit(x, y)
            else:
                model.fit(x, y)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [slope])

    def test_perfect_plane(self):
        intercept = 3
        slope1 = 2
        slope2 = 5
        n = 10
        np.random.seed(0)
        x = np.random.uniform(size=(n, 2))
        y = intercept + x.dot([slope1, slope2])

        model = LinearRegression(standardize=False)
        model.fit(x, y)
        np.testing.assert_almost_equal(model.predict(x), y)
        np.testing.assert_almost_equal(model.intercept, intercept)
        np.testing.assert_almost_equal(model.coef, [slope1, slope2])

        model = LinearRegression()
        model.fit(x, y)
        np.testing.assert_almost_equal(model.predict(x), y)
        np.testing.assert_almost_equal(model.intercept, intercept)
        np.testing.assert_almost_equal(model.coef, [slope1, slope2])

    def test_easy_1d(self):
        """Numbers from http://onlinestatbook.com/2/regression/intro.html"""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 1.3, 3.75, 2.25]

        model = LinearRegression(standardize=False)
        model.fit(x, y)
        intercept = model.intercept
        slope = float(model.coef)
        self.assertAlmostEqual(intercept, 0.785)
        self.assertAlmostEqual(slope, 0.425)

        model = LinearRegression(standardize=True)
        model.fit(x, y)
        intercept = model.intercept
        slope = float(model.coef)
        self.assertAlmostEqual(intercept, 0.785)
        self.assertAlmostEqual(slope, 0.425)

    def test_easy_2d(self):
        """Adapted from Exercises 3a, #3 in
            George A. F. Seber and Alan J. Lee. Linear Regression Analysis,
            Second Edition. Wiley Series in Probability and Statistics.
            Wiley-Interscience, Hoboken, NJ, 2003, pp. xvi+557.
            DOI: https://doi.org/10.1002/9780471722199
        """
        x = [[1, 0], [2, -1], [1, 2]]
        ys = np.linspace(-5, 5, num=10)
        for y1, y2, y3 in itertools.product(ys, ys, ys):
            y = [y1, y2, y3]
            model = LinearRegression(standardize=False, fit_intercept=False)
            model.fit(x, y)
            theta, phi = model.coef
            self.assertAlmostEqual(theta, (y1 + 2 * y2 + y3) / 6)
            self.assertAlmostEqual(phi, (2 * y3 - y2) / 5)
            self.assertEqual(model.intercept, 0.0)


if __name__ == "__main__":
    unittest.main()
