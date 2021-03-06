"""Unit tests for the LinearRegression class."""

import itertools
import unittest
import warnings

import numpy as np

from stattools.glm import LinearRegression
from stattools.optimization import GradientDescent, NewtonRaphson


class TestLinearRegression(unittest.TestCase):
    def test_horizontal_line(self):
        """Fit models of the form y=intercept, no error."""
        intercepts = np.arange(-3, 3)
        x = np.arange(5)
        gd = GradientDescent(rate=0.1, iterations=1000)
        nr = NewtonRaphson(iterations=100)
        solvers = (None, "lstsq", gd, nr)
        for intercept, solver in itertools.product(intercepts, solvers):
            y = np.repeat(intercept, repeats=len(x))

            model = LinearRegression(standardize=False, fit_intercept=True)
            model.fit(x, y, solver=solver)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [0])

            model = LinearRegression(standardize=True)
            with warnings.catch_warnings(record=True):
                model.fit(x, y, solver=solver)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [0])

    def test_perfect_line_through_origin(self):
        """Fit models of the form y=slope*x, no error."""
        slopes = np.arange(-3, 3)
        x = np.arange(5)
        gd = GradientDescent(rate=0.1, iterations=1000)
        nr = NewtonRaphson(iterations=100)
        solvers = (None, "lstsq", gd, nr)
        for slope, solver in itertools.product(slopes, solvers):
            if slope == 0:
                continue
            y = slope * x
            for kwargs in ({"standardize": False, "fit_intercept": False},
                           {"standardize": False, "fit_intercept": True},
                           {"standardize": True}):
                model = LinearRegression(**kwargs)
                model.fit(x, y, solver=solver)
                np.testing.assert_almost_equal(model.predict(x), y)
                np.testing.assert_almost_equal(model.intercept, 0)
                np.testing.assert_almost_equal(model.coef, [slope])

    def test_perfect_line(self):
        """Fit models of the form y=intercept + slope*x, no error."""
        intercepts = np.arange(-3, 3)
        slopes = np.arange(-3, 3)
        x = np.arange(5)
        gd = GradientDescent(rate=0.1, momentum=0.5, nesterov=True,
                             anneal=500, iterations=1000)
        nr = NewtonRaphson(iterations=100)
        solvers = (None, "lstsq", gd, nr)
        for intercept, slope, solver in itertools.product(intercepts, slopes,
                                                          solvers):
            if slope == 0:
                continue
            y = intercept + slope * x

            model = LinearRegression(standardize=False, fit_intercept=True)
            model.fit(x, y, solver=solver)
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
        """Fit models of the form y=b0 + b1*x1 + b2*x2, no error."""
        intercept = 3
        slope1 = 2
        slope2 = 5
        n = 10
        rs = np.random.RandomState(0)
        x = rs.uniform(size=(n, 2))
        y = intercept + x.dot([slope1, slope2])
        gd = GradientDescent(rate=0.1, momentum=0.9, iterations=1000)
        nr = NewtonRaphson(iterations=1000)
        solvers = (None, "lstsq", gd, nr)

        for solver in solvers:
            model = LinearRegression(standardize=False)
            model.fit(x, y, solver=solver)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [slope1, slope2])

            model = LinearRegression()
            model.fit(x, y, solver=solver)
            np.testing.assert_almost_equal(model.predict(x), y)
            np.testing.assert_almost_equal(model.intercept, intercept)
            np.testing.assert_almost_equal(model.coef, [slope1, slope2])

    def test_easy_1d(self):
        """Example from http://onlinestatbook.com/2/regression/intro.html"""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 1.3, 3.75, 2.25]

        gd = GradientDescent(rate=0.1, momentum=0.9, iterations=1000)
        nr = NewtonRaphson(iterations=100)
        solvers = (None, "lstsq", gd, nr)
        for solver in solvers:
            model = LinearRegression(standardize=False)
            model.fit(x, y, solver=solver)
            intercept = model.intercept
            slope = float(model.coef)
            self.assertAlmostEqual(intercept, 0.785)
            self.assertAlmostEqual(slope, 0.425)

            model = LinearRegression(standardize=True)
            model.fit(x, y, solver=solver)
            intercept = model.intercept
            slope = float(model.coef)
            self.assertAlmostEqual(intercept, 0.785)
            self.assertAlmostEqual(slope, 0.425)

    def test_easy_2d(self):
        """Example from Exercises 3a, #3 in Seber & Lee (2003)

        References
        ----------
        George A. F. Seber and Alan J. Lee. Linear Regression Analysis,
            Second Edition. Wiley Series in Probability and Statistics.
            Wiley-Interscience, Hoboken, NJ, 2003, pp. xvi+557.
            DOI: https://doi.org/10.1002/9780471722199
        """
        x = [[1, 0], [2, -1], [1, 2]]
        ys = np.linspace(-5, 5, num=5)
        nr = NewtonRaphson(iterations=100)
        solvers = (None, "lstsq", nr)
        for y1, y2, y3, solver in itertools.product(ys, ys, ys, solvers):
            y = [y1, y2, y3]
            model = LinearRegression(standardize=False, fit_intercept=False)
            model.fit(x, y, solver=solver)
            theta, phi = model.coef
            self.assertAlmostEqual(theta, (y1 + 2 * y2 + y3) / 6)
            self.assertAlmostEqual(phi, (2 * y3 - y2) / 5)
            self.assertEqual(model.intercept, 0.0)

    def test_fit_gradient_descent(self):
        """Test LinearRegression.fit() using the 'gd' option."""
        rs = np.random.RandomState(0)
        tries = 10
        n = 500
        p = 20
        intercept = 10
        coef = rs.uniform(low=-10, high=10, size=p)
        sigma = 0.001
        gd_params = [{},
                     dict(momentum=0.9, anneal=500, iterations=2000),
                     dict(momentum=0.5, nesterov=True, iterations=500)]
        for _, params in itertools.product(range(tries), gd_params):
            x = rs.uniform(low=-10, high=10, size=(n, p))
            y = intercept + x.dot(coef) + rs.normal(scale=sigma, size=n)
            model = LinearRegression()
            model.fit(x, y, solver="gd", **params)

            np.testing.assert_almost_equal(model.predict(x), y, decimal=2)
            np.testing.assert_almost_equal(model.intercept, intercept,
                                           decimal=2)
            np.testing.assert_almost_equal(model.coef, coef, decimal=2)


if __name__ == "__main__":
    unittest.main()
