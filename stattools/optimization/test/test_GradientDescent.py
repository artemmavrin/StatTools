import unittest

import numpy as np

from stattools.optimization import GradientDescent


class TestGradientDescent(unittest.TestCase):
    def test_quadratic_function_of_one_variable(self):
        def func(x):
            return x * x

        def grad(x):
            return 2.0 * x

        gd = GradientDescent(rate=0.01, iterations=1500)
        x = gd.optimize(x0=100, func=func, grad=grad)
        self.assertAlmostEqual(x, 0.0)

    def test_quadratic_function_of_several_variables(self):
        def func(x):
            return np.dot(x, x)

        func.grad = lambda x: 2.0 * np.asarray(x)

        gd = GradientDescent(rate=0.05, iterations=1000)
        n_tests = 100
        for _ in range(n_tests):
            size = np.random.randint(low=1, high=100)
            loc = np.random.uniform(low=-1000, high=1000)
            scale = np.random.uniform(low=1, high=10000)
            x0 = np.random.normal(loc=loc, scale=scale, size=size)
            x = gd.optimize(x0=x0, func=func)
            np.testing.assert_almost_equal(x, np.zeros(size))

    def test_quadratic_function_of_several_variables_momentum(self):
        def func(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * np.asarray(x)

        gd = GradientDescent(rate=0.05, iterations=500, momentum=0.5)
        n_tests = 100
        for _ in range(n_tests):
            size = np.random.randint(low=1, high=100)
            loc = np.random.uniform(low=-1000, high=1000)
            scale = np.random.uniform(low=1, high=10000)
            x0 = np.random.normal(loc=loc, scale=scale, size=size)
            x = gd.optimize(x0=x0, func=func, grad=grad)
            np.testing.assert_almost_equal(x, np.zeros(size))

    def test_quadratic_function_of_several_variables_annealing(self):
        def func(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * np.asarray(x)

        gd = GradientDescent(rate=0.05, iterations=500, momentum=0.5,
                             anneal=250)
        n_tests = 100
        for _ in range(n_tests):
            size = np.random.randint(low=1, high=100)
            loc = np.random.uniform(low=-1000, high=1000)
            scale = np.random.uniform(low=1, high=10000)
            x0 = np.random.normal(loc=loc, scale=scale, size=size)
            x = gd.optimize(x0=x0, func=func, grad=grad)
            np.testing.assert_almost_equal(x, np.zeros(size))

    def test_quadratic_function_of_several_variables_nesterov(self):
        def func(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * np.asarray(x)

        gd = GradientDescent(rate=0.05, iterations=500, momentum=0.5,
                             nesterov=True, anneal=250)
        n_tests = 100
        for _ in range(n_tests):
            size = np.random.randint(low=1, high=100)
            loc = np.random.uniform(low=-1000, high=1000)
            scale = np.random.uniform(low=1, high=10000)
            x0 = np.random.normal(loc=loc, scale=scale, size=size)
            x = gd.optimize(x0=x0, func=func, grad=grad)
            np.testing.assert_almost_equal(x, np.zeros(size))


if __name__ == "__main__":
    unittest.main()
