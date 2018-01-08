import unittest

import numpy as np

from mltools.optimization import NewtonRaphson


class TestNewtonRaphson(unittest.TestCase):
    def test_quadratic_function_of_one_variable(self):
        def func(x):
            return x * x

        def grad(x):
            return 2.0 * x

        def hess(x):
            return 2.0

        nr = NewtonRaphson(iterations=10)
        x = nr.minimize(x0=100, func=func, grad=grad, hess=hess)
        self.assertAlmostEqual(x, 0.0)

    def test_quadratic_function_of_several_variables(self):
        def func(x):
            return np.dot(x, x)

        func.grad = lambda x: 2. * np.asarray(x)
        func.hess = lambda x: 2. * np.identity(np.size(x))

        nr = NewtonRaphson(iterations=10)
        n_tests = 100
        for _ in range(n_tests):
            size = np.random.randint(low=1, high=100)
            loc = np.random.uniform(-1000, 1000)
            scale = np.random.uniform(10, 10000)
            x0 = np.random.normal(loc=loc, scale=scale, size=size)
            x = nr.minimize(x0=x0, func=func)
            np.testing.assert_almost_equal(x, np.zeros(size))


if __name__ == "__main__":
    unittest.main()
