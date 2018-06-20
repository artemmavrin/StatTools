import unittest

import numpy as np

from stattools.utils.validation import validate_func


class TestValidateFunc(unittest.TestCase):
    def test_callable(self):
        x = [1, 2, 3]
        y = [4, 5, 6]

        def func(x, y, plus):
            return np.mean(x) + np.mean(y) + plus

        func_new = validate_func(func, plus=10)

        self.assertEqual(func_new(x, y), func(x, y, 10))

    def test_name(self):
        x = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        func = validate_func("mean", axis=0)
        np.testing.assert_equal(func(x), np.mean(x, axis=0))

        func = validate_func("var", axis=1)
        np.testing.assert_equal(func(x), np.var(x, axis=1))

        func = validate_func("std", ddof=1)
        np.testing.assert_equal(func(x), np.std(x, ddof=1))


if __name__ == "__main__":
    unittest.main()
