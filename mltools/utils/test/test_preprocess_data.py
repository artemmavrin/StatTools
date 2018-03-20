import unittest

import numpy as np

from mltools.utils import preprocess_data


class TestPreprocessData(unittest.TestCase):
    def test_1(self):
        x = [1, 2, 3]
        x_ = preprocess_data(x)
        np.testing.assert_equal(x, x_)

        y = [[2, 3], [4, 5]]
        x_, y_ = preprocess_data(x, y)
        np.testing.assert_equal(x, x_)
        np.testing.assert_equal(y, y_)

        with self.assertRaises(ValueError):
            preprocess_data(x, y, equal_lengths=True)

        x_, y_ = preprocess_data(x, y, max_ndim=2)
        np.testing.assert_equal(x_.ndim, 2)
        np.testing.assert_equal(y_.ndim, 2)

        x_, y_ = preprocess_data(x, y, max_ndim=(1, 2))
        np.testing.assert_equal(x_.ndim, 1)
        np.testing.assert_equal(y_.ndim, 2)

        x_, y_ = preprocess_data(x, y, max_ndim=(None, 2))
        np.testing.assert_equal(x_.ndim, 1)
        np.testing.assert_equal(y_.ndim, 2)

        x_, y_ = preprocess_data(x, y, max_ndim=(1, None))
        np.testing.assert_equal(x_.ndim, 1)
        np.testing.assert_equal(y_.ndim, 2)


if __name__ == "__main__":
    unittest.main()
