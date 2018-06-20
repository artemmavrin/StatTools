import unittest

import numpy as np

from stattools.utils.validation import validate_samples


class TestValidateSamples(unittest.TestCase):
    def test_one_sample(self):
        x1 = [0, 1, 2, 3, 4]
        x2 = [[0, 1, 2], [3, 4, 5]]
        x3 = 0

        for x in (x1, x2, x3):
            x_val = validate_samples(x)
            np.testing.assert_equal(x_val, x)

            x_val = validate_samples(x, ret_list=True)
            np.testing.assert_equal(x_val, [x])

    def test_two_samples_equal_lengths(self):
        x = [0, 1, 2, 3, 4]
        y = [0, 0, 0, 0, 0]

        x_val, y_val = validate_samples(x, y)
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, y)

        x_val, y_val = validate_samples(x, y, equal_lengths=True)
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, y)

    def test_two_samples_different_lengths(self):
        x = [0, 1, 2, 3, 4]
        y = [0, 0, 0]

        x_val, y_val = validate_samples(x, y)
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, y)

        with self.assertRaises(ValueError):
            validate_samples(x, y, equal_lengths=True)

    def test_n_dim(self):
        x = [0, 1, 2, 3, 4]
        y = [1, 2, 3, 4, 5, 6]
        z = [[0, 1, 2], [3, 4, 5]]

        x_val, y_val, z_val = validate_samples(x, y, z)
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, y)
        np.testing.assert_equal(z_val, z)

        with self.assertRaises(ValueError):
            validate_samples(x, y, z, n_dim=1)

        x_val, y_val, z_val = validate_samples(x, y, z, n_dim=2)
        np.testing.assert_equal(x_val, np.atleast_2d(x).T)
        np.testing.assert_equal(y_val, np.atleast_2d(y).T)
        np.testing.assert_equal(z_val, z)

        x_val, y_val, z_val = validate_samples(x, y, z, n_dim=(1, 1, 2))
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, y)
        np.testing.assert_equal(z_val, z)

        x_val, y_val, z_val = validate_samples(x, y, z, n_dim=(1, 2, None))
        np.testing.assert_equal(x_val, x)
        np.testing.assert_equal(y_val, np.atleast_2d(y).T)
        np.testing.assert_equal(z_val, z)


if __name__ == "__main__":
    unittest.main()
