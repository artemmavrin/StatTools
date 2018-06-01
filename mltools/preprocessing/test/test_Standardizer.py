import unittest

import numpy as np

from mltools.generic.fit import UnfittedException
from mltools.preprocessing import Standardizer


class TestStandardizer(unittest.TestCase):
    def test_default_parameters(self):
        # Example input data
        x = [[0, 0, 0, 1],
             [0, 0, 1, 1],
             [0, 1, 2, 1]]
        n_samples, n_features = (3, 4)
        # Column-wise means and un-corrected standard deviations of x
        mean = [0, 1 / 3, 1, 1]
        std = [0, np.sqrt(2 / 9), np.sqrt(2 / 3), 0]
        # Indices of non-constant columns
        idx = [1, 2]
        # Desired transformed data; constant columns 0 and 3 are dropped
        z_true = [[-1 / 3 / np.sqrt(2 / 9), -1 / np.sqrt(2 / 3)],
                  [-1 / 3 / np.sqrt(2 / 9), 0],
                  [2 / 3 / np.sqrt(2 / 9), 1 / np.sqrt(2 / 3)]]

        # Create Normalizer with default parameters
        standardize = Standardizer()
        # Ensure that the Normalizer cannot transform before training
        with self.assertRaises(UnfittedException):
            standardize.transform(x)
        # Train and check attributes for correctness
        standardize.fit(x)
        np.testing.assert_almost_equal(standardize._mean, mean)
        np.testing.assert_almost_equal(standardize._std, std)
        np.testing.assert_equal(standardize._idx, idx)
        self.assertEqual(standardize._n_features, n_features)

        # Check transformation for correctness
        z_tran = standardize.transform(x)
        np.testing.assert_almost_equal(z_tran, z_true)
        np.testing.assert_almost_equal(z_tran.mean(axis=0), [0, 0])
        np.testing.assert_almost_equal(z_tran.var(axis=0), [1, 1])

        # Check inverse transformation for correctness
        x_inv_tran = standardize.inv_transform(z_tran)
        np.testing.assert_almost_equal(x_inv_tran, x)

    def test_reduce_False_bias_False(self):
        # Example input data
        x = [[0, 0, 0, 1],
             [0, 0, 1, 1],
             [0, 1, 2, 1]]
        n_samples, n_features = (3, 4)
        # Column-wise means and corrected standard deviations of x
        mean = [0, 1 / 3, 1, 1]
        std = [0, np.sqrt(1 / 3), 1, 0]
        # Indices of non-constant columns
        idx = [1, 2]
        # Desired transformed data; constant columns are unchanged
        z_true = [[0, -1 / 3 / np.sqrt(1 / 3), -1, 1],
                  [0, -1 / 3 / np.sqrt(1 / 3), 0, 1],
                  [0, 2 / 3 / np.sqrt(1 / 3), 1, 1]]

        # Create Normalizer
        standardize = Standardizer(reduce=False, bias=False)

        # Train and check attributes for correctness
        standardize.fit(x)
        np.testing.assert_almost_equal(standardize._mean, mean)
        np.testing.assert_almost_equal(standardize._std, std)
        np.testing.assert_equal(standardize._idx, idx)
        self.assertEqual(standardize._n_features, n_features)

        # Check transformation for correctness
        z_tran = standardize.transform(x)
        np.testing.assert_almost_equal(z_tran, z_true)
        np.testing.assert_almost_equal(z_tran.mean(axis=0), [0, 0, 0, 1])
        np.testing.assert_almost_equal(z_tran.var(axis=0, ddof=1), [0, 1, 1, 0])

        # Check inverse transformation for correctness
        x_inv_tran = standardize.inv_transform(z_tran)
        np.testing.assert_almost_equal(x_inv_tran, x)


if __name__ == "__main__":
    unittest.main()
