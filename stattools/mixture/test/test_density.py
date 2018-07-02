"""Unit tests for stattools.mixture.gaussian."""

import unittest
from itertools import product

import numpy as np

from stattools.mixture import GaussianMixtureDensity


class TestGaussianMixtureDensity(unittest.TestCase):
    def test_init_covs(self):
        """Check the different methods of initializing covariance matrices."""
        # Number of components
        ks = (2, 3, 5)
        # Number of features
        ps = (1, 3, 5)

        rs = np.random.RandomState(0)

        for k, p in product(ks, ps):
            means = np.arange(k * p).reshape(k, p)

            # No covs given: identity matrix for each component
            covs = np.asarray([np.eye(p) for _ in range(k)])
            gmd = GaussianMixtureDensity(means=means)
            np.testing.assert_equal(gmd.covs, covs)

            # Scalar multiple given: scalar mutliple times the identity matrix
            scalar = 5
            covs = scalar * np.asarray([np.eye(p) for _ in range(k)])
            gmd = GaussianMixtureDensity(means=means, covs=scalar)
            np.testing.assert_equal(gmd.covs, covs)

            # Vector of shape (k,): different scalar multiples of the identity
            # matrix
            scalars = np.arange(k)
            covs = np.asarray([scalar * np.eye(p) for scalar in scalars])
            gmd = GaussianMixtureDensity(means=means, covs=scalars)
            np.testing.assert_equal(gmd.covs, covs)

            # Vector of shape (p,): diagonal of covariances matrices
            if p == k:
                # If p=k, then vectors of shape (p,)=(k,) get interpreted as in
                # the previous case
                continue
            diag = np.arange(p)
            covs = np.asarray([np.diag(diag) for _ in range(k)])
            gmd = GaussianMixtureDensity(means=means, covs=diag)
            np.testing.assert_equal(gmd.covs, covs)

            # Matrix of shape (p, p): same covariance matrix for each component
            mat = np.arange(p * p).reshape(p, p)
            covs = np.asarray([mat for _ in range(k)])
            gmd = GaussianMixtureDensity(means=means, covs=mat)
            np.testing.assert_equal(gmd.covs, covs)

            # Matrix of shape (k, p, p): specify covariance matrix for each
            # component
            covs = np.arange(k * p * p).reshape(k, p, p)
            gmd = GaussianMixtureDensity(means=means, covs=covs)
            np.testing.assert_equal(gmd.covs, covs)



if __name__ == "__main__":
    unittest.main()
