"""Unit tests for the Permutation class."""

import unittest

import numpy as np

from mltools.hypothesis import PermutationTest


class TestPermutationTest(unittest.TestCase):
    def test_i_lost_the_labels(self):
        """Numerical example from Section 1.3, "Testing a Hypothesis", in
        Chapter 1 of
            Phillip Good. Permutation, parametric and bootstrap tests of
            hypotheses. Third. Springer Series in Statistics. Springer-Verlag,
            New York, 2005, pp. xx+315. DOI: https://doi.org/10.1007/b138696.

        This is a test of an exact permutation test, meaning that all possible
        permutations are to be sampled once.
        """
        # Vitamin E treatment group
        x = np.array([121, 118, 110])

        # Control group
        y = np.array([34, 12, 22])

        # Test statistic: sum of counts in the treatment group
        def statistic(treatment, _):
            return np.sum(treatment)

        pt = PermutationTest(x, y, statistic=statistic)
        res = pt.test(tail="right")

        self.assertEqual(res.statistic, np.sum(x))
        self.assertAlmostEqual(res.p_value, 1 / 20)


if __name__ == "__main__":
    unittest.main()
