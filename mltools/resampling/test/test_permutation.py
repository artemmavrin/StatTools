"""Unit tests for the Permutation class."""

import unittest

import numpy as np

from mltools.resampling import PermutationTest


class TestPermutationTest(unittest.TestCase):
    def test_vitamin_e_treatment(self):
        """Numerical example from Section 1.3, "Testing a Hypothesis", in
        Chapter 1 of
            Phillip Good. Permutation, parametric and bootstrap tests of
            hypotheses, Third Edition. Springer Series in Statistics.
            Springer-Verlag, New York, 2005, pp. xx+315.
            DOI: https://doi.org/10.1007/b138696.

        This is a test of an exact permutation test, meaning that all possible
        permutations are to be sampled once.
        """
        # Vitamin E treatment group counts
        treatment = [121, 118, 110]

        # Control group counts
        control = [34, 12, 22]

        # Test statistic: sum of counts in the treatment group
        pt = PermutationTest(treatment, control, stat="sum")

        self.assertEqual(pt.observed, np.sum(treatment))
        self.assertAlmostEqual(pt.p_value(tail="right"), 1 / 20)


if __name__ == "__main__":
    unittest.main()
