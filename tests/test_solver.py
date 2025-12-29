import unittest

import solver


EXPECTED_PREFIX_20 = [
    1,
    2,
    3,
    4,
    5,
    5,
    6,
    7,
    8,
    9,
    10,
    10,
    11,
    12,
    12,
    13,
    14,
    14,
    15,
    15,
]


class SolverTests(unittest.TestCase):
    def test_find_relation_detects_collision(self):
        # Known collision: {2,3,6} and {1} have equal reciprocal sums.
        rel = solver.find_relation([1, 2, 3, 4, 5, 6])
        self.assertIsNotNone(rel)
        plus_sum = sum(1 / i for i in rel.plus)
        minus_sum = sum(1 / i for i in rel.minus)
        self.assertAlmostEqual(plus_sum, minus_sum)

    def test_sequence_prefix_matches_known_values(self):
        # Compute R(n) for n=1..20 and compare sizes to known sequence prefix.
        sizes = []
        for n in range(1, 21):
            res = solver.solve_max_distinct(n, threads=4, verbose=False)
            self.assertTrue(solver.verify_relation_free(res.solution))
            sizes.append(res.size)
        self.assertEqual(sizes, EXPECTED_PREFIX_20)


if __name__ == "__main__":
    unittest.main()
