import unittest

import solver

ORTOOLS_AVAILABLE = solver.ORTOOLS_AVAILABLE


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
    @unittest.skipIf(
        ORTOOLS_AVAILABLE, "ortools installed; dependency failure test not applicable"
    )
    def test_solving_requires_ortools_dependency(self):
        with self.assertRaisesRegex(RuntimeError, "ortools"):
            solver.solve_max_distinct(3, threads=1)

    def test_find_relation_detects_collision(self):
        # Known collision: {2,3,6} and {1} have equal reciprocal sums.
        rel = solver.find_relation([1, 2, 3, 4, 5, 6])
        self.assertIsNotNone(rel)
        plus_sum = sum(1 / i for i in rel.plus)
        minus_sum = sum(1 / i for i in rel.minus)
        self.assertAlmostEqual(plus_sum, minus_sum)

    def test_sequential_counter_trivial_upper_bound(self):
        clauses, next_var = solver.sequential_counter_at_most([1, 2, 3], 0, start_var=4)
        self.assertEqual(clauses, [[-1], [-2], [-3]])
        self.assertEqual(next_var, 4)

    def test_build_cnf_includes_cuts_and_cardinality(self):
        cnf_text, num_vars, num_clauses = solver.build_cnf_for_min_size(
            3, 2, cuts=[[1, 2]]
        )
        self.assertEqual(num_vars, 9)
        self.assertEqual(num_clauses, 15)
        self.assertIn("p cnf 9 15", cnf_text.splitlines()[0])
        self.assertIn("-1 -2 0", cnf_text)

    @unittest.skipUnless(ORTOOLS_AVAILABLE, "ortools not installed")
    def test_collision_oracle_is_called(self):
        calls = []

        def oracle(elements):
            calls.append(tuple(elements))
            return None

        res = solver.solve_max_distinct(
            2, threads=1, verbose=False, collision_oracle=oracle
        )
        self.assertEqual(res.size, 2)
        self.assertTrue(calls)
        self.assertEqual(sorted(calls[-1]), [1, 2])

    @unittest.skipUnless(ORTOOLS_AVAILABLE, "ortools not installed")
    def test_static_cut_blocks_pair(self):
        res = solver.solve_max_distinct(
            2,
            threads=1,
            verbose=False,
            collision_oracle=lambda _: None,
            static_cuts=[[1, 2]],
        )
        self.assertEqual(res.size, 1)
        self.assertIn(res.solution[0], {1, 2})
        # Ensure the logged cut and support capture the static cut as well.
        self.assertIn([1, 2], res.cuts)
        self.assertEqual(res.collision_support, {1, 2})

    @unittest.skipUnless(ORTOOLS_AVAILABLE, "ortools not installed")
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
