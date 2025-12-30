import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import proofs
import solver
from pruning import PruningResult

ORTOOLS_AVAILABLE = solver.ORTOOLS_AVAILABLE


KNOWN_SEQUENCE = [
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
    15,
    16,
    17,
    17,
    18,
    19,
    20,
    20,
    21,
    21,
    22,
    23,
    23,
    24,
    24,
    25,
    26,
    27,
    28,
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

    def test_dedupe_cuts_sorts_and_uniquifies(self):
        cuts = solver.dedupe_cuts([[3, 1, 2], [2, 3, 1], [4], [4], [6, 5]])
        self.assertEqual(len(cuts), 3)
        self.assertIn([1, 2, 3], cuts)
        self.assertIn([4], cuts)
        self.assertIn([5, 6], cuts)

    def test_sequential_counter_trivial_upper_bound(self):
        clauses, next_var = proofs.sequential_counter_at_most([1, 2, 3], 0, start_var=4)
        self.assertEqual(clauses, [[-1], [-2], [-3]])
        self.assertEqual(next_var, 4)

    def test_build_cnf_includes_cuts_and_cardinality(self):
        cnf_text, num_vars, num_clauses = proofs.build_cnf_for_min_size(
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
        self.assertTrue(set(calls[-1]).issubset({1, 2}))

    @unittest.skipUnless(ORTOOLS_AVAILABLE, "ortools not installed")
    def test_p_adic_pruning_filters_collision_oracle(self):
        calls = []

        def oracle(elements):
            calls.append(tuple(elements))
            return None

        def fake_pruning(_):
            return PruningResult(
                safe_numbers={2, 3},
                all_or_none_groups=[],
                by_rule={"fake": {2, 3}},
            )

        res = solver.solve_max_distinct(
            3,
            threads=1,
            verbose=False,
            collision_oracle=oracle,
            use_p_adic_pruning=True,
            pruning_func=fake_pruning,
        )
        self.assertEqual(res.size, 3)
        # The oracle should see only the potentially colliding subset {1}.
        self.assertIn((1,), calls)

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
    def test_feasible_with_min_size_respects_additional_cuts(self):
        infeasible = solver.feasible_with_min_size(
            2, min_size=2, threads=1, additional_cuts=[[1, 2]]
        )
        self.assertFalse(infeasible)
        feasible = solver.feasible_with_min_size(
            3, min_size=2, threads=1, additional_cuts=[[1, 2]]
        )
        self.assertTrue(feasible)

    @unittest.skipUnless(ORTOOLS_AVAILABLE, "ortools not installed")
    def test_sequence_prefix_matches_known_values(self):
        # Compute R(n) for n=1..25 and compare to the known sequence prefix.
        # We keep the full known sequence for reference but only test the first 25 values.
        sizes = []
        for n in range(1, 26):
            res = solver.solve_max_distinct(n, threads=4, verbose=False)
            self.assertTrue(solver.verify_relation_free(res.solution))
            sizes.append(res.size)
        self.assertEqual(sizes, KNOWN_SEQUENCE[:25])

    def test_monotone_extension_skips_repeated_solves(self):
        class StubSolve:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                N,
                threads,
                verbose,
                collision_oracle=solver.find_relation,
                **_,
            ):
                self.calls.append(N)
                return solver.SolveResult(
                    size=N,
                    solution=list(range(1, N + 1)),
                    cuts=[],
                    collision_support=set(range(1, N + 1)),
                    runtime=0.01,
                )

        stub = StubSolve()
        with TemporaryDirectory() as tmpdir:
            solver.sequential_cert_run(
                target_N=3,
                out_dir=Path(tmpdir),
                threads=1,
                verbose=False,
                prove_optimal=False,
                monotone_window=2,
                monotone_collision_oracle=lambda elems: None,
                solve_func=stub,
            )
            self.assertEqual(stub.calls, [1])
            with open(Path(tmpdir) / "R_3.json") as fh:
                cert = json.load(fh)
            self.assertEqual(cert["size"], 3)
            self.assertIsNone(cert["runtime_seconds"])

    def test_monotone_extension_stops_when_collision_detected(self):
        class StubSolve:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                N,
                threads,
                verbose,
                collision_oracle=solver.find_relation,
                **_,
            ):
                self.calls.append(N)
                return solver.SolveResult(
                    size=N,
                    solution=list(range(1, N + 1)),
                    cuts=[],
                    collision_support=set(range(1, N + 1)),
                    runtime=0.01,
                )

        def collision_oracle(elements):
            # Declare a collision once element 3 is present.
            if 3 in elements:
                return solver.Relation(plus=[3], minus=[3])
            return None

        stub = StubSolve()
        with TemporaryDirectory() as tmpdir:
            solver.sequential_cert_run(
                target_N=3,
                out_dir=Path(tmpdir),
                threads=1,
                verbose=False,
                prove_optimal=False,
                monotone_window=2,
                monotone_collision_oracle=collision_oracle,
                solve_func=stub,
            )
            # Should solve for N=1 and N=3 (since extension by 1 only).
            self.assertEqual(stub.calls, [1, 3])
            with open(Path(tmpdir) / "R_2.json") as fh:
                cert = json.load(fh)
            self.assertEqual(cert["size"], 2)
            with open(Path(tmpdir) / "R_3.json") as fh:
                cert3 = json.load(fh)
            self.assertEqual(cert3["size"], 3)

    def test_adaptive_monotone_grows_on_success(self):
        class StubSolve:
            def __init__(self):
                self.calls = []

            def __call__(
                self, N, threads, verbose, collision_oracle=solver.find_relation, **_
            ):
                self.calls.append(N)
                return solver.SolveResult(
                    size=N,
                    solution=list(range(1, N + 1)),
                    cuts=[],
                    collision_support=set(range(1, N + 1)),
                    runtime=0.01,
                )

        stats = {}
        stub = StubSolve()
        with TemporaryDirectory() as tmpdir:
            solver.sequential_cert_run(
                target_N=7,
                out_dir=Path(tmpdir),
                threads=1,
                verbose=False,
                prove_optimal=False,
                monotone_window=3,
                monotone_collision_oracle=lambda elems: None,
                solve_func=stub,
                monotone_stats=stats,
            )
        self.assertEqual(stats["attempt_windows"], [2, 3])
        self.assertEqual(stats["extend_by"], [2, 3])
        # Successful full extensions should have increased the window to the cap.
        self.assertEqual(len(stub.calls), 2)

    def test_adaptive_monotone_shrinks_on_collision(self):
        class StubSolve:
            def __init__(self):
                self.calls = []

            def __call__(
                self, N, threads, verbose, collision_oracle=solver.find_relation, **_
            ):
                self.calls.append(N)
                return solver.SolveResult(
                    size=N,
                    solution=list(range(1, N + 1)),
                    cuts=[],
                    collision_support=set(range(1, N + 1)),
                    runtime=0.01,
                )

        def always_collide(_):
            return solver.Relation(plus=[1], minus=[1])

        stats = {}
        stub = StubSolve()
        with TemporaryDirectory() as tmpdir:
            solver.sequential_cert_run(
                target_N=4,
                out_dir=Path(tmpdir),
                threads=1,
                verbose=False,
                prove_optimal=False,
                monotone_window=3,
                monotone_collision_oracle=always_collide,
                solve_func=stub,
                monotone_stats=stats,
            )
        self.assertGreaterEqual(len(stats["attempt_windows"]), 2)
        # Window should shrink after the first failure (2 -> 1).
        self.assertEqual(stats["attempt_windows"][0], 2)
        self.assertEqual(stats["attempt_windows"][1], 1)
        self.assertTrue(all(v == 0 for v in stats["extend_by"]))


if __name__ == "__main__":
    unittest.main()
