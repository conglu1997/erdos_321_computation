import unittest

from pruning import PruningResult, compute_p_adic_exclusions
import solver


class PruningTests(unittest.TestCase):
    def test_unique_high_power_and_modular_rules_for_36(self):
        res: PruningResult = compute_p_adic_exclusions(36)
        expected_safe = {
            16,
            25,
            27,
            32,
            13,
            26,
            17,
            34,
            19,
            23,
            29,
            31,
        }
        self.assertEqual(res.safe_numbers, expected_safe)
        # Multiples of 11 behave like an all-or-none cluster at N=36.
        grouped = [sorted(g) for g in res.all_or_none_groups]
        self.assertIn([11, 22, 33], grouped)

    def test_pruning_summary_string(self):
        res = compute_p_adic_exclusions(20)
        summary = solver.format_pruning_summary(res, 20)
        self.assertIn("safe_count", summary)
        self.assertIn("group_count", summary)
        self.assertIn("by_rule", summary)


if __name__ == "__main__":
    unittest.main()
