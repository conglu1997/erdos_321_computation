import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import proofs


class ProofsTests(unittest.TestCase):
    def test_prove_optimality_uses_fallback_when_kissat_missing(self):
        fallback_calls = []

        def fake_runner(_, __):
            raise FileNotFoundError

        def fallback():
            fallback_calls.append(1)
            return True

        with TemporaryDirectory() as tmpdir:
            optimality, cnf_path, proof_path = proofs.prove_optimality(
                N=3,
                current_size=1,
                cuts=[[1, 2]],
                cnf_dir=Path(tmpdir),
                kissat_runner=fake_runner,
                fallback_feasible=fallback,
            )
            self.assertTrue(cnf_path.exists())
            self.assertEqual(cnf_path.parent, Path(tmpdir))
            self.assertEqual(proof_path.stem, cnf_path.stem)
        self.assertEqual(fallback_calls, [1])
        self.assertTrue(optimality)


if __name__ == "__main__":
    unittest.main()
