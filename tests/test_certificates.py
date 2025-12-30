import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from certificates import save_certificate


class CertificateTests(unittest.TestCase):
    def test_save_certificate_includes_metadata(self):
        calls = []

        def verifier(sol):
            calls.append(list(sol))
            return True

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cert.json"
            save_certificate(
                5,
                [1, 2],
                path,
                runtime=1.5,
                optimality_proved=False,
                verify_fn=verifier,
                monotone_extension_from=3,
                cnf_path=Path("cnf.cnf"),
                proof_path=Path("proof.drat"),
            )
            data = json.loads(path.read_text())
        self.assertEqual(calls, [[1, 2]])
        self.assertEqual(data["N"], 5)
        self.assertEqual(data["size"], 2)
        self.assertEqual(data["runtime_seconds"], 1.5)
        self.assertFalse(data["optimality_proved_no_larger"])
        self.assertEqual(data["monotone_extension_from"], 3)
        self.assertEqual(data["cnf"], "cnf.cnf")
        self.assertEqual(data["proof"], "proof.drat")
        self.assertTrue(data["verified_no_relation"])


if __name__ == "__main__":
    unittest.main()
