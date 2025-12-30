"""Certificate helpers for solver outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence


def save_certificate(
    N: int,
    sol: Sequence[int],
    path: Path,
    runtime: Optional[float],
    optimality_proved: Optional[bool],
    verify_fn: Callable[[Sequence[int]], bool],
    cnf_path: Optional[Path] = None,
    proof_path: Optional[Path] = None,
) -> None:
    """Write a JSON certificate with solution, verification, runtime, and proof paths."""
    data = {
        "N": N,
        "size": len(sol),
        "solution": list(sol),
        "verified_no_relation": verify_fn(sol),
        "runtime_seconds": runtime,
        "optimality_proved_no_larger": optimality_proved,
        "cnf": str(cnf_path) if cnf_path else None,
        "proof": str(proof_path) if proof_path else None,
    }
    path.write_text(json.dumps(data, indent=2))
