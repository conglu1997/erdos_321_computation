"""SAT/DRAT proof utilities for the collision-cut solver."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def sequential_counter_at_most(
    vars_ids: List[int], k: int, start_var: int
) -> Tuple[List[List[int]], int]:
    """Sinz sequential counter for sum(vars) <= k. Returns (clauses, next_free_var)."""
    if k >= len(vars_ids):
        return [], start_var
    if k == 0:
        # Enforce all vars false.
        clauses = [[-v] for v in vars_ids]
        return clauses, start_var
    if k < 0:
        # Unsat guard: empty clause.
        return [[]], start_var
    clauses: List[List[int]] = []
    s: List[List[int]] = [[0 for _ in range(k)] for _ in range(len(vars_ids))]
    next_var = start_var
    # allocate aux vars
    for i in range(len(vars_ids)):
        for j in range(k):
            s[i][j] = next_var
            next_var += 1
    # i = 0
    clauses.append([-vars_ids[0], s[0][0]])
    for j in range(1, k):
        clauses.append([-s[0][j]])  # cannot exceed
    # i > 0
    n = len(vars_ids)
    for i in range(1, n):
        # linking first column
        clauses.append([-vars_ids[i], s[i][0]])
        clauses.append([-s[i - 1][0], s[i][0]])
        for j in range(1, k):
            # propagation
            clauses.append([-s[i - 1][j], s[i][j]])
            clauses.append([-vars_ids[i], -s[i - 1][j - 1], s[i][j]])
        # overflow clause
        clauses.append([-vars_ids[i], -s[i - 1][k - 1]])
    # enforce upper bound on last row
    clauses.append([-s[n - 1][k - 1]])
    return clauses, next_var


def build_cnf_for_min_size(
    N: int, min_size: int, cuts: List[List[int]]
) -> Tuple[str, int, int]:
    """Build CNF (DIMACS string) expressing: sum x_i >= min_size and all logged cuts.

    Uses auxiliary y_i to encode y_i <-> ¬x_i, then at-most encoding on y for sum y <= N - min_size.
    Each cut C adds clause (∨_{i in C} ¬x_i). Returns (cnf_str, num_vars, num_clauses).
    """
    if min_size > N:
        # Impossible cardinality; encode as empty clause.
        return "p cnf 1 1\n0\n", 1, 1
    if min_size <= 0:
        # Trivially satisfied; encode empty clause set.
        return "p cnf 1 0\n", 1, 0
    clauses: List[List[int]] = []
    num_vars = N
    # Cuts
    for cut in cuts:
        clauses.append([-i for i in cut])
    # Cardinality: introduce y_i = ¬x_i
    y_vars = list(range(N + 1, 2 * N + 1))
    num_vars = 2 * N
    for xi, yi in zip(range(1, N + 1), y_vars):
        clauses.append([xi, yi])  # xi -> not yi
        clauses.append([-xi, -yi])  # yi -> not xi
    k = N - min_size  # at most k of the y's can be true
    card_clauses, next_var = sequential_counter_at_most(
        y_vars, k, start_var=num_vars + 1
    )
    clauses.extend(card_clauses)
    num_vars = max(num_vars, next_var - 1)
    cnf_lines = [f"p cnf {num_vars} {len(clauses)}"]
    for cl in clauses:
        cnf_lines.append(" ".join(str(lit) for lit in cl) + " 0")
    return "\n".join(cnf_lines) + "\n", num_vars, len(clauses)


def run_kissat_proof(cnf_path: Path, proof_path: Path) -> bool:
    """Run kissat with DRAT proof; returns True if UNSAT, False if SAT."""
    cmd = ["kissat", str(cnf_path), str(proof_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 10:
        return False  # SAT
    if result.returncode == 20:
        return True  # UNSAT
    raise RuntimeError(f"kissat failed: {result.stdout}\n{result.stderr}")


def prove_optimality(
    N: int,
    current_size: int,
    cuts: List[List[int]],
    cnf_dir: Path,
    kissat_runner: Callable[[Path, Path], bool] = run_kissat_proof,
    fallback_feasible: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[bool], Optional[Path], Optional[Path]]:
    """Attempt to prove no collision-free set larger than current_size exists.

    Returns (optimality, cnf_path, proof_path) where optimality is:
      True  -> UNSAT proof for size>=current_size+1
      False -> SAT instance found
      None  -> proof unavailable and no fallback provided
    """
    min_size = current_size + 1
    cnf_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = cnf_dir / f"cnf_N{N}_ge_{min_size}.cnf"
    proof_path = cnf_path.with_suffix(".drat")
    cnf_text, _, _ = build_cnf_for_min_size(N, min_size, cuts)
    cnf_path.write_text(cnf_text)
    try:
        unsat = kissat_runner(cnf_path, proof_path)
    except FileNotFoundError:
        unsat = None
    if unsat is None:
        optimality = fallback_feasible() if fallback_feasible else None
    elif unsat is False:
        optimality = False
    else:
        optimality = True
    return optimality, cnf_path, proof_path
