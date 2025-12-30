"""SAT/DRAT proof utilities for the collision-cut solver."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def sequential_counter_at_most(
    vars_ids: List[int], k: int, start_var: int
) -> Tuple[List[List[int]], int]:
    """Sinz sequential counter encoding for sum(vars_ids) <= k.

    This is the standard, sound encoding from:
      Carsten Sinz, "Towards an Optimal CNF Encoding of Boolean Cardinality Constraints", CP 2005.

    Returns (clauses, next_free_var).

    Notes:
    - If k >= n: no clauses needed.
    - If k == 0: enforce all vars false.
    - Uses auxiliary variables s[i][j] for i=0..n-2 and j=0..k-1 (1-based in the paper).
    """
    n = len(vars_ids)
    if k >= n:
        return [], start_var
    if k == 0:
        # Enforce all vars false.
        clauses = [[-v] for v in vars_ids]
        return clauses, start_var
    if k < 0:
        # Unsat guard: empty clause.
        return [[]], start_var
    if n == 0:
        return [], start_var

    clauses: List[List[int]] = []

    # Allocate auxiliary vars s(i,j) for i=0..n-2 and j=0..k-1.
    # These represent "among the first i+1 vars, at least j+1 are true".
    s: List[List[int]] = [[0 for _ in range(k)] for _ in range(n - 1)]
    next_var = start_var
    for i in range(n - 1):
        for j in range(k):
            s[i][j] = next_var
            next_var += 1

    # --- Base clauses for i=0 (first variable) ---
    # (¬x1 ∨ s1,1)
    clauses.append([-vars_ids[0], s[0][0]])
    # For j>1: s1,j must be false (cannot have >=2 trues among first var)
    for j in range(1, k):
        clauses.append([-s[0][j]])

    # --- Induction clauses for i = 1..n-2 (vars 2..n-1) ---
    for i in range(1, n - 1):
        xi = vars_ids[i]

        # (¬xi ∨ s(i,1))
        clauses.append([-xi, s[i][0]])
        # (¬s(i-1,1) ∨ s(i,1))
        clauses.append([-s[i - 1][0], s[i][0]])

        for j in range(1, k):
            # (¬s(i-1,j+1) ∨ s(i,j+1))
            clauses.append([-s[i - 1][j], s[i][j]])
            # (¬xi ∨ ¬s(i-1,j) ∨ s(i,j+1))
            clauses.append([-xi, -s[i - 1][j - 1], s[i][j]])

        # Overflow prevention for xi when already have k trues in prefix:
        # (¬xi ∨ ¬s(i-1,k))
        clauses.append([-xi, -s[i - 1][k - 1]])

    # --- Overflow prevention for the last variable x_n ---
    # If the first n-1 vars already include k trues (i.e., s(n-1,k) is true),
    # then x_n must be false: (¬x_n ∨ ¬s(n-1,k)).
    clauses.append([-vars_ids[n - 1], -s[n - 2][k - 1]])

    return clauses, next_var


def build_cnf_for_min_size(
    N: int, min_size: int, cuts: List[List[int]]
) -> Tuple[str, int, int]:
    """Build CNF (DIMACS string) expressing: sum x_i >= min_size and all logged cuts.

    Uses auxiliary y_i to encode y_i <-> ¬x_i, then an at-most encoding on y for:
        sum(y_i) <= N - min_size
    Each cut C adds clause (∨_{i in C} ¬x_i).

    Returns (cnf_str, num_vars, num_clauses).
    """
    if min_size > N:
        # Impossible cardinality; encode as empty clause.
        return "p cnf 1 1\n0\n", 1, 1
    if min_size <= 0:
        # Trivially satisfied; encode empty clause set.
        return "p cnf 1 0\n", 1, 0

    clauses: List[List[int]] = []

    # Cuts: for cut [i,j,k], add (¬xi ∨ ¬xj ∨ ¬xk)
    for cut in cuts:
        clauses.append([-i for i in cut])

    # Cardinality: introduce y_i = ¬x_i (i.e., y_i <-> ¬x_i)
    # Variables: x_i are 1..N, y_i are N+1..2N.
    y_vars = list(range(N + 1, 2 * N + 1))
    num_vars = 2 * N
    for xi, yi in zip(range(1, N + 1), y_vars):
        # (xi ∨ yi) and (¬xi ∨ ¬yi) enforce yi = ¬xi
        clauses.append([xi, yi])
        clauses.append([-xi, -yi])

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
