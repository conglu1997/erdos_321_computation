"""Compute R(N) with collision-cut MIP; emit resumable certificates and optional CNF/DRAT proofs.

Problem: R(N) is the max |S|⊆{1..N} such that all subset sums of {1/i : i∈S} are distinct.
Equivalently, there is no non-trivial signed relation Σ ε_i/i = 0 with ε_i∈{-1,0,1} supported on S.

Method:
- Binary vars x_i indicate inclusion. Objective: maximize Σ x_i.
- When a candidate solution S has a signed relation with plus-set A and minus-set B, we add the cut
  Σ_{i∈A} x_i + Σ_{i∈B} x_i ≤ |A| + |B| − 1, excluding that collision without removing any valid solution.
- Collision finder: exact meet-in-the-middle over coefficients in {-1,0,1}; no hashing shortcuts.

Capabilities:
- Single solve with optional verification and certificate (JSON includes runtime, verification flag,
  and if enabled CNF/DRAT paths for the “no larger set” proof).
- Sequential run up to N, writing `R_{n}.json` certificates into a folder and skipping existing ones
  to allow resume; with `--prove-optimal` it also writes CNF/DRAT for size≥k+1 if `kissat` is
  available (otherwise falls back to a feasibility check).

Example commands:
  python solver.py 36 --threads 12 --verify --prove-optimal --cert certificates/R_36.json
  python solver.py --seq 50 --threads 12 --cert-dir certificates --prove-optimal
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ortools.linear_solver import pywraplp

@dataclass
class Relation:
    plus: List[int]
    minus: List[int]


@dataclass
class SolveResult:
    size: int
    solution: List[int]
    cuts: List[List[int]]  # each cut is a list of vars that cannot all be 1
    runtime: float


def lcm_upto(n: int) -> int:
    lcm = 1
    for i in range(1, n + 1):
        lcm = lcm * i // math.gcd(lcm, i)
    return lcm


def find_relation(elements: Sequence[int]) -> Optional[Relation]:
    """Find a non-trivial signed zero-sum relation among {1/i} for i in elements.

    Exact meet-in-the-middle over coefficients in {-1,0,1}; no modular shortcuts.
    Returns disjoint plus/minus supports if found, else None.
    """
    if not elements:
        return None

    L = lcm_upto(max(elements))
    weights = [L // i for i in elements]
    g = math.gcd(*weights)
    weights = [w // g for w in weights]

    mid = len(elements) // 2
    left_idx = list(range(mid))
    right_idx = list(range(mid, len(elements)))

    table: Dict[int, Tuple[int, ...]] = {}
    for coeffs in itertools.product((-1, 0, 1), repeat=len(left_idx)):
        total = sum(weights[i] * c for i, c in zip(left_idx, coeffs))
        if total not in table:
            table[total] = coeffs

    for r_coeffs in itertools.product((-1, 0, 1), repeat=len(right_idx)):
        total_r = sum(weights[i] * c for i, c in zip(right_idx, r_coeffs))
        target = -total_r
        if target in table:
            l_coeffs = table[target]
            coeffs_full = list(l_coeffs) + list(r_coeffs)
            if all(c == 0 for c in coeffs_full):
                continue
            plus = [elements[i] for i, c in enumerate(coeffs_full) if c == 1]
            minus = [elements[i] for i, c in enumerate(coeffs_full) if c == -1]
            if plus or minus:
                return Relation(plus=plus, minus=minus)
    return None


def solve_max_distinct(N: int, threads: int = 8, verbose: bool = False) -> SolveResult:
    """Iteratively add collision cuts until the optimal valid solution is found."""
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    x: Dict[int, pywraplp.Variable] = {i: solver.BoolVar(f"x_{i}") for i in range(1, N + 1)}
    objective = solver.Objective()
    for var in x.values():
        objective.SetCoefficient(var, 1)
    objective.SetMaximization()

    cuts: List[List[int]] = []  # store collision sets for later CNF/DRAT proof
    t0 = time.perf_counter()
    while True:
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"Solver failed with status {status}")
        sol = [i for i in range(1, N + 1) if x[i].solution_value() > 0.5]
        relation = find_relation(sol)
        if relation is None:
            return SolveResult(size=len(sol), solution=sol, cuts=cuts, runtime=time.perf_counter() - t0)
        if verbose:
            print(f"collision found with plus={relation.plus} minus={relation.minus}")
        cut = solver.Constraint(-solver.infinity(), len(relation.plus) + len(relation.minus) - 1)
        for i in relation.plus:
            cut.SetCoefficient(x[i], 1)
        for i in relation.minus:
            cut.SetCoefficient(x[i], 1)
        cuts.append(relation.plus + relation.minus)


def feasible_with_min_size(
    N: int, min_size: int, threads: int = 8, verbose: bool = False
) -> bool:
    """Return True iff there exists a collision-free set of size >= min_size."""
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    x: Dict[int, pywraplp.Variable] = {i: solver.BoolVar(f"x_{i}") for i in range(1, N + 1)}
    model_sum = solver.Sum(x.values())
    solver.Add(model_sum >= min_size)
    solver.Maximize(model_sum)

    while True:
        status = solver.Solve()
        if status == pywraplp.Solver.INFEASIBLE:
            return False
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"Solver failed with status {status}")
        sol = [i for i in range(1, N + 1) if x[i].solution_value() > 0.5]
        relation = find_relation(sol)
        if relation is None:
            return True
        if verbose:
            print(f"[prove] collision found with plus={relation.plus} minus={relation.minus}")
        cut = solver.Constraint(-solver.infinity(), len(relation.plus) + len(relation.minus) - 1)
        for i in relation.plus:
            cut.SetCoefficient(x[i], 1)
        for i in relation.minus:
            cut.SetCoefficient(x[i], 1)


def verify_relation_free(elements: Sequence[int]) -> bool:
    """Exact check that the given set has no non-trivial signed relation."""
    return find_relation(elements) is None


def sequential_counter_at_most(vars_ids: List[int], k: int, start_var: int) -> Tuple[List[List[int]], int]:
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


def build_cnf_for_min_size(N: int, min_size: int, cuts: List[List[int]]) -> Tuple[str, int, int]:
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
        clauses.append([xi, yi])      # xi -> not yi
        clauses.append([-xi, -yi])    # yi -> not xi
    k = N - min_size  # at most k of the y's can be true
    card_clauses, next_var = sequential_counter_at_most(y_vars, k, start_var=num_vars + 1)
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


def save_certificate(
    N: int,
    sol: Sequence[int],
    path: Path,
    runtime: Optional[float],
    optimality_proved: Optional[bool],
    cnf_path: Optional[Path] = None,
    proof_path: Optional[Path] = None,
) -> None:
    """Write a simple JSON certificate of the solution and verification.

    The certificate records N, the claimed optimal size, one optimal set, a boolean from an explicit
    exact verification (`verify_relation_free`), and the runtime. If available, it also records CNF
    and DRAT proof paths for the size>=k+1 unsat proof. A certificate is a portable artifact another
    party can re-check by recomputing subset-sum collisions on the provided set; the embedded verification
    gives an immediate local check, and the data suffices to independently confirm correctness.
    """
    data = {
        "N": N,
        "size": len(sol),
        "solution": list(sol),
        "verified_no_relation": verify_relation_free(sol),
        "runtime_seconds": runtime,
        "optimality_proved_no_larger": optimality_proved,
        "cnf": str(cnf_path) if cnf_path else None,
        "proof": str(proof_path) if proof_path else None,
    }
    path.write_text(json.dumps(data, indent=2))


def sequential_cert_run(
    target_N: int, out_dir: Path, threads: int, verbose: bool, prove_optimal: bool
) -> None:
    """Compute R(n) for n=1..target_N, writing certificates and skipping existing ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split("_")[1]) for p in out_dir.glob("R_*.json") if p.stem.split("_")[1].isdigit()}
    start = max(existing) + 1 if existing else 1
    for n in range(start, target_N + 1):
        res = solve_max_distinct(n, threads=threads, verbose=verbose)
        optimality = None
        cnf_path = None
        proof_path = None
        if prove_optimal:
            cnf_text, _, _ = build_cnf_for_min_size(n, res.size + 1, res.cuts)
            cnf_path = out_dir / f"cnf_N{n}_ge_{res.size + 1}.cnf"
            proof_path = cnf_path.with_suffix(".drat")
            cnf_path.write_text(cnf_text)
            try:
                unsat = run_kissat_proof(cnf_path, proof_path)
            except FileNotFoundError:
                unsat = None
            if unsat is None:
                optimality = not feasible_with_min_size(n, res.size + 1, threads=threads, verbose=verbose)
            elif unsat is False:
                optimality = False
            else:
                optimality = True
        cert_path = out_dir / f"R_{n}.json"
        save_certificate(
            n,
            res.solution,
            cert_path,
            runtime=res.runtime,
            optimality_proved=optimality,
            cnf_path=cnf_path,
            proof_path=proof_path,
        )
        print(
            f"N={n} -> R({n})={res.size} | time {res.runtime:.3f}s | optimality_proved={optimality} | saved {cert_path}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute R(N) with iterative collision cuts.")
    parser.add_argument("N", type=int, nargs="?", help="Upper bound of the ground set {1..N}")
    parser.add_argument("--threads", type=int, default=8, help="Number of solver threads")
    parser.add_argument("--verbose", action="store_true", help="Print generated cuts.")
    parser.add_argument("--cert", type=Path, help="Write JSON certificate to this path.")
    parser.add_argument("--verify", action="store_true", help="Explicitly verify the final set.")
    parser.add_argument(
        "--seq",
        type=int,
        help="Run sequentially up to this N, writing certificates R_{n}.json in --cert-dir.",
    )
    parser.add_argument(
        "--cert-dir",
        type=Path,
        default=Path("certificates"),
        help="Directory for certificates in sequential mode.",
    )
    parser.add_argument(
        "--prove-optimal",
        action="store_true",
        help="After finding size k, attempt to prove no larger set (also emits CNF/DRAT if kissat is available).",
    )
    args = parser.parse_args()

    if args.seq:
        sequential_cert_run(args.seq, args.cert_dir, args.threads, args.verbose, args.prove_optimal)
        return

    if args.N is None:
        parser.error("Provide N or --seq.")

    res = solve_max_distinct(args.N, threads=args.threads, verbose=args.verbose)
    optimality = None
    cnf_path = None
    proof_path = None
    if args.prove_optimal:
        # Build CNF for size >= res.size+1 and run kissat proof if available.
        cnf_text, _, _ = build_cnf_for_min_size(args.N, res.size + 1, res.cuts)
        cnf_path = Path(f"cnf_N{args.N}_ge_{res.size + 1}.cnf")
        cnf_path.write_text(cnf_text)
        proof_path = cnf_path.with_suffix(".drat")
        try:
            unsat = run_kissat_proof(cnf_path, proof_path)
        except FileNotFoundError:
            unsat = None
        if unsat is None:
            # Kissat missing; fall back to feasibility check (non-proof).
            optimality = not feasible_with_min_size(args.N, res.size + 1, threads=args.threads, verbose=args.verbose)
        elif unsat is False:
            # CNF with logged cuts is satisfiable; no proof of optimality.
            optimality = False
        else:
            optimality = True
    print(f"R({args.N}) = {res.size}")
    print("one optimal set:", res.solution)
    print(f"time {res.runtime:.3f}s")
    if args.prove_optimal:
        print(f"no larger set exists (proof/solver): {optimality}")
    if args.verify:
        ok = verify_relation_free(res.solution)
        print(f"verification: {'passed' if ok else 'FAILED'}")
    if args.cert:
        save_certificate(
            args.N,
            res.solution,
            args.cert,
            runtime=res.runtime,
            optimality_proved=optimality,
            cnf_path=cnf_path,
            proof_path=proof_path,
        )
        print(f"wrote certificate to {args.cert}")


if __name__ == "__main__":
    main()
