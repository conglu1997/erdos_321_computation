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
- Optional monotone extension for sequential runs (`--monotone-window`): attempts to append new
  elements using exact collision checks before re-solving, adaptively tuning the window and emitting
  telemetry so repeated solves can be skipped when safe.

Example commands:
  # Single-N quick test run
  python solver.py 24 --threads 12 --verify --prove-optimal --cert certificates/R_24.json
  # Sequential max-speed run with proofs and monotone shortcut
  python solver.py --seq 50 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    from ortools.linear_solver import pywraplp
except ImportError as exc:
    pywraplp = None  # type: ignore[assignment]
    _ORTOOLS_IMPORT_ERROR: Optional[Exception] = exc
else:
    _ORTOOLS_IMPORT_ERROR = None

ORTOOLS_AVAILABLE = pywraplp is not None


@dataclass
class Relation:
    plus: List[int]
    minus: List[int]


@dataclass
class SolveResult:
    size: int
    solution: List[int]
    cuts: List[List[int]]  # each cut is a list of vars that cannot all be 1
    collision_support: Set[int]  # union of all elements appearing in cuts
    runtime: float


def _require_ortools() -> None:
    """Ensure ortools is available before solving."""
    if pywraplp is None:
        raise RuntimeError(
            "ortools is required for solving; install it with `pip install ortools`."
        ) from _ORTOOLS_IMPORT_ERROR


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


def solve_max_distinct(
    N: int,
    threads: int = 8,
    verbose: bool = False,
    collision_oracle: Callable[[Sequence[int]], Optional[Relation]] = find_relation,
    static_cuts: Optional[List[List[int]]] = None,
    forbidden: Optional[Sequence[int]] = None,
    groups: Optional[List[List[int]]] = None,
) -> SolveResult:
    """Iteratively add collision cuts until the optimal valid solution is found.

    collision_oracle: callable returning a Relation or None on a candidate set.
    static_cuts: cuts to apply before solving (each a list of vars that cannot all be 1).
    forbidden: variables that must stay 0.
    groups: list of disjoint groups; within each group, all vars must share the same value.
    """
    _require_ortools()
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    forbidden_set = set(forbidden or [])
    x: Dict[int, pywraplp.Variable] = {}
    for i in range(1, N + 1):
        var = solver.BoolVar(f"x_{i}")
        x[i] = var
        if i in forbidden_set:
            solver.Add(var == 0)

    group_list = groups or []
    seen_in_groups: Set[int] = set()
    for idx, group in enumerate(group_list):
        gvar = solver.BoolVar(f"g_{idx}")
        for item in group:
            if item in seen_in_groups:
                raise ValueError(f"overlapping group element {item}")
            seen_in_groups.add(item)
            if item not in x:
                raise ValueError(f"group element {item} outside 1..N")
            solver.Add(x[item] == gvar)

    objective = solver.Objective()
    for var in x.values():
        objective.SetCoefficient(var, 1)
    objective.SetMaximization()

    cuts: List[List[int]] = []  # store collision sets for later CNF/DRAT proof
    collision_support: Set[int] = set()
    for cut_vars in static_cuts or []:
        cut = solver.Constraint(-solver.infinity(), len(cut_vars) - 1)
        for i in cut_vars:
            cut.SetCoefficient(x[i], 1)
        cuts.append(list(cut_vars))
        collision_support.update(cut_vars)

    t0 = time.perf_counter()
    while True:
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"Solver failed with status {status}")
        sol = [i for i in range(1, N + 1) if x[i].solution_value() > 0.5]
        relation = collision_oracle(sol)
        if relation is None:
            return SolveResult(
                size=len(sol),
                solution=sol,
                cuts=cuts,
                collision_support=collision_support,
                runtime=time.perf_counter() - t0,
            )
        if verbose:
            print(f"collision found with plus={relation.plus} minus={relation.minus}")
        cut = solver.Constraint(
            -solver.infinity(), len(relation.plus) + len(relation.minus) - 1
        )
        for i in relation.plus:
            cut.SetCoefficient(x[i], 1)
        for i in relation.minus:
            cut.SetCoefficient(x[i], 1)
        cuts.append(relation.plus + relation.minus)
        collision_support.update(relation.plus)
        collision_support.update(relation.minus)


def feasible_with_min_size(
    N: int, min_size: int, threads: int = 8, verbose: bool = False
) -> bool:
    """Return True iff there exists a collision-free set of size >= min_size."""
    _require_ortools()
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    x: Dict[int, pywraplp.Variable] = {
        i: solver.BoolVar(f"x_{i}") for i in range(1, N + 1)
    }
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
            print(
                f"[prove] collision found with plus={relation.plus} minus={relation.minus}"
            )
        cut = solver.Constraint(
            -solver.infinity(), len(relation.plus) + len(relation.minus) - 1
        )
        for i in relation.plus:
            cut.SetCoefficient(x[i], 1)
        for i in relation.minus:
            cut.SetCoefficient(x[i], 1)


def verify_relation_free(elements: Sequence[int]) -> bool:
    """Exact check that the given set has no non-trivial signed relation."""
    return find_relation(elements) is None


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
    exact verification (`verify_relation_free`), and the runtime (None when produced via monotone
    extension without re-solving). If available, it also records CNF
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
    target_N: int,
    out_dir: Path,
    threads: int,
    verbose: bool,
    prove_optimal: bool,
    monotone_window: int = 0,
    monotone_collision_oracle: Callable[
        [Sequence[int]], Optional[Relation]
    ] = find_relation,
    solve_func: Callable[..., SolveResult] = solve_max_distinct,
    monotone_stats: Optional[Dict[str, List[int]]] = None,
) -> None:
    """Compute R(n) for n=1..target_N, writing certificates and skipping existing ones.

    monotone_window: upper bound on how many elements to attempt appending via monotone extension.
      An adaptive policy grows/shrinks the attempted window based on success/failure and cost
      estimates; when collision-free, certificates are emitted without re-solving.
    monotone_collision_oracle: collision finder used during monotone extension (Relation or None).
    solve_func: injectable solver (for testing or alternative backends).
    monotone_stats: optional dict to receive telemetry (attempt_windows, extend_by, collision_found).
    """

    def summarize_monotone_stats(stats: Dict[str, List[int]], window_cap: int) -> str:
        if not stats:
            return ""
        attempts = len(stats.get("attempt_windows", []))
        total_extended = sum(stats.get("extend_by", []))
        collision_hits = sum(stats.get("collision_found", []))
        successes = sum(1 for e in stats.get("extend_by", []) if e > 0)
        success_rate = successes / attempts if attempts else 0.0
        parts = [
            f"[monotone] attempts={attempts}",
            f"extended_steps={total_extended}",
            f"collisions={collision_hits}",
            f"success_rate={success_rate:.2f}",
            f"window_cap={window_cap}",
        ]
        recommendation = ""
        if attempts == 0:
            recommendation = "monotone disabled or never attempted."
        elif (
            success_rate > 0.8
            and collision_hits == 0
            and any(w >= window_cap for w in stats.get("attempt_windows", []))
        ):
            recommendation = (
                f"High success; consider raising --monotone-window above {window_cap}."
            )
        elif success_rate < 0.2 and collision_hits > 0:
            recommendation = "Low success with collisions; consider lowering --monotone-window or disabling monotone extension."
        elif total_extended == 0:
            recommendation = "No successful extensions; monotone shortcut not helping—reduce the window."
        else:
            recommendation = (
                "Monotone helping intermittently; window cap looks reasonable."
            )
        parts.append(f"recommendation={recommendation}")
        return " | ".join(parts)

    class MonotoneController:
        def __init__(self, max_window: int):
            self.max_window = max(0, max_window)
            self.current_window = min(2, self.max_window) if self.max_window > 0 else 0
            self.avg_solve: Optional[float] = None
            self.avg_oracle: Optional[float] = None

        def record_solve(self, runtime: Optional[float]) -> None:
            if runtime is None:
                return
            if self.avg_solve is None:
                self.avg_solve = runtime
            else:
                self.avg_solve = 0.5 * self.avg_solve + 0.5 * runtime

        def record_oracle(self, duration: float) -> None:
            if duration < 0:
                return
            if self.avg_oracle is None:
                self.avg_oracle = duration
            else:
                self.avg_oracle = 0.5 * self.avg_oracle + 0.5 * duration

        def planned_window(self, remaining: int) -> int:
            if self.current_window <= 0:
                return 0
            window = min(self.current_window, remaining)
            if window <= 0:
                return 0
            if self.avg_oracle is not None and self.avg_solve is not None:
                est_cost = window * self.avg_oracle
                if est_cost > self.avg_solve:
                    # reduce window so expected oracle cost stays under solve time
                    window_cap = max(
                        1, int(self.avg_solve / max(self.avg_oracle, 1e-9))
                    )
                    window = min(window, window_cap)
            return max(0, window)

        def update_after_attempt(self, extend_by: int, collision_found: bool) -> None:
            if extend_by == 0 or collision_found:
                if self.current_window > 1:
                    self.current_window = max(1, self.current_window // 2)
            elif (
                extend_by >= self.current_window
                and self.current_window < self.max_window
            ):
                self.current_window = min(self.max_window, self.current_window + 1)
            self.current_window = min(self.current_window, self.max_window)

    controller = MonotoneController(monotone_window) if monotone_window > 0 else None
    stats_store: Optional[Dict[str, List[int]]] = (
        monotone_stats if monotone_stats is not None else ({} if controller else None)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(p.stem.split("_")[1])
        for p in out_dir.glob("R_*.json")
        if p.stem.split("_")[1].isdigit()
    }
    start = max(existing) + 1 if existing else 1
    n = start
    while n <= target_N:
        res = solve_func(n, threads=threads, verbose=verbose)
        if controller:
            controller.record_solve(res.runtime)
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
                optimality = not feasible_with_min_size(
                    n, res.size + 1, threads=threads, verbose=verbose
                )
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
        extend_by = 0
        collision_found = False
        planned_window = controller.planned_window(target_N - n) if controller else 0
        if planned_window > 0:
            for j in range(1, planned_window + 1):
                candidate = sorted(res.solution + list(range(n + 1, n + j + 1)))
                t_oracle = time.perf_counter()
                relation = monotone_collision_oracle(candidate)
                t_oracle = time.perf_counter() - t_oracle
                if controller:
                    controller.record_oracle(t_oracle)
                if relation is None:
                    extend_by = j
                else:
                    collision_found = True
                    break
        if stats_store is not None and controller:
            stats_store.setdefault("attempt_windows", []).append(planned_window)
            stats_store.setdefault("extend_by", []).append(extend_by)
            stats_store.setdefault("collision_found", []).append(int(collision_found))
        if controller:
            controller.update_after_attempt(extend_by, collision_found)
        for j in range(1, extend_by + 1):
            extended_n = n + j
            extended_solution = sorted(
                res.solution + list(range(n + 1, extended_n + 1))
            )
            cert_path = out_dir / f"R_{extended_n}.json"
            save_certificate(
                extended_n,
                extended_solution,
                cert_path,
                runtime=None,
                optimality_proved=optimality,
                cnf_path=None,
                proof_path=None,
            )
            print(
                f"N={extended_n} -> R({extended_n})={len(extended_solution)} | monotone extension from N={n} | saved {cert_path}"
            )
        n += extend_by + 1

    # Emit a monotone summary when running in sequential mode with a controller.
    if controller and stats_store is not None:
        summary = summarize_monotone_stats(stats_store, monotone_window)
        if summary:
            print(summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute R(N) with iterative collision cuts."
    )
    parser.add_argument(
        "N", type=int, nargs="?", help="Upper bound of the ground set {1..N}"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of solver threads"
    )
    parser.add_argument("--verbose", action="store_true", help="Print generated cuts.")
    parser.add_argument(
        "--cert", type=Path, help="Write JSON certificate to this path."
    )
    parser.add_argument(
        "--verify", action="store_true", help="Explicitly verify the final set."
    )
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
    parser.add_argument(
        "--monotone-window",
        type=int,
        default=0,
        help="In sequential mode, attempt to extend each solution by up to this many new elements using exact collision checks, skipping solves when safe.",
    )
    args = parser.parse_args()

    if args.seq:
        sequential_cert_run(
            args.seq,
            args.cert_dir,
            args.threads,
            args.verbose,
            args.prove_optimal,
            monotone_window=args.monotone_window,
        )
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
            optimality = not feasible_with_min_size(
                args.N, res.size + 1, threads=args.threads, verbose=args.verbose
            )
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
