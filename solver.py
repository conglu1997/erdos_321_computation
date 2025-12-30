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
- P-adic pruning (default on): fixes provably safe numbers to 1 and groups “all-or-none” clusters
  in the collision oracle; show details with `--show-pruning`.
- Cut cache (optional): `--cuts-cache cuts.json` to persist learned collision cuts across runs.

Example commands:
  # Single-N quick test run
  python solver.py 24 --threads 12 --verify --prove-optimal --cert certificates/R_24.json
  # Sequential max-speed run with proofs and monotone shortcut
  python solver.py --seq 50 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3 --cuts-cache cuts.json --show-pruning
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from certificates import save_certificate
from pruning import PruningResult, compute_p_adic_exclusions
from proofs import prove_optimality

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
    pruning: Optional[PruningResult] = None


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


def format_pruning_summary(pruning: PruningResult, N: int) -> str:
    """Human-friendly one-line summary of pruning results."""
    safe = pruning.safe_numbers
    groups = pruning.all_or_none_groups
    parts = [
        f"[pruning] N={N}",
        f"safe_count={len(safe)}",
        f"group_count={len(groups)}",
        f"by_rule={ {k: len(v) for k, v in pruning.by_rule.items()} }",
    ]
    if groups:
        parts.append(f"groups={ [sorted(g) for g in groups] }")
    return " | ".join(parts)


def dedupe_cuts(cuts: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return sorted, deduplicated cuts as lists."""
    uniq = set()
    out: List[List[int]] = []
    for cut in cuts:
        key = tuple(sorted(set(cut)))
        if key in uniq:
            continue
        uniq.add(key)
        out.append(list(key))
    return out

def find_relation_grouped(
    elements: Sequence[int], groups: Sequence[Sequence[int]]
) -> Optional[Relation]:
    """Find a signed relation with the restriction that all elements in each group share a sign.

    A group may either be absent (all coefficients 0) or present with all +1 or all -1.
    This reduces the search space when modular reasoning shows only the “all equal sign”
    patterns are relevant (e.g., {11,22,33} at N=36).
    """
    if not elements:
        return None
    element_set = set(elements)
    grouped: List[List[int]] = []
    seen: Set[int] = set()
    for g in groups:
        g_in = [v for v in g if v in element_set]
        if len(g_in) <= 1:
            continue
        # Only keep disjoint groups; ignore overlaps for safety.
        if any(v in seen for v in g_in):
            continue
        grouped.append(g_in)
        seen.update(g_in)
    singles = [v for v in elements if v not in seen]
    agg_items: List[Tuple[int, List[int]]] = []
    # LCM-based weights as in find_relation, but aggregate by group.
    L = lcm_upto(max(elements))
    weight_map = {e: L // e for e in elements}
    g = math.gcd(*weight_map.values())
    for e in weight_map:
        weight_map[e] //= g
    for g_in in grouped:
        agg_items.append((sum(weight_map[v] for v in g_in), g_in))
    for s in singles:
        agg_items.append((weight_map[s], [s]))
    mid = len(agg_items) // 2
    left = agg_items[:mid]
    right = agg_items[mid:]
    table: Dict[int, Tuple[int, ...]] = {}
    for coeffs in itertools.product((-1, 0, 1), repeat=len(left)):
        total = sum(w * c for (w, _), c in zip(left, coeffs))
        if total not in table:
            table[total] = coeffs
    for r_coeffs in itertools.product((-1, 0, 1), repeat=len(right)):
        total_r = sum(w * c for (w, _), c in zip(right, r_coeffs))
        target = -total_r
        if target in table:
            l_coeffs = table[target]
            coeffs_full = list(l_coeffs) + list(r_coeffs)
            if all(c == 0 for c in coeffs_full):
                continue
            plus: List[int] = []
            minus: List[int] = []
            for c, (_, members) in zip(coeffs_full, agg_items):
                if c == 1:
                    plus.extend(members)
                elif c == -1:
                    minus.extend(members)
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
    use_p_adic_pruning: bool = True,
    pruning_func: Callable[[int], PruningResult] = compute_p_adic_exclusions,
    use_pruning_groups_in_oracle: bool = True,
    enforce_pruning_groups_in_model: bool = False,
    additional_cuts: Optional[List[List[int]]] = None,
) -> SolveResult:
    """Iteratively add collision cuts until the optimal valid solution is found.

    collision_oracle: callable returning a Relation or None on a candidate set.
    static_cuts: cuts to apply before solving (each a list of vars that cannot all be 1).
    forbidden: variables that must stay 0.
    groups: list of disjoint groups; within each group, all vars must share the same value.
    use_p_adic_pruning: if True, fix provably safe numbers to 1 and skip them in collision checks.
    pruning_func: supplier of p-adic exclusions; replaceable for testing/experiments.
    use_pruning_groups_in_oracle: if True, restrict collision signs to “all equal” on detected groups.
    enforce_pruning_groups_in_model: if True, tie all-or-none groups to a single Boolean; leave False unless you know the grouping corresponds to membership (not just sign-pattern) constraints.
    """
    _require_ortools()
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    pruning: Optional[PruningResult] = None
    safe_numbers: Set[int] = set()
    pruning_groups: List[List[int]] = []
    if use_p_adic_pruning:
        pruning = pruning_func(N)
        safe_numbers = set(pruning.safe_numbers)
        pruning_groups = [list(g) for g in pruning.all_or_none_groups]

    forbidden_set = set(forbidden or [])
    # Never force a number to 1 if the caller explicitly forbids it.
    safe_numbers -= forbidden_set
    x: Dict[int, pywraplp.Variable] = {}
    for i in range(1, N + 1):
        var = solver.BoolVar(f"x_{i}")
        x[i] = var
        if i in forbidden_set:
            solver.Add(var == 0)
        elif i in safe_numbers:
            solver.Add(var == 1)

    group_list = groups or []
    if enforce_pruning_groups_in_model and pruning_groups:
        group_list = group_list + pruning_groups
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
    for cut_vars in (static_cuts or []) + (additional_cuts or []):
        # Ignore cached/static cuts that reference indices outside 1..N.
        if any(i not in x for i in cut_vars):
            continue
        cut = solver.Constraint(-solver.infinity(), len(cut_vars) - 1)
        for i in cut_vars:
            cut.SetCoefficient(x[i], 1)
        cuts.append(list(cut_vars))
        collision_support.update(cut_vars)

    # Build a grouped oracle wrapper when applicable.
    active_groups_for_oracle = pruning_groups if use_pruning_groups_in_oracle else []

    t0 = time.perf_counter()
    while True:
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"Solver failed with status {status}")
        sol = [i for i in range(1, N + 1) if x[i].solution_value() > 0.5]
        # Collision search only needs the elements that can actually collide.
        active_sol = [i for i in sol if i not in safe_numbers]
        if active_groups_for_oracle:
            relation = find_relation_grouped(active_sol, active_groups_for_oracle)
        else:
            relation = collision_oracle(active_sol)
        if relation is None:
            return SolveResult(
                size=len(sol),
                solution=sol,
                cuts=cuts,
                collision_support=collision_support,
                runtime=time.perf_counter() - t0,
                pruning=pruning,
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
    N: int,
    min_size: int,
    threads: int = 8,
    verbose: bool = False,
    use_p_adic_pruning: bool = True,
    pruning_func: Callable[[int], PruningResult] = compute_p_adic_exclusions,
    additional_cuts: Optional[List[List[int]]] = None,
) -> bool:
    """Return True iff there exists a collision-free set of size >= min_size."""
    _require_ortools()
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("CBC solver unavailable")
    solver.SetNumThreads(threads)

    pruning: Optional[PruningResult] = None
    safe_numbers: Set[int] = set()
    if use_p_adic_pruning:
        pruning = pruning_func(N)
        safe_numbers = set(pruning.safe_numbers)

    x: Dict[int, pywraplp.Variable] = {
        i: solver.BoolVar(f"x_{i}") for i in range(1, N + 1)
    }
    for i in safe_numbers:
        solver.Add(x[i] == 1)
    for cut_vars in additional_cuts or []:
        if any(i not in x for i in cut_vars):
            continue
        cut = solver.Constraint(-solver.infinity(), len(cut_vars) - 1)
        for i in cut_vars:
            cut.SetCoefficient(x[i], 1)
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
        active_sol = [i for i in sol if i not in safe_numbers]
        relation = find_relation(active_sol)
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
    use_p_adic_pruning: bool = True,
    pruning_func: Callable[[int], PruningResult] = compute_p_adic_exclusions,
    use_pruning_groups_in_oracle: bool = True,
    enforce_pruning_groups_in_model: bool = False,
    show_pruning: bool = False,
    cuts_cache: Optional[Path] = None,
) -> None:
    """Compute R(n) for n=1..target_N, writing certificates and skipping existing ones.

    monotone_window: upper bound on how many elements to attempt appending via monotone extension.
      An adaptive policy grows/shrinks the attempted window based on success/failure and cost
      estimates; when collision-free, certificates are emitted without re-solving.
    monotone_collision_oracle: collision finder used during monotone extension (Relation or None).
    solve_func: injectable solver (for testing or alternative backends).
    monotone_stats: optional dict to receive telemetry (attempt_windows, extend_by, collision_found).
    use_p_adic_pruning/pruning_func: control p-adic exclusions (safe fixed-1 vars; all-or-none groups).
    use_pruning_groups_in_oracle: if True, grouped collision search enforces equal signs inside groups.
    enforce_pruning_groups_in_model: if True, tie each detected group to one Boolean in the MIP.
    show_pruning: if True, print a one-line summary of pruning for each solve.
    cuts_cache: optional path to persist and reload collision cuts across runs.
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
    # Load cached cuts if present.
    cached_cuts: List[List[int]] = []
    if cuts_cache and cuts_cache.exists():
        try:
            data = json.loads(cuts_cache.read_text())
            if isinstance(data, list):
                cached_cuts = [list(map(int, c)) for c in data if isinstance(c, list)]
        except Exception:
            cached_cuts = []

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(p.stem.split("_")[1])
        for p in out_dir.glob("R_*.json")
        if p.stem.split("_")[1].isdigit()
    }
    start = max(existing) + 1 if existing else 1
    n = start
    while n <= target_N:
        res = solve_func(
            n,
            threads=threads,
            verbose=verbose,
            use_p_adic_pruning=use_p_adic_pruning,
            pruning_func=pruning_func,
            use_pruning_groups_in_oracle=use_pruning_groups_in_oracle,
            enforce_pruning_groups_in_model=enforce_pruning_groups_in_model,
            additional_cuts=cached_cuts,
        )
        if controller:
            controller.record_solve(res.runtime)
        optimality = None
        cnf_path = None
        proof_path = None
        if prove_optimal:
            optimality, cnf_path, proof_path = prove_optimality(
                n,
                res.size,
                res.cuts,
                cnf_dir=out_dir,
                fallback_feasible=lambda: not feasible_with_min_size(
                    n,
                    res.size + 1,
                    threads=threads,
                    verbose=verbose,
                    use_p_adic_pruning=use_p_adic_pruning,
                    pruning_func=pruning_func,
                    additional_cuts=cached_cuts,
                ),
            )
        cert_path = out_dir / f"R_{n}.json"
        save_certificate(
            n,
            res.solution,
            cert_path,
            runtime=res.runtime,
            optimality_proved=optimality,
            verify_fn=verify_relation_free,
            cnf_path=cnf_path,
            proof_path=proof_path,
        )
        # Update the cached cuts (union) and persist if configured.
        if cuts_cache is not None:
            cached_cuts = dedupe_cuts(cached_cuts + res.cuts)
            cuts_cache.write_text(json.dumps(cached_cuts))
        if show_pruning and res.pruning:
            print(format_pruning_summary(res.pruning, n))
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
                verify_fn=verify_relation_free,
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
    parser.add_argument(
        "--pruning",
        choices=["on", "off"],
        default="on",
        help="Enable/disable p-adic pruning (default on).",
    )
    parser.add_argument(
        "--pruning-groups-oracle",
        choices=["on", "off"],
        default="on",
        help="Use all-or-none pruning groups in the collision oracle (default on).",
    )
    parser.add_argument(
        "--enforce-pruning-groups-in-model",
        action="store_true",
        help="Tie each detected all-or-none group to a single Boolean (experimental; enable only if grouping reflects true membership constraints).",
    )
    parser.add_argument(
        "--show-pruning",
        action="store_true",
        help="Print a summary of p-adic pruning (safe numbers and groups) for each solve.",
    )
    parser.add_argument(
        "--cuts-cache",
        type=Path,
        help="Path to persist collision cuts across runs (JSON list of cuts).",
    )
    args = parser.parse_args()

    if args.seq:
        pruning_on = args.pruning == "on"
        pruning_groups_on = args.pruning_groups_oracle == "on"
        sequential_cert_run(
            args.seq,
            args.cert_dir,
            args.threads,
            args.verbose,
            args.prove_optimal,
            monotone_window=args.monotone_window,
            use_p_adic_pruning=pruning_on,
            use_pruning_groups_in_oracle=pruning_groups_on,
            enforce_pruning_groups_in_model=args.enforce_pruning_groups_in_model,
            show_pruning=args.show_pruning,
            cuts_cache=args.cuts_cache,
        )
        return

    if args.N is None:
        parser.error("Provide N or --seq.")

    pruning_on = args.pruning == "on"
    pruning_groups_on = args.pruning_groups_oracle == "on"
    res = solve_max_distinct(
        args.N,
        threads=args.threads,
        verbose=args.verbose,
        use_p_adic_pruning=pruning_on,
        use_pruning_groups_in_oracle=pruning_groups_on,
        enforce_pruning_groups_in_model=args.enforce_pruning_groups_in_model,
        additional_cuts=[],
    )
    optimality = None
    cnf_path = None
    proof_path = None
    if args.prove_optimal:
        optimality, cnf_path, proof_path = prove_optimality(
            args.N,
            res.size,
            res.cuts,
            cnf_dir=Path("."),
            fallback_feasible=lambda: not feasible_with_min_size(
                args.N,
                res.size + 1,
                threads=args.threads,
                verbose=args.verbose,
                use_p_adic_pruning=pruning_on,
                additional_cuts=[],
            ),
        )
    print(f"R({args.N}) = {res.size}")
    print("one optimal set:", res.solution)
    print(f"time {res.runtime:.3f}s")
    if args.show_pruning and res.pruning:
        print(format_pruning_summary(res.pruning, args.N))
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
            verify_fn=verify_relation_free,
            cnf_path=cnf_path,
            proof_path=proof_path,
        )
        print(f"wrote certificate to {args.cert}")


if __name__ == "__main__":
    main()
