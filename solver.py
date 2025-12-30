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
- Backends: CBC (default), CP-SAT, MaxSAT, or a race of multiple backends for a “turbo” first-to-finish solve.
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
  python solver.py --seq 100 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3 --cuts-cache cuts.json --show-pruning
  # Turbo sequential run (race CP-SAT/MaxSAT/CBC; take first finisher)
  python solver.py --seq 100 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3 --cuts-cache cuts.json --backend race --show-pruning
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from collisions import (
    Relation,
    find_relation,
    find_relation_grouped,
    verify_relation_free,
)
from certificates import save_certificate
from monotone import MonotoneController, summarize_monotone_stats
from pruning import PruningResult, compute_p_adic_exclusions
from proofs import prove_optimality
from solver_backends import (
    BackendNotAvailable,
    BackendSpec,
    CPSAT_AVAILABLE,
    MAXSAT_AVAILABLE,
    ORTOOLS_LINEAR_AVAILABLE,
    BackendSession,
    create_backend_session,
    pywraplp,
)

ORTOOLS_AVAILABLE = ORTOOLS_LINEAR_AVAILABLE


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
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError(
            "ortools is required for solving; install it with `pip install ortools`."
        )


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


def _maybe_prove_optimal(
    n: int,
    res: SolveResult,
    prove_optimal: bool,
    out_dir: Path,
    threads: int,
    verbose: bool,
    use_p_adic_pruning: bool,
    pruning_func: Callable[[int], PruningResult],
    cached_cuts: List[List[int]],
) -> Tuple[Optional[bool], Optional[Path], Optional[Path]]:
    if not prove_optimal:
        return None, None, None
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
    return optimality, cnf_path, proof_path


def _attempt_monotone_extension(
    res: SolveResult,
    n: int,
    target_N: int,
    controller: Optional[MonotoneController],
    monotone_collision_oracle: Callable[[Sequence[int]], Optional[Relation]],
) -> Tuple[int, bool, int]:
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
    return extend_by, collision_found, planned_window


def _emit_extension_certs(
    res: SolveResult,
    n: int,
    extend_by: int,
    optimality: Optional[bool],
    out_dir: Path,
) -> None:
    for j in range(1, extend_by + 1):
        extended_n = n + j
        extended_solution = sorted(res.solution + list(range(n + 1, extended_n + 1)))
        cert_path = out_dir / f"R_{extended_n}.json"
        save_certificate(
            extended_n,
            extended_solution,
            cert_path,
            runtime=None,
            optimality_proved=optimality,
            verify_fn=verify_relation_free,
            monotone_extension_from=n,
            cnf_path=None,
            proof_path=None,
        )
        print(
            f"N={extended_n} -> R({extended_n})={len(extended_solution)} | monotone extension from N={n} | saved {cert_path}"
        )


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
    backend: str = "cbc",
    race_backends: Optional[Sequence[str]] = None,
    _runner_overrides: Optional[Dict[str, Callable[[], SolveResult]]] = None,
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
    backend: which backend to use (cbc, cpsat, maxsat, race). race runs multiple backends and returns the first to finish.
    race_backends: when backend == "race", which backends to run in parallel (defaults to available set).
    """
    if backend.lower() == "race":
        choices = list(race_backends) if race_backends else []
        if not choices:
            # Default race set: prefer CP-SAT and MaxSAT if available, otherwise CBC only.
            choices = [
                name
                for name, available in [
                    ("cpsat", CPSAT_AVAILABLE),
                    ("maxsat", MAXSAT_AVAILABLE),
                    ("cbc", True),
                ]
                if available
            ]
        return race_solve_max_distinct(
            backend_names=choices,
            N=N,
            threads=threads,
            verbose=verbose,
            collision_oracle=collision_oracle,
            static_cuts=static_cuts,
            forbidden=forbidden,
            groups=groups,
            use_p_adic_pruning=use_p_adic_pruning,
            pruning_func=pruning_func,
            use_pruning_groups_in_oracle=use_pruning_groups_in_oracle,
            enforce_pruning_groups_in_model=enforce_pruning_groups_in_model,
            additional_cuts=additional_cuts,
            runner_overrides=_runner_overrides,
        )

    return _solve_single_backend(
        N=N,
        backend=backend,
        threads=threads,
        verbose=verbose,
        collision_oracle=collision_oracle,
        static_cuts=static_cuts,
        forbidden=forbidden,
        groups=groups,
        use_p_adic_pruning=use_p_adic_pruning,
        pruning_func=pruning_func,
        use_pruning_groups_in_oracle=use_pruning_groups_in_oracle,
        enforce_pruning_groups_in_model=enforce_pruning_groups_in_model,
        additional_cuts=additional_cuts,
    )


def _solve_single_backend(
    N: int,
    backend: str,
    threads: int,
    verbose: bool,
    collision_oracle: Callable[[Sequence[int]], Optional[Relation]],
    static_cuts: Optional[List[List[int]]],
    forbidden: Optional[Sequence[int]],
    groups: Optional[List[List[int]]],
    use_p_adic_pruning: bool,
    pruning_func: Callable[[int], PruningResult],
    use_pruning_groups_in_oracle: bool,
    enforce_pruning_groups_in_model: bool,
    additional_cuts: Optional[List[List[int]]],
) -> SolveResult:
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
    group_list = groups or []
    if enforce_pruning_groups_in_model and pruning_groups:
        group_list = group_list + pruning_groups

    spec = BackendSpec(
        N=N,
        safe_numbers=safe_numbers,
        forbidden=forbidden_set,
        groups=group_list,
        static_cuts=[
            [i for i in cut if 1 <= i <= N]
            for cut in (static_cuts or [])
            if all(1 <= i <= N for i in cut)
        ],
        additional_cuts=[
            [i for i in cut if 1 <= i <= N]
            for cut in (additional_cuts or [])
            if all(1 <= i <= N for i in cut)
        ],
    )

    # Build a grouped oracle wrapper when applicable.
    active_groups_for_oracle = pruning_groups if use_pruning_groups_in_oracle else []

    cuts: List[List[int]] = []  # store collision sets for later CNF/DRAT proof
    collision_support: Set[int] = set()
    for cut_vars in spec.static_cuts + spec.additional_cuts:
        cuts.append(list(cut_vars))
        collision_support.update(cut_vars)

    t0 = time.perf_counter()
    session: Optional[BackendSession] = None
    try:
        session = create_backend_session(backend, spec, threads)
        while True:
            sol = session.solve()
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
                print(
                    f"collision found with plus={relation.plus} minus={relation.minus}"
                )
            cut_vars = relation.plus + relation.minus
            session.add_cut(cut_vars)
            cuts.append(cut_vars)
            collision_support.update(cut_vars)
    except BackendNotAvailable as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        if session is not None:
            session.close()


def _race_worker(backend_name: str, kwargs: Dict[str, Any]) -> SolveResult:
    return _solve_single_backend(backend=backend_name, **kwargs)


def race_solve_max_distinct(
    backend_names: Sequence[str],
    N: int,
    threads: int,
    verbose: bool,
    collision_oracle: Callable[[Sequence[int]], Optional[Relation]],
    static_cuts: Optional[List[List[int]]],
    forbidden: Optional[Sequence[int]],
    groups: Optional[List[List[int]]],
    use_p_adic_pruning: bool,
    pruning_func: Callable[[int], PruningResult],
    use_pruning_groups_in_oracle: bool,
    enforce_pruning_groups_in_model: bool,
    additional_cuts: Optional[List[List[int]]],
    runner_overrides: Optional[Dict[str, Callable[[], SolveResult]]] = None,
) -> SolveResult:
    """Run multiple backends in parallel and return the first successful result."""

    names = list(backend_names)
    if not names:
        raise ValueError("race requires at least one backend")

    base_kwargs = dict(
        N=N,
        threads=threads,
        verbose=verbose,
        collision_oracle=collision_oracle,
        static_cuts=static_cuts,
        forbidden=forbidden,
        groups=groups,
        use_p_adic_pruning=use_p_adic_pruning,
        pruning_func=pruning_func,
        use_pruning_groups_in_oracle=use_pruning_groups_in_oracle,
        enforce_pruning_groups_in_model=enforce_pruning_groups_in_model,
        additional_cuts=additional_cuts,
    )

    def _submit_jobs(
        executor: concurrent.futures.Executor,
    ) -> Dict[concurrent.futures.Future[SolveResult], str]:
        submitted: Dict[concurrent.futures.Future[SolveResult], str] = {}
        for name in names:
            if runner_overrides and name in runner_overrides:
                fut = executor.submit(runner_overrides[name])
            else:
                fut = executor.submit(_race_worker, name, base_kwargs)
            submitted[fut] = name
        return submitted

    errors: List[Tuple[str, Exception]] = []
    if runner_overrides:
        executor: concurrent.futures.Executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(names)
        )
    else:
        try:
            ctx = multiprocessing.get_context("spawn")
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=len(names), mp_context=ctx
            )
        except Exception:
            # Fallback to threads if process creation is not permitted (e.g., restricted env).
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(names))

    with executor as ex:
        futures = _submit_jobs(ex)
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
                # Cancel remaining work.
                for other in futures:
                    if other is not fut:
                        other.cancel()
                return res
            except Exception as exc:  # pragma: no cover - race error path
                errors.append((name, exc))
                continue

    msg = "; ".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(f"all race backends failed: {msg}")


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
    backend: str = "cbc",
    race_backends: Optional[Sequence[str]] = None,
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
    backend/race_backends: choose a backend or a set to race; defaults to CBC unless overridden.
    """

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
            backend=backend,
            race_backends=race_backends,
        )
        if controller:
            controller.record_solve(res.runtime)
        optimality, cnf_path, proof_path = _maybe_prove_optimal(
            n,
            res,
            prove_optimal,
            out_dir,
            threads,
            verbose,
            use_p_adic_pruning,
            pruning_func,
            cached_cuts,
        )
        cert_path = out_dir / f"R_{n}.json"
        save_certificate(
            n,
            res.solution,
            cert_path,
            runtime=res.runtime,
            optimality_proved=optimality,
            verify_fn=verify_relation_free,
            monotone_extension_from=None,
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
        extend_by, collision_found, planned_window = _attempt_monotone_extension(
            res, n, target_N, controller, monotone_collision_oracle
        )
        if stats_store is not None and controller:
            stats_store.setdefault("attempt_windows", []).append(planned_window)
            stats_store.setdefault("extend_by", []).append(extend_by)
            stats_store.setdefault("collision_found", []).append(int(collision_found))
        if controller:
            controller.update_after_attempt(extend_by, collision_found)
        _emit_extension_certs(res, n, extend_by, optimality, out_dir)
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
    parser.add_argument(
        "--backend",
        choices=["cbc", "cpsat", "maxsat", "race"],
        default="cbc",
        help="Solver backend: CBC (default), CP-SAT, MaxSAT, or a race of backends.",
    )
    parser.add_argument(
        "--race-backends",
        type=str,
        help="Comma-separated backend names to race when --backend=race (e.g., 'cpsat,cbc').",
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

    race_backends = (
        [b.strip() for b in args.race_backends.split(",") if b.strip()]
        if args.race_backends
        else None
    )

    if args.seq:
        pruning_on = args.pruning == "on"
        pruning_groups_on = args.pruning_groups_oracle == "on"
        sequential_cert_run(
            args.seq,
            args.cert_dir,
            args.threads,
            args.verbose,
            args.prove_optimal,
            backend=args.backend,
            race_backends=race_backends,
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
        backend=args.backend,
        race_backends=race_backends,
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
            monotone_extension_from=None,
            cnf_path=cnf_path,
            proof_path=proof_path,
        )
        print(f"wrote certificate to {args.cert}")


if __name__ == "__main__":
    main()
