"""Backend abstraction for collision-cut solving: CBC, CP-SAT, and MaxSAT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Set

try:
    from ortools.linear_solver import pywraplp
except ImportError as exc:  # pragma: no cover - import guard
    pywraplp = None  # type: ignore[assignment]
    _ORTOOLS_LINEAR_ERROR = exc
else:  # pragma: no cover - import guard
    _ORTOOLS_LINEAR_ERROR = None

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover - import guard
    cp_model = None  # type: ignore[assignment]
    _ORTOOLS_CP_ERROR = exc
else:  # pragma: no cover - import guard
    _ORTOOLS_CP_ERROR = None

try:
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF
except ImportError as exc:  # pragma: no cover - import guard
    RC2 = None  # type: ignore[assignment]
    WCNF = None  # type: ignore[assignment]
    _PYSAT_ERROR = exc
else:  # pragma: no cover - import guard
    _PYSAT_ERROR = None


ORTOOLS_LINEAR_AVAILABLE = pywraplp is not None
CPSAT_AVAILABLE = cp_model is not None
MAXSAT_AVAILABLE = RC2 is not None and WCNF is not None


class BackendNotAvailable(RuntimeError):
    """Raised when the requested backend is missing a dependency or unavailable."""


@dataclass
class BackendSpec:
    """Static model ingredients shared across backends."""

    N: int
    safe_numbers: Set[int]
    forbidden: Set[int]
    groups: List[List[int]]
    static_cuts: List[List[int]]
    additional_cuts: List[List[int]]


class BackendSession(Protocol):
    def solve(self) -> List[int]:
        """Solve the current model and return the selected elements."""

    def add_cut(self, cut: Sequence[int]) -> None:
        """Add a collision cut."""

    def close(self) -> None:
        """Release resources (optional)."""


def _require_ortools_linear() -> None:
    if pywraplp is None:
        raise BackendNotAvailable(
            "CBC backend requires ortools; install with `pip install ortools`."
        ) from _ORTOOLS_LINEAR_ERROR


def _require_ortools_cp() -> None:
    if cp_model is None:
        raise BackendNotAvailable(
            "CP-SAT backend requires ortools; install with `pip install ortools`."
        ) from _ORTOOLS_CP_ERROR


def _require_pysat() -> None:
    if not MAXSAT_AVAILABLE:
        raise BackendNotAvailable(
            "MaxSAT backend requires python-sat; install with `pip install python-sat[pblib,aiger]`."
        ) from _PYSAT_ERROR


class CBCBackendSession:
    def __init__(self, spec: BackendSpec, threads: int):
        _require_ortools_linear()
        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            raise BackendNotAvailable("CBC solver unavailable")
        solver.SetNumThreads(threads)

        self._spec = spec
        self._solver = solver
        self._x: Dict[int, pywraplp.Variable] = {}

        for i in range(1, spec.N + 1):
            var = solver.BoolVar(f"x_{i}")
            self._x[i] = var
            if i in spec.forbidden:
                solver.Add(var == 0)
            elif i in spec.safe_numbers:
                solver.Add(var == 1)

        seen_in_groups: Set[int] = set()
        for idx, group in enumerate(spec.groups):
            gvar = solver.BoolVar(f"g_{idx}")
            for item in group:
                if item in seen_in_groups:
                    raise ValueError(f"overlapping group element {item}")
                seen_in_groups.add(item)
                if item not in self._x:
                    raise ValueError(f"group element {item} outside 1..N")
                solver.Add(self._x[item] == gvar)

        objective = solver.Objective()
        for var in self._x.values():
            objective.SetCoefficient(var, 1)
        objective.SetMaximization()

        # Seed with static/additional cuts.
        for cut in spec.static_cuts + spec.additional_cuts:
            self._add_cut_internal(cut)

    def _add_cut_internal(self, cut: Sequence[int]) -> None:
        cons = self._solver.Constraint(-self._solver.infinity(), len(cut) - 1)
        for i in cut:
            cons.SetCoefficient(self._x[i], 1)

    def solve(self) -> List[int]:
        status = self._solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"CBC failed with status {status}")
        return [
            i for i in range(1, self._spec.N + 1) if self._x[i].solution_value() > 0.5
        ]

    def add_cut(self, cut: Sequence[int]) -> None:
        self._add_cut_internal(cut)

    def close(self) -> None:
        return


class CPSATBackendSession:
    def __init__(self, spec: BackendSpec, threads: int):
        _require_ortools_cp()
        self._spec = spec
        self._threads = threads
        self._cuts: List[List[int]] = list(spec.static_cuts + spec.additional_cuts)

    def add_cut(self, cut: Sequence[int]) -> None:
        self._cuts.append(list(cut))

    def _build_model(self) -> tuple[cp_model.CpModel, Dict[int, cp_model.IntVar]]:
        model = cp_model.CpModel()
        x: Dict[int, cp_model.IntVar] = {
            i: model.NewBoolVar(f"x_{i}") for i in range(1, self._spec.N + 1)
        }
        for i in self._spec.safe_numbers:
            model.Add(x[i] == 1)
        for i in self._spec.forbidden:
            model.Add(x[i] == 0)

        seen_in_groups: Set[int] = set()
        for idx, group in enumerate(self._spec.groups):
            gvar = model.NewBoolVar(f"g_{idx}")
            for item in group:
                if item in seen_in_groups:
                    raise ValueError(f"overlapping group element {item}")
                seen_in_groups.add(item)
                if item not in x:
                    raise ValueError(f"group element {item} outside 1..N")
                model.Add(x[item] == gvar)

        for cut in self._cuts:
            model.Add(sum(x[i] for i in cut) <= len(cut) - 1)

        model.Maximize(sum(x.values()))
        return model, x

    def solve(self) -> List[int]:
        model, x = self._build_model()
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self._threads
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(f"CP-SAT failed with status {status}")
        return [i for i in range(1, self._spec.N + 1) if solver.Value(x[i]) == 1]

    def close(self) -> None:
        return


class MaxSATBackendSession:
    def __init__(self, spec: BackendSpec, threads: int):
        _require_pysat()
        self._spec = spec
        self._threads = threads
        self._cuts: List[List[int]] = list(spec.static_cuts + spec.additional_cuts)

    def add_cut(self, cut: Sequence[int]) -> None:
        self._cuts.append(list(cut))

    def _build_wcnf(self) -> WCNF:
        wcnf = WCNF()
        N = self._spec.N
        # Hard clauses for safe/forbidden and cuts.
        for i in self._spec.safe_numbers:
            wcnf.append([i])  # must be true
        for i in self._spec.forbidden:
            wcnf.append([-i])  # must be false

        # Enforce equality inside groups via pairwise equivalence to the first element.
        for group in self._spec.groups:
            if not group:
                continue
            root = group[0]
            for item in group[1:]:
                wcnf.append([-root, item])
                wcnf.append([root, -item])

        for cut in self._cuts:
            wcnf.append([-i for i in cut])

        # Soft objective: reward each unfixed variable being true.
        for i in range(1, N + 1):
            if i in self._spec.forbidden:
                continue
            wcnf.append([i], weight=1)
        return wcnf

    def solve(self) -> List[int]:
        wcnf = self._build_wcnf()
        solver = RC2(wcnf)
        model = solver.compute()
        solver.delete()
        if model is None:
            raise RuntimeError("MaxSAT solver returned UNSAT/None")
        return sorted([i for i in model if 1 <= i <= self._spec.N and i > 0])

    def close(self) -> None:
        return


def create_backend_session(
    name: str, spec: BackendSpec, threads: int
) -> BackendSession:
    name = name.lower()
    if name == "cbc":
        return CBCBackendSession(spec, threads)
    if name in ("cpsat", "cp-sat", "cp_sat"):
        return CPSATBackendSession(spec, threads)
    if name == "maxsat":
        return MaxSATBackendSession(spec, threads)
    raise ValueError(f"Unknown backend {name}")
