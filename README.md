# Erdős Problem #321 – Computing R(N)

This repo contains a reproducible, certifying search for
R(N): the maximum size of \(S \subseteq \{1,\dots,N\}\) such that all
subset sums of \(\{1/i : i \in S\}\) are distinct (equivalently, no non-trivial
signed relation \(\sum \varepsilon_i/i = 0\) with \(\varepsilon_i \in \{-1,0,1\}\)).

## Requirements
- Python 3.11 (conda env recommended).
- `ortools` (installed via pip).
- `kissat` (Homebrew install) for UNSAT proofs of “no larger set”; if missing, optimality_proved will be False (no proof).

Setup example:
```bash
conda create -y -n erdos321 python=3.11
conda activate erdos321
python -m pip install ortools
brew install kissat
```

## Usage
Single run (with verification, proof attempt, and certificate):
```bash
python solver.py 36 --threads 12 --verify --prove-optimal --cert certificates/R_36.json
```

Sequential/resumable run to N (skips existing certs, emits CNF/DRAT if `kissat` is present):
```bash
python solver.py --seq 50 --threads 12 --cert-dir certificates --prove-optimal
```

Flags:
- `--verify`: exact check that the reported set has no signed zero-sum relation.
- `--prove-optimal`: after finding size k, build CNF for size ≥ k+1 using the
  encountered collision cuts and run `kissat` to produce a DRAT proof. If `kissat`
  is absent, falls back to a feasibility check (not proof-producing).
- `--threads`: passed to the MIP search (CBC).

## Tests
Run the unit tests (fast check of collision detection and the sequence prefix for N ≤ 20):
```bash
python -m unittest tests/test_solver.py
```

## Outputs
- Certificates: JSON files (e.g., `certificates/R_36.json`) with fields:
  - `N`, `size`, `solution` (one optimal set),
  - `verified_no_relation` (exact witness check),
  - `runtime_seconds`,
  - `optimality_proved_no_larger` (True if `kissat` UNSAT; False if SAT; None if `kissat` missing),
  - `cnf`, `proof` (paths to CNF and DRAT proof if produced).
- CNF/DRAT: when `--prove-optimal` and `kissat` are available, files
  `cnf_N{N}_ge_{k+1}.cnf` and `cnf_N{N}_ge_{k+1}.drat` are written alongside the cert.

## Verification (trusting `kissat`)
1) Witness: run the exact check on `solution` (automatic with `--verify`; anyone can re-run
   an independent checker).
2) Optimality: rerun `kissat cnf proof`; return code 20 means UNSAT, confirming no set of size k+1 exists under the recorded collision cuts. SAT leaves optimality_proved=False (no proof).

Why UNSAT on a subset of constraints is sufficient: the CNF we build uses only the collision cuts encountered during solving, plus the size ≥ k+1 cardinality constraint. This is a weaker formula than the full “no collisions” condition. If the weaker formula is UNSAT, adding more constraints (the remaining collisions) cannot make it satisfiable. Therefore an UNSAT result on this subset is a sound proof that no collision-free set of size k+1 exists. A SAT result does not imply feasibility for the full problem, so optimality_proved remains False in that case.
