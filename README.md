# Erdős Problem #321 – Computing R(N)

This repo contains a reproducible, certifying search for
R(N): the maximum size of a set S subset of {1,...,N} such that all
subset sums of {1/i : i in S} are distinct. Equivalently, there is no non-trivial
signed relation sum eps_i/i = 0 with eps_i in {-1,0,1} supported on S.

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
- `--monotone-window`: in sequential mode, try to extend solutions by up to this many
  new elements using exact collision checks before re-solving; 0 disables (default).
- `--pruning {on,off}`: toggle p-adic safe-variable fixing and grouped collision search (default on).
- `--pruning-groups-oracle {on,off}`: use p-adic all-or-none groups inside the collision oracle (default on).
- `--enforce-pruning-groups-in-model`: tie each detected all-or-none group to one Boolean (optional).
- `--show-pruning`: print a summary of the p-adic exclusions (safe numbers and groups) during runs.
- `--cuts-cache PATH`: persist and reuse collision cuts across runs (JSON list of cuts).

## Features and optimizations
- P-adic pruning (on by default): fixes provably safe numbers to 1 and treats detected
  all-or-none clusters (e.g., {11,22,33} at N=36) as grouped in the collision oracle.
 - Cut cache (optional): `--cuts-cache cuts.json` loads prior collision cuts as static cuts and
  appends newly found ones for reuse in subsequent runs.
- Monotone extension shortcut (`--monotone-window`): adaptively grows/shrinks the extension
  window based on collision hits and oracle/solve cost, emitting telemetry and skipping
  re-solves when the collision catalogue shows the next integers are safe to append.

## Tests
Run the unit tests (fast check of collision detection and the sequence prefix for N ≤ 20):
```bash
python -m unittest tests/test_solver.py
```

## Future roadmap
| Idea | What it changes | Expected speedup | Risk/notes | Ease |
| --- | --- | --- | --- | --- |
| Greedy Kraft-like warm start | Build a quick collision-free set as the initial incumbent/branch hint. | 1.2–2× fewer nodes typically | Low risk; only seeds search, does not prune valid solutions. | Easy |
| Modular-filtered collision oracle | Add fast modular signatures before the exact meet-in-the-middle `find_relation`; only exact-check true collisions. | 5–30× per collision check for sets of size ≈20+ | Low risk if exact check remains the final gate (no false negatives allowed). | Medium |
| Precomputed short collision cuts | Meet-in-the-middle catalogue of equalities up to small length (e.g., 6–8) added as static cuts (full n=36 catalogue had ~2.3M collisions; bounded-length to stay light; can block-dependent triplets like {11,22,33} as one choice). | 2–6× fewer solver iterations if catalogue stays small | Moderate: catalogue can blow up if length bound too high; correctness preserved if all cuts are exact. | Medium |
| Switch to CP-SAT/MaxSAT backend | Replace CBC with a modern SAT/MaxSAT solver that learns clauses natively. | 2–5× from better branching/learning (problem-size dependent) | Low-to-moderate: integration work; correctness fine if encoding matches current model. | Medium |
| Short-relation PSLQ/LLL prepass | Detect very short dependencies before full MITM to skip expensive searches. | 1.5–3× when short relations are common | Moderate: needs exact confirmation to avoid numeric pitfalls. | Medium |

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
