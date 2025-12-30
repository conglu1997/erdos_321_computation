# Erdős Problem #321 – Computing R(N)

This repo contains a reproducible, certifying search for
R(N): the maximum size of a set S subset of {1,...,N} such that all
subset sums of {1/i : i in S} are distinct. Equivalently, there is no non-trivial
signed relation sum eps_i/i = 0 with eps_i in {-1,0,1} supported on S.

Discussion thread: https://www.erdosproblems.com/forum/thread/321

## Requirements
- Python 3.11 (conda env recommended).
- `ortools` (installed via pip).
- `kissat` (Homebrew install) if you want DRAT proofs of “no larger set”. If missing, the solver falls back to a feasibility check for size k+1 (no DRAT proof, but it still records whether that check succeeds).
- `python-sat` (installed via pip) if you want to try the MaxSAT backend.
- `matplotlib` (only needed for `plot_certificates.py` visualizations).

Setup example:
```bash
conda create -y -n erdos321 python=3.11
conda activate erdos321
python -m pip install ortools
brew install kissat
```

## Usage
Single-N solve (prints the solution, verifies it, and optionally writes a certificate):
```bash
python solver.py 36 --threads 12 --verify --prove-optimal --cert certificates/R_36.json
```

Sequential/resumable run to N (skips existing certs; optional monotone shortcut, cut cache, and pruning diagnostics):
```bash
python solver.py --seq 100 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3 --cuts-cache cuts.json --show-pruning
# Turbo variant (race CP-SAT/MaxSAT/CBC and take the first finisher):
python solver.py --seq 100 --threads 12 --cert-dir certificates --prove-optimal --monotone-window 3 --cuts-cache cuts.json --backend race --show-pruning
```

Flags:
- `--cert PATH`: write a JSON certificate for a single-N run.
- `--verify`: exact check that the reported set has no signed zero-sum relation.
- `--prove-optimal`: after finding size k, build CNF for size ≥ k+1 using the
  encountered collision cuts and run `kissat` to produce a DRAT proof. If `kissat`
  is absent, it still writes the CNF and uses CBC to test feasibility of size k+1
  (no DRAT file, but `optimality_proved_no_larger` reflects the feasibility result).
- `--threads`: passed to the MIP search (CBC).
- `--backend {cbc,cpsat,maxsat,race}`: choose solver backend (CBC default). `race`
  runs multiple backends in parallel and takes the first solution (turbo mode).
- `--race-backends NAME,NAME`: when using `--backend race`, explicitly pick which
  backends to launch (e.g., `--race-backends cpsat,cbc`; defaults to CP-SAT+CBC if available).
- `--monotone-window`: in sequential mode, try to extend solutions by up to this many
  new elements using exact collision checks before re-solving; 0 disables (default).
- `--pruning {on,off}`: toggle p-adic safe-variable fixing and grouped collision search (default on).
- `--pruning-groups-oracle {on,off}`: use p-adic all-or-none groups inside the collision oracle (default on).
- `--enforce-pruning-groups-in-model`: tie each detected all-or-none group to one Boolean (optional).
- `--show-pruning`: print a summary of the p-adic exclusions (safe numbers and groups) during runs.
- `--cuts-cache PATH`: persist and reuse collision cuts across runs (JSON list of cuts).

## Features and optimizations
- Pluggable backends: CBC (default), CP-SAT, optional MaxSAT, or a race between backends.
- P-adic pruning (on by default): fixes provably safe numbers to 1 and treats detected
  all-or-none clusters (e.g., {11,22,33} at N=36) as grouped in the collision oracle.
- Cut cache (optional): `--cuts-cache cuts.json` loads prior collision cuts as static cuts and
  appends newly found ones for reuse in subsequent runs.
- Monotone extension shortcut (`--monotone-window`): adaptively grows/shrinks the extension
  window based on collision hits and oracle/solve cost, emitting telemetry and skipping
  re-solves when the collision catalogue shows the next integers are safe to append.

## Plotting and logging
- Visualize certificates and runtimes: `python plot_certificates.py --cert-dir certificates --out-dir plots`
  - Saves plots comparing the certificates against the known test sequence and showing runtime/density trends.

## Tests
- Full suite: `python -m unittest discover -s tests -p 'test_*.py'`
- Quick targeted run: `python -m unittest tests/test_solver.py`

## Future roadmap
| Idea | What it changes | Expected speedup | Risk/notes | Ease |
| --- | --- | --- | --- | --- |
| Short-relation PSLQ/LLL prepass | Detect very short dependencies before full MITM to skip expensive searches. | 1.5–3× when short relations are common | Moderate: needs exact confirmation to avoid numeric pitfalls. | Medium? |
| Switch to CP-SAT/MaxSAT backend | Replace CBC with a modern SAT/MaxSAT solver that learns clauses natively. | 2–5× from better branching/learning (problem-size dependent) | Low-to-moderate: integration work; correctness fine if encoding matches current model. | Medium-Hard? |

## Tried and not helpful (so far)
- Greedy Kraft-style warm start: seeding CBC with a collision-free incumbent showed no measurable gain up to N≈29 (sequential runs to 25 and partial to 30 were equal or slightly slower).
- Modular-filtered collision oracle: adding modular prefilters (multiple moduli, cached reachability) ahead of the exact meet-in-the-middle did not meaningfully speed up runs through N=30; the exact check remains the bottleneck.

## Outputs
- Certificates: JSON files (e.g., `certificates/R_36.json`) with fields:
  - `N`, `size`, `solution` (one optimal set),
  - `verified_no_relation` (exact witness check),
  - `runtime_seconds` (None for monotone-extended certificates),
  - `monotone_extension_from` (source N if the cert was produced by monotone extension),
  - `optimality_proved_no_larger` (True if `kissat` UNSAT or the size k+1 feasibility check fails; False if a size k+1 solution is found; None when no check was attempted),
  - `cnf`, `proof` (paths to CNF and DRAT proof if produced).
- CNF/DRAT: when `--prove-optimal` and `kissat` are available, files
  `cnf_N{N}_ge_{k+1}.cnf` and `cnf_N{N}_ge_{k+1}.drat` are written alongside the cert.

## Verification (trusting `kissat`)
1) Witness: run the exact check on `solution` (automatic with `--verify`; anyone can re-run
   an independent checker).
2) Optimality: rerun `kissat cnf proof`; return code 20 means UNSAT, confirming no set of size k+1 exists under the recorded collision cuts. If `kissat` is absent, the solver’s CBC feasibility check for size k+1 drives the `optimality_proved_no_larger` flag (no DRAT proof).

Why UNSAT on a subset of constraints is sufficient: the CNF we build uses only the collision cuts encountered during solving, plus the size ≥ k+1 cardinality constraint. This is a weaker formula than the full “no collisions” condition. If the weaker formula is UNSAT, adding more constraints (the remaining collisions) cannot make it satisfiable. Therefore an UNSAT result on this subset is a sound proof that no collision-free set of size k+1 exists. A SAT result does not imply feasibility for the full problem, so optimality_proved remains False in that case.
