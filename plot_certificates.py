#!/usr/bin/env python3
"""
Plot and log certificate results.

This script reads the JSON certificates emitted by `solver.py`, prints a small
summary table, and saves a couple of plots:
- R(N) progression: Uses a "bullseye" style to overlay discovered values
  atop the known sequence for clear visual comparison and shows configurable
  Bleicher-Erdos upper/lower bounds.
- Runtime and density trends, with proof status shown in colors.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt


@dataclass
class Certificate:
    N: int
    size: int
    runtime_seconds: float | None
    verified: bool | None
    optimality_proved: bool | None
    path: Path


def load_certificates(cert_dir: Path) -> List[Certificate]:
    certs: List[Certificate] = []
    if not cert_dir.exists():
        return []
    # Sort by N (extracted from filename R_N.json)
    for path in sorted(
        cert_dir.glob("R_*.json"), key=lambda p: int(p.stem.split("_")[1])
    ):
        with path.open() as fh:
            data = json.load(fh)
        certs.append(
            Certificate(
                N=int(data["N"]),
                size=int(data["size"]),
                runtime_seconds=data.get("runtime_seconds"),
                verified=data.get("verified_no_relation"),
                optimality_proved=data.get("optimality_proved_no_larger"),
                path=path,
            )
        )
    return certs


def load_known_sequence(test_file: Path) -> Sequence[int]:
    """
    Pull KNOWN_SEQUENCE from tests/test_solver.py without importing solver.py.
    """
    if not test_file.exists():
        return []
    tree = ast.parse(test_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(getattr(t, "id", None) == "KNOWN_SEQUENCE" for t in node.targets):
                try:
                    return list(ast.literal_eval(node.value))  # type: ignore[arg-type]
                except Exception:
                    return []
    return []


def iterated_logs(x: float, count: int) -> List[float]:
    """
    Compute log_i(x) for i=1..count where log_i is i-fold iterated natural log.
    Stops early if a non-positive value would be logged.
    """
    logs: List[float] = []
    current = float(x)
    for _ in range(count):
        if current <= 0:
            break
        current = math.log(current)
        logs.append(current)
    return logs


def compute_bounds(
    ns: Sequence[int], max_iter: int
) -> tuple[List[float | None], List[float | None], List[int | None], List[int | None]]:
    """
    Apply the Bleicher-Erdos bounds, choosing the *best available* parameters
    for each N:
      lower: maximize over k>=4 with log_k N >= k of (N/log N) * prod_{i=3}^k log_i N
      upper: minimize over r>=1 with log_{2r} N >= 1 of
             (1/log 2) * log_r N * (N/log N * prod_{i=3}^r log_i N)

    Returns (lower_vals, upper_vals, k_used, r_used) per N; entries are None when
    conditions fail.
    """
    lower_vals: List[float | None] = []
    upper_vals: List[float | None] = []
    k_used: List[int | None] = []
    r_used: List[int | None] = []
    log_two = math.log(2)

    for n in ns:
        # Precompute iterated logs until they drop non-positive.
        logs = iterated_logs(float(n), max_iter)
        lower_val: float | None = None
        upper_val: float | None = None
        k_best: int | None = None
        r_best: int | None = None

        # Lower bound: pick the largest valid k.
        for k in range(4, len(logs) + 1):
            if logs[0] <= 0 or logs[k - 1] < k:
                continue
            product = 1.0
            for val in logs[2:k]:  # i = 3..k
                product *= val
            candidate = (n / logs[0]) * product
            if lower_val is None or candidate > lower_val:
                lower_val = candidate
                k_best = k

        # Upper bound: pick the smallest valid r.
        for r in range(1, (len(logs) // 2) + 1):
            if logs[0] <= 0 or logs[2 * r - 1] < 1:
                continue
            product = 1.0
            if r >= 3:
                for val in logs[2:r]:  # i = 3..r
                    product *= val
            candidate = (1 / log_two) * logs[r - 1] * (n / logs[0]) * product
            if upper_val is None or candidate < upper_val:
                upper_val = candidate
                r_best = r

        lower_vals.append(lower_val)
        upper_vals.append(upper_val)
        k_used.append(k_best)
        r_used.append(r_best)

    return lower_vals, upper_vals, k_used, r_used


def print_summary(certs: Sequence[Certificate], known_seq: Sequence[int]) -> None:
    if not certs:
        print("No certificates found.")
        return
    proved = sum(1 for c in certs if c.optimality_proved)
    unproved = sum(1 for c in certs if c.optimality_proved is False)
    unknown = len(certs) - proved - unproved
    print(f"Loaded {len(certs)} certificates spanning N={certs[0].N}..{certs[-1].N}.")
    print(
        f"Optimality proofs: {proved} proved, {unproved} disproved, {unknown} missing or unknown."
    )

    missing_known = [
        c for c in certs if c.N <= len(known_seq) and c.size != known_seq[c.N - 1]
    ]
    if missing_known:
        mismatches = ", ".join(
            f"N={c.N} (got {c.size}, expected {known_seq[c.N-1]})"
            for c in missing_known
        )
        print(f"Known sequence mismatches: {mismatches}")
    else:
        print("Certificates match the known prefix (where available).")

    print(f"{'N':>3}  {'size':>5}  {'runtime (s)':>11}  {'proof':>5}  {'path'}")
    print("-" * 60)
    for cert in certs:
        runtime = (
            f"{cert.runtime_seconds:.2f}" if cert.runtime_seconds is not None else "—"
        )
        status = {True: "yes", False: "no"}.get(cert.optimality_proved, "?")
        print(
            f"{cert.N:>3} {cert.size:>5} {runtime:>11}  {status:>5}  {cert.path.name}"
        )


def save_fig(fig: plt.Figure, out_dir: Path, name: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig_path = out_dir / f"{name}.{fmt}"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved {fig_path}")
    plt.close(fig)


def plot_progression(
    certs: Sequence[Certificate],
    known_seq: Sequence[int],
    out_dir: Path,
    formats: Iterable[str],
    max_iter_logs: int,
) -> None:
    ns = [c.N for c in certs]
    sizes = [c.size for c in certs]

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Layer 1: Known Sequence (Background) ---
    # Plotted as large, hollow circles with a thick, faint line.
    if known_seq:
        known_ns = list(range(1, len(known_seq) + 1))
        ax.plot(
            known_ns,
            known_seq,
            linestyle="-",
            linewidth=3,  # Thick line
            color="#34495e",  # Dark gray/blue
            alpha=0.3,  # Very transparent
            marker="o",
            markersize=10,  # Large markers
            markerfacecolor="white",  # Hollow center
            markeredgewidth=1.5,
            label="Known sequence",
            zorder=1,
        )

    # --- Layer 2: Discovered Values (Foreground) ---
    # Plotted as small, solid dots with a thin, sharp line.
    # If they match, the dot fits inside the hollow circle (Bullseye effect).
    ax.plot(
        ns,
        sizes,
        marker="o",
        markersize=4,  # Small markers
        linestyle="-",
        linewidth=1.5,  # Thin line
        color="#2a9d8f",  # Teal
        label="Discovered values",
        zorder=2,
    )

    # --- Layer 3: Mismatches (Highlights) ---
    if known_seq:
        mismatched = [
            (c.N, c.size)
            for c in certs
            if c.N <= len(known_seq) and c.size != known_seq[c.N - 1]
        ]
        if mismatched:
            ax.scatter(
                [n for n, _ in mismatched],
                [s for _, s in mismatched],
                color="#e76f51",
                s=100,
                marker="x",  # Big red X for errors
                linewidth=2,
                label="Mismatched",
                zorder=3,
            )

    lower_bounds, upper_bounds, k_used, r_used = compute_bounds(
        ns, max_iter=max_iter_logs
    )
    lower_points = [(n, val) for n, val in zip(ns, lower_bounds) if val is not None]
    upper_points = [(n, val) for n, val in zip(ns, upper_bounds) if val is not None]
    lower_k_vals = {k for k in k_used if k is not None}
    upper_r_vals = {r for r in r_used if r is not None}
    print(
        f"Bounds coverage: lower {len(lower_points)}/{len(ns)} (k values used: {sorted(lower_k_vals)})"
    )
    print(
        f"Bounds coverage: upper {len(upper_points)}/{len(ns)} (r values used: {sorted(upper_r_vals)})"
    )

    if lower_points:
        ax.plot(
            [n for n, _ in lower_points],
            [v for _, v in lower_points],
            linestyle="--",
            linewidth=1.2,
            color="#8d99ae",
            label="Lower bound (best k per N)",
            zorder=0,
        )

    if upper_points:
        ax.plot(
            [n for n, _ in upper_points],
            [v for _, v in upper_points],
            linestyle=":",
            linewidth=1.2,
            color="#e9c46a",
            label="Upper bound (best r per N)",
            zorder=0,
        )

    ax.set_xlabel("N", fontsize=11)
    ax.set_ylabel("R(N)", fontsize=11)
    ax.set_title(
        "Discovered progression vs. known values", fontsize=13, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    save_fig(fig, out_dir, "progression", formats)


def plot_runtime_and_density(
    certs: Sequence[Certificate],
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    ns = [c.N for c in certs]
    runtimes = [c.runtime_seconds or 0.0 for c in certs]
    densities = [c.size / c.N for c in certs]
    colors = []
    for cert in certs:
        if cert.optimality_proved:
            colors.append("#2a9d8f")  # Teal
        elif cert.optimality_proved is False:
            colors.append("#e76f51")  # Orange/Red
        else:
            colors.append("#f4a261")  # Yellow/Tan

    fig, (ax_rt, ax_density) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Runtime Bar Chart
    ax_rt.bar(ns, runtimes, color=colors, width=0.8, alpha=0.9)
    ax_rt.set_ylabel("Runtime (s)", fontsize=11)
    ax_rt.set_yscale("symlog", linthresh=0.1)
    ax_rt.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_rt.set_title("Runtime and density trends", fontsize=13, fontweight="bold")

    legend_elements = [
        plt.Line2D([0], [0], color="#2a9d8f", lw=4, label="Proved optimal"),
        plt.Line2D([0], [0], color="#e76f51", lw=4, label="Refuted/Failed"),
        plt.Line2D([0], [0], color="#f4a261", lw=4, label="No proof yet"),
    ]
    ax_rt.legend(handles=legend_elements, fontsize="small", loc="upper left")

    # Density Line Chart
    ax_density.plot(ns, densities, marker="o", markersize=4, color="#264653")
    ax_density.fill_between(ns, densities, color="#264653", alpha=0.1)
    ax_density.set_xlabel("N", fontsize=11)
    ax_density.set_ylabel("Density (R(N)/N)", fontsize=11)
    ax_density.grid(True, linestyle="--", alpha=0.3)

    save_fig(fig, out_dir, "runtime_density", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cert-dir",
        type=Path,
        default=Path("certificates"),
        help="Directory containing R_*.json certificate files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Where to save generated plots.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Image formats to save (passed to matplotlib).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    parser.add_argument(
        "--known-sequence-file",
        type=Path,
        default=Path("tests") / "test_solver.py",
        help="Path to the test file containing KNOWN_SEQUENCE.",
    )
    parser.add_argument(
        "--max-iter-logs",
        type=int,
        default=20,
        help="Maximum iterated logs to consider (safety valve; defaults to 20, which is plenty for typical N).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set global style preferences
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    certs = load_certificates(args.cert_dir)
    known_seq = load_known_sequence(args.known_sequence_file)

    print_summary(certs, known_seq)
    if not certs:
        return

    plot_progression(
        certs,
        known_seq,
        args.out_dir,
        args.formats,
        args.max_iter_logs,
    )
    plot_runtime_and_density(certs, args.out_dir, args.formats)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
