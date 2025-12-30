#!/usr/bin/env python3
"""
Plot and log certificate results.

This script reads the JSON certificates emitted by `solver.py`, prints a small
summary table, and saves a couple of plots:
- R(N) progression from the certificates, overlaid on the known prefix from
  tests/test_solver.py.
- Runtime and density trends, with proof status shown in colors.
"""

from __future__ import annotations

import argparse
import ast
import json
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
    for path in sorted(cert_dir.glob("R_*.json"), key=lambda p: int(p.stem.split("_")[1])):
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


def print_summary(certs: Sequence[Certificate], known_seq: Sequence[int]) -> None:
    if not certs:
        print("No certificates found.")
        return
    proved = sum(1 for c in certs if c.optimality_proved)
    unproved = sum(1 for c in certs if c.optimality_proved is False)
    unknown = len(certs) - proved - unproved
    print(f"Loaded {len(certs)} certificates spanning N={certs[0].N}..{certs[-1].N}.")
    print(f"Optimality proofs: {proved} proved, {unproved} disproved, {unknown} unknown/missing.")
    missing_known = [c for c in certs if c.N <= len(known_seq) and c.size != known_seq[c.N - 1]]
    if missing_known:
        mismatches = ", ".join(f"N={c.N} (got {c.size}, expected {known_seq[c.N-1]})" for c in missing_known)
        print(f"Known-sequence mismatches: {mismatches}")
    else:
        print("Certificates match the known prefix (where available).")
    print("N  size  runtime(s)  proved  path")
    for cert in certs:
        runtime = (
            f"{cert.runtime_seconds:.2f}" if cert.runtime_seconds is not None else "—"
        )
        status = {True: "yes", False: "no"}.get(cert.optimality_proved, "?")
        print(f"{cert.N:>2} {cert.size:>5} {runtime:>11}  {status:>5}  {cert.path}")


def save_fig(fig: plt.Figure, out_dir: Path, name: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig_path = out_dir / f"{name}.{fmt}"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved {fig_path}")
    plt.close(fig)


def plot_progression(
    certs: Sequence[Certificate],
    known_seq: Sequence[int],
    out_dir: Path,
    formats: Iterable[str],
) -> None:
    ns = [c.N for c in certs]
    sizes = [c.size for c in certs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ns, sizes, marker="o", label="Certificates", color="#2a9d8f")

    if known_seq:
        known_ns = list(range(1, len(known_seq) + 1))
        ax.plot(
            known_ns,
            known_seq,
            linestyle="--",
            marker=".",
            color="#264653",
            label="Known Sequence",
        )
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
                label="Mismatch",
                zorder=5,
            )

    ax.set_xlabel("N")
    ax.set_ylabel("R(N)")
    ax.set_title("Certificate progression vs known values")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
            colors.append("#2a9d8f")
        elif cert.optimality_proved is False:
            colors.append("#e76f51")
        else:
            colors.append("#f4a261")

    fig, (ax_rt, ax_density) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax_rt.bar(ns, runtimes, color=colors, width=0.8)
    ax_rt.set_ylabel("Runtime (s)")
    ax_rt.set_yscale("symlog")
    ax_rt.grid(True, axis="y", alpha=0.3)
    ax_rt.set_title("Runtime and density trends (color = proof status)")

    legend_elements = [
        plt.Line2D([0], [0], color="#2a9d8f", lw=6, label="proved optimal"),
        plt.Line2D([0], [0], color="#e76f51", lw=6, label="refuted/failed proof"),
        plt.Line2D([0], [0], color="#f4a261", lw=6, label="no proof (yet)"),
    ]
    ax_rt.legend(handles=legend_elements, fontsize="small")

    ax_density.plot(ns, densities, marker="o", color="#264653")
    ax_density.fill_between(ns, densities, color="#264653", alpha=0.08)
    ax_density.set_xlabel("N")
    ax_density.set_ylabel("Density = R(N)/N")
    ax_density.grid(True, alpha=0.3)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    certs = load_certificates(args.cert_dir)
    known_seq = load_known_sequence(args.known_sequence_file)

    print_summary(certs, known_seq)
    if not certs:
        return

    plot_progression(certs, known_seq, args.out_dir, args.formats)
    plot_runtime_and_density(certs, args.out_dir, args.formats)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
