#!/usr/bin/env python3
"""Refresh the README's certified R(N) section from JSON certificates.

The README section is bounded by markers:
  <!-- CERT_TABLE:START -->
  ... auto-generated content ...
  <!-- CERT_TABLE:END -->

Usage examples:
  python update_readme_cert_table.py --cert-dir certificates --readme README.md
  python update_readme_cert_table.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


START_MARKER = "<!-- CERT_TABLE:START -->"
END_MARKER = "<!-- CERT_TABLE:END -->"


@dataclass
class CertEntry:
    N: int
    size: int


def load_certificates(cert_dir: Path) -> List[CertEntry]:
    """Load certificates sorted by N.

    Only the fields needed for the table are read to avoid heavier imports.
    """

    if not cert_dir.exists():
        raise FileNotFoundError(f"Certificate directory does not exist: {cert_dir}")

    entries: List[CertEntry] = []
    for path in sorted(cert_dir.glob("R_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with path.open() as fh:
            data = json.load(fh)
        entries.append(CertEntry(N=int(data["N"]), size=int(data["size"])))

    if not entries:
        raise RuntimeError(f"No certificates found in {cert_dir}")

    return entries


def _format_row(left: CertEntry, right: CertEntry | None) -> str:
    if right:
        return f"| {left.N} | {left.size} | {right.N} | {right.size} |"
    return f"| {left.N} | {left.size} |   |   |"


def format_table(entries: Sequence[CertEntry]) -> str:
    header = "| N | R(N) | N | R(N) |"
    divider = "|---|---|---|---|"
    rows = [header, divider]
    for i in range(0, len(entries), 2):
        left = entries[i]
        right = entries[i + 1] if i + 1 < len(entries) else None
        rows.append(_format_row(left, right))
    return "\n".join(rows)


def format_sequence(entries: Sequence[CertEntry]) -> str:
    values = ", ".join(str(entry.size) for entry in entries)
    return f"(R(1..{entries[-1].N})): `{values}`"


def render_chunk(entries: Sequence[CertEntry]) -> str:
    lines: List[str] = [
        "Derived from the `size` field in `certificates/R_*.json` (current repo state):",
        "",
        format_table(entries),
        "",
        format_sequence(entries),
        "",
    ]
    return "\n".join(lines)


def replace_chunk(readme_text: str, new_chunk: str) -> str:
    start_idx = readme_text.find(START_MARKER)
    end_idx = readme_text.find(END_MARKER)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise RuntimeError(
            "Could not find CERT_TABLE markers in README; please add them before running."
        )

    start_insert = start_idx + len(START_MARKER)
    before = readme_text[:start_insert]
    after = readme_text[end_idx:]
    new_body = "\n" + new_chunk.rstrip() + "\n"
    return before + new_body + after


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cert-dir",
        type=Path,
        default=Path("certificates"),
        help="Directory containing R_*.json certificate files.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to the README to update.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if an update would be made (no files are changed).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    entries = load_certificates(args.cert_dir)
    chunk = render_chunk(entries)

    readme_text = args.readme.read_text()
    updated_text = replace_chunk(readme_text, chunk)

    if args.check:
        if updated_text != readme_text:
            print("README is out of date; rerun without --check to update.")
            return 1
        print("README is up to date.")
        return 0

    if updated_text == readme_text:
        print("README already up to date.")
        return 0

    args.readme.write_text(updated_text)
    min_n, max_n = entries[0].N, entries[-1].N
    print(
        f"Updated {args.readme} with {len(entries)} certificates (N={min_n}..{max_n})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
