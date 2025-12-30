"""Collision and relation utilities for reciprocal-sum sets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


def lcm_upto(n: int) -> int:
    lcm = 1
    for i in range(1, n + 1):
        lcm = lcm * i // math.gcd(lcm, i)
    return lcm


@dataclass
class Relation:
    plus: List[int]
    minus: List[int]


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


def verify_relation_free(elements: Sequence[int]) -> bool:
    """Exact check that the given set has no non-trivial signed relation."""
    return find_relation(elements) is None
