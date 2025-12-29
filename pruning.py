"""P-adic pruning helpers.

These routines conservatively identify integers in {1..N} that can be fixed
up-front because they provably cannot participate in any signed zero-sum
relation of reciprocals. They also detect small “all or none” clusters like
{p, 2p, 3p} for p=11 at N=36, where any relation would have to use the whole
cluster with the same sign.

Use the returned `safe_numbers` to fix variables to 1 (they never create
collisions) and optionally exploit the `all_or_none_groups` to shrink search
spaces when enumerating candidate collisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass
class PruningResult:
    """Output of the p-adic exclusion engine."""

    safe_numbers: Set[int]
    all_or_none_groups: List[List[int]]
    by_rule: Dict[str, Set[int]]


def _primes_up_to(n: int) -> List[int]:
    """Simple sieve-less prime list; n is small (<= few hundred) in practice."""
    primes: List[int] = []
    for cand in range(2, n + 1):
        is_prime = True
        for p in primes:
            if p * p > cand:
                break
            if cand % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(cand)
    return primes


def _valuation(n: int, p: int) -> int:
    """Return v_p(n): largest k with p^k | n."""
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def _forced_by_unique_high_power(N: int) -> Set[int]:
    """Numbers that are the sole carrier of the highest p-adic exponent.

    For any prime p, if among the still-eligible numbers there is a single
    integer with maximal v_p, that integer cannot appear in a signed relation:
    its contribution to the lcm-weighted sum would be ±1 mod p while all other
    terms are 0 mod p. We iterate downward so once the unique top carrier is
    removed, the next level can become unique as well (e.g., 32 then 16 for p=2
    at N=36).
    """
    forced: Set[int] = set()
    primes = _primes_up_to(N)
    for p in primes:
        candidates: Dict[int, int] = {
            i: _valuation(i, p)
            for i in range(1, N + 1)
            if i not in forced and i % p == 0
        }
        if not candidates:
            continue
        while candidates:
            top_exp = max(candidates.values())
            top_carriers = [i for i, v in candidates.items() if v == top_exp]
            if len(top_carriers) == 1:
                only = top_carriers[0]
                forced.add(only)
                candidates.pop(only)
            else:
                break
    return forced


def _zero_sum_patterns_mod_p(
    p: int, multipliers: Sequence[int]
) -> List[Tuple[int, ...]]:
    """All sign patterns s_k in {-1,0,1} with Σ s_k/k ≡ 0 mod p (non-trivial)."""
    inv_cache = {k: pow(k, -1, p) for k in multipliers}
    patterns: List[Tuple[int, ...]] = []
    base = 3 ** len(multipliers)
    for mask in range(1, base):  # skip all-zero
        tmp = mask
        total = 0
        signs: List[int] = []
        for k in multipliers:
            tmp, digit = divmod(tmp, 3)
            if digit == 0:
                signs.append(0)
                continue
            coef = 1 if digit == 1 else -1
            signs.append(coef)
            total = (total + coef * inv_cache[k]) % p
        if total == 0:
            patterns.append(tuple(signs))
    return patterns


def _forced_by_modular_obstruction(
    N: int, already_forced: Set[int]
) -> Tuple[Set[int], List[List[int]]]:
    """Primes with p^2 > N: use modular sums to certify non-involvement.

    For such primes, any relation using a multiple kp contributes a factor
    L/(kp) ≡ (lcm_without_p / k) mod p. If no non-zero sign pattern on the
    available multiples makes the mod-p sum vanish, then none of those multiples
    can appear in a relation and they are safe. If the only solutions are “all
    +1”/“all -1” across every multiple, record an all-or-none group to shrink
    search when enumerating collisions.
    """
    forced: Set[int] = set()
    groups: List[List[int]] = []
    primes = [p for p in _primes_up_to(N) if p * p > N]
    for p in primes:
        multipliers = [k for k in range(1, N // p + 1) if k * p not in already_forced]
        if not multipliers:
            continue
        patterns = _zero_sum_patterns_mod_p(p, multipliers)
        if not patterns:
            forced.update({k * p for k in multipliers})
            continue
        # Detect the simple “all or none” structure: the only solutions are
        # all +1 or all -1 on the full support.
        full_support = tuple(1 for _ in multipliers)
        neg_full_support = tuple(-x for x in full_support)
        pattern_set = set(patterns)
        if pattern_set in ({full_support}, {full_support, neg_full_support}):
            groups.append([k * p for k in multipliers])
    return forced, groups


def compute_p_adic_exclusions(N: int) -> PruningResult:
    """Compute safe numbers and structured clusters for 1..N."""
    forced_high = _forced_by_unique_high_power(N)
    forced_mod, groups = _forced_by_modular_obstruction(N, forced_high)
    safe = forced_high | forced_mod
    return PruningResult(
        safe_numbers=safe,
        all_or_none_groups=groups,
        by_rule={"unique_high_power": forced_high, "modular": forced_mod},
    )
