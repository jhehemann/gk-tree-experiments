"""Mixed-rank adversarial key-set generator for Gᵏ⁺ height probes.

A Gᵏ⁺-tree defends against rank-1 pile-ups by converting an oversized
G-node set into a tree of the next-deeper dimension. Reaching depth d
therefore needs rank-1 collisions per dimension, while height *within* a
dimension needs rank diversity — two demands that compete for the same
keys. The shape families here occupy the middle ground between the two
pure strategies: each combines a fat rank-1 pile (drives conversion depth)
with thin engineered rank spines (drive per-dimension G-node paths).

Every constructor verifies the finished key set against the authoritative
rank calculation (``calc_ranks_multi_dims``) before returning it; the
search loop itself is never trusted.

Key-space layout: each engineered key is searched inside its own slot of
width ``SLOT_STRIDE``, slots ascending in the order the keys must appear.
Rank profiles only constrain *which* keys qualify; the slot layout
controls their relative order, which is what makes a rank-1 run
contiguous (one G-node) and places spine keys on the G-node path to it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

from gplus_trees.g_k_plus.utils import calc_ranks_multi_dims
from gplus_trees.utils import calc_rank_from_digest, get_group_size

# One entry per dimension, starting at dimension 1; None = unconstrained.
Profile = tuple[int | None, ...]

# Slots start above every frozen fixture key (those grow upward from 1)
# and stay far below the 2**60 ceiling of the random-baseline key draw:
# even hundreds of slots keep keys < 2**53.
KEY_BASE = 1 << 44
SLOT_STRIDE = 1 << 44


class ProfileMismatch(AssertionError):
    """A constructed key's recomputed ranks contradict its required profile."""


def find_key_with_profile(
    profile: Sequence[int | None],
    k: int,
    *,
    start_key: int,
    end_key: int | None = None,
    max_candidates: int = 500_000_000,
) -> int:
    """Smallest key ≥ ``start_key`` whose per-dimension ranks match ``profile``.

    ``profile[i]`` is the required rank in dimension i+1; None leaves that
    dimension unconstrained. Each dimension's digest is computed
    independently by domain separation, mirroring the authoritative
    ``gplus_trees.utils.get_digest``: dimension d hashes ⟨d⟩ ‖ key, a
    32-byte big-endian dimension tag prepended to the key (ADR-0004 / paper
    ADR-002). Dimensions are not chained, so an unconstrained dimension is
    skipped outright. Expected search cost is dominated by the profile's
    largest rank r (≈ k^(r-1) candidates) and its rank-1 prefix length j
    (≈ (k/(k-1))^j survival factor).

    Raises LookupError when ``end_key`` or ``max_candidates`` is exhausted.
    """
    if start_key < 1:
        raise ValueError("start_key must be ≥ 1 (0 is reserved as the non-existing split key)")
    group_size = get_group_size(k)
    dim_bytes = [d.to_bytes(32, "big") for d in range(1, len(profile) + 1)]
    sha = hashlib.sha256
    # Hot-loop shortcut for the dominant rank-1 checks: rank 1 means fewer
    # than group_size trailing zero bits, i.e. the digest's last byte has a
    # set bit inside the low group_size bits. Only valid for group_size ≤ 8
    # (k ≤ 256); rarer ranks and larger k stay on the authoritative
    # calc_rank_from_digest. Found keys are re-verified against the
    # authoritative calculation by verify_profiles either way.
    low_mask = (1 << group_size) - 1 if group_size <= 8 else 0
    key = start_key
    for _ in range(max_candidates):
        if end_key is not None and key >= end_key:
            break
        key_bytes = abs(key).to_bytes(32, "big")
        matched = True
        for i, want in enumerate(profile):
            if want is None:
                continue
            # Domain separation: dimension i+1's digest is H(⟨i+1⟩ ‖ key),
            # independent of every other dimension (ADR-0004 / paper ADR-002).
            digest = sha(dim_bytes[i] + key_bytes).digest()
            if want == 1 and low_mask:
                if digest[-1] & low_mask:
                    continue
                matched = False
                break
            if calc_rank_from_digest(digest, group_size) != want:
                matched = False
                break
        if matched:
            return key
        key += 1
    raise LookupError(
        f"no key matching profile {tuple(profile)} in [{start_key}, {end_key}) after {key - start_key} candidates"
    )


def verify_profiles(keys: Sequence[int], profiles: Sequence[Profile], k: int) -> None:
    """Check every key's recomputed ranks against its required profile.

    Recomputes with the authoritative ``calc_ranks_multi_dims`` — this is
    the independent check that lets callers treat generated key sets as
    ground truth rather than trusting the search loop.
    """
    if len(keys) != len(profiles):
        raise ValueError("keys and profiles must align")
    dims = max(len(p) for p in profiles)
    rank_lists = calc_ranks_multi_dims(list(keys), k, dims)
    for idx, profile in enumerate(profiles):
        for dim_idx, want in enumerate(profile):
            got = rank_lists[dim_idx][idx]
            if want is not None and got != want:
                raise ProfileMismatch(
                    f"key {keys[idx]} (index {idx}): dimension {dim_idx + 1} has rank {got}, profile requires {want}"
                )


@dataclass(frozen=True)
class MixedRankKeySet:
    """A verified adversarial key set plus the rank profiles it satisfies.

    ``keys`` are strictly ascending and meant to be inserted in this order
    (history independence makes the order irrelevant to the resulting
    tree; ascending keeps campaigns comparable). ``profiles[i]`` is the
    constraint ``keys[i]`` was searched for and verified against.
    """

    family: str
    label: str
    k: int
    depth: int
    params: dict
    keys: tuple[int, ...]
    profiles: tuple[Profile, ...]

    def verify(self) -> None:
        verify_profiles(self.keys, self.profiles, self.k)

    def shrink_pile(self, pile_size: int) -> MixedRankKeySet:
        """Same shape with the rank-1 pile truncated to its smallest ``pile_size`` keys.

        Valid because the pile keys are the largest keys of the set and all
        share one profile: dropping keys from the top preserves ordering,
        the pile's contiguity, and every remaining profile. Lets one
        expensive spine search serve a whole n-sweep.
        """
        old_size = self.params.get("pile_size")
        if old_size is None:
            raise ValueError(f"family {self.family!r} has no shrinkable pile")
        if not 0 < pile_size <= old_size:
            raise ValueError(f"pile_size must be in (0, {old_size}], got {pile_size}")
        cut = old_size - pile_size
        keep = len(self.keys) - cut
        return replace(
            self,
            keys=self.keys[:keep],
            profiles=self.profiles[:keep],
            params={**self.params, "pile_size": pile_size},
        )


def _slot_starts():
    """Ascending search-slot start keys, one slot per engineered key group."""
    slot = 0
    while True:
        yield KEY_BASE + slot * SLOT_STRIDE
        slot += 1


def _find_block(count: int, profile: Profile, k: int, slot_start: int) -> list[int]:
    """``count`` consecutive-slot keys matching one shared profile."""
    keys = []
    next_start = slot_start
    for _ in range(count):
        key = find_key_with_profile(profile, k, start_key=next_start, end_key=slot_start + SLOT_STRIDE)
        keys.append(key)
        next_start = key + 1
    return keys


def _build(
    family: str, label: str, k: int, depth: int, params: dict, entries: list[tuple[int, Profile]]
) -> MixedRankKeySet:
    """Assemble, order-check, and verify a key set from (key, profile) pairs."""
    keys = tuple(key for key, _ in entries)
    if list(keys) != sorted(set(keys)):
        raise ValueError("slot layout produced non-ascending or duplicate keys")
    key_set = MixedRankKeySet(
        family=family,
        label=label,
        k=k,
        depth=depth,
        params=params,
        keys=keys,
        profiles=tuple(profile for _, profile in entries),
    )
    key_set.verify()
    return key_set


def staircase(
    k: int,
    depth: int,
    spine_heights: int | Sequence[int],
    pile_size: int,
    *,
    family: str = "staircase",
    label: str | None = None,
) -> MixedRankKeySet:
    """Nested-spine staircase: engineered G-node path above the pile in every dimension.

    Per dimension j ≤ depth, ``spine_heights[j-1]`` spine keys carry ranks
    s+1 … 2 at dimension j (descending as keys ascend, so each spine node's
    right subtree leads toward the pile) and rank 1 in every dimension
    before j (so they survive inside all outer converted sets). The pile —
    ``pile_size`` keys, rank 1 through all ``depth`` dimensions — is the
    rightmost block; every carried entry of an outer dimension sorts left
    of the inner spine, so it can branch off but never interpose on the
    path to the pile.

    A uniform ``spine_heights=s`` probes multiplicative growth
    (≈ (s+1)·depth); front-loaded profiles probe one-off additive gains.
    """
    if isinstance(spine_heights, int):
        spine_heights = [spine_heights] * depth
    if len(spine_heights) != depth:
        raise ValueError(f"spine_heights must have one entry per dimension, got {len(spine_heights)} for depth {depth}")
    if label is None:
        label = f"{family}_s{max(spine_heights)}"

    slots = _slot_starts()
    entries: list[tuple[int, Profile]] = []
    for dim_idx, height in enumerate(spine_heights):  # dim_idx 0 = dimension 1
        for rank in range(height + 1, 1, -1):
            profile: Profile = (1,) * dim_idx + (rank,)
            key = find_key_with_profile(profile, k, start_key=next(slots))
            entries.append((key, profile))
    pile_profile: Profile = (1,) * depth
    entries += [(key, pile_profile) for key in _find_block(pile_size, pile_profile, k, next(slots))]

    params = {"spine_heights": list(spine_heights), "pile_size": pile_size}
    return _build(family, label, k, depth, params, entries)


def front_loaded_spine(k: int, depth: int, spine_height: int, pile_size: int) -> MixedRankKeySet:
    """Staircase variant with the whole spine budget in dimension 1 only.

    Probes the additive combination: one dimension of engineered height on
    top of a pure pile-up below it — the sum shape O(spine + depth), as
    opposed to the uniform staircase's product shape.
    """
    heights = [spine_height] + [0] * (depth - 1)
    return staircase(k, depth, heights, pile_size, family="front", label=f"front_s{spine_height}")


def collide_then_diversify(k: int, depth: int, bottom_spine_height: int, pile_size: int) -> MixedRankKeySet:
    """Pure pile-up to ``depth``, then an engineered spine at dimension depth+1.

    All keys collide (rank 1) through dimensions 1…depth; at the bottom
    dimension the spine keys carry ranks s+1 … 2 left of the bulk, whose
    ranks there are unconstrained (hash-random). Probes how tall a single
    dimension can be driven *at the bottom of* a maximal pile-up — the
    issue's "collide to depth j, then diversify" staircase profile.
    """
    slots = _slot_starts()
    entries: list[tuple[int, Profile]] = []
    for rank in range(bottom_spine_height + 1, 1, -1):
        profile: Profile = (1,) * depth + (rank,)
        key = find_key_with_profile(profile, k, start_key=next(slots))
        entries.append((key, profile))
    bulk_profile: Profile = (1,) * depth
    entries += [(key, bulk_profile) for key in _find_block(pile_size, bulk_profile, k, next(slots))]

    params = {"bottom_spine_height": bottom_spine_height, "pile_size": pile_size}
    return _build("dive", f"dive_s{bottom_spine_height}", k, depth, params, entries)


def parallel_blocks(k: int, depth: int, n_blocks: int, block_size: int, *, separator_rank: int = 2) -> MixedRankKeySet:
    """Independent rank-1 piles separated at dimension 1 — the parallelism control.

    ``n_blocks`` piles of ``block_size`` keys (rank 1 through ``depth``),
    kept apart by single keys of rank ``separator_rank`` at dimension 1.
    Each pile converts and recurses on its own; nesting-aware height is a
    maximum over branches, so this family should buy no height over a
    single pile of ``block_size`` — measuring that is its purpose.
    """
    slots = _slot_starts()
    entries: list[tuple[int, Profile]] = []
    block_profile: Profile = (1,) * depth
    separator_profile: Profile = (separator_rank,)
    for block_idx in range(n_blocks):
        entries += [(key, block_profile) for key in _find_block(block_size, block_profile, k, next(slots))]
        if block_idx < n_blocks - 1:
            key = find_key_with_profile(separator_profile, k, start_key=next(slots))
            entries.append((key, separator_profile))

    params = {"n_blocks": n_blocks, "block_size": block_size, "separator_rank": separator_rank}
    return _build("blocks", f"blocks_b{n_blocks}", k, depth, params, entries)
