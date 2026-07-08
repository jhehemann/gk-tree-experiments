"""Adversarial key-set construction for Gᵏ⁺ worst-case structure probes.

Produces key sets with engineered per-dimension rank profiles — beyond the
uniform rank-1 pile-up of the frozen fixtures (see
``benchmarks/adversarial_keys/PROVENANCE.md``) — and verifies every
constructed set against the authoritative rank calculation before handing
it out. The uniform rank-1 reference search stays in
``scripts/find_adversarial_keys.py``; this package covers the mixed-rank
shape families used by the height-bound probes.
"""

from adversarial.measure import analyze_chi, analyze_structure
from adversarial.mixed_rank import (
    MixedRankKeySet,
    ProfileMismatch,
    bushy_pile_tree,
    collide_then_diversify,
    find_key_with_profile,
    front_loaded_spine,
    parallel_blocks,
    staircase,
    verify_profiles,
)

__all__ = [
    "MixedRankKeySet",
    "ProfileMismatch",
    "analyze_chi",
    "analyze_structure",
    "bushy_pile_tree",
    "collide_then_diversify",
    "find_key_with_profile",
    "front_loaded_spine",
    "parallel_blocks",
    "staircase",
    "verify_profiles",
]
