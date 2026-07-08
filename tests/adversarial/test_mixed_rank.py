"""Tests for the mixed-rank adversarial key-set generator.

Rank assertions here recompute ranks with the authoritative
``calc_ranks_multi_dims`` instead of trusting the generator's own
verification — the acceptance bar is "verified against the rank
calculation, not assumed".
"""

import pytest

from adversarial import (
    ProfileMismatch,
    collide_then_diversify,
    find_key_with_profile,
    front_loaded_spine,
    parallel_blocks,
    staircase,
    verify_profiles,
)
from adversarial.mixed_rank import KEY_BASE
from gplus_trees.base import ItemData, LeafItem
from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.utils import calc_ranks, calc_ranks_multi_dims
from gplus_trees.tree_stats import gtree_stats_

K = 4


def ranks_of(key: int, dims: int) -> list[int]:
    """Authoritative per-dimension ranks of a single key."""
    rank_lists = calc_ranks_multi_dims([key], K, dims)
    return [rank_lists[dim][0] for dim in range(dims)]


class TestFindKeyWithProfile:
    def test_found_key_matches_requested_profile(self):
        for profile in [(2,), (1, 2), (1, 1, 3), (4,)]:
            key = find_key_with_profile(profile, K, start_key=1)
            assert ranks_of(key, len(profile)) == list(profile)

    def test_respects_start_key_and_wildcard(self):
        key = find_key_with_profile((None, 2), K, start_key=KEY_BASE)
        assert key >= KEY_BASE
        assert ranks_of(key, 2)[1] == 2

    def test_deterministic(self):
        args = ((1, 1, 2), K)
        assert find_key_with_profile(*args, start_key=1) == find_key_with_profile(*args, start_key=1)

    def test_raises_when_range_exhausted(self):
        with pytest.raises(LookupError):
            find_key_with_profile((12,), K, start_key=KEY_BASE, end_key=KEY_BASE + 50)

    def test_rejects_reserved_key_zero(self):
        with pytest.raises(ValueError):
            find_key_with_profile((1,), K, start_key=0)


class TestVerifyProfiles:
    def test_detects_mismatch(self):
        key = find_key_with_profile((1,), K, start_key=1)
        with pytest.raises(ProfileMismatch):
            verify_profiles([key], [(2,)], K)

    def test_accepts_matching_and_wildcard(self):
        key = find_key_with_profile((2, 1), K, start_key=1)
        verify_profiles([key], [(2, None)], K)


class TestStaircase:
    def test_ranks_and_ordering_verified_independently(self):
        depth, spine, pile = 3, 2, 12
        key_set = staircase(K, depth, spine, pile_size=pile)
        keys = list(key_set.keys)
        assert keys == sorted(set(keys))
        assert len(keys) == depth * spine + pile

        rank_lists = calc_ranks_multi_dims(keys, K, depth)
        for dim_idx in range(depth):  # spine of dimension dim_idx+1
            lo = dim_idx * spine
            spine_ranks = [rank_lists[dim_idx][i] for i in range(lo, lo + spine)]
            assert spine_ranks == [3, 2], f"dimension {dim_idx + 1} spine must descend toward the pile"
            for outer_dim in range(dim_idx):
                assert all(rank_lists[outer_dim][i] == 1 for i in range(lo, lo + spine))
        pile_lo = depth * spine
        for dim_idx in range(depth):
            assert all(rank_lists[dim_idx][i] == 1 for i in range(pile_lo, len(keys)))
        assert max(keys[:pile_lo]) < min(keys[pile_lo:]), "pile must be the rightmost key block"

    def test_shrink_pile(self):
        key_set = staircase(K, 2, 1, pile_size=12)
        smaller = key_set.shrink_pile(5)
        assert smaller.keys == key_set.keys[: 2 + 5]
        assert smaller.profiles == key_set.profiles[: 2 + 5]
        assert smaller.params["pile_size"] == 5
        smaller.verify()
        with pytest.raises(ValueError):
            key_set.shrink_pile(0)
        with pytest.raises(ValueError):
            key_set.shrink_pile(13)

    def test_front_loaded_spine_only_in_dimension_one(self):
        depth, spine, pile = 3, 2, 8
        key_set = front_loaded_spine(K, depth, spine, pile_size=pile)
        assert len(key_set.keys) == spine + pile
        rank_lists = calc_ranks_multi_dims(list(key_set.keys), K, depth)
        assert [rank_lists[0][0], rank_lists[0][1]] == [3, 2]
        for dim_idx in range(depth):
            assert all(rank_lists[dim_idx][i] == 1 for i in range(spine, spine + pile))


class TestParallelBlocks:
    def test_blocks_and_separators(self):
        depth, n_blocks, block_size = 2, 3, 6
        key_set = parallel_blocks(K, depth, n_blocks, block_size)
        keys = list(key_set.keys)
        assert len(keys) == n_blocks * block_size + (n_blocks - 1)
        assert keys == sorted(set(keys))

        rank_lists = calc_ranks_multi_dims(keys, K, depth)
        pos = 0
        for block_idx in range(n_blocks):
            for dim_idx in range(depth):
                assert all(rank_lists[dim_idx][i] == 1 for i in range(pos, pos + block_size))
            pos += block_size
            if block_idx < n_blocks - 1:
                assert rank_lists[0][pos] == 2, "separator must split blocks at dimension 1"
                pos += 1


class TestCollideThenDiversify:
    def test_bottom_spine_below_full_pile(self):
        depth, spine, pile = 2, 3, 8
        key_set = collide_then_diversify(K, depth, spine, pile_size=pile)
        keys = list(key_set.keys)
        assert len(keys) == spine + pile

        rank_lists = calc_ranks_multi_dims(keys, K, depth + 1)
        for dim_idx in range(depth):
            assert all(rank_lists[dim_idx][i] == 1 for i in range(len(keys)))
        bottom = [rank_lists[depth][i] for i in range(spine)]
        assert bottom == [4, 3, 2], "bottom spine must descend toward the bulk"


class TestTreeIntegration:
    def test_staircase_drives_conversion_depth(self):
        """The generated keys must actually reach the engineered nesting depth."""
        depth = 2
        key_set = staircase(K, depth, 1, pile_size=20)
        tree_cls, _, _, _ = make_gkplustree_classes(K)
        tree = tree_cls(l_factor=1.0)
        keys = list(key_set.keys)
        for item, rank in zip(
            (LeafItem(ItemData(key, f"value_{key}")) for key in keys),
            calc_ranks(keys, K),
            strict=True,
        ):
            tree, _, _ = tree.insert(item, rank)

        assert tree.get_max_dim() >= depth + 1, "pile must convert through every engineered dimension"
        stats = gtree_stats_(tree)
        assert stats.real_item_count == len(keys)
        # Deliberately not asserted: all_leaf_values_present — known checker
        # false positive on converted inner trees (c8ed7ee follow-up issue).
        for flag in ("is_heap", "is_search_tree", "leaf_keys_in_order", "linked_leaf_nodes"):
            assert getattr(stats, flag), f"invariant {flag} violated"
