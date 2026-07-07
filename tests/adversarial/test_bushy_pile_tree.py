"""Tests for the bushy pile-tree generator and the χ measurement.

The decisive assertions:
- profiles verify independently against ``calc_ranks_multi_dims``;
- L08 Part 4's exact worked example (and the ``bushy`` generalization at
  depth 2 / branching 2) reproduces the measured nesting-aware height **6**
  with conversion multiplicity **χ = 2** (the file's headline number);
- χ stays at ``branching`` and the height obeys ``H_d = B·H_{d-1} + B`` as
  depth grows — i.e. the branching genuinely nests (ancestor→descendant,
  inner heights SUM), instead of degenerating into side-by-side siblings;
- the ``parallel_blocks`` sibling control measures χ = 1, confirming the χ
  metric actually distinguishes nesting from siblings.
"""

import pytest

from adversarial import analyze_chi, analyze_structure, bushy_pile_tree, parallel_blocks
from adversarial.mixed_rank import KEY_BASE, find_key_with_profile, verify_profiles
from gplus_trees.base import ItemData, LeafItem
from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.utils import calc_ranks, calc_ranks_multi_dims

K = 4


def build_tree(keys, k, l_factor):
    tree_cls, _, _, _ = make_gkplustree_classes(k)
    tree = tree_cls(l_factor=l_factor)
    for item, rank in zip(
        (LeafItem(ItemData(key, f"v{key}")) for key in keys),
        calc_ranks(list(keys), k),
        strict=True,
    ):
        tree, _, _ = tree.insert(item, rank)
    return tree


def _l08_part4_key_set(k, l_factor):
    """L08 Part 4's exact CE(k,τ) family, calibrated to the code's threshold.

    Order (≤_U):  a_1 < b_1 < … < b_P < a_2 < … < a_P.
    Profiles: a_1=(2,2); a_{≥2}=(2,1); b_1=(1,2); b_{≥2}=(1,1).
    Pile size P = int(k*l_factor) makes A convert while the dimension-2
    residue (size P-1) sits exactly at the threshold and does not -- L08's
    calibrated hinge, giving dim=2, chi=2, ht=6.
    """
    P = int(k * l_factor)
    stride = 1 << 44
    slot = 0

    def next_slot():
        nonlocal slot
        s = KEY_BASE + slot * stride
        slot += 1
        return s

    entries = [(find_key_with_profile((2, 2), k, start_key=next_slot()), (2, 2))]
    entries.append((find_key_with_profile((1, 2), k, start_key=next_slot()), (1, 2)))
    start = next_slot()
    for _ in range(P - 1):  # b_2 … b_P
        key = find_key_with_profile((1, 1), k, start_key=start, end_key=start + stride)
        entries.append((key, (1, 1)))
        start = key + 1
    start = next_slot()
    for _ in range(P - 1):  # a_2 … a_P
        key = find_key_with_profile((2, 1), k, start_key=start, end_key=start + stride)
        entries.append((key, (2, 1)))
        start = key + 1

    keys = [key for key, _ in entries]
    profiles = [p for _, p in entries]
    assert keys == sorted(set(keys))
    verify_profiles(keys, profiles, k)
    return keys


class TestBushyProfiles:
    def test_profiles_verified_independently(self):
        key_set = bushy_pile_tree(K, depth=3, branching=2, leaf_size=3)
        keys = list(key_set.keys)
        assert keys == sorted(set(keys))
        assert len(keys) == 2**3 * 3  # branching**depth * leaf_size
        # Independent recompute of every per-dimension rank.
        rank_lists = calc_ranks_multi_dims(keys, K, key_set.depth)
        for idx, profile in enumerate(key_set.profiles):
            got = [rank_lists[d][idx] for d in range(key_set.depth)]
            assert got == list(profile)

    def test_branching_three_profiles(self):
        key_set = bushy_pile_tree(K, depth=2, branching=3, leaf_size=3)
        assert len(key_set.keys) == 3**2 * 3
        key_set.verify()  # raises ProfileMismatch on any disagreement
        # Ranks are drawn from {1, …, branching}.
        assert {r for profile in key_set.profiles for r in profile} <= {1, 2, 3}

    def test_rejects_degenerate_params(self):
        with pytest.raises(ValueError):
            bushy_pile_tree(K, depth=2, branching=1, leaf_size=3)
        with pytest.raises(ValueError):
            bushy_pile_tree(K, depth=0, branching=2, leaf_size=3)


class TestL08Part4BaseCase:
    def test_exact_ce_family_reproduces_height_six(self):
        """L08 Part 4's own worked example: dim=2, χ=2, ht=6."""
        keys = _l08_part4_key_set(K, l_factor=1.0)
        tree = build_tree(keys, K, l_factor=1.0)
        h_nested, dim_max, _ = analyze_structure(tree)
        max_chi, max_prod_chi = analyze_chi(tree)
        assert dim_max == 2
        assert h_nested == 6, "L08 Part 4 fixes ht(T(S)) = 6"
        assert max_chi == 2, "one dim-1 skeleton path crosses both converted piles A and B"
        assert max_prod_chi == 2

    def test_bushy_depth2_reproduces_the_mechanism(self):
        """bushy(depth=2, branching=2) is L08 Part 4's χ=2 mechanism."""
        key_set = bushy_pile_tree(K, depth=2, branching=2, leaf_size=3)
        tree = build_tree(key_set.keys, K, l_factor=1.0)
        h_nested, dim_max, _ = analyze_structure(tree)
        max_chi, _ = analyze_chi(tree)
        assert dim_max == 2
        assert h_nested == 6
        assert max_chi == 2


class TestBushyNestingSustains:
    def test_chi_stays_at_branching_and_height_recurs(self):
        """χ = branching sustained + H_d = B·H_{d-1}+B ⇒ genuine nesting."""
        B, leaf = 2, 3
        heights = {}
        for depth in (2, 3, 4, 5):
            key_set = bushy_pile_tree(K, depth=depth, branching=B, leaf_size=leaf)
            tree = build_tree(key_set.keys, K, l_factor=1.0)
            h_nested, dim_max, _ = analyze_structure(tree)
            max_chi, _ = analyze_chi(tree)
            assert dim_max == depth, "piles must convert through every engineered dimension"
            assert max_chi == B, f"χ must stay at branching={B}, not collapse to 1 (siblings)"
            heights[depth] = h_nested
        # Exact height recurrence of nested (not sibling) converted piles.
        for depth in (3, 4, 5):
            assert heights[depth] == B * heights[depth - 1] + B

    def test_branching_three_sustains_chi_three(self):
        B, leaf = 3, 3
        key_set = bushy_pile_tree(K, depth=3, branching=B, leaf_size=leaf)
        tree = build_tree(key_set.keys, K, l_factor=1.0)
        _, dim_max, _ = analyze_structure(tree)
        max_chi, max_prod_chi = analyze_chi(tree)
        assert dim_max == 3
        assert max_chi == B
        assert max_prod_chi >= B  # product form Π χ grows with the branching


class TestChiDistinguishesSiblings:
    def test_parallel_blocks_measure_chi_one(self):
        """The sibling control: side-by-side piles give χ = 1, not ≥ 2.

        Blocks are sized just above the threshold (``K + 1`` vs. threshold
        ``int(K·1.0) = K``) so each converts exactly once, without the deep
        random-rank tail that oversized piles grow — that tail can nest by
        chance and is what the bushy campaign avoids by keeping leaf piles
        below the threshold. Here the three piles are genuine siblings (each
        in its own gap of the dimension-1 separator node), so no single
        skeleton path crosses two converted nodes: χ = 1.
        """
        key_set = parallel_blocks(K, depth=2, n_blocks=3, block_size=K + 1)
        tree = build_tree(key_set.keys, K, l_factor=1.0)
        max_chi, max_prod_chi = analyze_chi(tree)
        assert max_chi == 1, "independent sibling piles must not register as nested (χ≥2)"
        assert max_prod_chi == 1
