"""
Comprehensive tests for rank calculation and hash/digest functions.
Tests ensure consistency between different ways of calculating ranks and hashes.
"""

import hashlib
import unittest

from gplus_trees.base import InternalItem, ItemData
from gplus_trees.g_k_plus.utils import calc_rank, calc_rank_from_group_size, calc_ranks_multi_dims
from gplus_trees.utils import (
    calc_rank_from_digest,
    calc_rank_from_digest_k,
    count_trailing_zero_bits,
    find_keys_for_rank_lists,
    get_group_size,
)


class TestRankHashConsistency(unittest.TestCase):
    """Test consistency between different rank calculation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_keys = [1, 2, 3, 4, 5, 10, 42, 100, 1000, 9999]
        self.test_k_values = [4, 8, 16, 32, 64]  # All powers of 2
        self.test_dimensions = [1, 2, 3, 4, 5]

    def test_calculate_group_size_valid_inputs(self):
        """Test calculate_group_size with valid power-of-2 inputs."""
        expected_results = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10}

        for k, expected in expected_results.items():
            with self.subTest(k=k):
                result = get_group_size(k)
                self.assertEqual(result, expected, f"calculate_group_size({k}) = {result}, expected {expected}")

    def test_calculate_group_size_invalid_inputs(self):
        """Test calculate_group_size with invalid inputs."""
        invalid_k_values = [0, -1, 3, 5, 6, 7, 9, 10, 15, 17]

        for k in invalid_k_values:
            with self.subTest(k=k), self.assertRaises(ValueError):
                get_group_size(k)

    def test_count_trailing_zero_bits(self):
        """Test count_trailing_zero_bits function."""
        test_cases = [
            (b"\x00\x00\x00\x01", 0),  # binary: ...00000001
            (b"\x00\x00\x00\x02", 1),  # binary: ...00000010
            (b"\x00\x00\x00\x04", 2),  # binary: ...00000100
            (b"\x00\x00\x00\x08", 3),  # binary: ...00001000
            (b"\x00\x00\x01\x00", 8),  # binary: ...00000001 00000000
            (b"\x00\x01\x00\x00", 16),  # binary: ...00000001 00000000 00000000
            (b"\x00\x00\x00\x00", 32),  # All zeros in 4-byte sequence
        ]

        for digest, expected_zeros in test_cases:
            with self.subTest(digest=digest.hex()):
                result = count_trailing_zero_bits(digest)
                self.assertEqual(result, expected_zeros)

    def test_calc_rank_from_digest_consistency(self):
        """Test consistency between calc_rank_from_digest and calc_rank_from_digest_k."""
        test_digest = hashlib.sha256(b"test_data").digest()

        for k in self.test_k_values:
            with self.subTest(k=k):
                group_size = get_group_size(k)

                rank1 = calc_rank_from_digest(test_digest, group_size)
                rank2 = calc_rank_from_digest_k(test_digest, k)

                self.assertEqual(
                    rank1,
                    rank2,
                    f"Rank calculation inconsistency for k={k}: "
                    f"calc_rank_from_digest={rank1}, calc_rank_from_digest_k={rank2}",
                )

    def test_calc_rank_for_dim_vs_calc_rank_from_group_size(self):
        """Test consistency between calc_rank_for_dim and calc_rank_from_group_size."""
        for key in self.test_keys:
            for k in self.test_k_values:
                for dim in self.test_dimensions:
                    with self.subTest(key=key, k=k, dim=dim):
                        group_size = get_group_size(k)

                        rank1 = calc_rank(key, k, dim)
                        rank2 = calc_rank_from_group_size(key, group_size, dim)

                        self.assertEqual(
                            rank1,
                            rank2,
                            f"Rank calculation inconsistency for key={key}, k={k}, dim={dim}: "
                            f"calc_rank_for_dim={rank1}, calc_rank_from_group_size={rank2}",
                        )

    def test_internal_item_get_digest_for_dim_consistency(self):
        """Test InternalItem.get_digest_for_dim consistency with utils functions."""
        for key in self.test_keys:
            item_data = ItemData(key)
            internal_item = InternalItem(item_data)

            for dim in self.test_dimensions:
                with self.subTest(key=key, dim=dim):
                    # Get digest using InternalItem method
                    digest1 = internal_item.get_digest_for_dim(dim)

                    # Calculate digest manually using the same pattern as get_digest_for_dim
                    # Start with dimension 1: hash(abs(key) + 1)
                    digest2 = hashlib.sha256(abs(key).to_bytes(32, "big") + (1).to_bytes(32, "big")).digest()

                    # For subsequent dimensions, hash(prev_digest + target_dim)
                    for target_dim in range(2, dim + 1):
                        digest2 = hashlib.sha256(digest2 + target_dim.to_bytes(32, "big")).digest()

                    self.assertEqual(digest1, digest2, f"Digest calculation inconsistency for key={key}, dim={dim}")

    def test_internal_item_digest_caching(self):
        """Test that InternalItem caches digests correctly."""
        key = 42
        item_data = ItemData(key)
        internal_item = InternalItem(item_data)

        # First call should calculate and cache
        digest1 = internal_item.get_digest_for_dim(3)

        # Second call should use cached value
        digest2 = internal_item.get_digest_for_dim(3)

        self.assertEqual(digest1, digest2)
        self.assertIn(3, internal_item.dim_hashes)

        # Lower dimensions should also be cached
        self.assertIn(1, internal_item.dim_hashes)
        self.assertIn(2, internal_item.dim_hashes)

    def test_calc_ranks_multi_dims_consistency(self):
        """Test consistency of calc_ranks_multi_dims with individual calculations."""
        keys = [1, 5, 10, 42]
        k = 16
        dimensions = 3

        # Calculate using multi-dimensional function
        rank_lists = calc_ranks_multi_dims(keys, k, dimensions)

        # Calculate individually for each key and dimension
        for key_idx, key in enumerate(keys):
            for dim_idx in range(dimensions):
                expected_rank = calc_rank(key, k, dim_idx + 1)
                actual_rank = rank_lists[dim_idx][key_idx]

                self.assertEqual(
                    actual_rank,
                    expected_rank,
                    f"Rank inconsistency for key={key}, dim={dim_idx + 1}: "
                    f"expected={expected_rank}, actual={actual_rank}",
                )

    def test_find_keys_for_rank_lists_consistency(self):
        """Test that find_keys_for_rank_lists produces keys with correct ranks."""
        # Create some rank lists
        rank_lists = [
            [1, 2, 1],  # dim 1 ranks
            [1, 1, 3],  # dim 2 ranks
            [2, 1, 1],  # dim 3 ranks
        ]
        k = 8

        # Find keys that match these rank lists
        keys = find_keys_for_rank_lists(rank_lists, k)

        # Verify each key has the correct ranks
        for key_idx, key in enumerate(keys):
            for dim_idx, rank_list in enumerate(rank_lists):
                expected_rank = rank_list[key_idx]
                actual_rank = calc_rank(key, k, dim_idx + 1)

                self.assertEqual(
                    actual_rank,
                    expected_rank,
                    f"Key {key} at position {key_idx} has rank {actual_rank} "
                    f"for dimension {dim_idx + 1}, expected {expected_rank}",
                )

    def test_negative_keys_handled_correctly(self):
        """Test that negative keys are handled correctly using absolute value."""
        negative_keys = [-1, -5, -10, -42]
        positive_keys = [1, 5, 10, 42]

        for neg_key, pos_key in zip(negative_keys, positive_keys, strict=False):
            for k in self.test_k_values:
                for dim in self.test_dimensions:
                    with self.subTest(neg_key=neg_key, pos_key=pos_key, k=k, dim=dim):
                        # Both should give same rank due to abs() in calculations
                        rank_neg = calc_rank(neg_key, k, dim)
                        rank_pos = calc_rank(pos_key, k, dim)

                        self.assertEqual(
                            rank_neg,
                            rank_pos,
                            f"Negative key {neg_key} and positive key {pos_key} "
                            f"should have same rank for k={k}, dim={dim}",
                        )

    def test_rank_calculation_deterministic(self):
        """Test that rank calculations are deterministic."""
        key = 42
        k = 16
        dim = 3

        # Calculate rank multiple times
        ranks = [calc_rank(key, k, dim) for _ in range(10)]

        # All should be the same
        self.assertTrue(all(r == ranks[0] for r in ranks), f"Rank calculations should be deterministic: {ranks}")

    def test_rank_range_validity(self):
        """Test that calculated ranks are in valid range (≥ 1)."""
        for key in self.test_keys:
            for k in self.test_k_values:
                for dim in self.test_dimensions:
                    with self.subTest(key=key, k=k, dim=dim):
                        rank = calc_rank(key, k, dim)
                        self.assertGreaterEqual(
                            rank, 1, f"Rank should be ≥ 1, got {rank} for key={key}, k={k}, dim={dim}"
                        )

    def test_different_keys_different_distributions(self):
        """Test that different keys produce different rank distributions."""
        k = 16
        dim = 1

        # Test with many keys to see distribution
        keys = list(range(1, 101))
        ranks = [calc_rank(key, k, dim) for key in keys]

        # Should have variety in ranks (not all the same)
        unique_ranks = set(ranks)
        self.assertGreater(len(unique_ranks), 1, "Different keys should produce different ranks")

        # All ranks should be valid
        self.assertTrue(all(r >= 1 for r in ranks), "All ranks should be ≥ 1")

    def test_hash_dimension_independence(self):
        """Test that hash calculations for different dimensions are independent."""
        key = 42
        item_data = ItemData(key)
        internal_item = InternalItem(item_data)

        # Get digests for different dimensions
        digest1 = internal_item.get_digest_for_dim(1)
        digest2 = internal_item.get_digest_for_dim(2)
        digest3 = internal_item.get_digest_for_dim(3)

        # Should all be different
        self.assertNotEqual(digest1, digest2)
        self.assertNotEqual(digest1, digest3)
        self.assertNotEqual(digest2, digest3)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with key = 0
        key = 0
        k = 8
        dim = 1

        # Should not raise an error
        rank = calc_rank(key, k, dim)
        self.assertGreaterEqual(rank, 1)

        # Test with very large key
        large_key = 2**63 - 1
        rank_large = calc_rank(large_key, k, dim)
        self.assertGreaterEqual(rank_large, 1)

        # Test with minimum k
        min_k = 2
        rank_min_k = calc_rank(key, min_k, dim)
        self.assertGreaterEqual(rank_min_k, 1)


if __name__ == "__main__":
    unittest.main()
