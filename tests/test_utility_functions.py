"""
Additional comprehensive tests for specific utility functions.
"""

import unittest
import hashlib
from typing import List

from gplus_trees.base import InternalItem, ItemData
from gplus_trees.utils import (
    count_trailing_zero_bits,
    calculate_group_size,
    calc_rank_from_digest,
    calc_rank_from_digest_k,
    find_keys_for_rank_lists
)
from gplus_trees.g_k_plus.utils import (
    calc_rank_for_dim,
    calc_rank_from_group_size,
    calc_ranks,
    calc_ranks_multi_dims
)


class TestUtilityFunctions(unittest.TestCase):
    """Test specific utility functions in detail."""
    
    def test_count_trailing_zero_bits_comprehensive(self):
        """Comprehensive test of count_trailing_zero_bits with various inputs."""
        test_cases = [
            # (input_bytes, expected_zeros)
            (b'\x01', 0),  # 00000001
            (b'\x02', 1),  # 00000010
            (b'\x03', 0),  # 00000011
            (b'\x04', 2),  # 00000100
            (b'\x08', 3),  # 00001000
            (b'\x10', 4),  # 00010000
            (b'\x20', 5),  # 00100000
            (b'\x40', 6),  # 01000000
            (b'\x80', 7),  # 10000000
            (b'\x01\x00', 8),  # 00000001 00000000 (0x01 is MSB, 0x00 is LSB)
            (b'\x02\x00', 9),  # 00000010 00000000 (0x02 is MSB, 0x00 is LSB)
            (b'\x00\x01\x00', 8),  # 00000000 00000001 00000000 (middle byte 0x01, LSB is 0x00)
            (b'\x00\x00\x01\x00', 8),  # 00000000 00000000 00000001 00000000 (0x01 in 3rd pos, LSB is 0x00)
            (b'\x00\x00\x00\x00', 32),  # All zeros in 4-byte sequence
        ]
        
        for input_bytes, expected_zeros in test_cases:
            with self.subTest(input_bytes=input_bytes.hex()):
                result = count_trailing_zero_bits(input_bytes)
                self.assertEqual(result, expected_zeros,
                               f"Input {input_bytes.hex()} should have {expected_zeros} trailing zeros, got {result}")
    
    def test_calc_rank_from_digest_edge_cases(self):
        """Test calc_rank_from_digest with various edge cases."""
        # Test with digest that has no trailing zeros
        digest_no_zeros = b'\xFF\xFF\xFF\xFF'  # All bits set
        result = calc_rank_from_digest(digest_no_zeros, 3)
        self.assertEqual(result, 1, "Digest with no trailing zeros should have rank 1")
        
        # Test with digest that has many trailing zeros
        digest_many_zeros = b'\x00\x00\x00\x00'  # All zeros
        result = calc_rank_from_digest(digest_many_zeros, 3)
        expected = (32 // 3) + 1  # 10 + 1 = 11
        self.assertEqual(result, expected, f"Digest with 32 trailing zeros should have rank {expected}")
        
        # Test with group_size = 1
        result = calc_rank_from_digest(digest_many_zeros, 1)
        expected = 32 + 1  # 33
        self.assertEqual(result, expected, f"With group_size=1, should have rank {expected}")
        
        # Test with group_size = 0 (should raise ValueError)
        with self.assertRaises(ValueError):
            calc_rank_from_digest(digest_many_zeros, 0)
    
    def test_calc_ranks_function(self):
        """Test the calc_ranks function with Entry objects."""
        # First, let's create some mock entries
        class MockEntry:
            def __init__(self, key):
                self.item = MockItem(key)
        
        class MockItem:
            def __init__(self, key):
                self.key = key
        
        entries = [MockEntry(key) for key in [1, 5, 10, 42]]
        group_size = 3
        DIM = 2
        
        ranks = calc_ranks(entries, group_size, DIM)
        
        # Verify we got the right number of ranks
        self.assertEqual(len(ranks), len(entries))
        
        # Verify each rank is valid
        for rank in ranks:
            self.assertGreaterEqual(rank, 1, "All ranks should be >= 1")
        
        # Verify ranks match individual calculations
        for i, entry in enumerate(entries):
            expected_rank = calc_rank_from_group_size(entry.item.key, group_size, DIM)
            self.assertEqual(ranks[i], expected_rank,
                           f"Rank for entry {i} should match individual calculation")
    
    def test_find_keys_for_rank_lists_edge_cases(self):
        """Test find_keys_for_rank_lists with edge cases."""
        # Single dimension
        rank_lists = [[1, 2, 3]]
        k = 8
        keys = find_keys_for_rank_lists(rank_lists, k)
        self.assertEqual(len(keys), 3)
        
        # Verify each key has the correct rank
        for i, key in enumerate(keys):
            expected_rank = rank_lists[0][i]
            actual_rank = calc_rank_for_dim(key, k, 1)
            self.assertEqual(actual_rank, expected_rank)
        
        # Single key, multiple dimensions
        rank_lists = [[1], [2], [1]]
        keys = find_keys_for_rank_lists(rank_lists, k)
        self.assertEqual(len(keys), 1)
        
        key = keys[0]
        for dim in range(1, 4):
            expected_rank = rank_lists[dim-1][0]
            actual_rank = calc_rank_for_dim(key, k, dim)
            self.assertEqual(actual_rank, expected_rank)
    
    def test_calc_ranks_multi_dims_edge_cases(self):
        """Test calc_ranks_multi_dims with edge cases."""
        # Empty key list
        keys = []
        k = 8
        dimensions = 3
        
        rank_lists = calc_ranks_multi_dims(keys, k, dimensions)
        self.assertEqual(len(rank_lists), dimensions)
        for rank_list in rank_lists:
            self.assertEqual(len(rank_list), 0)
        
        # Single key
        keys = [42]
        rank_lists = calc_ranks_multi_dims(keys, k, dimensions)
        self.assertEqual(len(rank_lists), dimensions)
        for rank_list in rank_lists:
            self.assertEqual(len(rank_list), 1)
        
        # Verify consistency with individual calculations
        for dim in range(dimensions):
            expected_rank = calc_rank_for_dim(keys[0], k, dim + 1)
            actual_rank = rank_lists[dim][0]
            self.assertEqual(actual_rank, expected_rank)
        
        # Single dimension
        keys = [1, 5, 10]
        dimensions = 1
        rank_lists = calc_ranks_multi_dims(keys, k, dimensions)
        self.assertEqual(len(rank_lists), 1)
        self.assertEqual(len(rank_lists[0]), 3)
        
        # Verify consistency
        for i, key in enumerate(keys):
            expected_rank = calc_rank_for_dim(key, k, 1)
            actual_rank = rank_lists[0][i]
            self.assertEqual(actual_rank, expected_rank)
    
    def test_internal_item_digest_caching_efficiency(self):
        """Test that InternalItem caches digests efficiently."""
        key = 42
        item_data = ItemData(key)
        internal_item = InternalItem(item_data)
        
        # Request dimensions out of order
        digest3 = internal_item.get_digest_for_dim(3)
        digest1 = internal_item.get_digest_for_dim(1)
        digest2 = internal_item.get_digest_for_dim(2)
        
        # All should be cached now
        self.assertIn(1, internal_item.dim_hashes)
        self.assertIn(2, internal_item.dim_hashes)
        self.assertIn(3, internal_item.dim_hashes)
        
        # Request again - should use cached values
        digest1_again = internal_item.get_digest_for_dim(1)
        digest2_again = internal_item.get_digest_for_dim(2)
        digest3_again = internal_item.get_digest_for_dim(3)
        
        self.assertEqual(digest1, digest1_again)
        self.assertEqual(digest2, digest2_again)
        self.assertEqual(digest3, digest3_again)
        
        # Request higher dimension - should build from existing cache
        digest5 = internal_item.get_digest_for_dim(5)
        self.assertIn(4, internal_item.dim_hashes)
        self.assertIn(5, internal_item.dim_hashes)
    
    def test_internal_item_with_different_key_types(self):
        """Test InternalItem with different key types."""
        keys = [0, 1, -1, 42, -42, 2**31 - 1, -(2**31)]
        
        for key in keys:
            with self.subTest(key=key):
                item_data = ItemData(key)
                internal_item = InternalItem(item_data)
                
                # Should not raise an error
                digest = internal_item.get_digest_for_dim(1)
                self.assertIsInstance(digest, bytes)
                self.assertGreater(len(digest), 0)
                
                # Should be consistent across calls
                digest2 = internal_item.get_digest_for_dim(1)
                self.assertEqual(digest, digest2)
    
    def test_hash_consistency_across_functions(self):
        """Test that hash calculations are consistent across different functions."""
        keys = [1, 5, 10, 42, 100]
        k = 16
        dimensions = 4
        
        for key in keys:
            item_data = ItemData(key)
            internal_item = InternalItem(item_data)
            
            for dim in range(1, dimensions + 1):
                with self.subTest(key=key, dim=dim):
                    # Method 1: Using g_k_plus utils
                    rank1 = calc_rank_for_dim(key, k, dim)
                    
                    # Method 2: Using InternalItem + calc_rank_from_digest_k
                    digest = internal_item.get_digest_for_dim(dim)
                    rank2 = calc_rank_from_digest_k(digest, k)
                    
                    # Method 3: Using calc_rank_from_group_size
                    group_size = calculate_group_size(k)
                    rank3 = calc_rank_from_group_size(key, group_size, dim)
                    
                    # All should be equal
                    self.assertEqual(rank1, rank2,
                                   f"Rank mismatch for key={key}, dim={dim}: "
                                   f"calc_rank_for_dim={rank1}, InternalItem+calc_rank_from_digest_k={rank2}")
                    self.assertEqual(rank1, rank3,
                                   f"Rank mismatch for key={key}, dim={dim}: "
                                   f"calc_rank_for_dim={rank1}, calc_rank_from_group_size={rank3}")
    
    def test_find_keys_for_rank_lists_with_spacing(self):
        """Test find_keys_for_rank_lists with spacing parameter."""
        rank_lists = [[1, 1], [2, 2]]
        k = 8
        
        # Test with spacing=False (default)
        keys_no_spacing = find_keys_for_rank_lists(rank_lists, k, spacing=False)
        self.assertEqual(len(keys_no_spacing), 2)
        
        # Test with spacing=True
        keys_with_spacing = find_keys_for_rank_lists(rank_lists, k, spacing=True)
        self.assertEqual(len(keys_with_spacing), 2)
        
        # With spacing, keys should be different (and typically larger)
        # This is a behavioral test - the exact values might vary
        self.assertIsInstance(keys_with_spacing, list)
        self.assertEqual(len(keys_with_spacing), 2)
    
    def test_large_key_values(self):
        """Test with very large key values."""
        large_keys = [2**32, 2**48, 2**63 - 1]
        k = 16
        
        for key in large_keys:
            with self.subTest(key=key):
                # Should not raise an error
                rank = calc_rank_for_dim(key, k, 1)
                self.assertGreaterEqual(rank, 1)
                
                # Should be consistent with InternalItem
                item_data = ItemData(key)
                internal_item = InternalItem(item_data)
                digest = internal_item.get_digest_for_dim(1)
                rank2 = calc_rank_from_digest_k(digest, k)
                self.assertEqual(rank, rank2)
    
    def test_rank_distribution_properties(self):
        """Test that rank distributions have expected properties."""
        k = 16
        keys = list(range(1, 101))  # 100 keys
        
        # Calculate ranks for all keys
        ranks = [calc_rank_for_dim(key, k, 1) for key in keys]
        
        # Should have variety in ranks
        unique_ranks = set(ranks)
        self.assertGreater(len(unique_ranks), 1, "Should have variety in ranks")
        
        # All ranks should be >= 1
        self.assertTrue(all(r >= 1 for r in ranks), "All ranks should be >= 1")
        
        # Most ranks should be relatively small (property of exponential distribution)
        rank_1_count = ranks.count(1)
        self.assertGreater(rank_1_count, 10, "Should have many rank-1 items")
        
        # Should have some higher ranks but fewer
        max_rank = max(ranks)
        self.assertGreater(max_rank, 1, "Should have some ranks > 1")


if __name__ == '__main__':
    unittest.main()
