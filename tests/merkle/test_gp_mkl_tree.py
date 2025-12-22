"""Tests for Merkle G+-trees"""
# pylint: skip-file

import unittest
import numpy as np
import random
from binascii import hexlify

from gplus_trees.gplus_tree_base import gtree_stats_
from gplus_trees.merkle import create_merkle_gplustree
from tests.test_base import GPlusTreeTestCase
from tests.utils import assert_tree_invariants_tc

class TestMerkleGPlusTree(GPlusTreeTestCase):
    """Base class for Merkle GPlus-tree tests"""

    def setUp(self):
        """Set up a new Merkle GPlus-tree for each test"""
        self.K = 4  # Default capacity
        self.tree = create_merkle_gplustree(self.K)
        
    def tearDown(self):
        """Verify tree integrity after each test"""
        if not hasattr(self, 'tree') or self.tree.is_empty():
            return
            
        # Check standard invariants
        stats = gtree_stats_(self.tree, {})
        assert_tree_invariants_tc(self, self.tree, stats)
        
        # Also check Merkle tree integrity
        self.assertTrue(self.tree.verify_integrity(), 
                       "Merkle tree integrity check failed")
                       
    def _create_sample_tree(self, keys=None, values=None, ranks=None):
        """Create a sample tree with specified keys, values, and ranks"""
        if keys is None:
            keys = [1, 3, 5, 7, 9]
        
        if values is None:
            values = [f"val_{k}" for k in keys]
            
        if ranks is None:
            ranks = [1] * len(keys)
            
        items = [self.make_item(k, v) for k, v in zip(keys, values)]
        
        for item, rank in zip(items, ranks):
            self.tree.insert(item, rank)
            
        return items


# class TestMerkleGPlusTreeBasic(TestMerkleGPlusTree):
#     """Test basic operations on Merkle GPlus-trees"""
    
#     def test_create_empty_tree(self):
#         """Test that a new empty tree has no root hash"""
#         self.assertTrue(self.tree.is_empty())
#         self.assertIsNone(self.tree.get_root_hash())
        
#     def test_insert_single_item(self):
#         """Test inserting a single item and verifying the hash"""
#         item = self.make_item(1, "value_1")
#         self.tree.insert(item, 1)
        
#         # Tree should not be empty and should have a hash
#         self.assertFalse(self.tree.is_empty())
#         self.assertIsNotNone(self.tree.get_root_hash())
        
#         # Hash should be deterministic
#         hash1 = self.tree.get_root_hash()
        
#         # Reset the hash and recompute
#         self.tree._invalidate_all_hashes()
#         hash2 = self.tree.get_root_hash()
        
#         # Hashes should be identical
#         self.assertEqual(hash1, hash2)
        
#     def test_hash_depends_on_values(self):
#         """Test that the hash changes when values change"""
#         # Insert an item
#         item1 = self.make_item(1, "value_1")
#         self.tree.insert(item1, 1)
#         hash1 = self.tree.get_root_hash()
        
#         # Create a new tree with a different value for the same key
#         tree2 = create_merkle_gplustree(self.K)
#         item2 = self.make_item(1, "value_2")
#         tree2.insert(item2, 1)
#         hash2 = tree2.get_root_hash()
        
#         # Hashes should be different
#         self.assertNotEqual(hash1, hash2)
        
#     def test_hash_depends_on_structure(self):
#         """Test that the hash depends on the tree structure"""
#         # Create two trees with same items but different rank assignments
#         tree1 = create_merkle_gplustree(self.K)
#         tree1.insert(self.make_item(1, "val"), 1)
#         tree1.insert(self.make_item(2, "val"), 1)
#         hash1 = tree1.get_root_hash()
        
#         tree2 = create_merkle_gplustree(self.K)
#         tree2.insert(self.make_item(1, "val"), 1)
#         tree2.insert(self.make_item(2, "val"), 2)  # Different rank
#         hash2 = tree2.get_root_hash()
        
#         # Hashes should be different
#         self.assertNotEqual(hash1, hash2)


# class TestMerkleGPlusTreeComplex(TestMerkleGPlusTree):
#     """Test more complex scenarios with Merkle GPlus-trees"""
    
#     def test_insert_multiple_items_leaf_node(self):
#         """Test inserting multiple items into a leaf node"""
#         keys = [1, 3, 5]
#         items = self._create_sample_tree(keys=keys)
        
#         # Tree should still be a single leaf node
#         self.assertEqual(self.tree.node.rank, 1)
#         self.assertIsNotNone(self.tree.get_root_hash())
        
#         # Adding another item should still work
#         self.tree.insert(self.make_item(7, "val_7"), 1)
        
#         # Hash should change
#         new_hash = self.tree.get_root_hash()
#         self.assertIsNotNone(new_hash)
        
#     def test_insert_creates_internal_nodes(self):
#         """Test that inserting with higher ranks creates internal nodes with valid hashes"""
#         # Insert with different ranks to create internal structure
#         self.tree.insert(self.make_item(1, "val_1"), 1)
#         self.tree.insert(self.make_item(3, "val_3"), 2)
#         self.tree.insert(self.make_item(5, "val_5"), 3)
        
#         # Root should be at rank 3
#         self.assertEqual(self.tree.node.rank, 3)
        
#         # All nodes should have valid hashes
#         root_hash = self.tree.get_root_hash()
#         self.assertIsNotNone(root_hash)
        
#     def test_multi_level_hashing(self):
#         """Test that hashes are properly computed in a multi-level tree"""
#         # Create a complex tree
#         self.tree.insert(self.make_item(1, "val_1"), 3)
#         self.tree.insert(self.make_item(3, "val_3"), 2)
#         self.tree.insert(self.make_item(5, "val_5"), 1)
#         self.tree.insert(self.make_item(7, "val_7"), 2)
#         self.tree.insert(self.make_item(9, "val_9"), 1)
        
#         # Get the root hash
#         root_hash = self.tree.get_root_hash()
#         self.assertIsNotNone(root_hash)
        
#         # Now modify a leaf and check hash change
#         leaf_item = self.make_item(9, "modified")
#         self.tree.insert(leaf_item, 1)  # Update leaf
        
#         # Root hash should change
#         new_hash = self.tree.get_root_hash()
#         self.assertNotEqual(root_hash, new_hash)
        
#     def test_insert_only_invalidates_hash(self):
#         """Test that insert only invalidates the hash without recomputing it"""
#         self._create_sample_tree()
        
#         # Get initial hash
#         initial_hash = self.tree.get_root_hash()
        
#         # Insert with mockery to detect hash computation
#         old_compute_hash = MerkleGPlusNodeBase.compute_hash
#         compute_called = [False]
        
#         def mock_compute_hash(self):
#             compute_called[0] = True
#             return old_compute_hash(self)
            
#         try:
#             MerkleGPlusNodeBase.compute_hash = mock_compute_hash
            
#             # Insert should only invalidate, not recompute
#             self.tree.insert(self.make_item(11, "val_11"), 1)
#             self.assertFalse(compute_called[0], 
#                             "Hash should not be recomputed during insert")
                            
#             # Now request the hash explicitly
#             new_hash = self.tree.get_root_hash()
#             self.assertTrue(compute_called[0], 
#                            "Hash should be computed on demand")
#             self.assertNotEqual(initial_hash, new_hash)
            
#         finally:
#             # Restore the method
#             MerkleGPlusNodeBase.compute_hash = old_compute_hash


# class TestMerkleGPlusTreeProofs(TestMerkleGPlusTree):
#     """Test Merkle proof functionality"""
    
#     def test_inclusion_proof_basic(self):
#         """Test basic inclusion proof for a simple tree"""
#         # Create tree with a few items
#         keys = [1, 3, 5, 7, 9]
#         items = self._create_sample_tree(keys=keys)
        
#         # Get proof for key 5
#         proof = self.tree.get_inclusion_proof(5)
        
#         # In a leaf node, proof should be empty as all items are in the same hash
#         self.assertEqual(len(proof), 0)
        
#     def test_inclusion_proof_complex(self):
#         """Test inclusion proof in a more complex tree"""
#         # Create a tree with internal nodes
#         self.tree.insert(self.make_item(1, "val_1"), 3)
#         self.tree.insert(self.make_item(3, "val_3"), 2)
#         self.tree.insert(self.make_item(5, "val_5"), 1)
#         self.tree.insert(self.make_item(7, "val_7"), 2)
        
#         # Get proof for key 5
#         proof = self.tree.get_inclusion_proof(5)
        
#         # Complex tree should have non-empty proof
#         self.assertGreater(len(proof), 0)
        
#     def test_inclusion_proof_missing_key(self):
#         """Test inclusion proof for a missing key"""
#         self._create_sample_tree()
        
#         # Get proof for non-existent key
#         proof = self.tree.get_inclusion_proof(999)
        
#         # Should return an empty proof
#         self.assertEqual(len(proof), 0)
        

# class TestMerkleGPlusTreeIntegrity(TestMerkleGPlusTree):
#     """Test tree integrity verification"""
    
#     def test_verify_integrity_empty(self):
#         """Test integrity verification on empty tree"""
#         self.assertTrue(self.tree.verify_integrity())
        
#     def test_verify_integrity_single(self):
#         """Test integrity verification on tree with one item"""
#         self.tree.insert(self.make_item(1, "val_1"), 1)
#         self.assertTrue(self.tree.verify_integrity())
        
#     def test_verify_integrity_complex(self):
#         """Test integrity verification on complex tree"""
#         # Create a complex tree
#         self._create_sample_tree(ranks=[3, 2, 1, 2, 1])
#         self.assertTrue(self.tree.verify_integrity())
        
#     def test_integrity_tampered_node(self):
#         """Test integrity check fails when a node is tampered with"""
#         self._create_sample_tree()
        
#         # Manually tamper with a node's hash
#         if not self.tree.is_empty() and isinstance(self.tree.node, MerkleGPlusNodeBase):
#             # Store real hash
#             real_hash = self.tree.node.get_hash()
            
#             # Create a fake hash
#             fake_hash = bytearray(real_hash)
#             fake_hash[0] = (fake_hash[0] + 1) % 256  # Change one byte
            
#             # Inject the fake hash
#             self.tree.node.merkle_hash = bytes(fake_hash)
            
#             # Integrity check should fail
#             self.assertFalse(self.tree.verify_integrity())


# class TestMerkleGPlusTreeLargeScale(TestMerkleGPlusTree):
#     """Test Merkle trees with larger numbers of insertions"""
    
#     def test_random_insertions(self):
#         """Test many random insertions maintain consistent hashing"""
#         num_items = 50
#         random.seed(42)  # For reproducibility
        
#         # Generate random items
#         items = []
#         for i in range(num_items):
#             key = random.randint(1, 10000)
#             value = f"val_{key}"
#             rank = random.randint(1, 3)
#             items.append((self.make_item(key, value), rank))
        
#         # Insert into tree
#         for item, rank in items:
#             self.tree.insert(item, rank)
            
#         # Tree should be valid
#         stats = gtree_stats_(self.tree, {})
#         assert_tree_invariants_tc(self, self.tree, stats)
        
#         # Get the final hash
#         final_hash = self.tree.get_root_hash()
#         self.assertIsNotNone(final_hash)
        
#         # Reset all hashes and verify they recompute to the same value
#         self.tree._invalidate_all_hashes()
#         recomputed_hash = self.tree.get_root_hash()
#         self.assertEqual(final_hash, recomputed_hash)


class TestMerkleGPlusTreeSpecialCases(TestMerkleGPlusTree):
    """Test special cases and edge conditions"""
    
    def test_update_existing_value(self):
        """Test updating an existing value changes the hash"""
        # Insert initial item
        key = 5
        self.tree.insert(self.make_item(key, "original"), 1)
        original_hash = self.tree.get_root_hash()
        
        # Update the same key with new value
        self.tree.insert(self.make_item(key, "updated"), 1)
        updated_hash = self.tree.get_root_hash()
        
        # Hash should change
        self.assertNotEqual(original_hash, updated_hash)
        
    def test_hash_visualization(self):
        """Test hash visualization for debugging purposes"""
        self._create_sample_tree()
        
        # Get the hash
        hash_bytes = self.tree.get_root_hash()
        
        # Convert to hex for human-readable representation
        hash_hex = hexlify(hash_bytes).decode('utf-8')
        
        # Should be a valid hex string of appropriate length (SHA-256 = 64 hex chars)
        self.assertEqual(len(hash_hex), 64)
        
    def test_different_capacity_trees(self):
        """Test trees with different capacities have same hashes for same content"""
        # Create two trees with different K values but same content
        tree1 = create_merkle_gplustree(4)
        tree2 = create_merkle_gplustree(8)
        
        # Insert the same items
        for key in [1, 3, 5, 9, 11]:
            tree1.insert(self.make_item(key, f"val_{key}"), 1)
            tree2.insert(self.make_item(key, f"val_{key}"), 1)
            
        # Get hashes
        hash1 = tree1.get_root_hash()
        hash2 = tree2.get_root_hash()
        
        # Trees should have different capacities so different structure
        # which should lead to different hashes
        self.assertEqual(hash1, hash2)

    def test_random_order_insertions(self):
        """Test that random order insertions result in the same hash. """
        n = 100
        K = 2
        p = 1.0 - (1.0 / (K))

        # we need at least n unique values; 2^24 = 16 777 216 > 1 000 000
        space = 1 << 24
        if space <= n:
            raise ValueError(f"Key-space too small! Required: {n + 1}, Available: {space}")

        indices = random.sample(range(1, space), k=n) # Exclude dummy key 0

        # Pre-allocate items list
        items = [(None, None)] * n

        ranks = np.random.geometric(p, size=n)

        # Process all items in a single pass
        for i, idx in enumerate(indices):
            # Use the index directly as the key
            key = idx
            val = "val"
            items[i] = (self.make_item(key, val), int(ranks[i]))

        hashes = []
        for i in range(100):
            # Create a new tree for each iteration
            tree = create_merkle_gplustree(K)
            random.shuffle(items)
            tree_insert = tree.insert
            for (item, rank) in items:
                tree_insert(item, rank)
            hashes.append(tree.get_root_hash())
        # Check if all hashes are the same
        for i in range(1, len(hashes)):
            self.assertEqual(hashes[i], hashes[0], f"Hashes should be the same for all trees, but differ at index {i}")

            

if __name__ == "__main__":
    unittest.main()