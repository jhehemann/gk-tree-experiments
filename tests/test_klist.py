"""Tests for k-lists with factory pattern"""
# pylint: skip-file

import unittest
import random

# Import factory function instead of concrete classes
from gplus_trees.base import Item
from gplus_trees.factory import make_gplustree_classes
import logging

# Configure logging for test
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test with different capacities to ensure factory works correctly
TEST_CAPACITIES = [4, 8, 16]

class TestKListFactory(unittest.TestCase):
    """Test the factory pattern itself with various capacities"""
    
    def test_factory_creates_different_classes(self):
        """Verify that factory creates different class types for different K values"""
        classes = {}
        for K in TEST_CAPACITIES:
            tree_class, node_class, klist_class, knode_class = make_gplustree_classes(K)
            classes[K] = (tree_class, node_class, klist_class, knode_class)
            
            # Check class names contain K value
            self.assertIn(str(K), tree_class.__name__)
            self.assertIn(str(K), klist_class.__name__)
            self.assertIn(str(K), knode_class.__name__)
            
            # Check that capacity is correctly set
            self.assertEqual(knode_class.CAPACITY, K)
            
            # Verify proper class relationships
            self.assertEqual(tree_class.SetClass, klist_class)
            self.assertEqual(node_class.SetClass, klist_class)
            self.assertEqual(klist_class.KListNodeClass, knode_class)
        
        # Different K values should create different classes
        for k1 in TEST_CAPACITIES:
            for k2 in TEST_CAPACITIES:
                if k1 != k2:
                    self.assertNotEqual(classes[k1][2], classes[k2][2], 
                                       f"KList classes for K={k1} and K={k2} should be different")


class TestKListBase(unittest.TestCase):
    """Base class for all KList factory tests"""
    
    def setUp(self):
        # Use the factory to create classes with the test capacity
        self.K = 4  # Default capacity for tests
        _, _, self.KListClass, self.KListNodeClass = make_gplustree_classes(self.K)
        self.klist = self.KListClass()
        self.cap = self.K  # Use factory-defined capacity
        # logger.debug(f"Created KList test with K={self.K}, using class {self.KListClass.__name__}")
        # logger.debug(f"Created KListNode test with K={self.K}, using class {self.KListNodeClass.__name__}")
    
    def tearDown(self):
        # Verify invariants after each test
        self.klist.check_invariant()

    def _count_nodes(self, klist):
        count = 0
        node = klist.head
        while node:
            count += 1
            node = node.next
        return count
    
    def insert_sequence(self, keys):
        """Helper to insert integer keys with dummy values."""
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        # ensure invariants
        self.klist.check_invariant()

class TestKList(TestKListBase):
    """Basic insertion tests for factory-created klists"""

    def test_insert_in_order(self):
        for key in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]:
            self.klist.insert(Item(key, f"val_{key}"))
        # invariant is checked in tearDown()

    def test_insert_out_of_order(self):
        for key in [4, 2, 1, 3, 5, 8, 7, 6, 10, 9, 8]:
            self.klist.insert(Item(key, f"val_{key}"))
        # invariant is checked in tearDown()
    
    def test_capacity_matches_factory(self):
        """Verify that the KListNodeClass has the correct CAPACITY value"""
        node = self.KListNodeClass()
        self.assertEqual(node.__class__.CAPACITY, self.K)
        
        # Fill to capacity and ensure overflow behavior
        for i in range(self.K + 1):
            self.klist.insert(Item(i, f"val_{i}"))
        
        # Should have created a second node for the overflow
        self.assertIsNotNone(self.klist.head.next)
        self.assertEqual(len(self.klist.head.entries), self.K)
        self.assertEqual(len(self.klist.head.next.entries), 1)


class TestKListInsert(TestKListBase):
    def extract_all_keys(self):
        """Traverse the KList and collect all item keys in order."""
        keys = []
        node = self.klist.head
        while node:
            keys.extend(e.item.key for e in node.entries)
            node = node.next
        return keys

    def test_insert_into_empty(self):
        # Inserting into an empty list should set head and tail
        self.assertIsNone(self.klist.head)
        self.klist.insert(Item(1, "val_1"))
        self.assertIsNotNone(self.klist.head)
        self.assertIs(self.klist.head, self.klist.tail)
        self.assertEqual(self.extract_all_keys(), [1])

    def test_insert_in_order(self):
        # Insert keys in sorted order one by one
        for key in [1, 2, 3, 4, 5]:
            self.klist.insert(Item(key, f"val_{key}"))
        self.assertEqual(self.extract_all_keys(), [1, 2, 3, 4, 5])

    def test_insert_out_of_order(self):
        # Insert keys in random order, final list must be sorted
        for key in [4, 1, 5, 2, 3]:
            self.klist.insert(Item(key, f"val_{key}"))
        self.assertEqual(self.extract_all_keys(), [1, 2, 3, 4, 5])

    def test_single_node_overflow(self):
        # Fill exactly one node to capacity, then insert one more
        keys = list(range(self.cap))
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        # one more causes a second node
        extra = self.cap
        self.klist.insert(Item(extra, f"val_{extra}"))

        all_keys = self.extract_all_keys()
        self.assertEqual(len(all_keys), self.cap + 1)

        # First node must have cap entries, second node the overflow
        node = self.klist.head
        self.assertEqual([e.item.key for e in node.entries], keys)
        self.assertIsNotNone(node.next)
        self.assertEqual([e.item.key for e in node.next.entries], [extra])
        self.assertIs(self.klist.tail, node.next)

    def test_multiple_node_overflows(self):
        # Insert 3*cap + 2 items, ensure proper node counts
        total = 3 * self.cap + 2
        keys = list(range(total))
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))

        # Traverse and count nodes & entries
        node = self.klist.head
        counts = []
        while node:
            counts.append(len(node.entries))
            node = node.next

        # All but the last should be full
        for cnt in counts[:-1]:
            self.assertEqual(cnt, self.cap)
        # Last node has the remainder
        self.assertEqual(sum(counts), total)

    def test_duplicate_keys(self):
        # Insert duplicate keys – they all should appear in order
        for _ in range(3):
            self.klist.insert(Item(7, "duplicate"))
        self.assertEqual(self.extract_all_keys(), [7, 7, 7])

    def test_tail_fast_path(self):
        # Repeatedly append monotonic integer keys
        for i in range(100):
            self.klist.insert(Item(i, f"val_{i}"))
        self.assertEqual(self.extract_all_keys(), list(range(100)))

    def test_interleaved_inserts_and_checks(self):
        # Interleave inserts with invariant checks to catch transient issues
        sequence = [5, 2, 8, 3, 9, 1, 7]
        for key in sequence:
            self.klist.insert(Item(key, f"val_{key}"))
            so_far = self.extract_all_keys()
            self.assertEqual(so_far, sorted(so_far))

    def test_complex_pattern(self):
        # Insert a complex shuffled pattern repeatedly and verify final sort
        import random
        keys = list(range(self.cap * 2))
        for _ in range(5):
            random.shuffle(keys)
            for k in keys:
                self.klist.insert(Item(k, f"val_{k}"))

        all_keys = self.extract_all_keys()
        self.assertEqual(all_keys, sorted(all_keys))
        self.assertEqual(len(all_keys), self.cap * 2 * 5)


class TestKListDelete(TestKListBase):
    def insert_keys(self, keys):
        """Helper: insert a sequence of integer keys with dummy values."""
        for k in keys:
            self.klist.insert(Item(k, f"val_{k}"))
        self.klist.check_invariant()

    def extract_all_keys(self):
        """Helper: traverse KList and return all keys in order."""
        keys = []
        node = self.klist.head
        while node:
            keys.extend(e.item.key for e in node.entries)
            node = node.next
        return keys

    def test_delete_on_empty_list(self):
        # deleting from an empty KList should do nothing
        before = self.extract_all_keys()
        self.klist.delete(999)           # nonexistent int
        after = self.extract_all_keys()
        self.assertEqual(before, after)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_nonexistent_key(self):
        # insert some items, then delete a missing key
        self.insert_keys([1, 2, 3])
        before = self.extract_all_keys()
        self.klist.delete(999)
        after = self.extract_all_keys()
        self.assertEqual(before, after)

    def test_delete_only_item(self):
        # after deleting the sole element, head and tail should be None
        self.insert_keys([5])
        self.klist.delete(5)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_head_key(self):
        # delete the first key in a multi-element, single-node list
        keys = [1, 2, 3]
        self.insert_keys(keys)
        self.klist.delete(1)
        result = self.extract_all_keys()
        self.assertEqual(result, [2, 3])
        # head should remain the same node
        self.assertIsNotNone(self.klist.head)
        self.klist.check_invariant()

    def test_delete_tail_key(self):
        # delete the last key in a single-node list
        keys = [1, 2, 3]
        self.insert_keys(keys)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2])
        self.klist.check_invariant()

    def test_delete_middle_key(self):
        # delete a middle key and ensure rebalance keeps packing
        keys = [1, 2, 3, 4, 5]
        self.insert_keys(keys)
        # ensure at least two nodes exist
        self.assertGreater(len(self.klist.head.entries), 0)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2, 4, 5])
        self.klist.check_invariant()

    def test_delete_causes_node_removal(self):
        # build exactly two nodes: first full, second with 1 entry
        keys = list(range(self.cap + 1))
        self.insert_keys(keys)
        # delete the lone entry in the second node
        last_key = keys[-1]
        self.klist.delete(last_key)
        # the second node should be spliced out
        self.assertIsNone(self.klist.head.next)
        # head still has all capacity entries
        self.assertEqual(len(self.klist.head.entries), self.cap)
        self.klist.check_invariant()

    def test_multiple_deletes(self):
        # delete multiple keys in succession
        keys = [1, 2, 3, 4, 5, 6, 7]
        self.insert_keys(keys)
        for k in [2, 5, 1, 7, 4]:
            self.klist.delete(k)
            self.assertNotIn(k, self.extract_all_keys())
            self.klist.check_invariant()
        # remaining should be [3,6]
        self.assertEqual(self.extract_all_keys(), [3, 6])

    def test_repeated_delete_same_key(self):
        # inserting duplicates—only first matching should be removed each time
        dup_key = 7
        self.insert_keys([dup_key, dup_key, dup_key])
        self.klist.delete(dup_key)
        # exactly two remain
        self.assertEqual(self.extract_all_keys(), [dup_key, dup_key])
        self.klist.delete(dup_key)
        self.klist.delete(dup_key)
        # now list is empty
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_all_nodes(self):
        # insert enough to create 3+ nodes, then delete everything one by one
        keys = list(range(3 * self.cap + 2))
        self.insert_keys(keys)
        for k in keys:
            self.klist.delete(k)
        # list should be empty afterwards
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)


class TestKListRetrieve(TestKListBase):
    def assertRetrieval(self, key, found_key, next_key):
        """
        Helper: call retrieve(key) and assert that
          result.found_entry.item.key == found_key  (or None)
          result.next_entry.item.key == next_key    (or None)
        """
        res = self.klist.retrieve(key)
        if found_key is None:
            self.assertIsNone(res.found_entry, f"Expected no entry for {key}")
        else:
            self.assertIsNotNone(res.found_entry)
            self.assertEqual(res.found_entry.item.key, found_key)
        if next_key is None:
            self.assertIsNone(res.next_entry, f"Expected no successor for {key}")
        else:
            self.assertIsNotNone(res.next_entry)
            self.assertEqual(res.next_entry.item.key, next_key)

    def test_retrieve_empty(self):
        # empty list returns (None, None)
        res = self.klist.retrieve(123)
        self.assertIsNone(res.found_entry)
        self.assertIsNone(res.next_entry)

    def test_type_error_on_non_int(self):
        with self.assertRaises(TypeError):
            self.klist.retrieve("not-an-int")

    def test_single_node_exact_middle(self):
        # fill one node without overflow
        keys = [10, 20, 30]
        self.insert_sequence(keys)
        # exact match in middle
        self.assertRetrieval(20, 20, 30)

    def test_single_node_exact_first(self):
        keys = [5, 15, 25]
        self.insert_sequence(keys)
        self.assertRetrieval(5, 5, 15)

    def test_single_node_exact_last(self):
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        # found at last position → successor None
        self.assertRetrieval(3, 3, None)

    def test_single_node_between(self):
        keys = [100, 200, 300]
        self.insert_sequence(keys)
        # between 100 and 200
        self.assertRetrieval(150, None, 200)

    def test_single_node_below_min(self):
        keys = [50, 60]
        self.insert_sequence(keys)
        # below first
        self.assertRetrieval(40, None, 50)

    def test_single_node_above_max(self):
        keys = [7, 8, 9]
        self.insert_sequence(keys)
        # above last
        self.assertRetrieval(100, None, None)

    def test_cross_node_exact_and_successor(self):
        # overflow into two nodes
        # capacity = 4, so use 5 items
        keys = [1, 2, 3, 4, 5]
        self.insert_sequence(keys)
        # 4 is last of first node, successor should be first of second node (5)
        self.assertRetrieval(4, 4, 5)
        # 5 is in second node, exact last → successor None
        self.assertRetrieval(5, 5, None)

    def test_cross_node_between(self):
        # retrieve only accepts int keys, so passing a float should raise
        with self.assertRaises(TypeError):
            self.klist.retrieve(4.5)

    def test_cross_node_below_head(self):
        keys = [10, 20, 30, 40, 50]
        self.insert_sequence(keys)
        # below first in head
        self.assertRetrieval(5, None, 10)

    def test_cross_node_above_tail(self):
        keys = [10, 20, 30, 40, 50]
        self.insert_sequence(keys)
        # above max across all nodes
        self.assertRetrieval(1000, None, None)

    def test_bulk_retrieval_all_keys(self):
        # retrieve each real key should find itself and next
        keys = list(range(1, self.cap * 2 + 1))  # generate enough to overflow
        self.insert_sequence(keys)
        for idx, k in enumerate(keys):
            expected_next = keys[idx+1] if idx+1 < len(keys) else None
            self.assertRetrieval(k, k, expected_next)

    def test_random_nonexistent(self):
        keys = list(range(0, self.cap * 3))
        self.insert_sequence(keys)
        low, high = -10, max(keys) + 10

        for _ in range(20):
            x = random.randint(low, high)
            if x in keys:
                continue  # skip existing keys

            # find the smallest key > x
            next_candidates = [k for k in keys if k > x]
            nxt = min(next_candidates) if next_candidates else None

            # now x is an int, so retrieve(x) works
            self.assertRetrieval(x, None, nxt)


class TestKlistGetMinMax(TestKListBase):   
    def test_empty(self):
        # empty list should return N
        with self.subTest("max"): 
            res = self.klist.get_max()
            self.assertIsNone(res.found_entry)
            self.assertIsNone(res.next_entry)

        with self.subTest("min"):
            res = self.klist.get_min()
            self.assertIsNone(res.found_entry)
            self.assertIsNone(res.next_entry)

    def test_single_node_single_entry(self):
        # fill one node with a single entry
        keys = [10]
        self.insert_sequence(keys)
        
        with self.subTest("max"):    
            res = self.klist.get_max()
            self.assertEqual(res.found_entry.item.key, 10)
            self.assertIsNone(res.next_entry)
        
        with self.subTest("min"):
            res = self.klist.get_min()
            self.assertEqual(res.found_entry.item.key, 10)
            self.assertIsNone(res.next_entry)

    def test_single_node_cap(self):
        # fill one node up to capacity
        keys = list(range(self.cap))
        self.insert_sequence(keys)
        
        with self.subTest("max"):    
            res = self.klist.get_max()
            self.assertEqual(res.found_entry.item.key, self.cap - 1)
            self.assertIsNone(res.next_entry)
        
        with self.subTest("min"):
            res = self.klist.get_min()
            self.assertEqual(res.found_entry.item.key, 0)
            self.assertEqual(res.next_entry.item.key, 1)

    def test_multi_nodes(self):
        # fill first node, overflow into second
        keys = list(range(self.cap * 3 + 1))
        self.insert_sequence(keys)

        with self.subTest("max"):
            res = self.klist.get_max()
            self.assertEqual(res.found_entry.item.key, self.cap * 3)
            self.assertIsNone(res.next_entry)

        with self.subTest("min"):
            res = self.klist.get_min()
            self.assertEqual(res.found_entry.item.key, 0)
            self.assertEqual(res.next_entry.item.key, 1)

class TestKListIndex(TestKListBase):
    def test_empty_index(self):
        # Before any operations, index lists should exist and be empty
        self.assertTrue(hasattr(self.klist, "_nodes"))
        self.assertEqual(self.klist._nodes, [], "_nodes should be initialized empty")
        self.assertTrue(hasattr(self.klist, "_prefix_counts"))
        self.assertEqual(self.klist._prefix_counts, [], "_prefix_counts should be initialized empty")

    def test_index_after_single_insert(self):
        # Insert one item, rebuild, then index should have 1 node, prefix_counts [1]
        self.klist.insert(Item(10, "val_10"))
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts, [1])
        # Node in list should be the head
        self.assertIs(self.klist._nodes[0], self.klist.head)

    def test_index_after_multiple_inserts_no_overflow(self):
        # Insert fewer than CAPACITY items
        keys = list(range(self.cap - 1))
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # Still one node, prefix_counts = [len(keys)]
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts, [len(keys)])
    
    def test_index_after_overflow(self):
        # Insert exactly CAPACITY + 2 items → 2 nodes
        total = self.cap + 2
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # Should have 2 nodes
        self.assertEqual(len(self.klist._nodes), 2)
        # prefix_counts: first node cap, second cap+2
        expected = [self.cap, total]
        self.assertEqual(self.klist._prefix_counts, expected)
        # Check that _nodes entries match actual chain
        node = self.klist.head
        for idx, n in enumerate(self.klist._nodes):
            self.assertIs(n, node)
            node = node.next

    def test_prefix_counts_monotonic_and_correct(self):
        # Random insertion pattern, then check prefix sums
        keys = [5, 1, 9, 2, 8, 3, 7, 4, 6, 0]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # Now delete a few to force structure change
        for k in [9, 0]:
            self.klist.delete(k)
        # Compute expected prefix sums by traversing
        running = 0
        expected = []
        node = self.klist.head
        while node:
            running += len(node.entries)
            expected.append(running)
            node = node.next
        self.assertEqual(self.klist._prefix_counts, expected)
        # Ensure strictly increasing
        for a, b in zip(expected, expected[1:]):
            self.assertLess(a, b)

    def test_index_after_bulk_deletes(self):
        # Fill three nodes exactly, then remove the middle node
        total = 3 * self.cap
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # delete all keys in the middle node
        middle_start = self.cap
        middle_end   = 2 * self.cap - 1
        for k in range(middle_start, middle_end + 1):
            self.klist.delete(k)
        # Now exactly two nodes remain (the head and the tail)
        self.assertEqual(len(self.klist._nodes), 2)
        # And the prefix sums should be [CAPACITY, 3*CAPACITY]
        self.assertEqual(
            self.klist._prefix_counts,
            [self.cap, 2 * self.cap]
        )


# class TestUpdateLeftSubtree(TestKListBase):
#     def setUp(self):
#         super().setUp()
#         # Use factory to create tree instances
#         self.TreeClass, _, _, _ = make_gplustree_classes(self.K)
#         # Trees to attach
#         self.treeA = self.TreeClass()
#         self.treeB = self.TreeClass()
#         # logger.debug(f"Created trees of type {type(self.treeA).__name__}")

#     def extract_left_subtrees(self):
#         """Helper to collect left_subtree pointers for all entries."""
#         subs = []
#         node = self.klist.head
#         while node:
#             subs.extend(entry.left_subtree for entry in node.entries)
#             node = node.next
#         return subs

#     def test_update_on_empty_list(self):
#         # Updating an empty list should do nothing and return self
#         returned = self.klist.update_left_subtree(1, self.treeA)
#         self.assertIs(returned, self.klist)
#         # List still empty
#         self.assertIsNone(self.klist.head)
#         self.assertIsNone(self.klist.tail)

#     def test_update_nonexistent_key(self):
#         # Insert some keys, then update a non-existent one
#         keys = [1, 2, 3]
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         before = self.extract_left_subtrees()
#         returned = self.klist.update_left_subtree(99, self.treeA)
#         self.assertIs(returned, self.klist)
#         after = self.extract_left_subtrees()
#         self.assertEqual(before, after)

#     def test_update_first_entry(self):
#         # Insert keys and update the first key
#         keys = [10, 20, 30]
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         returned = self.klist.update_left_subtree(10, self.treeA)
#         self.assertIs(returned, self.klist)
#         # First entry gets treeA, others remain None
#         subs = self.extract_left_subtrees()
#         self.assertEqual(subs[0], self.treeA)
#         self.assertTrue(all(s is None for s in subs[1:]))

#     def test_update_last_entry(self):
#         # Insert keys and update the last key
#         keys = [5, 6, 7]
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         returned = self.klist.update_left_subtree(7, self.treeB)
#         self.assertIs(returned, self.klist)
#         subs = self.extract_left_subtrees()
#         # Last subtree updated
#         self.assertEqual(subs[-1], self.treeB)
#         self.assertTrue(all(s is None for s in subs[:-1]))

#     def test_update_middle_entry(self):
#         # Insert multiple keys and update a middle one
#         keys = [1, 2, 3, 4, 5]
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         returned = self.klist.update_left_subtree(3, self.treeA)
#         self.assertIs(returned, self.klist)
#         subs = self.extract_left_subtrees()
#         # Only the third entry is updated
#         for i, s in enumerate(subs):
#             if keys[i] == 3:
#                 self.assertIs(s, self.treeA)
#             else:
#                 self.assertIsNone(s)

#     def test_update_after_overflow(self):
#         # Force two nodes and update an entry in second node
#         total = self.cap + 2
#         for k in range(total):
#             self.klist.insert(Item(k, f"val_{k}"))
#         # Update key = cap (first entry in second node)
#         returned = self.klist.update_left_subtree(self.cap, self.treeB)
#         self.assertIs(returned, self.klist)
#         # Traverse to the second node:
#         node = self.klist.head.next
#         # First entry in that node should have left_subtree = treeB
#         self.assertIs(node.entries[0].left_subtree, self.treeB)

#     def test_chained_updates(self):
#         # Multiple updates in sequence
#         keys = [1, 2, 3]
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         self.klist.update_left_subtree(1, self.treeA)
#         self.klist.update_left_subtree(2, self.treeB)
#         subs = self.extract_left_subtrees()
#         self.assertIs(subs[0], self.treeA)
#         self.assertIs(subs[1], self.treeB)
#         self.assertIsNone(subs[2])

#     def test_type_error_on_non_int_key(self):
#         with self.assertRaises(TypeError):
#             self.klist.update_left_subtree("not-int", self.treeA)


class TestSplitInplace(TestKListBase):
    def extract_keys(self, kl):
        """Return list of keys in order from KList."""
        keys = []
        node = kl.head
        while node:
            keys.extend(entry.item.key for entry in node.entries)
            node = node.next
        return keys

    def test_empty_split(self):
        left, subtree, right = self.klist.split_inplace(5)
        # both sides empty
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        # invariants hold
        left.check_invariant()
        right.check_invariant()

    def test_split_before_all_keys(self):
        # Insert some keys
        keys = [10, 20, 30]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split before smallest
        left, subtree, right = self.klist.split_inplace(5)
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), keys)
        left.check_invariant()
        right.check_invariant()

    def test_split_after_all_keys(self):
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split after largest
        left, subtree, right = self.klist.split_inplace(10)
        self.assertEqual(self.extract_keys(left), keys)
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        left.check_invariant()
        right.check_invariant()

    def test_split_exact_middle(self):
        keys = [1, 2, 3, 4, 5]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on 3
        left, subtree, right = self.klist.split_inplace(3)
        self.assertEqual(self.extract_keys(left), [1, 2])
        self.assertIsNone(subtree)  # default left_subtree None
        self.assertEqual(self.extract_keys(right), [4, 5])
        left.check_invariant()
        right.check_invariant()

    def test_split_nonexistent_between(self):
        keys = [10, 20, 30, 40]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on a key not present but between 20 and 30
        left, subtree, right = self.klist.split_inplace(25)
        self.assertEqual(self.extract_keys(left), [10, 20])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [30, 40])
        left.check_invariant()
        right.check_invariant()

    def test_split_at_node_boundary_max(self):
        # make at least two nodes
        total = self.cap + 1
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # first node has keys 0..cap-1, second has [cap+1]
        # split exactly at cap (first of second node)
        left, subtree, right = self.klist.split_inplace(self.cap)

        self.assertEqual(self.extract_keys(left), list(range(self.cap)))
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        left.check_invariant()
        right.check_invariant()

    def test_split_at_node_boundary(self):
        # make at least two nodes
        total = self.cap + 2
        for k in range(total):
            self.klist.insert(Item(k, f"v{k}"))
        # first node has keys 0..cap-1, second has [cap+1]
        # split exactly at cap (first of second node)
        left, subtree, right = self.klist.split_inplace(self.cap)

        self.assertEqual(self.extract_keys(left), list(range(self.cap)))
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [self.cap + 1])
        left.check_invariant()
        right.check_invariant()

    def test_split_with_subtree_propagation(self):
        # insert and assign a left_subtree for a particular key
        keys = [1, 2, 3]
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # update left_subtree for key=2
        # Create tree using the factory
        self.TreeClass, _, _, _ = make_gplustree_classes(self.K)
        subtree = self.TreeClass()
        found = self.klist.retrieve(2).found_entry
        self.assertIsNotNone(found)
        found.left_subtree = subtree
        left, st, right = self.klist.split_inplace(2)
        # left contains [1], subtree returned
        self.assertEqual(self.extract_keys(left), [1])
        self.assertIs(st, subtree)
        self.assertEqual(self.extract_keys(right), [3])
        left.check_invariant()
        right.check_invariant()

    def test_split_multiple_times(self):
        # perform sequential splits
        keys = list(range(6))
        for k in keys:
            self.klist.insert(Item(k, f"v{k}"))
        # split on 2
        l1, _, r1 = self.klist.split_inplace(2)
        # split r1 on 4
        l2, _, r2 = r1.split_inplace(4)
        self.assertEqual(self.extract_keys(l1), [0, 1])
        self.assertEqual(self.extract_keys(l2), [3])
        self.assertEqual(self.extract_keys(r2), [5])
        l1.check_invariant()
        l2.check_invariant()
        r2.check_invariant()

    def test_type_error_non_int_key(self):
        with self.assertRaises(TypeError):
            self.klist.split_inplace("not-int")



class TestKListCompactionInvariant(TestKListBase):
    """Tests specifically for the compaction invariant that only the last node should have fewer than K items"""
    
    def setUp(self):
        # Use the factory to create classes with the test capacity
        self.K = 16  # Default capacity for tests
        _, _, self.KListClass, self.KListNodeClass = make_gplustree_classes(self.K)
        self.klist = self.KListClass()
        self.cap = self.K  # Use factory-defined capacity
    
    def _count_nodes(self, klist):
        """Helper to count nodes in a klist"""
        count = 0
        node = klist.head
        while node:
            count += 1
            node = node.next
        return count
    
    def _verify_compaction_invariant(self, klist):
        """Helper to manually verify the compaction invariant without using check_invariant"""
        node_count = 0
        node = klist.head
        violations = []
        
        while node:
            node_count += 1
            if node.next is not None and len(node.entries) < node.__class__.CAPACITY:
                violations.append(
                    f"Non-tail node at position {node_count} has {len(node.entries)} entries, "
                    f"but should have {node.__class__.CAPACITY}"
                )
            node = node.next
            
        return violations
    
    def test_compaction_after_single_deletion(self):
        """Test compaction after deleting a single item"""
        # Create a klist with multiple nodes
        items = 2 * self.cap + 1  # Creates 3 nodes
        for i in range(items):
            self.klist.insert(Item(i, f"val_{i}"))
            
        # Verify initial structure is valid
        self.assertFalse(self._verify_compaction_invariant(self.klist))
        
        # Delete an item from the first node
        self.klist.delete(0)
        
        # Check for violations of compaction invariant
        violations = self._verify_compaction_invariant(self.klist)
        if violations:
            self.fail(f"Compaction invariant violated after deletion: {violations}")
    
    def test_compaction_after_middle_deletion(self):
        """Test compaction after deleting an item from the middle of the list"""
        # Create a klist with 3 nodes (full capacity)
        total_items = 3 * self.cap 
        for i in range(total_items):
            self.klist.insert(Item(i, f"val_{i}"))
            
        # Delete an item from the middle node
        middle_key = self.cap + self.cap//2
        self.klist.delete(middle_key)
        
        # Check for violations
        violations = self._verify_compaction_invariant(self.klist)
        if violations:
            self.fail(f"Compaction invariant violated after middle deletion: {violations}")
    
    def test_compaction_after_multiple_deletions(self):
        """Test compaction after multiple deletions from different nodes"""
        # Create a klist with multiple nodes
        total_items = 4 * self.cap
        for i in range(total_items):
            self.klist.insert(Item(i, f"val_{i}"))
            
        # Delete items from various positions
        delete_keys = [
            1,                      # First node
            self.cap + 1,           # Second node
            2 * self.cap + 1,       # Third node
            3 * self.cap - 1        # Fourth node 
        ]
        
        # Track violations after each deletion
        for i, key in enumerate(delete_keys):
            self.klist.delete(key)
            violations = self._verify_compaction_invariant(self.klist)
            if violations:
                self.fail(f"Compaction invariant violated after deletion {i+1} (key={key}): {violations}")
    
    def test_compaction_after_sparse_deletions(self):
        """Test compaction when many deletions create sparse nodes that need redistribution"""
        # Create a klist with exactly 3 full nodes
        total_items = 3 * self.cap
        for i in range(total_items):
            self.klist.insert(Item(i, f"val_{i}"))
        
        # Delete every other item in the first two nodes
        for i in range(0, 2 * self.cap, 2):
            self.klist.delete(i)
        
        # Check for violations
        violations = self._verify_compaction_invariant(self.klist)
        if violations:
            self.fail(f"Compaction invariant violated after sparse deletions: {violations}")
    
    def test_mixed_operations(self):
        """Test compaction invariant with mixed insert/delete operations"""
        # Start with some initial data
        for i in range(2 * self.cap):
            self.klist.insert(Item(i, f"val_{i}"))
            
        operations = [
            ("delete", 0),           # Delete first item
            ("insert", 100),         # Insert new item
            ("delete", self.cap),    # Delete from second node
            ("delete", self.cap+1),  # Delete another from second node
            ("insert", 101),         # Insert new item
            ("delete", 2),           # Delete from first node again
            ("delete", 3),           # Delete from first node again
        ]
        
        # Execute operations and check invariant after each
        for i, (op, key) in enumerate(operations):
            if op == "delete":
                self.klist.delete(key)
            else:
                self.klist.insert(Item(key, f"val_{key}"))
                
            violations = self._verify_compaction_invariant(self.klist)
            if violations:
                self.fail(f"Compaction invariant violated after operation {i+1} ({op} {key}): {violations}")
    
    def test_delete_entire_middle_node(self):
        """Test compaction when an entire node's contents are deleted"""
        # Create 3 nodes
        for i in range(3 * self.cap):
            self.klist.insert(Item(i, f"val_{i}"))
            
        # Delete all items from the middle node
        for key in range(self.cap, 2 * self.cap):
            self.klist.delete(key)
            
        # Check for violations
        violations = self._verify_compaction_invariant(self.klist)
        if violations:
            self.fail(f"Compaction invariant violated after emptying middle node: {violations}")
            
    def test_split_inplace_maintains_compaction(self):
        """Test that split_inplace operation maintains the compaction invariant"""
        # Create a multi-node klist
        for i in range(3 * self.cap):
            self.klist.insert(Item(i, f"val_{i}"))
            
        # Split at different points and verify compaction
        split_points = [self.cap//2, self.cap, self.cap + self.cap//2, 2*self.cap]
        
        for split_key in split_points:
            # Create a fresh klist for each test
            test_klist = self.KListClass()
            for i in range(3 * self.cap):
                test_klist.insert(Item(i, f"val_{i}"))
                
            # Perform split
            left, subtree, right = test_klist.split_inplace(split_key)
            
            # Check both resulting klists
            for name, klist in [("left", left), ("right", right)]:
                violations = self._verify_compaction_invariant(klist)
                if violations:
                    self.fail(f"Compaction invariant violated in {name} klist after split at {split_key}: {violations}")


if __name__ == "__main__":
    unittest.main()


# class TestKListGetEntry(TestKListBase):
#     def insert_sequence(self, keys):
#         """Helper: insert integer keys with dummy values."""
#         for k in keys:
#             self.klist.insert(Item(k, f"val_{k}"))
#         self.klist.check_invariant()

#     def assertGet(self, index, found_key, next_key):
#         """Helper: assert get_entry(index) returns expected keys."""
#         res = self.klist.get_entry(index)
#         if found_key is None:
#             self.assertIsNone(res.found_entry, f"Expected no entry at index {index}")
#         else:
#             self.assertIsNotNone(res.found_entry, f"Expected entry at index {index}")
#             self.assertEqual(res.found_entry.item.key, found_key)
#         if next_key is None:
#             self.assertIsNone(res.next_entry, f"Expected no successor for index {index}")
#         else:
#             self.assertIsNotNone(res.next_entry, f"Expected successor for index {index}")
#             self.assertEqual(res.next_entry.item.key, next_key)

#     def test_empty_list(self):
#         # retrieving any index from empty list returns (None, None)
#         for idx in [0, 1, -1, 100]:
#             self.assertGet(idx, None, None)

#     def test_type_error_non_int(self):
#         with self.assertRaises(TypeError):
#             self.klist.get_entry('0')
#         with self.assertRaises(TypeError):
#             self.klist.get_entry(1.5)

#     def test_single_node_boundaries(self):
#         keys = [10, 20, 30]
#         self.insert_sequence(keys)
#         # valid indices
#         self.assertGet(0, 10, 20)
#         self.assertGet(1, 20, 30)
#         self.assertGet(2, 30, None)
#         # out of range
#         self.assertGet(-1, None, None)
#         self.assertGet(3, None, None)

#     def test_single_node_varied(self):
#         keys = [5]
#         self.insert_sequence(keys)
#         self.assertGet(0, 5, None)
#         self.assertGet(1, None, None)

#     def test_two_nodes_indexing(self):
#         # fill first node, overflow one into second
#         keys = list(range(self.cap + 1))
#         self.insert_sequence(keys)
#         # first node: indices 0..cap-1
#         for i in range(self.cap):
#             next_key = i+1
#             if next_key == self.cap:
#                 # next entry is first of second node
#                 expected_next = self.cap
#             else:
#                 expected_next = next_key
#             self.assertGet(i, i, expected_next)
#         # index cap is first of second node
#         self.assertGet(self.cap, self.cap, None)

#     def test_multi_node_full_scan(self):
#         total = 3 * self.cap + 2
#         keys = list(range(total))
#         self.insert_sequence(keys)
#         # test all indices
#         for i in range(total):
#             expected_next = i+1 if i+1 < total else None
#             self.assertGet(i, i, expected_next)
#         # out of bounds
#         self.assertGet(total, None, None)
#         self.assertGet(total+5, None, None)

#     def test_rebuild_index_on_modification(self):
#         # if index is maintained, ensure it updates
#         if not hasattr(self.klist, '_prefix_counts'):
#             self.skipTest("Index not implemented")
#         # initial insert
#         keys = [1, 2, 3, 4]
#         self.insert_sequence(keys)
#         # delete middle
#         self.klist.delete(2)
#         self.klist._rebuild_index()
#         # now index 1 should be key=3
#         self.assertGet(1, 3, 4)