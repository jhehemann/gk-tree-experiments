"""Tests for k-lists with factory pattern"""

import itertools
import logging
import random
import unittest

# Import factory function instead of concrete classes
from gplus_trees.base import Entry
from gplus_trees.factory import make_gplustree_classes
from tests.test_base import BaseTestCase

logger = logging.getLogger(__name__)

# Test with different capacities to ensure factory works correctly
TEST_CAPACITIES = [4, 8, 16]


class TestKListFactory(BaseTestCase):
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
                    self.assertNotEqual(
                        classes[k1][2], classes[k2][2], f"KList classes for K={k1} and K={k2} should be different"
                    )


class TestKListBase(BaseTestCase):
    """Base class for all KList factory tests"""

    K = 4  # Default capacity; override in subclasses

    def setUp(self):
        _, _, self.KListClass, self.KListNodeClass = make_gplustree_classes(self.K)
        self.klist = self.KListClass()
        self.cap = self.K

    def tearDown(self):
        self.validate_klist(self.klist)

    def insert_sequence(self, keys):
        """Helper to insert integer keys with dummy values."""
        for k in keys:
            entry = Entry(self.make_item(k, f"val_{k}"), None)
            self.klist.insert_entry(entry)
        # ensure invariants
        self.klist.check_invariant()

    def extract_all_keys(self, klist=None):
        """Traverse a KList and collect all item keys in order."""
        if klist is None:
            klist = self.klist
        keys = []
        node = klist.head
        while node:
            keys.extend(e.item.key for e in node.entries)
            node = node.next
        return keys

    extract_keys = extract_all_keys  # alias used by TestSplitInplace


class TestKListInsert(TestKListBase):
    def test_insert_into_empty(self):
        # Inserting into an empty list should set head and tail
        self.assertIsNone(self.klist.head)
        self.klist.insert_entry(Entry(self.make_item(1, "val_1"), None))
        self.assertIsNotNone(self.klist.head)
        self.assertIs(self.klist.head, self.klist.tail)
        self.assertEqual(self.extract_all_keys(), [1])
        self.assertEqual(self.klist.item_count(), 1)

    def test_insert_in_order(self):
        # Insert keys in sorted order one by one
        keys = [1, 2, 3, 4, 5]
        for key in keys:
            self.klist.insert_entry(Entry(self.make_item(key, f"val_{key}"), None))
        self.assertEqual(self.extract_all_keys(), keys)
        self.assertEqual(self.klist.item_count(), len(keys))

    def test_insert_out_of_order(self):
        # Insert keys in random order, final list must be sorted
        keys = [4, 1, 5, 2, 3]
        for key in keys:
            self.klist.insert_entry(Entry(self.make_item(key, f"val_{key}"), None))
        self.assertEqual(self.extract_all_keys(), sorted(keys))
        self.assertEqual(self.klist.item_count(), len(keys))

    def test_single_node_overflow(self):
        # Fill exactly one node to capacity, then insert one more
        keys = list(range(self.cap))
        for k in keys:
            self.klist.insert_entry(Entry(self.make_item(k, f"val_{k}"), None))
        # one more causes a second node
        extra = self.cap
        self.klist.insert_entry(Entry(self.make_item(extra, f"val_{extra}"), None))
        all_keys = self.extract_all_keys()
        self.assertEqual(len(all_keys), self.cap + 1)
        self.assertEqual(self.klist.item_count(), len(all_keys))

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
            self.klist.insert_entry(Entry(self.make_item(k, f"val_{k}"), None))

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
        # Insert duplicate keys - they all should appear in order
        _, inserted, _ = self.klist.insert_entry(Entry(self.make_item(7, "duplicate"), None))
        self.assertTrue(inserted, "First insert should succeed")
        for _ in range(3):
            _, inserted, _ = self.klist.insert_entry(Entry(self.make_item(7, "duplicate"), None))
            self.assertFalse(inserted, "Subsequent inserts should fail for duplicates")
        self.assertEqual(self.extract_all_keys(), [7])

    def test_tail_fast_path(self):
        # Repeatedly append monotonic integer keys
        for i in range(100):
            self.klist.insert_entry(Entry(self.make_item(i, f"val_{i}"), None))
        self.assertEqual(self.extract_all_keys(), list(range(100)))

    def test_interleaved_inserts_and_checks(self):
        # Interleave inserts with invariant checks to catch transient issues
        sequence = [5, 2, 8, 3, 9, 1, 7]
        for key in sequence:
            self.klist.insert_entry(Entry(self.make_item(key, f"val_{key}"), None))
            so_far = self.extract_all_keys()
            self.assertEqual(so_far, sorted(so_far))

    def test_random_insert(self):
        # Insert a shuffled pattern and verify final sort
        keys = list(range(self.cap * 5))
        random.shuffle(keys)
        for k in keys:
            self.klist.insert_entry(Entry(self.make_item(k, f"val_{k}"), None))
        all_keys = self.extract_all_keys()
        self.assertEqual(all_keys, sorted(all_keys))
        self.assertEqual(len(all_keys), self.cap * 5)


class TestKListDelete(TestKListBase):
    def test_delete_on_empty_list(self):
        # deleting from an empty KList should do nothing
        before = self.extract_all_keys()
        self.klist.delete(999)  # nonexistent int
        after = self.extract_all_keys()
        self.assertEqual(before, after)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_nonexistent_key(self):
        # insert some items, then delete a missing key
        self.insert_sequence([1, 2, 3])
        before = self.extract_all_keys()
        self.klist.delete(999)
        after = self.extract_all_keys()
        self.assertEqual(before, after)

    def test_delete_only_item(self):
        # after deleting the sole element, head and tail should be None
        self.insert_sequence([5])
        self.klist.delete(5)
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)

    def test_delete_head_key(self):
        # delete the first key in a multi-element, single-node list
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        self.klist.delete(1)
        result = self.extract_all_keys()
        self.assertEqual(result, [2, 3])
        # head should remain the same node
        self.assertIsNotNone(self.klist.head)
        self.klist.check_invariant()

    def test_delete_tail_key(self):
        # delete the last key in a single-node list
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2])
        self.klist.check_invariant()

    def test_delete_middle_key(self):
        # delete a middle key and ensure rebalance keeps packing
        keys = [1, 2, 3, 4, 5]
        self.insert_sequence(keys)
        # ensure at least two nodes exist
        self.assertGreater(len(self.klist.head.entries), 0)
        self.klist.delete(3)
        result = self.extract_all_keys()
        self.assertEqual(result, [1, 2, 4, 5])
        self.klist.check_invariant()

    def test_delete_causes_node_removal(self):
        # build exactly two nodes: first full, second with 1 entry
        keys = list(range(self.cap + 1))
        self.insert_sequence(keys)
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
        self.insert_sequence(keys)
        for k in [2, 5, 1, 7, 4]:
            self.klist.delete(k)
            self.assertNotIn(k, self.extract_all_keys())
            self.klist.check_invariant()
        # remaining should be [3,6]
        self.assertEqual(self.extract_all_keys(), [3, 6])

    def test_delete_all_nodes(self):
        # insert enough to create 3+ nodes, then delete everything one by one
        keys = list(range(3 * self.cap + 2))
        self.insert_sequence(keys)
        for k in keys:
            self.klist.delete(k)
        # list should be empty afterwards
        self.assertIsNone(self.klist.head)
        self.assertIsNone(self.klist.tail)


class TestKListRetrieve(TestKListBase):
    def assertRetrieval(self, key, found_key, next_key):
        """
        Helper: call retrieve(key) and assert that
          result[0].item.key == found_key  (or None)
          result[1].item.key == next_key    (or None)
        """
        found_entry, next_entry = self.klist.retrieve(key)
        if found_key is None:
            self.assertIsNone(found_entry, f"Expected no entry for {key}")
        else:
            self.assertIsNotNone(found_entry)
            self.assertEqual(found_entry.item.key, found_key)
        if next_key is None:
            self.assertIsNone(next_entry, f"Expected no successor for {key}")
        else:
            self.assertIsNotNone(next_entry)
            self.assertEqual(next_entry.item.key, next_key)

    def test_retrieve_empty(self):
        # empty list returns (None, None)
        found_entry, next_entry = self.klist.retrieve(123)
        self.assertIsNone(found_entry)
        self.assertIsNone(next_entry)

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

    def test_type_error_on_float(self):
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
            expected_next = keys[idx + 1] if idx + 1 < len(keys) else None
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
        # empty list should return None
        with self.subTest("max"):
            found_entry, next_entry = self.klist.get_max()
            self.assertIsNone(found_entry)
            self.assertIsNone(next_entry)

        with self.subTest("min"):
            found_entry, next_entry = self.klist.get_min()
            self.assertIsNone(found_entry)
            self.assertIsNone(next_entry)

    def test_single_node_single_entry(self):
        # fill one node with a single entry
        keys = [10]
        self.insert_sequence(keys)

        with self.subTest("max"):
            found_entry, next_entry = self.klist.get_max()
            self.assertEqual(found_entry.item.key, 10)
            self.assertIsNone(next_entry)

        with self.subTest("min"):
            found_entry, next_entry = self.klist.get_min()
            self.assertEqual(found_entry.item.key, 10)
            self.assertIsNone(next_entry)

    def test_single_node_cap(self):
        # fill one node up to capacity
        keys = list(range(self.cap))
        self.insert_sequence(keys)

        with self.subTest("max"):
            found_entry, next_entry = self.klist.get_max()
            self.assertEqual(found_entry.item.key, self.cap - 1)
            self.assertIsNone(next_entry)

        with self.subTest("min"):
            found_entry, next_entry = self.klist.get_min()
            self.assertEqual(found_entry.item.key, 0)
            self.assertEqual(next_entry.item.key, 1)

    def test_multi_nodes(self):
        # fill first node, overflow into second
        keys = list(range(self.cap * 3 + 1))
        self.insert_sequence(keys)

        with self.subTest("max"):
            found_entry, next_entry = self.klist.get_max()
            self.assertEqual(found_entry.item.key, self.cap * 3)
            self.assertIsNone(next_entry)

        with self.subTest("min"):
            found_entry, next_entry = self.klist.get_min()
            self.assertEqual(found_entry.item.key, 0)
            self.assertEqual(next_entry.item.key, 1)


class TestKListIndex(TestKListBase):
    def test_empty_index(self):
        # Before any operations, index lists should exist and be empty
        self.assertTrue(hasattr(self.klist, "_nodes"))
        self.assertEqual(self.klist._nodes, [], "_nodes should be initialized empty")
        self.assertTrue(hasattr(self.klist, "_prefix_counts_tot"))
        self.assertEqual(self.klist._prefix_counts_tot, [], "_prefix_counts_tot should be initialized empty")

    def test_index_after_single_insert(self):
        # Insert one item, rebuild, then index should have 1 node, prefix_counts [1]
        self.klist.insert_entry(Entry(self.make_item(10, "val_10"), None))
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts_tot, [1])
        # Node in list should be the head
        self.assertIs(self.klist._nodes[0], self.klist.head)

    def test_index_after_multiple_inserts_no_overflow(self):
        # Insert fewer than CAPACITY items
        keys = list(range(self.cap - 1))
        self.insert_sequence(keys)
        # Still one node, prefix_counts = [len(keys)]
        self.assertEqual(len(self.klist._nodes), 1)
        self.assertEqual(self.klist._prefix_counts_tot, [len(keys)])

    def test_index_after_overflow(self):
        # Insert exactly CAPACITY + 2 items → 2 nodes
        total = self.cap + 2
        self.insert_sequence(range(total))
        # Should have 2 nodes
        self.assertEqual(len(self.klist._nodes), 2)
        # prefix_counts: first node cap, second cap+2
        expected = [self.cap, total]
        self.assertEqual(self.klist._prefix_counts_tot, expected)
        # Check that _nodes entries match actual chain
        node = self.klist.head
        for _idx, n in enumerate(self.klist._nodes):
            self.assertIs(n, node)
            node = node.next

    def test_prefix_counts_monotonic_and_correct(self):
        # Random insertion pattern, then check prefix sums
        keys = [5, 1, 9, 2, 8, 3, 7, 4, 6, 0]
        self.insert_sequence(keys)
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
        self.assertEqual(self.klist._prefix_counts_tot, expected)
        # Ensure strictly increasing
        for a, b in itertools.pairwise(expected):
            self.assertLess(a, b)

    def test_index_after_bulk_deletes(self):
        # Fill three nodes exactly, then remove the middle node
        total = 3 * self.cap
        self.insert_sequence(range(total))
        # delete all keys in the middle node
        middle_start = self.cap
        middle_end = 2 * self.cap - 1
        for k in range(middle_start, middle_end + 1):
            self.klist.delete(k)
        # Now exactly two nodes remain (the head and the tail)
        self.assertEqual(len(self.klist._nodes), 2)
        # And the prefix sums should be [CAPACITY, 3*CAPACITY]
        self.assertEqual(self.klist._prefix_counts_tot, [self.cap, 2 * self.cap])


class TestSplitInplace(TestKListBase):
    def test_empty_split(self):
        left, subtree, right, next_entry = self.klist.split_inplace(5)
        # both sides empty
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        self.assertIsNone(next_entry)
        # invariants hold
        left.check_invariant()
        right.check_invariant()

    def test_split_before_all_keys(self):
        # Insert some keys
        keys = [10, 20, 30]
        self.insert_sequence(keys)
        # split before smallest
        left, subtree, right, next_entry = self.klist.split_inplace(5)
        self.assertEqual(self.extract_keys(left), [])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), keys)
        self.assertEqual(next_entry.item.key, 10)
        left.check_invariant()
        right.check_invariant()

    def test_split_at_dummy(self):
        # Insert some keys
        keys = [-1, 20, 30]
        self.insert_sequence(keys)
        # split before smallest
        left, subtree, right, next_entry = self.klist.split_inplace(-1)
        self.assertEqual(self.extract_keys(left), [])
        self.assertEqual(left.real_item_count(), 0)
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [20, 30])
        self.assertEqual(right.real_item_count(), 2)
        self.assertEqual(next_entry.item.key, 20)
        left.check_invariant()
        right.check_invariant()

    def test_split_after_all_keys(self):
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        # split after largest
        left, subtree, right, next_entry = self.klist.split_inplace(10)
        self.assertEqual(self.extract_keys(left), keys)
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        self.assertIsNone(next_entry)
        left.check_invariant()
        right.check_invariant()

    def test_split_exact_middle(self):
        keys = [1, 2, 3, 4, 5]
        self.insert_sequence(keys)
        # split on 3
        left, subtree, right, next_entry = self.klist.split_inplace(3)
        self.assertEqual(self.extract_keys(left), [1, 2])
        self.assertIsNone(subtree)  # default left_subtree None
        self.assertEqual(self.extract_keys(right), [4, 5])
        self.assertEqual(next_entry.item.key, 4)
        left.check_invariant()
        right.check_invariant()

    def test_split_nonexistent_between(self):
        keys = [10, 20, 30, 40]
        self.insert_sequence(keys)
        # split on a key not present but between 20 and 30
        left, subtree, right, next_entry = self.klist.split_inplace(25)
        self.assertEqual(self.extract_keys(left), [10, 20])
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [30, 40])
        self.assertEqual(next_entry.item.key, 30)
        left.check_invariant()
        right.check_invariant()

    def test_split_at_node_boundary_max(self):
        # make at least two nodes
        total = self.cap + 1
        self.insert_sequence(range(total))
        # first node has keys 0..cap-1, second has [cap+1]
        # split exactly at cap (first of second node)
        left, subtree, right, next_entry = self.klist.split_inplace(self.cap)

        self.assertEqual(self.extract_keys(left), list(range(self.cap)))
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [])
        self.assertIsNone(next_entry)
        left.check_invariant()
        right.check_invariant()

    def test_split_at_node_boundary(self):
        # make at least two nodes
        total = self.cap + 2
        self.insert_sequence(range(total))
        # first node has keys 0..cap-1, second has [cap+1]
        # split exactly at cap (first of second node)
        left, subtree, right, next_entry = self.klist.split_inplace(self.cap)

        self.assertEqual(self.extract_keys(left), list(range(self.cap)))
        self.assertIsNone(subtree)
        self.assertEqual(self.extract_keys(right), [self.cap + 1])
        self.assertEqual(next_entry.item.key, self.cap + 1)
        left.check_invariant()
        right.check_invariant()

    def test_split_with_subtree_propagation(self):
        # insert and assign a left_subtree for a particular key
        keys = [1, 2, 3]
        self.insert_sequence(keys)
        # update left_subtree for key=2
        # Create tree using the factory
        self.TreeClass, _, _, _ = make_gplustree_classes(self.K)
        subtree = self.TreeClass()
        found = self.klist.retrieve(2)[0]
        self.assertIsNotNone(found)
        found.left_subtree = subtree
        left, st, right, next_entry = self.klist.split_inplace(2)
        # left contains [1], subtree returned
        self.assertEqual(self.extract_keys(left), [1])
        self.assertIs(st, subtree)
        self.assertEqual(self.extract_keys(right), [3])
        self.assertEqual(next_entry.item.key, 3)
        left.check_invariant()
        right.check_invariant()

    def test_split_multiple_times(self):
        # perform sequential splits
        keys = list(range(6))
        self.insert_sequence(keys)
        # split on 2
        l1, _, r1, next_entry = self.klist.split_inplace(2)
        self.assertEqual(next_entry.item.key, 3)
        # split r1 on 4
        l2, _, r2, next_entry = r1.split_inplace(4)
        self.assertEqual(next_entry.item.key, 5)
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

    K = 16

    def test_compaction_after_single_deletion(self):
        """Test compaction after deleting a single item"""
        # Create a klist with multiple nodes
        items = 2 * self.cap + 1  # Creates 3 nodes
        self.insert_sequence(range(items))

        # Delete an item from the first node
        self.klist.delete(0)

        # Check compaction invariant
        self.klist.check_invariant()

    def test_compaction_after_middle_deletion(self):
        """Test compaction after deleting an item from the middle of the list"""
        # Create a klist with 3 nodes (full capacity)
        total_items = 3 * self.cap
        self.insert_sequence(range(total_items))

        # Delete an item from the middle node
        middle_key = self.cap + self.cap // 2
        self.klist.delete(middle_key)

        # Check compaction invariant
        self.klist.check_invariant()

    def test_compaction_after_multiple_deletions(self):
        """Test compaction after multiple deletions from different nodes"""
        # Create a klist with multiple nodes
        total_items = 4 * self.cap
        self.insert_sequence(range(total_items))

        # Delete items from various positions
        delete_keys = [
            1,  # First node
            self.cap + 1,  # Second node
            2 * self.cap + 1,  # Third node
            3 * self.cap - 1,  # Fourth node
        ]

        # Check invariant after each deletion
        for key in delete_keys:
            self.klist.delete(key)
            self.klist.check_invariant()

    def test_compaction_after_sparse_deletions(self):
        """Test compaction when many deletions create sparse nodes that need redistribution"""
        # Create a klist with exactly 3 full nodes
        total_items = 3 * self.cap
        self.insert_sequence(range(total_items))

        # Delete every other item in the first two nodes
        for i in range(0, 2 * self.cap, 2):
            self.klist.delete(i)

        # Check compaction invariant
        self.klist.check_invariant()

    def test_mixed_operations(self):
        """Test compaction invariant with mixed insert/delete operations"""
        # Start with some initial data
        self.insert_sequence(range(2 * self.cap))

        operations = [
            ("delete", 0),  # Delete first item
            ("insert", 100),  # Insert new item
            ("delete", self.cap),  # Delete from second node
            ("delete", self.cap + 1),  # Delete another from second node
            ("insert", 101),  # Insert new item
            ("delete", 2),  # Delete from first node again
            ("delete", 3),  # Delete from first node again
        ]

        # Execute operations and check invariant after each
        for op, key in operations:
            if op == "delete":
                self.klist.delete(key)
            else:
                self.klist.insert_entry(Entry(self.make_item(key, f"val_{key}"), None))
            self.klist.check_invariant()

    def test_delete_entire_middle_node(self):
        """Test compaction when an entire node's contents are deleted"""
        # Create 3 nodes
        self.insert_sequence(range(3 * self.cap))

        # Delete all items from the middle node
        for key in range(self.cap, 2 * self.cap):
            self.klist.delete(key)

        # Check compaction invariant
        self.klist.check_invariant()

    def test_split_inplace_maintains_compaction(self):
        """Test that split_inplace operation maintains the compaction invariant"""
        # Create a multi-node klist
        self.insert_sequence(range(3 * self.cap))

        # Split at different points and verify compaction
        split_points = [self.cap // 2, self.cap, self.cap + self.cap // 2, 2 * self.cap]

        for split_key in split_points:
            # Create a fresh klist for each test
            test_klist = self.KListClass()
            for i in range(3 * self.cap):
                test_klist.insert_entry(Entry(self.make_item(i, f"val_{i}"), None))

            # Perform split
            left, _subtree, right, _next_entry = test_klist.split_inplace(split_key)

            # Check both resulting klists
            left.check_invariant()
            right.check_invariant()


class TestKListNodeHelpers(TestKListBase):
    """Direct tests for KListNodeBase helper methods that keep parallel lists in sync."""

    def _make_node(self, keys):
        """Create a node and populate it with entries for the given keys."""
        node = self.KListNodeClass()
        for k in keys:
            node._append_entry(Entry(self.make_item(k, f"val_{k}"), None))
        return node

    def _extract(self, node):
        """Return (entry_keys, keys, real_keys) from a node."""
        return (
            [e.item.key for e in node.entries],
            list(node.keys),
            list(node.real_keys),
        )

    # --- _append_entry ---

    def test_append_entry_to_empty(self):
        node = self.KListNodeClass()
        entry = Entry(self.make_item(10, "v10"), None)
        node._append_entry(entry)
        self.assertEqual(self._extract(node), ([10], [10], [10]))

    def test_append_entry_preserves_order(self):
        node = self._make_node([1, 2, 3])
        node._append_entry(Entry(self.make_item(4, "v4"), None))
        self.assertEqual(self._extract(node), ([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))

    def test_append_dummy_entry(self):
        """Dummy (negative-key) entries should not appear in real_keys."""
        node = self.KListNodeClass()
        node._append_entry(Entry(self.make_item(-5, "d1"), None))
        node._append_entry(Entry(self.make_item(-1, "d2"), None))
        ek, k, rk = self._extract(node)
        self.assertEqual(ek, [-5, -1])
        self.assertEqual(k, [-5, -1])
        self.assertEqual(rk, [])

    # --- _insert_entry_at ---

    def test_insert_entry_at_beginning(self):
        node = self._make_node([20, 30])
        node._insert_entry_at(0, Entry(self.make_item(10, "v10"), None))
        self.assertEqual(self._extract(node), ([10, 20, 30], [10, 20, 30], [10, 20, 30]))

    def test_insert_entry_at_middle(self):
        node = self._make_node([10, 30])
        node._insert_entry_at(1, Entry(self.make_item(20, "v20"), None))
        self.assertEqual(self._extract(node), ([10, 20, 30], [10, 20, 30], [10, 20, 30]))

    def test_insert_entry_at_end(self):
        node = self._make_node([10, 20])
        node._insert_entry_at(2, Entry(self.make_item(30, "v30"), None))
        self.assertEqual(self._extract(node), ([10, 20, 30], [10, 20, 30], [10, 20, 30]))

    # --- _pop_last_entry ---

    def test_pop_last_entry(self):
        node = self._make_node([10, 20, 30])
        popped = node._pop_last_entry()
        self.assertEqual(popped.item.key, 30)
        self.assertEqual(self._extract(node), ([10, 20], [10, 20], [10, 20]))

    def test_pop_last_entry_single(self):
        node = self._make_node([5])
        popped = node._pop_last_entry()
        self.assertEqual(popped.item.key, 5)
        self.assertEqual(self._extract(node), ([], [], []))

    # --- _pop_first_entry ---

    def test_pop_first_entry(self):
        node = self._make_node([10, 20, 30])
        popped = node._pop_first_entry()
        self.assertEqual(popped.item.key, 10)
        self.assertEqual(self._extract(node), ([20, 30], [20, 30], [20, 30]))

    def test_pop_first_entry_single(self):
        node = self._make_node([42])
        popped = node._pop_first_entry()
        self.assertEqual(popped.item.key, 42)
        self.assertEqual(self._extract(node), ([], [], []))

    # --- _remove_entry_at ---

    def test_remove_entry_at_beginning(self):
        node = self._make_node([10, 20, 30])
        removed = node._remove_entry_at(0)
        self.assertEqual(removed.item.key, 10)
        self.assertEqual(self._extract(node), ([20, 30], [20, 30], [20, 30]))

    def test_remove_entry_at_middle(self):
        node = self._make_node([10, 20, 30])
        removed = node._remove_entry_at(1)
        self.assertEqual(removed.item.key, 20)
        self.assertEqual(self._extract(node), ([10, 30], [10, 30], [10, 30]))

    def test_remove_entry_at_end(self):
        node = self._make_node([10, 20, 30])
        removed = node._remove_entry_at(2)
        self.assertEqual(removed.item.key, 30)
        self.assertEqual(self._extract(node), ([10, 20], [10, 20], [10, 20]))

    # --- _transfer_first_n_from ---

    def test_transfer_zero(self):
        """Transferring 0 entries is a no-op."""
        src = self._make_node([10, 20, 30])
        dst = self._make_node([1, 2])
        dst._transfer_first_n_from(src, 0)
        self.assertEqual(self._extract(dst), ([1, 2], [1, 2], [1, 2]))
        self.assertEqual(self._extract(src), ([10, 20, 30], [10, 20, 30], [10, 20, 30]))

    def test_transfer_partial(self):
        """Transfer some entries from source to dest."""
        src = self._make_node([10, 20, 30])
        dst = self._make_node([1, 2])
        dst._transfer_first_n_from(src, 2)
        self.assertEqual(self._extract(dst), ([1, 2, 10, 20], [1, 2, 10, 20], [1, 2, 10, 20]))
        self.assertEqual(self._extract(src), ([30], [30], [30]))

    def test_transfer_all(self):
        """Transfer all entries from source to dest."""
        src = self._make_node([10, 20, 30])
        dst = self._make_node([1, 2])
        dst._transfer_first_n_from(src, 3)
        self.assertEqual(self._extract(dst), ([1, 2, 10, 20, 30], [1, 2, 10, 20, 30], [1, 2, 10, 20, 30]))
        self.assertEqual(self._extract(src), ([], [], []))

    def test_transfer_into_empty(self):
        """Transfer into an empty destination node."""
        src = self._make_node([5, 6, 7])
        dst = self.KListNodeClass()
        dst._transfer_first_n_from(src, 2)
        self.assertEqual(self._extract(dst), ([5, 6], [5, 6], [5, 6]))
        self.assertEqual(self._extract(src), ([7], [7], [7]))

    def test_transfer_with_dummy_keys(self):
        """Dummy (negative) keys should not appear in real_keys after transfer."""
        src = self.KListNodeClass()
        src._append_entry(Entry(self.make_item(-3, "d1"), None))
        src._append_entry(Entry(self.make_item(-1, "d2"), None))
        src._append_entry(Entry(self.make_item(5, "v5"), None))

        dst = self.KListNodeClass()
        dst._transfer_first_n_from(src, 2)
        self.assertEqual(self._extract(dst), ([-3, -1], [-3, -1], []))
        self.assertEqual(self._extract(src), ([5], [5], [5]))

    def test_transfer_negative(self):
        """Transferring negative n is a no-op."""
        src = self._make_node([10, 20])
        dst = self._make_node([1])
        dst._transfer_first_n_from(src, -1)
        self.assertEqual(self._extract(dst), ([1], [1], [1]))
        self.assertEqual(self._extract(src), ([10, 20], [10, 20], [10, 20]))


class TestKListProperties(TestKListBase):
    """Tests for O(1) property methods: __bool__, is_empty, item_slot_count,
    physical_height, find_pivot, __iter__."""

    def test_bool_empty(self):
        self.assertFalse(bool(self.klist))

    def test_bool_nonempty(self):
        self.klist.insert_entry(Entry(self.make_item(1, "v"), None))
        self.assertTrue(bool(self.klist))

    def test_bool_after_delete_to_empty(self):
        self.klist.insert_entry(Entry(self.make_item(1, "v"), None))
        self.klist.delete(1)
        self.assertFalse(bool(self.klist))

    def test_is_empty(self):
        self.assertTrue(self.klist.is_empty())
        self.klist.insert_entry(Entry(self.make_item(1, "v"), None))
        self.assertFalse(self.klist.is_empty())

    def test_item_slot_count_empty(self):
        self.assertEqual(self.klist.item_slot_count(), 0)

    def test_item_slot_count_single_node(self):
        self.klist.insert_entry(Entry(self.make_item(1, "v"), None))
        self.assertEqual(self.klist.item_slot_count(), self.cap)

    def test_item_slot_count_multi_node(self):
        for i in range(self.cap + 1):
            self.klist.insert_entry(Entry(self.make_item(i, f"v{i}"), None))
        self.assertEqual(self.klist.item_slot_count(), 2 * self.cap)

    def test_physical_height_empty(self):
        self.assertEqual(self.klist.physical_height(), 0)

    def test_physical_height_one_node(self):
        self.klist.insert_entry(Entry(self.make_item(1, "v"), None))
        self.assertEqual(self.klist.physical_height(), 1)

    def test_physical_height_multi_nodes(self):
        for i in range(3 * self.cap + 1):
            self.klist.insert_entry(Entry(self.make_item(i, f"v{i}"), None))
        self.assertEqual(self.klist.physical_height(), 4)

    def test_find_pivot_empty(self):
        entry, nxt = self.klist.find_pivot()
        self.assertIsNone(entry)
        self.assertIsNone(nxt)

    def test_find_pivot_returns_min(self):
        self.insert_sequence([10, 20, 30])
        entry, nxt = self.klist.find_pivot()
        self.assertEqual(entry.item.key, 10)
        self.assertEqual(nxt.item.key, 20)

    def test_iter_empty(self):
        self.assertEqual(list(self.klist), [])

    def test_iter_ordered(self):
        keys = [5, 1, 9, 3, 7]
        self.insert_sequence(sorted(keys))
        iter_keys = [e.item.key for e in self.klist]
        self.assertEqual(iter_keys, sorted(keys))

    def test_iter_multi_node(self):
        keys = list(range(self.cap * 3 + 2))
        self.insert_sequence(keys)
        iter_keys = [e.item.key for e in self.klist]
        self.assertEqual(iter_keys, keys)

    def test_insert_entry_type_error(self):
        """insert_entry raises TypeError for non-Entry input."""
        with self.assertRaises(TypeError):
            self.klist.insert_entry("not an Entry")
        with self.assertRaises(TypeError):
            self.klist.insert_entry(42)

    def test_real_item_count_empty(self):
        self.assertEqual(self.klist.real_item_count(), 0)

    def test_real_item_count_with_items(self):
        self.insert_sequence([10, 20, 30])
        self.assertEqual(self.klist.real_item_count(), 3)

    def test_item_count_consistency(self):
        """item_count matches the number of entries yielded by __iter__."""
        keys = list(range(self.cap * 2 + 3))
        self.insert_sequence(keys)
        self.assertEqual(self.klist.item_count(), len(list(self.klist)))


class TestCheckInvariant(TestKListBase):
    """Tests that check_invariant correctly detects structural violations."""

    def tearDown(self):
        # Skip the default invariant check — these tests intentionally corrupt the klist.
        pass

    def test_passes_on_valid_klist(self):
        """check_invariant should not raise on a correctly-built k-list."""
        self.insert_sequence(list(range(self.cap * 2 + 1)))
        self.klist.check_invariant()  # should not raise

    def test_detects_tail_not_last(self):
        """Tail pointer not referencing the final node should fail."""
        self.insert_sequence(list(range(self.cap + 1)))
        # Corrupt: set tail to head even though head has a successor
        self.klist.tail = self.klist.head
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_detects_intra_node_sort_violation(self):
        """Entries out of order within a single node should fail."""
        self.klist.insert_entry(Entry(self.make_item(10, "v10"), None))
        self.klist.insert_entry(Entry(self.make_item(20, "v20"), None))
        # Corrupt: swap entries within the node
        node = self.klist.head
        node.entries[0], node.entries[1] = node.entries[1], node.entries[0]
        node.keys[0], node.keys[1] = node.keys[1], node.keys[0]
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_detects_keys_list_out_of_sync(self):
        """keys list not matching entries should fail."""
        self.insert_sequence([10, 20, 30])
        node = self.klist.head
        # Corrupt: tamper with keys list
        node.keys[1] = 999
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_detects_real_keys_out_of_sync(self):
        """real_keys list not matching entries should fail."""
        self.insert_sequence([10, 20, 30])
        node = self.klist.head
        # Corrupt: add a phantom key to real_keys
        node.real_keys.append(999)
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_detects_inter_node_order_violation(self):
        """Keys not strictly increasing across nodes should fail."""
        # Build two nodes
        for i in range(self.cap + 1):
            self.klist.insert_entry(Entry(self.make_item(i, f"v{i}"), None))
        # Corrupt: set the first entry of the second node to a key <= the last in the first
        second_node = self.klist.head.next
        first_node_max = self.klist.head.entries[-1].item.key
        # Overwrite key to violate ordering
        second_node.entries[0] = Entry(self.make_item(first_node_max, "dup"), None)
        second_node.keys[0] = first_node_max
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_detects_compaction_violation(self):
        """A non-tail node below capacity should fail."""
        # Build two nodes
        for i in range(self.cap + 2):
            self.klist.insert_entry(Entry(self.make_item(i, f"v{i}"), None))
        self.assertEqual(len(self.klist.head.entries), self.cap)
        # Corrupt: steal an entry from the first node without rebalancing
        self.klist.head._pop_last_entry()
        # First node now has cap-1 entries but is not the tail -> violation
        with self.assertRaises(AssertionError):
            self.klist.check_invariant()

    def test_passes_on_empty(self):
        """check_invariant should not raise on an empty k-list."""
        self.klist.check_invariant()


if __name__ == "__main__":
    unittest.main()
