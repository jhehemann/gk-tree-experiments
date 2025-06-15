"""Tests for GK+-trees with factory pattern"""

from typing import List, Tuple, Optional
from gplus_trees.g_k_plus.g_k_plus_base import Item, Entry
from tests.test_base import GKPlusTreeTestCase

# Inherit from the unified base class but keep backward compatibility
class TreeTestCase(GKPlusTreeTestCase):
    """Legacy TreeTestCase that extends GKPlusTreeTestCase with backward compatibility."""
    
    def _assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
        self, node, items: List[Item]
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is a rank 1 leaf containing exactly `items` in order,
        and that all its subtrees are empty.

        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, 1, f"Leaf node rank should be 1")
        actual_len   = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Leaf node has {actual_len} items; expected {expected_len}"
        )
        self.assertIsNone(node.right_subtree, 
                          "Expected leaf node's right_subtree to be None")

        # verify each entry's key/value
        for i, (entry, expected) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected.key,
                f"Entry #{i} key: expected {expected.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected.value,
                f"Entry #{i} ({expected.key}) value: expected "
                f"{expected.value!r}, "
                f"got {entry.item.value!r}"
            )
