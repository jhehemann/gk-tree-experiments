"""Tests for GK+-trees with factory pattern"""

from typing import List, Tuple, Optional
from gplus_trees.base import Entry, LeafItem
from gplus_trees.g_k_plus.utils import calc_ranks_multi_dims
from tests.test_base import GKPlusTreeTestCase
from gplus_trees.logging_config import get_test_logger

logger = get_test_logger("GKPlusTree")

# Inherit from the unified base class but keep backward compatibility
class TreeTestCase(GKPlusTreeTestCase):
    """Legacy TreeTestCase that extends GKPlusTreeTestCase with backward compatibility."""

    def _log_ranks(self, K: int, keys: Optional[List] = None, dimensions: int = 10) -> None:
        """
        Print the ranks of keys provided in <list> for debugging purposes.
        Args:
            K (int): The number of desired entries per G-node.
            keys (Optional[List]): List of keys to log ranks for. Defaults to [-1, -2, ..., -10].
            dimensions (int): Number of dimensions to consider for rank calculation. Defaults to 10.
        Returns:
            None
        """
        if keys is None:
            keys = list(range(-1, -11, -1))

        # key_map = {i: get_dummy(dim=abs(i)) for i in keys}
        absolute_keys = [abs(key) for key in keys]
        ranks = calc_ranks_multi_dims(absolute_keys, K, dimensions=dimensions)

        # Log ranks for debugging
        logger.debug(f"Ranks for keys: {list(keys)}")
        for dim, ranks in enumerate(ranks):
            logger.debug(f"Dimension {dim + 1}: {ranks}")
    
    def _assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
        self, node, items: List[LeafItem]
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
