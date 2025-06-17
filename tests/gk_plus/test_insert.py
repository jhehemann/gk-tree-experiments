import sys
import os
import unittest
import random
from typing import List, TYPE_CHECKING
from itertools import product
from tqdm import tqdm
import copy
from statistics import median_low

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.g_k_plus.utils import calc_rank
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from tests.logconfig import logger

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

class TestGKPlusInsert(GKPlusTreeTestCase):

    def test_split_abcdefghij(self):
        keys  =  [761, 346, 990, 874, 340, 250]
        # keys  =  [4, 3, 6, 5, 2, 1]
        ranks =  [1, 1, 1, 1, 1, 2]
        items = [Item(k, f"val_{k}") for k in keys]

        # build the tree once
        base_tree = create_gkplus_tree(K=4, l_factor=1.0)
        msg = ""
        for item, rank in zip(items, ranks):
            base_tree, _ = base_tree.insert(item, rank)
            msg += f"Tree after inserting {item.key}: {print_pretty(base_tree)}"

        exp_keys = keys
        dummies_left = self.get_dummies(base_tree)
        exp_keys = sorted(dummies_left + exp_keys)
        self.validate_tree(base_tree, exp_keys, msg)