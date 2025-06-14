"""GKPlusTree factory module"""

from typing import Type, Tuple
import logging

from gplus_trees.klist_base import KListBase, KListNodeBase
from gplus_trees.factory import make_gplustree_classes
from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
    GKPlusNodeBase,
    DEFAULT_L_FACTOR
)

logger = logging.getLogger(__name__)

def make_gkplustree_classes(K: int, dimension: int = 1) -> Tuple[
    Type[GKPlusTreeBase], 
    Type[GKPlusNodeBase], 
    Type[KListBase], 
    Type[KListNodeBase]
]:
    """
    Factory function to generate GKPlus-tree and KList classes specialized 
    for a given capacity K and dimension.
    
    Args:
        K: The capacity value for KListNodes
        dimension: The initial dimension for the tree (default: 1)
        
    Returns:
        GKPlusTreeK: Subclass of GKPlusTreeBase with NodeClass=GKPlusNodeK and SetClass=KListK
        GKPlusNodeK: Subclass of GKPlusNodeBase with SetClass=KListK
        KListK: Subclass of KListBase with KListNodeClass=KListNodeK
        KListNodeK: Subclass of KListNodeBase with CAPACITY=K
    """
    logger.debug(f"Creating GKPlusTree classes with K={K}, dimension={dimension}")
    
    # Get the base G+tree classes from the main factory
    GPlusTreeK, GPlusNodeK, KListK, KListNodeK = make_gplustree_classes(K)
    
    # 1) Create the GKPlusNode class extending GPlusNodeBase
    GKPlusNodeK = type(
        f"GKPlusNode_K{K}_Dim{dimension}",
        (GKPlusNodeBase,),
        {
            "SetClass": KListK,  # Inherit KList class from base factory
            "__slots__": GKPlusNodeBase.__slots__
        }
    )
    
    # 2) Create the GKPlusTree class extending GPlusTreeBase
    GKPlusTreeK = type(
        f"GKPlusTree_K{K}_Dim{dimension}",
        (GKPlusTreeBase,),
        {
            "NodeClass": GKPlusNodeK,
            "SetClass": KListK,
            "KListClass": KListK,
            "DIM": dimension,
            "__slots__": GKPlusTreeBase.__slots__
        }
    )
    
    # 3) Set the TreeClass reference in the Node class
    GKPlusNodeK.TreeClass = GKPlusTreeK
    
    return GKPlusTreeK, GKPlusNodeK, KListK, KListNodeK

def create_gkplus_tree(K: int, dimension: int = 1, l_factor: float = DEFAULT_L_FACTOR) -> GKPlusTreeBase:
    """
    Create a new GKPlusTree with the specified capacity K and dimension.
    
    Args:
        K: The capacity of the tree's KListNodes
        dimension: The initial dimension for the tree
        l_factor: The threshold factor for KList <-> GKPlusTree conversions
        
    Returns:
        A new empty GKPlusTree with the specified capacity and dimension
    """
    GKPlusTreeK, _, _, _ = make_gkplustree_classes(K, dimension)
    tree = GKPlusTreeK(node=None, l_factor=l_factor)
    return tree