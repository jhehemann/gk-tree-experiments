"""Factory for Merkle-enabled GPlus-trees"""

from typing import Type, Tuple, Dict

from gplus_trees.factory import make_gplustree_classes
from gplus_trees.merkle.gplus_mkl_base import (
    MerkleGPlusTreeBase,
    MerkleGPlusNodeBase
)

# Cache for previously created Merkle classes to avoid recreating them
_merkle_class_cache: Dict[int, Tuple[Type, Type]] = {}


def make_merkle_gplustree_classes(K: int) -> Tuple[
    Type[MerkleGPlusTreeBase],
    Type[MerkleGPlusNodeBase]
]:
    """
    Factory function to generate Merkle-enabled GPlus-tree classes for a given capacity K.

    Parameters:
        K (int): The capacity of the tree's KListNodes

    Returns:
        Tuple[Type[MerkleGPlusTreeBase], Type[MerkleGPlusNodeBase]]: The tree and node classes
    """
    # Check if we've already created classes for this K value
    if K in _merkle_class_cache:
        return _merkle_class_cache[K]
        
    # Get the standard classes first
    GPlusTreeK, GPlusNodeK, KListK, KListNodeK = make_gplustree_classes(K)
    
    # Create Merkle node class
    MerkleGPlusNodeK = type(
        f"MerkleGPlusNode_K{K}",
        (MerkleGPlusNodeBase,),
        {
            "SetClass": KListK,
            "__slots__": MerkleGPlusNodeBase.__slots__
        }
    )
    
    # Create Merkle tree class
    MerkleGPlusTreeK = type(
        f"MerkleGPlusTree_K{K}",
        (MerkleGPlusTreeBase,),
        {
            "SetClass": KListK,
            "NodeClass": MerkleGPlusNodeK,
            "__slots__": MerkleGPlusTreeBase.__slots__
        }
    )
    
    # Set the TreeClass on the node
    setattr(MerkleGPlusNodeK, "TreeClass", MerkleGPlusTreeK)
    
    # Cache the created classes
    _merkle_class_cache[K] = (MerkleGPlusTreeK, MerkleGPlusNodeK)
    
    return MerkleGPlusTreeK, MerkleGPlusNodeK


def create_merkle_gplustree(K: int) -> MerkleGPlusTreeBase:
    """
    Create a new Merkle-enabled GPlusTree with the specified capacity K.
    
    Parameters:
        K (int): The capacity of the tree's KListNodes
        
    Returns:
        MerkleGPlusTreeBase: A new empty Merkle-enabled GPlusTree
    """
    MerkleGPlusTreeK, _ = make_merkle_gplustree_classes(K)
    tree = MerkleGPlusTreeK()
    return tree