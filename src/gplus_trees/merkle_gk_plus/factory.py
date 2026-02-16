"""Factory for Merkle-enabled GKPlus-trees"""

from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.merkle_gk_plus.gk_plus_mkl_base import MerkleGKPlusNodeBase, MerkleGKPlusTreeBase

# Cache for previously created Merkle classes to avoid recreating them
_gk_plus_merkle_class_cache: dict[int, tuple[type, type]] = {}


def make_merkle_gk_plustree_classes(K: int) -> tuple[type[MerkleGKPlusTreeBase], type[MerkleGKPlusNodeBase]]:
    """
    Factory function to generate Merkle-enabled GKPlus-tree classes for a given capacity K.

    Parameters:
        K (int): The capacity of the tree's KListNodes

    Returns:
        Tuple[Type[MerkleGKPlusTreeBase], Type[MerkleGKPlusNodeBase]]: The tree and node classes
    """
    # Check if we've already created classes for this K value
    if K in _gk_plus_merkle_class_cache:
        return _gk_plus_merkle_class_cache[K]

    # Get the standard classes first
    _GKPlusTreeK, _GKPlusNodeK, KListK, _KListNodeK = make_gkplustree_classes(K)

    # Create Merkle node class
    MerkleGKPlusNodeK = type(
        f"MerkleGKPlusNode_K{K}",
        (MerkleGKPlusNodeBase,),
        {"SetClass": KListK, "__slots__": MerkleGKPlusNodeBase.__slots__},
    )

    # Create Merkle tree class
    MerkleGKPlusTreeK = type(
        f"MerkleGKPlusTree_K{K}",
        (MerkleGKPlusTreeBase,),
        {"SetClass": KListK, "NodeClass": MerkleGKPlusNodeK, "__slots__": MerkleGKPlusTreeBase.__slots__},
    )

    # Set the TreeClass on the node
    MerkleGKPlusNodeK.TreeClass = MerkleGKPlusTreeK

    # Cache the created classes
    _gk_plus_merkle_class_cache[K] = (MerkleGKPlusTreeK, MerkleGKPlusNodeK)

    return MerkleGKPlusTreeK, MerkleGKPlusNodeK


def create_merkle_gk_plustree(K: int) -> MerkleGKPlusTreeBase:
    """
    Create a new Merkle-enabled GPlusTree with the specified capacity K.

    Parameters:
        K (int): The capacity of the tree's KListNodes

    Returns:
        MerkleGKPlusTreeBase: A new empty Merkle-enabled GPlusTree
    """
    MerkleGKPlusTreeK, _ = make_merkle_gk_plustree_classes(K)
    tree = MerkleGKPlusTreeK()
    return tree
