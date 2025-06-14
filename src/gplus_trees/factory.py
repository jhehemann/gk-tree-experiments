"""Factory for the creation of set data structures"""

from typing import Type, Tuple, Dict
from gplus_trees.gplus_tree_base import GPlusTreeBase, GPlusNodeBase
from gplus_trees.klist_base import KListBase, KListNodeBase

# Cache for previously created classes to avoid recreating them
_class_cache: Dict[int, Tuple[Type, Type, Type, Type]] = {}


def make_gplustree_classes(K: int) -> Tuple[
    Type[GPlusTreeBase],
    Type[GPlusNodeBase],
    Type[KListBase],
    Type[KListNodeBase]
]:
    """
    Factory function to generate GPlus-tree and KList classes specialized for a given capacity K.

    Returns:
        GPlusTreeK    – subclass of GPlusTreeBase with NodeClass=GPlusNodeK and SetClass=KListK.
        GPlusNodeK    – subclass of GPlusNodeBase with SetClass=KListK.
        KListK        – subclass of KListBase with KListNodeClass=KListNodeK.
        KListNodeK    – subclass of KListNodeBase with CAPACITY=K.
    """
    # Check if we've already created classes for this K value
    if K in _class_cache:
        return _class_cache[K]
    
    # 1) Leaf-list node: capacity K
    KListNodeK = type(
        f"KListNode_K{K}",
        (KListNodeBase,),
        {
            "CAPACITY": K,
            "__slots__": KListNodeBase.__slots__
        }
    )

    # 2) KList class points at the node class
    KListK = type(
        f"KList_K{K}",
        (KListBase,),
        {
            "KListNodeClass": KListNodeK,
            "__slots__": KListBase.__slots__
        }
    )

    # 3) Create GPlusTreeK with a forward reference (to be set later)
    GPlusTreeK = type(
        f"GPlusTree_K{K}",
        (GPlusTreeBase,),
        {
            "SetClass": KListK,
            "__slots__": GPlusTreeBase.__slots__
        }
    )
    
    # 4) GPlus-node class uses the KListK and references the already created GPlusTreeK
    GPlusNodeK = type(
        f"GPlusNode_K{K}",
        (GPlusNodeBase,),
        {
            "SetClass": KListK,
            "TreeClass": GPlusTreeK,
            "__slots__": GPlusNodeBase.__slots__
        }
    )
    

    # 5) Set NodeClass on GPlusTreeK now that GPlusNodeK exists
    setattr(GPlusTreeK, "NodeClass", GPlusNodeK)

    # Cache the created classes
    _class_cache[K] = (GPlusTreeK, GPlusNodeK, KListK, KListNodeK)

    return GPlusTreeK, GPlusNodeK, KListK, KListNodeK


def create_gplustree(K: int) -> GPlusTreeBase:
    """
    Create a new GPlusTree with the specified capacity K.
    
    Args:
        K (int): The capacity of the tree's KListNodes
        
    Returns:
        A new empty GPlusTree with the specified capacity
    """
    GPlusTreeK, _, _, _ = make_gplustree_classes(K)
    tree = GPlusTreeK()
    return tree
