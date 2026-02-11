from abc import abstractmethod

from typing import Optional, Tuple, TypeVar, Iterator

from gplus_trees.base import (
    AbstractSetDataStructure,
    Entry
)
from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound="GKTreeSetDataStructure")

class GKTreeSetDataStructure(AbstractSetDataStructure):
    """
    Abstract base class for G+ trees and k-lists.
    """

    @abstractmethod
    def get_min(self) -> Tuple[Optional['Entry'], Optional['Entry']]:
        """
        Retrieve the minimum entry in the set data structure.
        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (found_entry, next_entry) where:
                - found_entry: The minimum entry in the set
                - next_entry: The next entry in sorted order after the minimum entry
        """
        pass

    @abstractmethod
    def item_count(self) -> int:
        """
        Get the count of items in the set data structure.
        Returns:
            int: The number of items in the set.
        """
        pass

    @abstractmethod
    def real_item_count(self) -> int:
        """
        Get the count of real items (excluding dummy items) in the set data structure.
        Returns:
            int: The number of real items in the set.
        """
        pass

    @abstractmethod
    def item_slot_count(self) -> int:
        """
        Get the total number of item slots reserved by the set data structure. This is the sum of the capacity of all nodes.
        Returns:
            int: The number of item slots in the set.
        """
        pass
    
    @abstractmethod
    def physical_height(self) -> int:
        """
        Get the physical height of the set data structure which is the maximum number of traversal steps needed to reach the deepest node.
        Returns:
            int: The physical height of the set.
        """
        pass

    # @abstractmethod
    # def split_inplace(
    #         self, key: int
    # ) -> Tuple[T, Optional[T], T]:
    #     """
    #     Split the set into two parts based on the provided key.
        
    #     The first part contains all items less than or equal to the key,
    #     and the second part contains all items greater than the key.
        
    #     Parameters:
    #         key (int): The key to split the set by.
        
    #     Returns:
    #         Tuple[AbstractSetDataStructure, Optional[AbstractSetDataStructure], AbstractSetDataStructure]:
    #             A tuple containing:
    #                 - The left set (items < key).
    #                 - The left subtree of the item (item = key).
    #                 - The right subtree (items > key).
    #     """
    #     pass

    @abstractmethod
    def __iter__(self) -> Iterator['Entry']:
        """
        Iterate over the entries in the set data structure.
        
        Returns:
            Iterator[Entry]: An iterator over the entries in the set.
        """
        pass