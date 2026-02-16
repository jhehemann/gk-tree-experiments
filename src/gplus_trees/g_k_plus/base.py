"""Abstract base class for GK+-tree set data structures."""

from abc import abstractmethod
from collections.abc import Iterator
from typing import Optional

from gplus_trees.base import AbstractSetDataStructure, Entry


class GKTreeSetDataStructure(AbstractSetDataStructure):
    """
    Abstract base class for G+ trees and k-lists.
    """

    @abstractmethod
    def get_min(self) -> tuple[Optional["Entry"], Optional["Entry"]]:
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

    @abstractmethod
    def __iter__(self) -> Iterator["Entry"]:
        """
        Iterate over the entries in the set data structure.

        Returns:
            Iterator[Entry]: An iterator over the entries in the set.
        """
        pass
