from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import NamedTuple, Optional, TypeVar, Generic, List
import hashlib
import logging
from gplus_trees.logging_config import get_logger
import numpy as np

# Get logger for this module
logger = get_logger(__name__)


class Item:
    """
    Represents an item (a key-value pair) for insertion in G-trees and k-lists.
    """
    __slots__ = ("key", "value")  # Define slots for memory efficiency

    def __init__(
            self,
            key: int,
            value: str = None
    ):
        """
        Initialize an Item.

        Parameters:
            key (int): The item's key.
            value (str): The item's value.
        """
        self.key = key
        self.value = value
    
    def short_key(self) -> str:
        """Create a short representation of the key for display purposes."""
        if isinstance(self.key, (bytes, bytearray)):
            s = self.key.hex()
        else:
            # treat everything else—including int—as decimal-string
            s = str(self.key)

        # 2) If it’s already short, just return it; otherwise elide the middle
        return s if len(s) <= 10 else f"{s[:3]}...{s[-3:]}"
    
    

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(key={self.key!r}, value={self.value!r})"

    def __str__(self):
        cls = self.__class__.__name__
        return f"{cls}(key={self.short_key()}, value={self.value})"


T = TypeVar("T", bound="AbstractSetDataStructure")

class AbstractSetDataStructure(ABC, Generic[T]):
    """
    Abstract base class for a set data structure storing tuples of items and their left subtrees.
    """
    
    # @abstractmethod
    # def insert_entry(self, entry: 'Entry', rank: int) -> T:
    #     """
    #     Insert an entry into the set with the provided rank.
        
    #     Parameters:
    #         entry (Entry): The item to be inserted.
    #         rank (int): The rank for the item.
        
    #     Returns:
    #         AbstractSetDataStructure: The set data structure instance where the item was inserted.
    #     """
    #     pass

    @abstractmethod
    def delete(self, key: int) -> T:
        """
        Delete the item corresponding to the given key from the corresponding set data structure.
        
        Parameters:
            key (int): The key of the item to be deleted.
        
        Returns:
            AbstractSetDataStructure: The set data structure instance after deletion.
        """
        pass

    @abstractmethod
    def retrieve(
        self, key: int
    ) -> 'RetrievalResult':
        """
        Retrieve the entry associated with the given key from the set data structure.

        Parameters:
            key (int): The key of the entry to retrieve.
        
        Returns:
            RetrievalResult: A named tuple containing:
                - found_entry: A tuple (item, left_subtree) if the key is found; otherwise, None.
                - next_entry: A tuple (next_item, left_subtree) representing the next entry in sorted order,
                            or None if no subsequent entry exists.
        """
        pass


@dataclass
class Entry:
    """
    Represents an entry in the KList or G⁺-tree.

    Attributes:
        item (Item): The item contained in the entry.
        left_subtree (AbstractSetDataStructure): The left subtree associated with this item.
            This is always provided, even if the subtree is empty.
    """
    __slots__ = ("item", "left_subtree")

    item: Item
    left_subtree: T


class RetrievalResult(NamedTuple):
    """
    A container for the result of a lookup in an AbstractSetDataStructure.

    Attributes:
        found_entry (Optional[Entry]):
            The entry corresponding to the searched key if found;
            otherwise, None.
        next_entry (Optional[Entry]):
            The subsequent entry in the sorted order, which serves as a candidate for
            further operations or in-order traversal; None if no subsequent entry exists.
    """
    found_entry: Optional[Entry]
    next_entry: Optional[Entry]


def _create_replica(key):
    """Create a replica item with given key and no value."""
    return Item(key, None)


def debug_log(message, *args, **kwargs):
    """Log a debug message only if debug logging is enabled"""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(message, *args, **kwargs)
