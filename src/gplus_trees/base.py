from abc import ABC, abstractmethod
from dataclasses import dataclass

import hashlib
from typing import Optional, TypeVar, Generic, Tuple, Union
import logging

# from gplus_trees.klist_base import KListBase
from gplus_trees.logging_config import get_logger

# Get logger for this module
logger = get_logger("GPlusTree")

class ItemData:
    __slots__ = ("key", "value", "dim_rank_hashes")
    def __init__(self, key, value=None):
        self.key   = key
        self.value = value
        self.dim_rank_hashes  = {}         # ← per-entry storage

class LeafItem:
    __slots__ = ("_item",)
    def __init__(self, item_data: ItemData):
        self._item = item_data

    @property
    def key(self):
        return self._item.key

    @property
    def value(self):
        return self._item.value

    @value.setter
    def value(self, v):
        self._item.value = v

    @property
    def dim_hashes(self):
        return self._item.dim_hashes
    
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

class InternalItem:
    __slots__ = ("_item",)
    def __init__(self, item_data: ItemData):
        self._item = item_data

    @property
    def key(self):
        return self._item.key

    @property
    def dim_hashes(self):
        return self._item.dim_hashes

    @property
    def value(self):
        return None
    
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
    

class DummyItem(InternalItem):
    """Represents a dummy item with no value and negative key, used for internal operations."""
    __slots__ = () 


class Item:
    """
    Represents an item (a key-value pair) for insertion in G-trees and k-lists.
    """
    __slots__ = ("key", "value", "rank_hash")  # Define slots for memory efficiency

    def __init__(
            self,
            key: int,
            value: str = None,
            rank_hash: Optional[bytes] = None
    ):
        """
        Initialize an Item.

        Parameters:
            key (int): The item's key.
            value (str): The item's value.
            rank_hash (bytes): The hash digest for rank calculation. Uses a provided rank hash or the absolute value of the key (to ensure negative keys (dummy keys) are hashable).
        """
        self.key = key
        self.value = value
        self.rank_hash = rank_hash or hashlib.sha256(abs(self.key).to_bytes(32, 'big')).digest()
    
    
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
    ) -> Tuple[Optional['Entry'], Optional['Entry']]:
        """
        Retrieve the entry associated with the given key from the set data structure.

        Parameters:
            key (int): The key of the entry to retrieve.
        
        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (found_entry, next_entry) where:
                - found_entry: The entry with the matching key if found, otherwise None
                - next_entry: The subsequent entry in sorted order, or None if no next entry exists
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

def _create_replica(key):
    """Create a replica item with given key and no value."""
    return InternalItem(ItemData(key=key))

def _get_replica(item: Union[LeafItem, InternalItem, DummyItem]) -> InternalItem:
    """Get a replica item with given key and no value."""
    data = item._item
    return InternalItem(data)

def debug_log(message, *args, **kwargs):
    """Log a debug message only if debug logging is enabled"""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(message, *args, **kwargs)
