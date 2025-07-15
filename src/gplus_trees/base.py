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
    __slots__ = ("key", "value", "dim_hash_map")
    def __init__(self, key, value=None):
        self.key   = key
        self.value = value
        self.dim_hash_map  = {}         # ← per-entry storage


class InternalItem:
    __slots__ = ("_item",)
    def __init__(self, item_data: ItemData):
        self._item = item_data

    @property
    def key(self):
        return self._item.key

    @property
    def dim_hashes(self):
        return self._item.dim_hash_map

    @property
    def value(self):
        return None
    
    def get_digest_for_dim(self, dim: int) -> int:
        """
        Get the hash digest for the specified dimension.

        Args:
            dim: The dimension level to get the rank for.
        
        Returns:
            The rank for the specified dimension.
        """
        # TODO: include the dimension when hashing to differentiate between dimensions
        # TODO: use strings to avoid collisions with negative keys
        digest = self._item.dim_hash_map.get(dim, None)
        if digest is None:
            key = self.key
            # Find the next lower existing dim hash
            lower_dims = [d for d in self._item.dim_hash_map if d < dim]
            if not lower_dims:
                # If no lower dims, create the digest for dim 1 and continue from there
                digest = hashlib.sha256(abs(key).to_bytes(32, 'big')).digest()
                self._item.dim_hash_map[1] = digest
                lower_dims = [1]

            next_lower_dim = max(lower_dims)
            digest = self._item.dim_hash_map[next_lower_dim]
            # Rehash and store digest until dim is reached
            for d in range(next_lower_dim + 1, dim + 1):
                digest = hashlib.sha256(digest).digest()

                # h = hashlib.sha256()
                # h.update(digest)
                # h.update(d.to_bytes(32, 'big'))
                # digest = h.digest()
                self._item.dim_hash_map[d] = digest

        return digest

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
    

class LeafItem(InternalItem):
    __slots__ = ()
    def __init__(self, item_data: ItemData):
        self._item = item_data

    @property
    def value(self):
        return self._item.value

    @value.setter
    def value(self, v):
        self._item.value = v


class DummyItem(InternalItem):
    """Represents a dummy item with no value and negative key, used for internal operations."""
    __slots__ = () 

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
        item (Union[InternalItem, LeafItem]): The item contained in the entry.
        left_subtree (AbstractSetDataStructure): The left subtree associated with this item.
            This is always provided, even if the subtree is empty.
    """
    __slots__ = ("item", "left_subtree")

    item: Union[InternalItem, LeafItem]
    left_subtree: T

def _create_replica(key):
    """Create a replica item with given key and no value."""
    return InternalItem(ItemData(key=key))

def _get_replica(item: Union[LeafItem, InternalItem]) -> InternalItem:
    """Get a replica item with given key and no value."""
    data = item._item
    return InternalItem(data)

def debug_log(message, *args, **kwargs):
    """Log a debug message only if debug logging is enabled"""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(message, *args, **kwargs)
