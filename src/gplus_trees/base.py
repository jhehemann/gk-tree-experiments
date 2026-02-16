from abc import ABC, abstractmethod
from dataclasses import dataclass

import hashlib
import logging
from typing import NamedTuple, Optional, TypeVar, Generic, Tuple, Union

# from gplus_trees.klist_base import KListBase
from gplus_trees.utils import get_digest
from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)

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
    
    def __lt__(self, other: 'InternalItem') -> bool:
        """Compare two InternalItem instances based on their keys."""
        return self.key < other.key
    
    def __eq__(self, other: 'InternalItem') -> bool:
        """Check equality of two InternalItem instances based on their keys."""
        return self.key == other.key
    
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
        digest = self._item.dim_hash_map.get(dim)
        if digest is not None:
            return digest
        
        key = self.key
        dim_hash_map = self._item.dim_hash_map
        
        # Ensure dim 1 is always present
        if 1 not in dim_hash_map:
            dim_hash_map[1] = get_digest(key, 1)

        # Find the next lower existing dim hash
        # Find the closest lower dimension
        lower_dims = [d for d in dim_hash_map if d < dim]
        next_lower_dim = max(lower_dims) if lower_dims else 1
        digest = dim_hash_map[next_lower_dim]
        
        # Only hash for missing dimensions
        # For dimension d, hash the previous dimension's digest with d itself
        for target_dim in range(next_lower_dim + 1, dim + 1):
            digest = get_digest(digest, target_dim)
            dim_hash_map[target_dim] = digest

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

class InsertResult(NamedTuple):
    """Result of an insert operation on a G⁺-tree or set data structure.

    Backward-compatible with the legacy ``(tree, inserted, next_entry)`` tuple
    — existing ``tree, inserted, next_entry = t.insert(…)`` unpacking still works.
    """
    tree: 'AbstractSetDataStructure'
    inserted: bool
    next_entry: Optional[Entry] = None


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
