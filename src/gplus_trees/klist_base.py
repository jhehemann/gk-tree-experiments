"""K-list implementation"""
from typing import TYPE_CHECKING, Optional, Tuple, Type, Any
from bisect import bisect_left
from itertools import chain
import copy

from gplus_trees.base import (
    AbstractSetDataStructure,
    Entry,
)
if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase

from gplus_trees.logging_config import get_logger
logger = get_logger("KList")

class KListNodeBase:
    """
    A node in the k-list.

    Each node stores up to CAPACITY entries.
    Each entry is a tuple of the form:
        (item, left_subtree)
    where `left_subtree` is a G-tree associated with this entry.
    """
    __slots__ = ("entries", "keys", "real_keys", "next")
    
    # Default capacity that will be overridden by factory-created subclasses
    CAPACITY: int  # Default value, will usually be overridden by factory

    def __init__(
            self, entries: Optional[list[Entry]] = None,
            keys: Optional[list[int]] = None,
            real_keys: Optional[list[int]] = None,
            next_node: Optional['KListNodeBase'] = None
        ):
        self.entries: list[Entry] = entries if entries is not None else []
        self.keys: list[int] = keys if keys is not None else []  # Inverted keys in ascending order for efficient binary search
        self.real_keys: list[int] = real_keys if real_keys is not None else []  # without dummy keys
        self.next: Optional['KListNodeBase'] = next_node

    def _set_attributes(self, entries: list[Entry], keys: list[int], real_keys: list[int], next_node: Optional['KListNodeBase']):
        self.entries = entries
        self.keys = keys
        self.real_keys = real_keys
        self.next = next_node
        return self

    def insert_entry(
            self, 
            entry: Entry,
    ) -> Optional[Entry]:
        """
        Inserts an entry into a reverse-sorted KListNode by key (descending order).
        If capacity exceeds, last entry is returned for further processing.
        
        Attributes:
            entry (Entry): The entry to insert into the KListNode.
        Returns:
            Optional[Entry]: The last entry if the node overflows; otherwise, None.
        """
        
        entries = self.entries
        keys = self.keys
        real_keys = self.real_keys
        x_key = entry.item.key
        is_dummy = x_key < 0
        next_entry = None

        # Empty list case
        if not entries:
            entries.append(entry)
            keys.append(-x_key)  # Store inverted key
            if not is_dummy:
                real_keys.append(x_key)
            return None, True, next_entry

        # Fast path: Append at end (smallest key goes last)
        if x_key < entries[-1].item.key:
            entries.append(entry)
            keys.append(-x_key)  # Store inverted key
            if not is_dummy:
                real_keys.append(x_key)
            next_entry = entries[-2]
        # Fast path: Insert at beginning (largest key goes first)
        elif x_key > entries[0].item.key:
            entries.insert(0, entry)
            keys.insert(0, -x_key)  # Store inverted key
            if not is_dummy:
                real_keys.insert(0, x_key)
        else:
            # Choose algorithm based on list length
            i = search_idx(x_key, keys)
            next_entry = entries[i-1] if i > 0 else None
            if -keys[i] == x_key:  # Compare with inverted key
                return None, False, next_entry
            entries.insert(i, entry)
            keys.insert(i, -x_key)  # Store inverted key
            

            if not is_dummy:
                # Insert into real_keys in descending order
                # Fast path: if real_keys is empty or x_key < smallest element, append
                real_keys_len = len(real_keys)
                if real_keys_len == 0 or x_key < real_keys[-1]:
                    real_keys.append(x_key)
                else:
                    i = search_idx_descending(x_key, real_keys)
                    real_keys.insert(i, x_key)

        # Handle overflow - pop the smallest (last) entry
        if len(entries) > self.__class__.CAPACITY:
            pop_entry = entries.pop()
            keys.pop()
            if pop_entry.item.key >= 0:
                real_keys.pop()
            return pop_entry, True, next_entry
        return None, True, next_entry

    def retrieve_entry(
        self, x_key: int
    ) -> Tuple[Optional[Entry], Optional[Entry], bool]:
        """
        Returns (found, in_node_successor, go_next):
         - found: the Entry with .item.key == key, or None
         - in_node_successor: the next Entry *within this node* if key exists
                              else None
         - go_next: True if key < min_key OR (found at min_key) → caller should
                    continue into next node to find the true successor.
        
        Note: Since entries are in descending order, the successor is the next smaller key.
        """
        entries = self.entries
        if not entries:
            return None, None, True

        keys = self.keys
        
        # Find the index of the key in the sorted (inverted, ascending) keys list
        i = search_idx(x_key, keys)

        # Case A: found exact
        if i < len(entries) and -keys[i] == x_key:  # Compare with inverted key
            found = entries[i]
            # if not last in this node, return node-local successor (next smaller)
            if i + 1 < len(entries):
                return found, entries[i+1], False
            # otherwise we must scan next node
            return found, None, True

        # Case B: i < len → this entries[i] is the in-node successor (next smaller)
        if i < len(entries):
            return None, entries[i], False

        # Case C: key < min_key in this node → skip to next
        return None, None, True
    
    def get_by_offset(self, offset: int) -> Tuple[Entry, Optional[Entry], bool]:
        """
        offset: 0 <= offset < len(self.entries)
        Returns (entry, in_node_successor, needs_next_node)
        """
        entry = self.entries[offset]
        if offset + 1 < len(self.entries):
            return entry, self.entries[offset+1], False
        else:
            return entry, None, True

class KListBase(AbstractSetDataStructure):
    """
    A k-list implemented as a linked list of nodes.
    Each node holds up to CAPACITY sorted entries.
    An entry is of the form (item, left_subtree), where left_subtree is a G+-tree (or None).
    The overall order is maintained lexicographically by key.
    """
    __slots__ = ("head", "tail", "_nodes", "_prefix_counts_tot", "_prefix_counts_real", "_bounds")

    # Will be assigned by factory
    KListNodeClass: Type[KListNodeBase]
    
    def __init__(
            self,
            head: Optional[KListNodeBase] = None,
            tail: Optional[KListNodeBase] = None,
            nodes: Optional[list[KListNodeBase]] = None,
            prefix_counts_tot: Optional[list[int]] = None,
            prefix_counts_real: Optional[list[int]] = None,
            bounds: Optional[list[int]] = None
        ):
        self.head = head
        self.tail = tail
        # auxiliary index
        self._nodes = nodes if nodes is not None else []
        self._prefix_counts_tot = prefix_counts_tot if prefix_counts_tot is not None else []
        self._prefix_counts_real = prefix_counts_real if prefix_counts_real is not None else []
        self._bounds = bounds if bounds is not None else []
    
    def __bool__(self) -> bool:
        # return False when empty, True when non-empty
        return not self.is_empty()

    def reset(self):
        self.head = None
        self.tail = None
        self._nodes = []
        self._prefix_counts_tot = []
        self._prefix_counts_real = []
        self._bounds = []
        return self

    def clone(self):
        return self.__class__(self.head, self.tail, self._nodes, self._prefix_counts_tot,
                              self._prefix_counts_real, self._bounds)

    def _rebuild_index(self):
        """Rebuild the node list and prefix-sum of entry counts."""
        nodes = self._nodes
        prefix_counts_tot = self._prefix_counts_tot
        prefix_counts_real = self._prefix_counts_real
        bounds = self._bounds
        
        nodes.clear()
        prefix_counts_tot.clear()
        prefix_counts_real.clear()
        bounds.clear()

        total = 0
        real = 0
        node = self.head
        while node:
            nodes.append(node)
            entries = node.entries

            total += len(entries)
            prefix_counts_tot.append(total)
            real += len(node.real_keys)
            prefix_counts_real.append(real)

            # Add the minimum key in this node to bounds (since entries are in descending order)
            if entries:
                bounds.append(entries[-1].item.key)  # Last entry has minimum key

            node = node.next
    
    def is_empty(self) -> bool:
        return self.head is None
    
    def item_count(self) -> int:
        """Returns the total number of entries in the KList in O(1) time."""
        if not self._prefix_counts_tot:
            return 0
        return self._prefix_counts_tot[-1]

    def real_item_count(self) -> int:
        """Returns the total number of real entries in the KList in O(1) time."""
        if not self._prefix_counts_real:
            return 0
        return self._prefix_counts_real[-1]

    def item_slot_count(self) -> int:
        """
        Returns the total number of slots available
        in the k-list, which is the sum of the capacities of all nodes.
        """
        count = 0
        current = self.head
        while current is not None:
            count += self.KListNodeClass.CAPACITY
            current = current.next
        return count
    
    def physical_height(self) -> int:
        """
        Returns the number of KListNode segments in this k-list.
        (i.e. how many times you must follow `next` before you reach None).
        """
        height = 0
        node = self.head
        # print(f"\nKList Item Count: {self.item_count()}")
        while node is not None:
            # print(f"Node Item Count: {len(node.entries)}")
            height += 1
            node = node.next
        # print(f"Klist Height: {height}")
        return height

    def _handle_empty_insert(self, entry: Entry) -> 'KListBase':
        key = entry.item.key
        node = self.KListNodeClass(
            entries=[entry],
            keys=[-key],  # Store inverted key
            real_keys=[key] if key >= 0 else []
        )
        self.head = self.tail = node
        self._nodes.append(node)
        self._prefix_counts_tot.append(1)
        self._prefix_counts_real.append(1 if key >= 0 else 0)
        self._bounds.append(key)
        return self, True, None

    def _handle_min_insert(self, entry: Entry) -> 'KListBase':
        key = entry.item.key
        tail = self.tail
        tail_entries = tail.entries
        is_real = key >= 0
        if len(tail_entries) >= self.KListNodeClass.CAPACITY:
            next_entry = tail_entries[-1]
            tail.next = self.KListNodeClass(
                entries=[entry],
                keys=[-key],  # Store inverted key
                real_keys=[key] if is_real else []
            )
            self.tail = tail.next
            
            # Manually update index and return
            self._nodes.append(self.tail)
            self._prefix_counts_tot.append(self._prefix_counts_tot[-1] + 1)
            if is_real:
                self._prefix_counts_real.append(self._prefix_counts_real[-1] + 1)
            self._bounds.append(key)
            return self, True, next_entry

        next_entry = tail_entries[-1]
        tail_entries.append(entry)
        tail.keys.append(-key)  # Store inverted key
        if is_real:  # Only add to real_keys if it's not a dummy key
            tail.real_keys.append(key)
            self._prefix_counts_real[-1] += 1 # partial index update

        # Complete the index update
        self._prefix_counts_tot[-1] += 1
        self._bounds[-1] = key

        return self, True, next_entry


    def insert_entry(self, entry: Entry, rank: Optional[int] = None) -> 'KListBase':
        """
        Inserts an existing Entry object into the k-list, preserving the Entry object's identity.

        The insertion ensures that the keys are kept in reverse lexicographic order (descending).
        If a node overflows (more than k entries), the extra entry is recursively inserted into the next node.
        
        Optimized for O(1) minimum key inserts using tail fast-path.

        Parameters:
            entry (Entry): The Entry object to insert (containing item and left_subtree).
            rank (Optional[int]): The rank of the entry, if applicable. Not used in this implementation. Only for compatibility with the G+-tree interface.
            
        Returns:
            KListBase: The updated k-list.
        """
        if not isinstance(entry, Entry):
            raise TypeError(f"entry must be Entry, got {type(entry).__name__!r}")
        
        key = entry.item.key
        bounds = self._bounds
        tail = self.tail

        # If the k-list is empty, create a new node.
        if self.head is None:
            return self._handle_empty_insert(entry)
        elif key < bounds[-1]:
            return self._handle_min_insert(entry)
        elif key > self._nodes[0].entries[0].item.key:
            node = self.head
            node_idx = 0
        else:
            nodes = self._nodes
            node = tail  # Default fallback
            node_idx = -1 # Default fallback
            node_idx = search_idx_descending(key, bounds)
            if node_idx < len(nodes):
                node = nodes[node_idx]

        node = self._nodes[node_idx]
        # overflow, inserted = node.insert_entry(entry)
        res = node.insert_entry(entry)
        overflow, inserted, next_entry = res[0], res[1], res[2]

        if next_entry is None and node_idx > 0:
            # If next_entry is None, we need to find the next larger entry in the previous node
            next_entry = self._nodes[node_idx - 1].entries[-1]
            
        if not inserted:
            return self, False, next_entry


        original_next_entry = next_entry
        # Handle successful insertion with potential overflow
        if node is tail and overflow is None:
            # manually update index
            self._prefix_counts_tot[-1] += 1
            if key >= 0:
                self._prefix_counts_real[-1] += 1
            return self, True, original_next_entry

        MAX_OVERFLOW_DEPTH = 10000
        depth = 0

        # Propagate overflow if needed
        while overflow is not None:
            if node.next is None:
                overflow_key = overflow.item.key
                node.next = self.KListNodeClass(
                    entries=[overflow],
                    keys=[-overflow_key],  # Store inverted key
                    real_keys=[overflow_key] if overflow_key >= 0 else []
                )
                self.tail = node.next
                self._rebuild_index()
                return self, True, original_next_entry

            node = node.next
            res = node.insert_entry(overflow)
            overflow, inserted, _ = res[0], res[1], res[2]  # Ignore next_entry from overflow
            depth += 1
            if depth > MAX_OVERFLOW_DEPTH:
                raise RuntimeError("KList insert_entry overflowed too deeply – likely infinite loop.")
        self._rebuild_index()
        return self, True, original_next_entry
        
    def delete(self, key: int) -> "KListBase":
        node = self.head
        found = False

        # 1) Find and remove the entry.
        while node:
            # Search within the node (descending order)
            keys = node.keys
            if keys:
                left = search_idx(key, keys)
                
                if left < len(keys) and -keys[left] == key:  # Compare with inverted key
                    # Found the key
                    del node.entries[left]
                    del node.keys[left]
                    if key >= 0:  # Only remove from real_keys if it's not a dummy key
                        # Find the key in real_keys and remove it (also in descending order)
                        real_keys = node.real_keys
                        left_rk = search_idx_descending(key, real_keys)
                        if left_rk < len(real_keys) and real_keys[left_rk] == key:
                            del node.real_keys[left_rk]
                    found = True
                    break
            
            node = node.next

        if not found:
            self._rebuild_index()
            return self

        # 2) If head is now empty, advance head
        if node is self.head and not node.entries:
            self.head = node.next
            if not self.head:
                # list became empty
                self.tail = None
            self._rebuild_index()
            return self

        # 3) Remove any empty nodes first
        current = self.head
        prev = None
        while current:
            if not current.entries:
                # Remove this empty node
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                if current is self.tail:
                    self.tail = prev
                current = current.next
            else:
                prev = current
                current = current.next
        
        # If list is now empty, we're done
        if not self.head:
            self.tail = None
            self._rebuild_index()
            return self

        # 4) Start a comprehensive rebalancing pass through the entire list
        # We need to ensure that all non-tail nodes are at capacity
        capacity = self.KListNodeClass.CAPACITY
        
        restart_rebalancing = True
        while restart_rebalancing:
            restart_rebalancing = False
            current = self.head
            
            while current and current.next:
                next_node = current.next
                
                # Continue moving items from next_node to current until current is at capacity
                # or next_node is empty

                while len(current.entries) < capacity and next_node.entries:
                    # Move the largest item from next_node (from the beginning) to current
                    shifted = next_node.entries.pop(0)  # Pop from beginning (largest key)
                    shifted_key = next_node.keys.pop(0)   # Pop from beginning (inverted largest key)
                    
                    # Insert into current in the correct position to maintain descending order
                    if not current.entries:
                        # Empty current, just append
                        current.entries.append(shifted)
                        current.keys.append(shifted_key)  # Already inverted
                    elif -shifted_key >= current.entries[0].item.key:  # Compare with original key
                        # Insert at beginning (largest position)
                        current.entries.insert(0, shifted)
                        current.keys.insert(0, shifted_key)  # Already inverted
                    elif -shifted_key <= current.entries[-1].item.key:  # Compare with original key
                        # Insert at end (smallest position)
                        current.entries.append(shifted)
                        current.keys.append(shifted_key)  # Already inverted
                    else:
                        # Find correct position in descending order
                        pos = search_idx(-shifted_key, current.keys)  # Search for original key
                        current.entries.insert(pos, shifted)
                        current.keys.insert(pos, shifted_key)  # Already inverted
                    
                    # Move from real_keys if it's not a dummy key
                    shifted_original_key = -shifted_key
                    if shifted_original_key >= 0:
                        # Remove from next_node real_keys if present
                        if next_node.real_keys and next_node.real_keys[0] == shifted_original_key:
                            next_node.real_keys.pop(0)
                        
                        # Insert into current real_keys in descending order
                        if not current.real_keys:
                            current.real_keys.append(shifted_original_key)
                        elif shifted_original_key >= current.real_keys[0]:
                            current.real_keys.insert(0, shifted_original_key)
                        elif shifted_original_key <= current.real_keys[-1]:
                            current.real_keys.append(shifted_original_key)
                        else:
                            pos = search_idx_descending(shifted_original_key, current.real_keys)
                            current.real_keys.insert(pos, shifted_original_key)
                    
                    # If next_node became empty, splice it out and restart rebalancing
                    if not next_node.entries:
                        current.next = next_node.next
                        if next_node is self.tail:
                            self.tail = current
                        # Restart rebalancing from the beginning to ensure compaction
                        restart_rebalancing = True
                        break
                
                if restart_rebalancing:
                    break
                    
                # Move to next node for the next iteration
                current = current.next
        
        self._rebuild_index()
        return self

    def _find_next_larger_entry(self, node_idx: int, entry_idx: int) -> Optional[Entry]:
        """
        Helper method to find the next larger entry (predecessor in physical list order).
        
        Args:
            node_idx: Index of the current node
            entry_idx: Index within the current node, or -1 if searching from previous node
            
        Returns:
            The next larger entry, or None if no larger entry exists
        """
        # Fast path: Next larger entry is in the same node
        if entry_idx > 0:
            return self._nodes[node_idx].entries[entry_idx - 1]
        
        # Search in previous node (which contains larger keys)
        if node_idx > 0:
            prev_node = self._nodes[node_idx - 1]
            if prev_node.entries:
                return prev_node.entries[-1]
        
        return None

    def retrieve(self, key: int, with_next: bool = True) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Search for `key` and return the found entry and its successor (the next larger key).
        
        In a descending-ordered KList, the "next larger key" is actually the predecessor
        in the physical list order.
        
        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (found_entry, next_entry) where:
                - found_entry: The entry with the matching key if found, otherwise None
                - next_entry: The next larger key entry, or None if no larger key exists
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        # Fast path: Empty K-List
        if self.is_empty():
            return None, None
            
        nodes = self._nodes
        
        # Fast path: Key is larger than the largest key
        if key > nodes[0].entries[0].item.key:
            return None, None
        
        # Fast path: Key is smaller than the smallest key
        if key < self._bounds[-1]:
            return None, nodes[-1].entries[-1]

        # Find target node using bounds
        node_idx = search_idx_descending(key, self._bounds)
        node = self._nodes[node_idx]
        entries = node.entries
        
        found_entry = None
        next_entry = None

        i = search_idx(key, node.keys)
        # i must be smaller than len(entries) since we checked bounds above
        if entries[i].item.key == key:
            found_entry = entries[i]
            next_entry = self._find_next_larger_entry(node_idx, i)
        elif i > 0:
            # Key would be inserted at position i, so next larger is at i-1
            next_entry = entries[i-1]
        else:
            # Key is larger than all entries in this node
            next_entry = self._find_next_larger_entry(node_idx, -1)

        
        return found_entry, next_entry

    
    def find_pivot(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Find the pivot entry (minimum entry) in the KList."""
        return self.get_min()

    def get_min(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Retrieve the minimum entry from the reverse-sorted KList (last entry of tail node)."""
        if self.is_empty():
            return None, None
        
        tail_entries = self.tail.entries  
        min_entry = tail_entries[-1]

        # Find successor (next larger entry)
        if len(tail_entries) > 1:
            return min_entry, tail_entries[-2]
        nodes = self._nodes
        return min_entry, nodes[-2].entries[-1] if len(nodes) > 1 else None

    def get_max(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Retrieve the maximum entry from the reverse-sorted KList (first entry of head node)."""
        if self.is_empty():
            return None, None
        
        head_entries = self.head.entries
        max_entry = head_entries[0]
        return max_entry, None # Maximum has no successor

    def split_inplace(
        self, key: int
    ) -> Tuple["KListBase", Optional["GPlusTreeBase"], "KListBase"]:

        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        # Fast path: Empty K-List
        if self.head is None:
            right = type(self)()
            return self, None, right

        nodes = self._nodes

        # If key is smaller than any key in the list, all entries have keys > key
        # Interface expects: left (keys < key), right (keys > key)
        if key < nodes[-1].entries[-1].item.key:
            right_return = copy.copy(self)
            return self.reset(), None, right_return  # Empty left, all entries go to right

        # If key is larger than the maximum key, all entries have keys < key
        if key > nodes[0].entries[0].item.key:
            right_return = type(self)()
            return self, None, right_return  # All entries go to left, empty right

        # --- locate split node ------------------------------------------------
        # Find the first node where key >= min_key
        node_idx = search_idx_descending(key, self._bounds)
        split_node = nodes[node_idx]
        prev_node = nodes[node_idx - 1] if node_idx else None
        original_next = split_node.next

        # --- locate target inside that node (reverse order) -------------------------
        node_entries = split_node.entries
        node_keys = split_node.keys

        i = search_idx(key, node_keys)
        exact = i < len(node_keys) and -node_keys[i] == key  # Compare with inverted key and check bounds
        lo_idx = i + 1 if exact else i

        # In reverse order: entries with keys > split_key go to greater_part, <= split_key go to lesser_part
        greater_entries = node_entries[:i]  # Keys > split_key -> will become RIGHT return
        lesser_entries = node_entries[lo_idx:]  # Keys < split_key -> will become LEFT return
        key_subtree = node_entries[i].left_subtree if exact else None

        greater_keys = node_keys[:i]
        lesser_keys = node_keys[lo_idx:]

        real_keys = split_node.real_keys
        j = search_idx_descending(key, real_keys)
        greater_real_keys = real_keys[:j]
        lesser_real_keys = real_keys[j + 1 if exact else j :]

        # ------------- build RIGHT RETURN (keys > split_key) ------------
        if greater_entries:                          
            # Right: previous nodes + split_node
            split_node = split_node._set_attributes(    # reuse split_node
                entries=greater_entries,
                keys=greater_keys,
                real_keys=greater_real_keys,
                next_node=None
            )
            right_return = type(self)(head=self.head, tail=split_node)
        else:
            # Right: previous nodes
            if prev_node:
                prev_node.next = None
                right_return = type(self)(head=self.head, tail=prev_node)
            else:                                    # empty greater part
                right_return = type(self)()

        # ------------- build LEFT RETURN (keys < split_key) ------------
        if lesser_entries:
            new_node = self.KListNodeClass(
                entries=lesser_entries,
                keys=lesser_keys,
                real_keys=lesser_real_keys,
                next_node=original_next
            )
            self.head = new_node
        else:
            self.head = original_next

        # Check if we need to update tail for left return
        if self.tail is split_node:
            self.tail = self.head

        # Ensure all non-tail nodes in left return are at full capacity
        if self.head and lesser_entries:
            self._rebalance_for_compaction(self)
        
        # ------------- rebuild indexes ---------------------------------------
        self._rebuild_index()
        right_return._rebuild_index()

        # Return in interface order: left (keys < split_key), key_subtree, right (keys > split_key)
        return self, key_subtree, right_return

    def _rebalance_for_compaction(self, klist: 'KListBase') -> None:
        """
        Helper method to ensure the compaction invariant is maintained in a klist.
        Redistributes entries to ensure all non-tail nodes are at full capacity.
        
        Parameters:
            klist: The KList to rebalance
        """
        current = klist.head
        capacity = klist.KListNodeClass.CAPACITY
        
        # Rebalance all nodes
        while current and current.next:
            next_node = current.next
            
            # Continue moving items from next_node to current until current is at capacity
            # or next_node is empty
            needed = capacity - len(current.entries)
            available = len(next_node.entries)
            move_count = min(needed, available)
            if move_count > 0:
                # Move a slice of entries and keys
                current.entries.extend(next_node.entries[:move_count])
                current.keys.extend(next_node.keys[:move_count])
                del next_node.entries[:move_count]
                del next_node.keys[:move_count]
                # Move real_keys if needed (only for non-dummy keys)
                if next_node.real_keys:
                    # Find how many of the moved keys are real (>= 0)
                    moved_real_keys = []
                    for k in current.keys[-move_count:]:
                        if -k >= 0:  # Check original key value
                            moved_real_keys.append(k)  # Store original key
                        else:
                            break
                    if moved_real_keys:
                        current.real_keys.extend(next_node.real_keys[:len(moved_real_keys)])
                        del next_node.real_keys[:len(moved_real_keys)]
            # If next_node became empty, splice it out and update tail
            if not next_node.entries:
                current.next = next_node.next
                if next_node is klist.tail:
                    klist.tail = current
                # Exit inner loop as we've emptied next_node
                break
            
            # Move to next node for the next iteration
            current = current.next

    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 2):
        """
        Returns a string representation of the k-list for debugging.
        
        Parameters:
            indent (int): Number of spaces for indentation.
            depth (int): Current recursion depth.
            max_depth (int): Maximum allowed recursion depth.
        """
        if self.head is None:
            return f"{' ' * indent}Empty"

        if depth > max_depth:
            return f"{' ' * indent}... (max depth reached)"

        result = []
        node = self.head
        index = 0
        while node:
            result.append(f"{' ' * indent}KListNode(idx={index}, K={self.KListNodeClass.CAPACITY})")
            for entry in node.entries:
                result.append(f"{' ' * indent}• {str(entry.item)}")
                if entry.left_subtree is None:
                    result.append(f"{' ' * indent}  Left: None")
                else:
                    result.append(entry.left_subtree.print_structure(indent + 2, depth + 1, max_depth))
            node = node.next
            index += 1
        return "\n".join(result)

    def __iter__(self):
        """Yield each entry of the k-list in ascending order of keys."""
        for node in reversed(self._nodes):
            yield from reversed(node.entries)

    def iter_reverse(self):
        """Yield each entry of the k-list in descending order of keys."""
        for node in self._nodes:
            yield from node.entries

    def __str__(self):
        """
        Returns a string representation of the k-list for debugging.
        """
        result = []
        node = self.head
        index = 0
        while node:
            result.append(f"Node {index}: {node.entries}")
            node = node.next
            index += 1
        return "\n".join(result)
    
    def check_invariant(self) -> None:
        """
        Verifies that:
          1) Each KListNode.entries is internally sorted by item.key in descending order.
          2) For each consecutive pair of nodes, 
             first_key(node_i) > last_key(node_{i+1}) (since we store in reverse order).
          3) self.tail.next is always None (tail really is the last node).
          4) All nodes except the last one must be at full capacity (have k items).

        Raises:
            AssertionError: if any of these conditions fails.
        """
        # 1) Tail pointer must point to the true last node
        assert (self.head is None and self.tail is None) or (
            self.tail is not None and self.tail.next is None
        ), "Invariant violated: tail must reference the final node"

        node = self.head
        previous_last_key = None
        
        # For counting nodes to check compaction invariant
        nodes_seen = 0
        is_last_node = False

        # 2) Walk through all nodes
        while node is not None:
            nodes_seen += 1
            is_last_node = (node.next is None)
            
            # 2a) Entries within this node are sorted in descending order
            for i in range(1, len(node.entries)):
                k0 = node.entries[i-1].item.key
                k1 = node.entries[i].item.key
                assert k0 >= k1, (
                    f"Intra-node reverse sort order violated in node {node}: "
                    f"{k0} < {k1} (should be descending)"
                )

            # 2b) Boundary with the previous node (reverse order)
            if previous_last_key is not None and node.entries:
                first_key = node.entries[0].item.key
                assert previous_last_key > first_key, (
                    f"Inter-node reverse invariant violated between nodes: "
                    f"{previous_last_key} <= {first_key} (should be strictly decreasing)"
                )
                
            # 2c) All non-tail nodes must be at full capacity
            if not is_last_node:  # Only check non-tail nodes
                assert len(node.entries) == node.__class__.CAPACITY, (
                    f"Non-tail node at position {nodes_seen} has {len(node.entries)} entries, "
                    f"but should have {node.__class__.CAPACITY} (compaction invariant violated)"
                )

            # Update for the next iteration (in reverse order, track the last key which is minimum)
            if node.entries:
                previous_last_key = node.entries[-1].item.key

            node = node.next


def search_idx(comp_elem: Any, list: list[Any]) -> int:
    """
    Searches for the index in the given list (with inverted keys in ascending order) where the comparison element `comp_elem` fits.
    For lists with fewer than 8 elements, performs a linear search, returning the index of the first element where `comp_elem >= -bound`.
    For longer lists, performs a binary search using `bisect_left` directly (no lambda needed due to inverted keys).
    Args:
        comp_elem (Any): The element to compare against the list elements.
        list (list[Any]): The list of inverted elements to search, sorted in ascending order.
    Returns:
        int: The index where `comp_elem` fits in the list according to the comparison logic.
    """
    
    # for i, bound in enumerate(list):
    #         if comp_elem >= -bound:  # Compare with original key value
    #             return i
    # return len(list)

    list_len = len(list)
    if list_len < 8:
        # Linear search for small number of nodes
        for i, bound in enumerate(list):
            if comp_elem >= -bound:  # Compare with original key value
                return i
        return list_len
    else:
        # Binary search for larger node count - now efficient without lambda!
        i = bisect_left(list, -comp_elem)
        return i


def search_idx_descending(comp_elem: Any, list: list[Any]) -> int:
    """
    Searches for the index in the given list (sorted in descending order) where the comparison element `comp_elem` fits.
    For lists with fewer than 8 elements, performs a linear search, returning the index of the first element where `comp_elem >= bound`.
    For longer lists, performs a binary search using `bisect_left` with a custom key.
    Args:
        comp_elem (Any): The element to compare against the list elements.
        list (list[Any]): The list of elements to search sorted in descending order.
    Returns:
        int: The index where `comp_elem` fits in the list according to the comparison logic.
    """
    list_len = len(list)
    if list_len < 8:
        # Linear search for small number of nodes
        for i, bound in enumerate(list):
            if comp_elem >= bound:
                return i
        return list_len
    else:
        # Binary search for larger node count
        i = bisect_left(list, -comp_elem, key=lambda v: -v)
        return i