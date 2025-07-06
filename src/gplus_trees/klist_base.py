"""K-list implementation"""
from typing import TYPE_CHECKING, Optional, Tuple, Type
from bisect import bisect_left, insort_left
from itertools import chain

from gplus_trees.base import (
    Item,
    AbstractSetDataStructure,
    Entry
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
    
    def __init__(self):
        self.entries: list[Entry] = []
        self.keys: list[int] = []       # Sorted list of keys for fast binary search
        self.real_keys: list[int] = []  # Sorted list of real keys (excluding dummy keys)
        self.next: Optional['KListNodeBase'] = None
        
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

        # Empty list case
        if not entries:
            entries.append(entry)
            keys.append(x_key)
            if not is_dummy:
                real_keys.append(x_key)
            return None, True

        # Fast path: Append at end (smallest key goes last)
        if x_key < entries[-1].item.key:
            entries.append(entry)
            keys.append(x_key)
            if not is_dummy:
                real_keys.append(x_key)
        # Fast path: Insert at beginning (largest key goes first)
        elif x_key > entries[0].item.key:
            entries.insert(0, entry)
            keys.insert(0, x_key)
            if not is_dummy:
                real_keys.insert(0, x_key)
        else:
            # Choose algorithm based on list length
            entries_len = len(entries)
            if entries_len < 8:
                # Linear search for very small lists (descending order)
                for i in range(entries_len):
                    if x_key > entries[i].item.key:
                        entries.insert(i, entry)
                        keys.insert(i, x_key)
                        break
                    elif x_key == entries[i].item.key:
                        # If we find an exact match, we can choose to replace or ignore
                        # Here we choose to ignore the insertion if the key already exists
                        return None, False
            else:
                # Binary search for larger lists using reverse bisect
                i = bisect_left(keys, -x_key, key=lambda v: -v)
                if i < len(entries) and keys[i] == x_key:
                    return None, False
                entries.insert(i, entry)
                keys.insert(i, x_key)

            if not is_dummy:
                # Insert into real_keys in descending order
                # Fast path: if real_keys is empty or x_key < smallest element, append
                real_keys_len = len(real_keys)
                if real_keys_len == 0 or x_key < real_keys[-1]:
                    real_keys.append(x_key)
                else:
                    if real_keys_len < 8:
                        # Linear search for very small lists (descending order)
                        for i in range(real_keys_len):
                            if x_key > real_keys[i]:
                                real_keys.insert(i, x_key)
                                break
                    else:
                        # Binary search for larger lists (descending order)
                        insort_left(real_keys, x_key, key=lambda v: -v)

        # Handle overflow - pop the smallest (last) entry
        if len(entries) > self.__class__.CAPACITY:
            pop_entry = entries.pop()
            # logger.debug(f"POP ENTRY {pop_entry.item.key}")
            keys.pop()
            if pop_entry.item.key >= 0:
                real_keys.pop()
            return pop_entry, True
        return None, True
     
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
        
        # Binary search in descending order
        i = bisect_left(keys, -x_key, key=lambda v: -v)

        # Case A: found exact
        if i < len(entries) and keys[i] == x_key:
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
    
    def __init__(self):
        self.head = self.tail = None
        # auxiliary index
        self._nodes = []               # List[KListNodeBase]
        self._prefix_counts_tot = []   # List[int]
        self._prefix_counts_real = []  # List[int], real items count (excluding dummy items)
        self._bounds = []              # List[int], max key per node (optional)
    
    def __bool__(self) -> bool:
        # return False when empty, True when non-empty
        return not self.is_empty()

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
            node = self.KListNodeClass()
            node.entries.append(entry)
            node.keys.append(key)
            if key >= 0: # Only add to real_keys if it's not a dummy key
                node.real_keys.append(key)
            self.head = self.tail = node
            self._rebuild_index()
            return self, True
        elif key < bounds[-1]:
            tail_entries = tail.entries
            if len(tail_entries) >= self.KListNodeClass.CAPACITY:
                tail.next = self.KListNodeClass()
                self.tail = tail.next
                tail = self.tail
                tail_entries = tail.entries

            tail_entries.append(entry)
            tail.keys.append(key)
            if key >= 0:  # Only add to real_keys if it's not a dummy key
                tail.real_keys.append(key)
            self._rebuild_index()
            return self, True
        elif key > self._nodes[0].entries[0].item.key:
            node = self.head
        else:
            # Find appropriate node with threshold-based search
            nodes = self._nodes
            node = tail  # Default fallback
            
            if len(nodes) < 8:
                # Linear search for small number of nodes
                for idx, bound in enumerate(bounds):
                    if key >= bound:
                        node = nodes[idx]
                        break
            else:
                # Binary search for larger node count
                i = bisect_left(bounds, -key, key=lambda v: -v)
                if i < len(nodes):
                    node = nodes[i]
        
        overflow, inserted = node.insert_entry(entry)

        if not inserted:
            return self, False

        # Handle successful insertion with potential overflow
        if node is tail and overflow is None:
            self._rebuild_index()
            return self, True

        MAX_OVERFLOW_DEPTH = 10000
        depth = 0

        # Propagate overflow if needed
        while overflow is not None:
            if node.next is None:
                overflow_key = overflow.item.key
                node.next = self.KListNodeClass()
                self.tail = node.next
                tail = self.tail
                tail.entries.append(overflow)
                tail.keys.append(overflow_key)
                if overflow_key >= 0:  # Only add to real_keys if it's not a dummy key
                    tail.real_keys.append(overflow_key)
                self._rebuild_index()
                return self, True

            node = node.next
            overflow, inserted = node.insert_entry(overflow)
            depth += 1
            if depth > MAX_OVERFLOW_DEPTH:
                raise RuntimeError("KList insert_entry overflowed too deeply – likely infinite loop.")
        self._rebuild_index()
        return self, True
        
    def delete(self, key: int) -> "KListBase":
        node = self.head
        found = False

        # 1) Find and remove the entry.
        while node:
            # Search within the node using bisect (descending order)
            keys = node.keys
            if keys:
                # Use bisect to find the key position
                left = bisect_left(keys, -key, key=lambda v: -v)
                
                if left < len(keys) and keys[left] == key:
                    # Found the key
                    del node.entries[left]
                    del node.keys[left]
                    if key >= 0:  # Only remove from real_keys if it's not a dummy key
                        # Find the key in real_keys and remove it (also in descending order)
                        real_keys = node.real_keys
                        left_rk = bisect_left(real_keys, -key, key=lambda v: -v)
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
                    shifted_key = next_node.keys.pop(0)   # Pop from beginning (largest key)
                    
                    # Insert into current in the correct position to maintain descending order
                    if not current.entries:
                        # Empty current, just append
                        current.entries.append(shifted)
                        current.keys.append(shifted_key)
                    elif shifted_key >= current.entries[0].item.key:
                        # Insert at beginning (largest position)
                        current.entries.insert(0, shifted)
                        current.keys.insert(0, shifted_key)
                    elif shifted_key <= current.entries[-1].item.key:
                        # Insert at end (smallest position)
                        current.entries.append(shifted)
                        current.keys.append(shifted_key)
                    else:
                        # Use bisect to find correct position in descending order
                        pos = bisect_left(current.keys, -shifted_key, key=lambda v: -v)
                        current.entries.insert(pos, shifted)
                        current.keys.insert(pos, shifted_key)
                    
                    # Move from real_keys if it's not a dummy key
                    if shifted_key >= 0:
                        # Remove from next_node real_keys if present
                        if next_node.real_keys and next_node.real_keys[0] == shifted_key:
                            next_node.real_keys.pop(0)
                        
                        # Insert into current real_keys in descending order
                        if not current.real_keys:
                            current.real_keys.append(shifted_key)
                        elif shifted_key >= current.real_keys[0]:
                            current.real_keys.insert(0, shifted_key)
                        elif shifted_key <= current.real_keys[-1]:
                            current.real_keys.append(shifted_key)
                        else:
                            # Use bisect for real_keys insertion
                            pos = bisect_left(current.real_keys, -shifted_key, key=lambda v: -v)
                            current.real_keys.insert(pos, shifted_key)
                    
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
            
        # Fast path: Key is larger than the largest key
        if key > self._nodes[0].entries[0].item.key:
            return None, None
        
        # Fast path: Key is smaller than the smallest key
        if key < self._bounds[-1]:
            return None, self._nodes[-1].entries[-1]

        # Find target node using binary search on bounds
        node_idx = bisect_left(self._bounds, -key, key=lambda v: -v)
        node = self._nodes[node_idx]
        entries = node.entries
        keys = node.keys
        
        # Fast path: Key is larger than max key in this node
        if key > entries[0].item.key:
            return None, self._find_next_larger_entry(node_idx, -1)
        
        # Choose search strategy based on node size (threshold = 8)
        found_entry = None
        found_idx = -1
        next_entry = None
        
        if len(entries) < 8:
            # Linear search for small nodes
            for i, entry in enumerate(entries):
                entry_key = entry.item.key
                if entry_key == key:
                    found_entry = entry
                    found_idx = i
                    break
                elif entry_key < key:
                    # Found insertion point, next larger entry is previous in list
                    next_entry = entries[i-1] if i > 0 else self._find_next_larger_entry(node_idx, -1)
                    break
            else:
                # All entries are larger than key, next larger is in previous node
                next_entry = self._find_next_larger_entry(node_idx, -1)
        else:
            # Binary search for larger nodes
            i = bisect_left(keys, -key, key=lambda v: -v)
            if i < len(entries) and entries[i].item.key == key:
                found_entry = entries[i]
                found_idx = i
            elif i > 0:
                # Key would be inserted at position i, so next larger is at i-1
                next_entry = entries[i-1]
            else:
                # Key is larger than all entries in this node
                next_entry = self._find_next_larger_entry(node_idx, -1)
        
        # Handle successor finding for exact matches
        if found_entry is not None and with_next:
            next_entry = self._find_next_larger_entry(node_idx, found_idx)
        elif found_entry is not None:
            next_entry = None
        
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

        if self.head is None:                        # ··· (1) empty
            self = type(self)()  # Create new instances of the same class
            right = type(self)()
            return self, None, right

        # --- locate split node ------------------------------------------------
        # Find the first node where key >= min_key (using reverse order bounds)
        left_search, right_search = 0, len(self._bounds)
        while left_search < right_search:
            mid = (left_search + right_search) // 2
            if key >= self._bounds[mid]:  # key >= min_key of node
                right_search = mid
            else:
                left_search = mid + 1
        
        # If key is smaller than any key in the list, all entries have keys > key
        # Interface expects: left (keys < key), right (keys > key)
        if left_search >= len(self._nodes):         # ··· (2) key < min
            empty_klist = type(self)()  # Empty klist for keys < key
            return empty_klist, None, self  # Empty left, all entries go to right

        # If key is larger than the maximum key, all entries have keys < key  
        if self._nodes and key > self._nodes[0].entries[0].item.key:  # key > max
            empty_klist = type(self)()  # Empty klist for keys > key
            return self, None, empty_klist  # All entries go to left, empty right

        node_idx = left_search
        split_node = self._nodes[node_idx]
        prev_node = self._nodes[node_idx - 1] if node_idx else None
        original_next = split_node.next

        # --- bisect inside that node (reverse order) -------------------------
        node_entries = split_node.entries
        node_keys = split_node.keys

        # Binary search in descending order
        i = bisect_left(node_keys, -key, key=lambda v: -v)
        exact = i < len(node_keys) and node_keys[i] == key

        # In reverse order: entries with keys > split_key go to greater_part, <= split_key go to lesser_part
        greater_entries = node_entries[:i]  # Keys > split_key -> will become interface RIGHT
        lesser_entries = node_entries[i + 1 if exact else i :]  # Keys < split_key -> will become interface LEFT
        left_subtree = node_entries[i].left_subtree if exact else None

        greater_keys = node_keys[:i]
        lesser_keys = node_keys[i + 1 if exact else i :]

        real_keys = split_node.real_keys
        # Find position in real_keys using bisect (also in descending order)
        j = bisect_left(real_keys, -key, key=lambda v: -v)
        
        greater_real_keys = real_keys[:j]
        lesser_real_keys = real_keys[j + 1 if exact else j :]

        # ------------- build GREATER PART (keys > split_key) -> INTERFACE RIGHT ------------
        interface_right = type(self)()
        if greater_entries:                          # reuse split_node
            split_node.entries = greater_entries
            split_node.keys = greater_keys
            split_node.next    = None
            split_node.real_keys = greater_real_keys
            
            # Build the right part: previous nodes + split_node
            interface_right.head = self.head
            interface_right.tail = split_node
            if prev_node:
                prev_node.next = split_node
            else:
                interface_right.head = split_node
        else:                                        # nothing in greater part
            if prev_node:                            # just the previous nodes
                prev_node.next = None
                interface_right.head = self.head
                interface_right.tail = prev_node
            else:                                    # empty greater part
                interface_right.head = interface_right.tail = None

        # ------------- build LESSER PART (keys < split_key) -> INTERFACE LEFT ------------
        interface_left = type(self)()
        if lesser_entries:
            new_node = self.KListNodeClass()
            new_node.entries   = lesser_entries
            new_node.keys      = lesser_keys
            new_node.real_keys = lesser_real_keys
            new_node.next      = original_next
            interface_left.head = new_node
        else:                                        # no lesser_entries
            interface_left.head = original_next

        # find interface_left.tail
        tail = interface_left.head
        while tail and tail.next:
            tail = tail.next
        interface_left.tail = tail

        # Rebalance both lists for compaction
        if interface_left.head:
            self._rebalance_for_compaction(interface_left)
        if interface_right.head:
            self._rebalance_for_compaction(interface_right)
        
        # ------------- rebuild indexes ---------------------------------------
        interface_left._rebuild_index()
        interface_right._rebuild_index()

        # Return in interface order: left (keys < split_key), subtree, right (keys > split_key)
        return interface_left, left_subtree, interface_right
        
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
            while len(current.entries) < capacity and next_node.entries:
                # Move an item from next_node to current
                shifted = next_node.entries.pop(0)
                shifted_key = next_node.keys.pop(0)
                current.entries.append(shifted)
                current.keys.append(shifted_key)
                if shifted_key >= 0:  # Only update real_keys if it's not a dummy key
                    shifted_real_key = next_node.real_keys.pop(0)
                    current.real_keys.append(shifted_real_key)
                
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

    def count_ge(self, key: int) -> int:
        """
        Return the count of items with keys greater than or equal to the input key.
        
        This method leverages the existing index system for O(log l + log k) performance,
        where l is the number of nodes and k is the capacity per node.
        
        Since entries are stored in descending order, we count from the beginning.
        
        Args:
            key (int): The key threshold
            
        Returns:
            int: Number of items with keys >= key
            
        Raises:
            TypeError: If key is not an integer
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        # Empty list case
        if not self._prefix_counts_tot:
            return 0
        
        # If key is greater than the maximum key (first entry), return 0
        if self.head and key > self.head.entries[0].item.key:
            return 0
            
        # If key is less than or equal to the minimum key (last entry), return total count
        if key <= self._bounds[-1]:  # _bounds contains minimum keys
            return self._prefix_counts_tot[-1]
        
        # Find the first node that might contain keys >= key
        # We need to find the first node where key >= min_key
        # Since _bounds contains minimum keys in descending order, search for -key
        left = bisect_left(self._bounds, -key, key=lambda v: -v)
        
        # If key is smaller than all minimum keys, count all items
        if left >= len(self._nodes):
            return self._prefix_counts_tot[-1]
            
        count = 0
        
        # Count items from the beginning up to but not including the target node
        if left > 0:
            count = self._prefix_counts_tot[left - 1]
        
        # Count items in the target node that are >= key
        node = self._nodes[left]
        node_keys = node.keys
        if node_keys:
            # Binary search within the node to find first position >= key (in descending order)
            left_bs, right_bs = 0, len(node_keys)
            while left_bs < right_bs:
                mid = (left_bs + right_bs) // 2
                if node_keys[mid] >= key:
                    left_bs = mid + 1
                else:
                    right_bs = mid
            
            # In descending order, all items from 0 to left_bs-1 are >= key
            count += left_bs
        
        return count

    # def get_entry(self, index: int) -> RetrievalResult:
    #         """
    #         Returns the entry at the given overall index in the sorted KList along with the next entry. O(log l) node-lookup plus O(1) in-node offset.

    #         Parameters:
    #             index (int): Zero-based index to retrieve.

    #         Returns:
    #             RetrievalResult: A structured result containing:
    #                 - found_entry: The requested Entry if present, otherwise None.
    #                 - next_entry: The subsequent Entry, or None if no next entry exists.
    #         """
    #         # 0) validate
    #         if not isinstance(index, int):
    #             raise TypeError(f"index must be int, got {type(index).__name__!r}")

    #         # 1) empty list?
    #         if not self._prefix_counts_tot:
    #             return RetrievalResult(found_entry=None, next_entry=None)

    #         total_items = self._prefix_counts_tot[-1]
    #         # 2) out‐of‐bounds?
    #         if index < 0 or index >= total_items:
    #             return RetrievalResult(found_entry=None, next_entry=None)

    #         # 3) find the node in O(log l)
    #         node_idx = bisect_right(self._prefix_counts_tot, index)
    #         node = self._nodes[node_idx]

    #         # 4) compute offset within that node
    #         prev_count = self._prefix_counts_tot[node_idx - 1] if node_idx else 0
    #         offset = index - prev_count

    #         # 5) delegate to node
    #         entry, in_node_succ, needs_next = node.get_by_offset(offset)

    #         # 6) if we hit the end of this node, pull the true successor
    #         if needs_next:
    #             if node.next and node.next.entries:
    #                 next_entry = node.next.entries[0]
    #             else:
    #                 next_entry = None
    #         else:
    #             next_entry = in_node_succ
