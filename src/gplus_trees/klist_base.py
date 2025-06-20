"""K-list implementation"""
from typing import TYPE_CHECKING, Optional, Tuple, Type
from bisect import bisect_left, insort_left

from gplus_trees.base import (
    Item,
    AbstractSetDataStructure,
    RetrievalResult,
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
        Inserts an entry into a sorted KListNode by key.
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


        # logger.debug(f"NODE INSERT {x_key}: entries={[e.item.key for e in entries]}, keys={keys}, real_keys={real_keys}")

        # Empty list case
        if not entries:
            entries.append(entry)
            keys.append(x_key)
            if not is_dummy:
                real_keys.append(x_key)
            return None, True

        # Fast path: Append at end
        if x_key > entries[-1].item.key:
            entries.append(entry)
            keys.append(x_key)
            if not is_dummy:
                real_keys.append(x_key)
        # Fast path: Insert at beginning
        elif x_key < entries[0].item.key:
            entries.insert(0, entry)
            keys.insert(0, x_key)
            if not is_dummy:
                real_keys.insert(0, x_key)
        else:
            # Choose algorithm based on list length
            entries_len = len(entries)
            if entries_len < 8:
                # Linear search for very small lists
                for i in range(entries_len):
                    if x_key < entries[i].item.key:
                        entries.insert(i, entry)
                        keys.insert(i, x_key)
                        break
                    elif x_key == entries[i].item.key:
                        # If we find an exact match, we can choose to replace or ignore
                        # Here we choose to ignore the insertion if the key already exists
                        return None, False
            else:
                # Binary search for larger lists - more efficient with higher capacities
                # i = bisect_left([e.item.key for e in entries], x_key)
                i = bisect_left(keys, x_key)
                if i < len(entries) and keys[i] == x_key:
                    return None, False
                entries.insert(i, entry)
                keys.insert(i, x_key)

            if not is_dummy:
                # Insert into real_keys only if it's not a dummy key
                real_keys_len = len(real_keys)
                if real_keys_len < 8:
                    # Linear search for very small lists
                    for i in range(real_keys_len):
                        if x_key < real_keys[i]:
                            real_keys.insert(i, x_key)
                            break
                else:
                    # Binary search for larger lists
                    insort_left(real_keys, x_key)

        # Handle overflow
        # logger.debug(f"ENTRY INSERTED {x_key}: entries={[e.item.key for e in entries]}, keys={keys}, real_keys={real_keys}")
        if len(entries) > self.__class__.CAPACITY:
            pop_entry = entries.pop()
            # logger.debug(f"POP ENTRY {pop_entry.item.key}")
            keys.pop()
            # logger.debug(f"NODE after POP: entries={[e.item.key for e in entries]}, keys={keys}")
            if pop_entry.item.key >= 0:
                real_keys.pop()
            # logger.debug(f"REAL KEYS after POP: real_keys={real_keys}")
            return pop_entry, True
        # logger.debug(f"ENTRY INSERTED {x_key} (POP none): entries={[e.item.key for e in entries]}, keys={keys}, real_keys={real_keys}")
        return None, True
     
    def retrieve_entry(
        self, key: int
    ) -> Tuple[Optional[Entry], Optional[Entry], bool]:
        """
        Returns (found, in_node_successor, go_next):
         - found: the Entry with .item.key == key, or None
         - in_node_successor: the next Entry *within this node* if key < max_key
                              else None
         - go_next: True if key > max_key OR (found at max_key) → caller should
                    continue into next node to find the true successor.
        """
        entries = self.entries
        if not entries:
            return None, None, True

        keys = [e.item.key for e in entries]
        i = bisect_left(keys, key)

        # Case A: found exact
        if i < len(entries) and keys[i] == key:
            found = entries[i]
            # if not last in this node, return node-local successor
            if i + 1 < len(entries):
                return found, entries[i+1], False
            # otherwise we must scan next node
            return found, None, True

        # Case B: i < len → this entries[i] is the in-node successor
        if i < len(entries):
            return None, entries[i], False

        # Case C: key > max_key in this node → skip to next
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
        

    def __lt__(self, other):
        return self.key < other.key

    def _rebuild_index(self):
        """Rebuild the node list and prefix-sum of entry counts."""
        self._nodes.clear()
        self._prefix_counts_tot.clear()
        self._prefix_counts_real.clear()
        self._bounds.clear()
        
        total = 0
        real = 0
        node = self.head
        while node:
            self._nodes.append(node)
            entries = node.entries
            keys = node.keys
            real_keys = node.real_keys

            total += len(entries)
            self._prefix_counts_tot.append(total)
            real += len(real_keys)
            self._prefix_counts_real.append(real)

            # Add the maximum key in this node to bounds
            if entries:
                self._bounds.append(entries[-1].item.key)
            
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
    
    # def insert(
    #         self, 
    #         item: Item,
    #         left_subtree: Optional['GPlusTreeBase'] = None
    # ) -> 'KListBase':
    #     """
    #     Inserts an item with an optional left subtree into the k-list.
    #     It is stored as an Entry(item, left_subtree).

    #     The insertion ensures that the keys are kept in lexicographic order.
    #     If a node overflows (more than k entries), the extra entry is recursively inserted into the next node.

    #     Parameters:
    #         item (Item): The item to insert.
    #         left_subtree (GPlusTreeBase or None): Optional G+-tree to attach as the left subtree.
    #     """
    #     entry = Entry(item, left_subtree)
        
    #     # If the k-list is empty, create a new node.
    #     if self.head is None:
    #         node = self.KListNodeClass()
    #         self.head = self.tail = node
    #     else:
    #         # Fast-Path: If the new key > the last key in the tail, insert there.
    #         if self.tail.entries and item.key > self.tail.entries[-1].item.key:
    #             node = self.tail
    #         else:
    #             # linear search from the head
    #             node = self.head
    #             while node.next is not None and node.entries and item.key > node.entries[-1].item.key:
    #                 node = node.next
        
    #     overflow = node.insert_entry(entry)

    #     if node is self.tail and overflow is None:
    #         self._rebuild_index()
    #         return self

    #     MAX_OVERFLOW_DEPTH = 10000
    #     depth = 0

    #     # Propagate overflow if needed.
    #     while overflow is not None:
    #         if node.next is None:
    #             node.next = self.KListNodeClass()
    #             self.tail = node.next
    #         node = node.next
    #         overflow = node.insert_entry(overflow)
    #         depth += 1
    #         if depth > MAX_OVERFLOW_DEPTH:
    #             raise RuntimeError("KList insert overflowed too deeply – likely infinite loop.")
            
    #     self._rebuild_index()

        # return self

    def insert_entry(self, entry: Entry, rank: Optional[int] = None) -> 'KListBase':
        """
        Inserts an existing Entry object into the k-list, preserving the Entry object's identity.

        The insertion ensures that the keys are kept in lexicographic order.
        If a node overflows (more than k entries), the extra entry is recursively inserted into the next node.

        Parameters:
            entry (Entry): The Entry object to insert (containing item and left_subtree).
            rank (Optional[int]): The rank of the entry, if applicable. Not used in this implementation. Only for compatibility with the G+-tree interface.
            
        Returns:
            KListBase: The updated k-list.
        """
        if not isinstance(entry, Entry):
            raise TypeError(f"insert_entry(): expected Entry, got {type(entry).__name__}")
        
        key = entry.item.key
        # If the k-list is empty, create a new node.
        if self.head is None:
            node = self.KListNodeClass()
            self.head = self.tail = node
        else:
            # Fast-Path: If the new key > the last key in the tail, insert there.
            if self.tail.entries and key > self.tail.entries[-1].item.key:
                node = self.tail
            else:
                # linear search from the head
                node = self.head
                # Using bisect_left to find the insertion node (key >= split key)
                node_idx = bisect_left(self._bounds, key)
                node = self._nodes[node_idx]
                # while node.next is not None and node.entries and key > node.entries[-1].item.key:
                #     node = node.next
        
        overflow, inserted = node.insert_entry(entry)

        if inserted:
            if node is self.tail and overflow is None:
                self._rebuild_index()
                return self, True

            MAX_OVERFLOW_DEPTH = 10000
            depth = 0

            # Propagate overflow if needed.
            while overflow is not None:
                if node.next is None:
                    node.next = self.KListNodeClass()
                    self.tail = node.next
                node = node.next
                overflow, inserted = node.insert_entry(overflow)
                depth += 1
                if depth > MAX_OVERFLOW_DEPTH:
                    raise RuntimeError("KList insert_entry overflowed too deeply – likely infinite loop.")
            self._rebuild_index()
            return self, True
        
        return self, False

    def delete(self, key: int) -> "KListBase":
        node = self.head
        prev = None
        found = False

        # 1) Find and remove the entry.
        while node:
            for i, entry in enumerate(node.entries):
                if entry.item.key == key:
                    del node.entries[i]
                    found = True
                    break
            if found:
                break
            prev, node = node, node.next

        if not found:
            self._rebuild_index()
            return self

        # 2) If head is now empty, advance head.
        if node is self.head and not node.entries:
            self.head = node.next
            if self.head is None:
                # list became empty
                self.tail = None
                self._rebuild_index()
                return self
            # reset for possible rebalance, but prev stays None
            node = self.head

        # 3) If *any other* node is now empty, splice it out immediately.
        elif not node.entries:
            # remove node from chain
            prev.next = node.next
            # if we removed the tail, update it
            if prev.next is None:
                self.tail = prev
            self._rebuild_index()
            return self

        # 4) Start a rebalancing pass through the entire list
        current = node
        capacity = self.KListNodeClass.CAPACITY
        
        # Rebalance all nodes starting from the node where deletion occurred
        while current and current.next:
            next_node = current.next
            
            # Continue moving items from next_node to current until current is at capacity
            # or next_node is empty

            while len(current.entries) < capacity and next_node.entries:
                # Move an item from next_node to current
                shifted = next_node.entries.pop(0)
                current.entries.append(shifted)
                
                # If next_node became empty, splice it out and update tail if needed
                if not next_node.entries:
                    current.next = next_node.next
                    if current.next is None:
                        self.tail = current
                    # Exit inner loop as we've emptied next_node
                    break
            
            # Move to next node for the next iteration
            current = current.next
        
        self._rebuild_index()
        return self
    
    def retrieve(self, key: int) -> RetrievalResult:
        """
        Search for `key` using linear search on the list or binary search O(log l + log k) on the index, based on the number of entries in the node.
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        # Empty list case
        if not self._bounds:
            return RetrievalResult(found_entry=None, next_entry=None)
        
        # Find node that might contain key using binary search on max keys
        node_idx = bisect_left(self._bounds, key)
        
        # Case: key > max of any node
        if node_idx >= len(self._nodes):
            return RetrievalResult(found_entry=None, next_entry=None)
        
        # Get the target node
        node = self._nodes[node_idx]
        entries = node.entries
        
        # Empty node (shouldn't happen if index is maintained)
        if not entries:
            return RetrievalResult(found_entry=None, next_entry=None)
        
        # Case: key < first entry in this node
        if key < entries[0].item.key:
            return RetrievalResult(found_entry=None, next_entry=entries[0])
        
        if len(entries) < 8:
            # Linear search for very small lists
            for i, entry in enumerate(entries):
                if key <= entry.item.key:
                    # Exact match?
                    if entry.item.key == key:
                        found = entry
                        # Find successor
                        if i + 1 < len(entries):
                            succ = entries[i+1]
                        else:
                            succ = (node.next.entries[0] if node.next and node.next.entries else None)
                        return RetrievalResult(found_entry=found, next_entry=succ)
                    # Not found, but we know the successor
                    return RetrievalResult(found_entry=None, next_entry=entry)
            
            # Fell off the end of this node
            if node.next and node.next.entries:
                return RetrievalResult(found_entry=None, next_entry=node.next.entries[0])
            return RetrievalResult(found_entry=None, next_entry=None)
        else:
            # Binary search for larger lists
            # TODO: Store keys in a separate list for O(log k) search
            keys = [e.item.key for e in entries]
            i = bisect_left(keys, key)
        
        # Exact match?
        if i < len(entries) and entries[i].item.key == key:
            found = entries[i]
            # Find successor (in-node or from next node)
            if i + 1 < len(entries):
                succ = entries[i+1]
            else:
                succ = (node.next.entries[0] if node.next and node.next.entries else None)
            return RetrievalResult(found_entry=found, next_entry=succ)
        
        # Not found, but we know the successor
        if i < len(entries):
            return RetrievalResult(found_entry=None, next_entry=entries[i])
        
        # Check next node for successor if we fell off the end of this node
        if node.next and node.next.entries:
            return RetrievalResult(found_entry=None, next_entry=node.next.entries[0])
        
        # No successor found
        return RetrievalResult(found_entry=None, next_entry=None)
    
    def find_pivot(self) -> RetrievalResult:
        """Find the pivot entry (minimum entry) in the KList."""
        return self.get_min()

    def get_min(self) -> RetrievalResult:
        """Retrieve the minimum entry from the sorted KList."""
        if not self._prefix_counts_tot:
            return RetrievalResult(found_entry=None, next_entry=None)
        node = self.head
        entry, in_node_succ, needs_next = node.get_by_offset(0)
        if needs_next:
            if node.next and node.next.entries:
                next_entry = node.next.entries[0]
            else:
                next_entry = None
        else:
            next_entry = in_node_succ

        return RetrievalResult(found_entry=entry, next_entry=next_entry)
    
    def get_max(self) -> RetrievalResult:
        """Retrieve the maximum entry from the sorted KList."""
        if not self._prefix_counts_tot:
            return RetrievalResult(found_entry=None, next_entry=None)
        node = self.tail
        entries = node.entries
        entry, in_node_succ, _ = node.get_by_offset(len(entries) - 1)

        return RetrievalResult(found_entry=entry, next_entry=in_node_succ)

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
        # Using bisect_left to find the first node that contains a key >= split key
        node_idx = bisect_left(self._bounds, key)
        
        # If key is greater than any key in the list
        if node_idx >= len(self._nodes):             # ··· (2) key > max
            right = type(self)()
            return self, None, right

        split_node = self._nodes[node_idx]
        # logger.debug(f"Split node at key {key}: entries={[e.item.key for e in split_node.entries]}, keys={split_node.keys}, real_keys={split_node.real_keys}")
        prev_node = self._nodes[node_idx - 1] if node_idx else None
        original_next = split_node.next

        # --- bisect inside that node -----------------------------------------
        keys = [e.item.key for e in split_node.entries]
        node_entries = split_node.entries
        node_keys = split_node.keys
        
        i = bisect_left(keys, key)
        exact = i < len(keys) and keys[i] == key

        left_entries = node_entries[:i]
        right_entries = node_entries[i + 1 if exact else i :]
        left_subtree = node_entries[i].left_subtree if exact else None

        left_keys = keys[:i]
        right_keys = keys[i + 1 if exact else i :]

        real_keys = split_node.real_keys
        j = bisect_left(real_keys, key)
        left_real_keys = real_keys[:j]
        right_real_keys = real_keys[j + 1 if exact else j :]
        
        # logger.debug(f"Split at key {key}: left_entries={[e.item.key for e in left_entries]}, right_entries={[e.item.key for e in right_entries]}, left_keys={left_keys}, right_keys={right_keys}, left_real_keys={left_real_keys}, right_real_keys={right_real_keys}")

        # ------------- build LEFT --------------------------------------------
        # left = type(self)()
        if left_entries:                          # reuse split_node
            split_node.entries = left_entries
            split_node.keys = left_keys
            split_node.next    = None
            self.tail = split_node              
            split_node.real_keys = left_real_keys
        else:                                        # nothing in split node
            if prev_node:                            # skip it
                prev_node.next = None
                self.tail = prev_node
            else:                                    # key at very first entry
                self.head = self.tail = None

        # ------------- build RIGHT -------------------------------------------
        right = type(self)()
        if right_entries:
            if left_entries:                         # both halves non-empty
                new_node = self.KListNodeClass()
                new_node.entries   = right_entries
                new_node.keys      = right_keys
                new_node.real_keys = right_real_keys
                new_node.next      = original_next
                right.head         = new_node
            else:                                    # left empty → reuse split_node
                split_node.entries   = right_entries
                split_node.keys      = right_keys
                split_node.real_keys = right_real_keys
                split_node.next      = original_next
                right.head           = split_node
        else:                                        # no right_entries
            right.head = original_next

        # find right.tail
        tail = right.head
        while tail and tail.next:
            tail = tail.next
        right.tail = tail

        # Rebalance right list for compaction
        if right.head:
            self._rebalance_for_compaction(right)
        
        # ------------- rebuild indexes ---------------------------------------
        self._rebuild_index()
        right._rebuild_index()

        return self, left_subtree, right
        
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
                
                # If next_node became empty, splice it out and update tail if needed
                if not next_node.entries:
                    current.next = next_node.next
                    if current.next is None:
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
        """
        Yields each entry of the k-list in order.
        Each entry is of the form (item, left_subtree).
        """
        node = self.head
        while node:
            for entry in node.entries:
                yield entry
            node = node.next

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
          1) Each KListNode.entries is internally sorted by item.key.
          2) For each consecutive pair of nodes, 
             last_key(node_i) <= first_key(node_{i+1}).
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
            
            # 2a) Entries within this node are sorted
            for i in range(1, len(node.entries)):
                k0 = node.entries[i-1].item.key
                k1 = node.entries[i].item.key
                assert k0 <= k1, (
                    f"Intra-node sort order violated in node {node}: "
                    f"{k0} > {k1}"
                )

            # 2b) Boundary with the previous node
            if previous_last_key is not None and node.entries:
                first_key = node.entries[0].item.key
                assert previous_last_key <= first_key, (
                    f"Inter-node invariant violated between nodes: "
                    f"{previous_last_key} > {first_key}"
                )
                
            # 2c) All non-tail nodes must be at full capacity
            if not is_last_node:  # Only check non-tail nodes
                assert len(node.entries) == node.__class__.CAPACITY, (
                    f"Non-tail node at position {nodes_seen} has {len(node.entries)} entries, "
                    f"but should have {node.__class__.CAPACITY} (compaction invariant violated)"
                )

            # Update for the next iteration
            if node.entries:
                previous_last_key = node.entries[-1].item.key

            node = node.next

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
