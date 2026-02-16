"""K-list -- a cache-friendly sorted linked list of fixed-capacity node segments.

Provides :class:`KListNodeBase` (a single segment) and :class:`KListBase`
(the linked list over those segments), both parameterised via factory-created
subclasses.
"""
from typing import TYPE_CHECKING, Optional, Tuple, Type
from bisect import bisect_left, insort_left

from gplus_trees.base import (
    AbstractSetDataStructure,
    Entry,
)
if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase

from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)

_LINEAR_SEARCH_THRESHOLD = 8


class KListNodeBase:
    """
    A single segment of a k-list, storing up to ``CAPACITY`` :class:`Entry` objects.

    Three parallel sorted lists are maintained for performance:

    - ``entries`` -- the :class:`Entry` objects themselves.
    - ``keys``    -- ``[e.item.key for e in entries]``, enabling O(log k)
      binary search via :func:`bisect_left`.
    - ``real_keys`` -- the subset of ``keys`` with non-negative values
      (excludes dummy sentinel keys).

    All mutations go through helper methods (``_append_entry``,
    ``_insert_entry_at``, ``_pop_last_entry``, ``_pop_first_entry``,
    ``_remove_entry_at``, ``_transfer_first_n_from``) so the three lists
    stay in sync.
    """
    __slots__ = ("entries", "keys", "real_keys", "next")
    
    # Default capacity that will be overridden by factory-created subclasses
    CAPACITY: int  # Default value, will usually be overridden by factory
    
    def __init__(self):
        self.entries: list[Entry] = []
        self.keys: list[int] = []       # Sorted list of keys for fast binary search
        self.real_keys: list[int] = []  # Sorted list of real keys (excluding dummy keys)
        self.next: Optional['KListNodeBase'] = None

    def _append_entry(self, entry: Entry) -> None:
        """Append entry at the end (assumes key > all existing keys)."""
        x_key = entry.item.key
        self.entries.append(entry)
        self.keys.append(x_key)
        if x_key >= 0:
            self.real_keys.append(x_key)

    def _insert_entry_at(self, i: int, entry: Entry) -> None:
        """Insert entry at position i, maintaining all parallel lists."""
        x_key = entry.item.key
        self.entries.insert(i, entry)
        self.keys.insert(i, x_key)
        if x_key >= 0:
            insort_left(self.real_keys, x_key)

    def _pop_last_entry(self) -> Entry:
        """Remove and return the last entry, updating all parallel lists."""
        entry = self.entries.pop()
        self.keys.pop()
        if entry.item.key >= 0:
            self.real_keys.pop()
        return entry

    def _pop_first_entry(self) -> Entry:
        """Remove and return the first entry, updating all parallel lists."""
        entry = self.entries.pop(0)
        key = self.keys.pop(0)
        if key >= 0 and self.real_keys and self.real_keys[0] == key:
            self.real_keys.pop(0)
        return entry

    def _remove_entry_at(self, i: int) -> Entry:
        """Remove and return entry at position i, updating all parallel lists."""
        entry = self.entries.pop(i)
        key = self.keys.pop(i)
        if key >= 0:
            j = bisect_left(self.real_keys, key)
            if j < len(self.real_keys) and self.real_keys[j] == key:
                del self.real_keys[j]
        return entry

    def _transfer_first_n_from(self, source: 'KListNodeBase', n: int) -> None:
        """Move the first *n* entries from *source* to the end of *self*.

        Assumes sorted order is maintained (last key in *self* < first key
        in *source*).  Uses list slicing for **O(k)** total work instead of
        O(n * k) individual ``pop(0)`` calls.
        """
        if n <= 0:
            return

        # Determine how many real_keys belong to the transferred range.
        # All transferred keys are < source.keys[n] (the first key NOT moved).
        if n < len(source.keys):
            boundary = source.keys[n]
            real_count = bisect_left(source.real_keys, boundary)
        else:
            real_count = len(source.real_keys)  # moving everything

        # Batch move entries and keys
        self.entries.extend(source.entries[:n])
        self.keys.extend(source.keys[:n])
        del source.entries[:n]
        del source.keys[:n]

        # Batch move real_keys
        if real_count > 0:
            self.real_keys.extend(source.real_keys[:real_count])
            del source.real_keys[:real_count]

    def insert_entry(
            self, 
            entry: Entry,
    ) -> Tuple[Optional[Entry], bool, Optional[Entry]]:
        """
        Inserts an entry into a sorted KListNode by key.
        If capacity exceeds, last entry is returned for further processing.
        
        Args:
            entry (Entry): The entry to insert into the KListNode.
        Returns:
            Tuple[Optional[Entry], bool, Optional[Entry]]:
                (overflow_entry, was_inserted, next_entry)
        """
        entries = self.entries
        keys = self.keys
        x_key = entry.item.key
        next_entry = None

        # Empty list case
        if not entries:
            self._append_entry(entry)
            return None, True, next_entry

        # Fast path: Append at end
        if x_key > keys[-1]:
            self._append_entry(entry)
        # Fast path: Insert at beginning
        elif x_key < keys[0]:
            self._insert_entry_at(0, entry)
            next_entry = entries[1]
        else:
            # Find insertion position
            entries_len = len(entries)
            if entries_len < _LINEAR_SEARCH_THRESHOLD:
                # Linear search for very small lists
                for i in range(entries_len):
                    if x_key < keys[i]:
                        self._insert_entry_at(i, entry)
                        next_entry = entries[i + 1]
                        break
                    elif x_key == keys[i]:
                        next_entry = entries[i + 1] if i + 1 < entries_len else None
                        return None, False, next_entry
            else:
                # Binary search for larger lists
                i = bisect_left(keys, x_key)
                if i < entries_len and keys[i] == x_key:
                    next_entry = entries[i + 1] if i + 1 < entries_len else None
                    return None, False, next_entry
                self._insert_entry_at(i, entry)
                next_entry = entries[i + 1]

        # Handle overflow
        if len(entries) > self.__class__.CAPACITY:
            pop_entry = self._pop_last_entry()
            return pop_entry, True, next_entry
        return None, True, next_entry

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

        keys = self.keys
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
    A k-list: a sorted linked list of :class:`KListNodeBase` segments.

    Each node holds up to *k* (``CAPACITY``) sorted :class:`Entry` objects.
    An auxiliary index (``_nodes``, ``_bounds``, ``_prefix_counts_tot``,
    ``_prefix_counts_real``) is rebuilt after every mutation for fast lookups.

    Complexity summary (l = number of nodes, k = ``CAPACITY``):

    +-----------------------+----------------------------+
    | Operation             | Time                       |
    +=======================+============================+
    | ``insert_entry``      | O(log l + k) amortised     |
    | ``delete``            | O(log l + k)               |
    | ``retrieve``          | O(log l + log k)           |
    | ``split_inplace``     | O(log l + k)               |
    | ``item_count``        | O(1)                       |
    | ``real_item_count``   | O(1)                       |
    | ``item_slot_count``   | O(1)                       |
    | ``physical_height``   | O(1)                       |
    | ``get_min / get_max`` | O(1)                       |
    +-----------------------+----------------------------+
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
        """Returns the total number of slots available in the k-list in O(1) time."""
        return len(self._nodes) * self.KListNodeClass.CAPACITY
    
    def physical_height(self) -> int:
        """Returns the number of KListNode segments in this k-list in O(1) time."""
        return len(self._nodes)

    def insert_entry(
        self, entry: Entry, rank: Optional[int] = None
    ) -> Tuple['KListBase', bool, Optional[Entry]]:
        """Insert an Entry into the k-list, maintaining sorted order.

        O(log l) node lookup, O(k) in-node insert.  Overflows cascade to
        successor nodes; the index is rebuilt afterwards.

        Parameters:
            entry (Entry): The Entry object to insert.
            rank (Optional[int]): Unused -- accepted for interface
                compatibility with the G+-tree.

        Returns:
            Tuple[KListBase, bool, Optional[Entry]]:
                ``(self, was_inserted, next_entry)`` where *was_inserted* is
                False when the key already exists, and *next_entry* is the
                in-order successor of the inserted key (or None).
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

        overflow, inserted, next_entry = node.insert_entry(entry)

        if inserted:
            # Preserve the original next_entry from the first insertion
            original_next_entry = next_entry
            
            if node is self.tail and overflow is None:
                self._rebuild_index()
                return self, True, original_next_entry

            MAX_OVERFLOW_DEPTH = 10000
            depth = 0

            # Propagate overflow if needed.
            while overflow is not None:
                if node.next is None:
                    node.next = self.KListNodeClass()
                    self.tail = node.next
                node = node.next
                res = node.insert_entry(overflow)
                overflow, inserted, _ = res[0], res[1], res[2]  # Ignore next_entry from overflow
                depth += 1
                if depth > MAX_OVERFLOW_DEPTH:
                    raise RuntimeError("KList insert_entry overflowed too deeply – likely infinite loop.")
            self._rebuild_index()
            return self, True, original_next_entry

        return self, False, next_entry

    def delete(self, key: int) -> "KListBase":
        """Delete the entry with the given key from the k-list.

        Uses the index for O(log l) node lookup and O(log k) binary search
        within the node, then rebalances to maintain the compaction invariant.
        Overall: **O(log l + k)** -- dominated by the rebalance step.

        Parameters:
            key (int): The key to delete.

        Returns:
            KListBase: The updated k-list (self).
        """
        # Empty list -> O(1) early return
        if not self._bounds:
            return self

        # O(log l) node lookup using the index
        node_idx = bisect_left(self._bounds, key)
        if node_idx >= len(self._nodes):
            return self  # key > max, not found

        node = self._nodes[node_idx]
        prev = self._nodes[node_idx - 1] if node_idx > 0 else None

        # O(log k) search within the node
        keys = node.keys
        i = bisect_left(keys, key)
        if i >= len(keys) or keys[i] != key:
            return self  # not found

        node._remove_entry_at(i)

        # If head is now empty, advance head.
        if node is self.head and not node.entries:
            self.head = node.next
            if self.head is None:
                self.tail = None
                self._rebuild_index()
                return self
            node = self.head

        # If any other node is now empty, splice it out.
        elif not node.entries:
            prev.next = node.next
            if prev.next is None:
                self.tail = prev
            self._rebuild_index()
            return self

        # Rebalance starting from the affected node
        self._rebalance_for_compaction(self, start_node=node)

        self._rebuild_index()
        return self

    def retrieve(self, key: int, with_next: bool = True) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Search for *key* in the k-list.

        O(log l) node lookup via ``_bounds``, then O(log k) binary search
        (or linear scan for small nodes) within the target node.
        Overall: **O(log l + log k)**.

        Parameters:
            key (int): The key to search for.
            with_next (bool): If True (default), also return the successor.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]:
                ``(found_entry, next_entry)`` -- found_entry is the Entry
                with the matching key (or None); next_entry is the in-order
                successor (or None).
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        # Empty list case
        if not self._bounds:
            return None, None
        
        # Find node that might contain key using binary search on max keys
        node_idx = bisect_left(self._bounds, key)
        
        # Case: key > max of any node
        if node_idx >= len(self._nodes):
            return None, None
        
        # Get the target node
        node = self._nodes[node_idx]
        entries = node.entries
        keys = node.keys
        
        # Empty node (shouldn't happen if index is maintained)
        if not entries:
            return None, None
        
        # Case: key < first entry in this node
        if key < entries[0].item.key:
            return None, entries[0]
        
        if len(entries) < _LINEAR_SEARCH_THRESHOLD:
            # Linear search for very small lists
            for i, entry in enumerate(entries):
                if key <= entry.item.key:
                    # Exact match?
                    if entry.item.key == key:
                        found = entry
                        
                        # Early return if we don't need the next entry
                        if not with_next:
                            return found, None
                        
                        # Find successor
                        if i + 1 < len(entries):
                            succ = entries[i+1]
                        else:
                            succ = (node.next.entries[0] if node.next and node.next.entries else None)
                        return found, succ
                    # Not found, but we know the successor
                    return None, entry
            
            # Fell off the end of this node
            if node.next and node.next.entries:
                return None, node.next.entries[0]
            return None, None
        else:
            # Binary search for larger lists
            i = bisect_left(keys, key)
        
        # Exact match?
        if i < len(entries) and entries[i].item.key == key:
            found = entries[i]
            
            if not with_next:
                return found, None
            
            # Find successor (in-node or from next node)
            if i + 1 < len(entries):
                succ = entries[i+1]
            else:
                succ = (node.next.entries[0] if node.next and node.next.entries else None)
            return found, succ
        
        # Not found, but we know the successor
        if i < len(entries):
            return None, entries[i]
        
        # Check next node for successor if we fell off the end of this node
        if node.next and node.next.entries:
            return None, node.next.entries[0]
        
        # No successor found
        return None, None
    
    def find_pivot(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Find the pivot entry (minimum entry) in the KList."""
        return self.get_min()

    def get_min(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Retrieve the minimum entry from the sorted KList."""
        if not self._prefix_counts_tot:
            return None, None
        node = self.head
        entry, in_node_succ, needs_next = node.get_by_offset(0)
        if needs_next:
            if node.next and node.next.entries:
                next_entry = node.next.entries[0]
            else:
                next_entry = None
        else:
            next_entry = in_node_succ

        return entry, next_entry
    
    def get_max(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Retrieve the maximum entry from the sorted KList."""
        if not self._prefix_counts_tot:
            return None, None
        node = self.tail
        entries = node.entries
        entry, in_node_succ, _ = node.get_by_offset(len(entries) - 1)

        return entry, in_node_succ

    def _locate_split_node(
        self, key: int
    ) -> Optional[Tuple[KListNodeBase, Optional[KListNodeBase], Optional[KListNodeBase]]]:
        """Locate the node containing the split key using the index.

        O(log l) via binary search on ``_bounds``.

        Returns:
            ``(split_node, prev_node, original_next)`` or ``None`` if
            key > max key in the list.
        """
        node_idx = bisect_left(self._bounds, key)
        if node_idx >= len(self._nodes):
            return None
        split_node = self._nodes[node_idx]
        prev_node = self._nodes[node_idx - 1] if node_idx else None
        original_next = split_node.next
        return split_node, prev_node, original_next

    def _bisect_node_entries(self, split_node: KListNodeBase, key: int) -> tuple:
        """Split a node's parallel lists at *key* using binary search.

        O(log k) for the bisect, plus O(k) for the list slices.

        Returns:
            ``(left_entries, left_keys, left_real_keys,
            right_entries, right_keys, right_real_keys,
            left_subtree)``
        """
        node_entries = split_node.entries
        node_keys = split_node.keys

        i = bisect_left(node_keys, key)
        exact = i < len(node_keys) and node_keys[i] == key

        left_entries = node_entries[:i]
        right_entries = node_entries[i + 1 if exact else i :]
        left_subtree = node_entries[i].left_subtree if exact else None

        left_keys = node_keys[:i]
        right_keys = node_keys[i + 1 if exact else i :]

        real_keys = split_node.real_keys
        j = bisect_left(real_keys, key)
        exact_in_real = j < len(real_keys) and real_keys[j] == key
        left_real_keys = real_keys[:j]
        right_real_keys = real_keys[j + 1 if exact_in_real else j :]

        return (left_entries, left_keys, left_real_keys,
                right_entries, right_keys, right_real_keys,
                left_subtree)

    def split_inplace(
        self, key: int
    ) -> Tuple["KListBase", Optional["GPlusTreeBase"], "KListBase", Optional[Entry]]:

        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        if self.head is None:                        # (1) empty
            self = type(self)()
            right = type(self)()
            return self, None, right, None

        # --- locate split node ------------------------------------------------
        loc = self._locate_split_node(key)
        if loc is None:                              # key > max
            right = type(self)()
            return self, None, right, None

        split_node, prev_node, original_next = loc
        next_entry = None

        # --- bisect inside that node -----------------------------------------
        (left_entries, left_keys, left_real_keys,
         right_entries, right_keys, right_real_keys,
         left_subtree) = self._bisect_node_entries(split_node, key)

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
            next_entry = right_entries[0]
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
            if right.head is not None:
                next_entry = right.head.entries[0]

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

        return self, left_subtree, right, next_entry
        
    def _rebalance_for_compaction(self, klist: 'KListBase', start_node: Optional[KListNodeBase] = None) -> None:
        """Ensure the compaction invariant: all non-tail nodes at full capacity.

        Redistributes entries by batch-transferring from successor nodes using
        ``_transfer_first_n_from`` for **O(k)** per node pair rather than
        individual pops which would be O(k²).

        Parameters:
            klist: The KList to rebalance.
            start_node: Node to start rebalancing from (defaults to ``klist.head``).
        """
        current = start_node if start_node is not None else klist.head
        capacity = klist.KListNodeClass.CAPACITY

        while current and current.next:
            next_node = current.next
            deficit = capacity - len(current.entries)

            if deficit > 0 and next_node.entries:
                to_move = min(deficit, len(next_node.entries))
                current._transfer_first_n_from(next_node, to_move)

                # If next_node became empty, splice it out
                if not next_node.entries:
                    current.next = next_node.next
                    if current.next is None:
                        klist.tail = current

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
        """Verify structural invariants of the k-list.

          1) ``self.tail.next`` is ``None`` (tail is truly the last node).
          2a) Entries within each node are sorted by ``item.key``.
          2b) Parallel lists ``keys`` and ``real_keys`` mirror ``entries``.
          2c) Consecutive nodes satisfy ``last_key(node_i) < first_key(node_{i+1})``.
          2d) All non-tail nodes are at full capacity (compaction invariant).

        Raises:
            AssertionError: if any condition fails.
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

            # 2b) Parallel lists mirror entries
            expected_keys = [e.item.key for e in node.entries]
            assert node.keys == expected_keys, (
                f"keys list out of sync in node {nodes_seen}: "
                f"keys={node.keys}, expected={expected_keys}"
            )
            expected_real = [k for k in expected_keys if k >= 0]
            assert node.real_keys == expected_real, (
                f"real_keys list out of sync in node {nodes_seen}: "
                f"real_keys={node.real_keys}, expected={expected_real}"
            )

            # 2c) Boundary with the previous node
            if previous_last_key is not None and node.entries:
                first_key = node.entries[0].item.key
                assert previous_last_key < first_key, (
                    f"Inter-node invariant violated between nodes: "
                    f"{previous_last_key} >= {first_key}"
                )
                
            # 2d) All non-tail nodes must be at full capacity
            if not is_last_node:  # Only check non-tail nodes
                assert len(node.entries) == node.__class__.CAPACITY, (
                    f"Non-tail node at position {nodes_seen} has {len(node.entries)} entries, "
                    f"but should have {node.__class__.CAPACITY} (compaction invariant violated)"
                )

            # Update for the next iteration
            if node.entries:
                previous_last_key = node.entries[-1].item.key

            node = node.next

