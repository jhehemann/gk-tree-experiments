# Gᵏ-trees

History-independent ordered-set trees (G⁺ / Gᵏ⁺) building on the
[G-trees paper](https://g-trees.github.io/g_trees/). Where the paper formally
defines a term, we adopt it unchanged; terms the paper does not fix may be
sharpened here (see ADR-0001).

## Language

### Trees & structure

**G-tree**:
The tree family from the G-trees paper: a recursively defined search tree whose
items of maximal rank form the root G-node. Family term — name the specific
variant (G⁺-tree, Gᵏ⁺-tree) when talking about our implementations.

**G⁺-tree**:
A leafy G-tree with linked leaves, the B⁺-tree analogue of the family: all items
live in leaf G-nodes, internal G-nodes hold replicas for routing.
Single-dimension.

**Gᵏ⁺-tree**:
The multi-dimension extension of the G⁺-tree: when a G-node's set exceeds its
conversion threshold, that set becomes a Gᵏ⁺-tree of the next-deeper dimension.
_Avoid_: GK tree, G^k tree (spelling drift — write Gᵏ⁺-tree, or GK+-tree in
ASCII-only contexts)

**G-node**:
A node of a G⁺-/Gᵏ⁺-tree: the maximal group of items sharing one rank, holding
the node's item set and a right subtree. Generic term for both variants;
qualify as G⁺-node / Gᵏ⁺-node only when the variant matters.
_Avoid_: bare "node" (ambiguous with segment)

**k-list**:
A sorted linked list of fixed-capacity segments; the recursion anchor at the
deepest dimension of every node set. (`KList` in code.)

**Segment**:
A block of a k-list holding up to K entries in sorted order.
_Avoid_: KList node, bucket ("node" is reserved for G-nodes)

**Item**:
An element of the universe, identified by its key and optionally carrying a
value; the unit whose per-dimension rank determines placement. Real items,
dummies, and replicas are its kinds.
_Avoid_: element, record

**Entry**:
One slot in a G-node's set or a segment: an item paired with the link to its
left subtree.
_Avoid_: slot, pair

**Leaf chain**:
The linked list threading all leaf G-nodes in key order — the "linked" in the
G⁺-tree's linked leaves, and the path for ordered iteration.
_Avoid_: leaf list, linked leaves (as a noun)

### Ranks & placement

**Rank**:
The pseudorandom, geometrically distributed rank of an *item*, derived from its
key hash per dimension; items of equal rank collide into one G-node.
Unqualified "rank" always means the item rank.
_Avoid_: "rank" for the node attribute — that is the node rank

**Node rank**:
The rank a G-node sits at: the minimum item rank in its set when the set holds
more than one item, or 1 when it holds exactly one. The minimum ignores any
higher-ranked replica propagated down from a parent, so the node rank tracks
the node's own items, not the routing replica; it fixes the G-node's layer.
_Avoid_: rank (unqualified — that is the item rank), level

**Dimension**:
The nesting depth of a Gᵏ⁺-tree: dimension 1 is the outermost tree, a higher
dimension is nested deeper (inside a G-node's set). Ranks for dimension d+1 are
derived by hashing the dimension-d digest.
_Avoid_: level (collides with tree height)

**Layer**:
All G-nodes sharing one node rank, viewed as one horizontal slice of a tree;
layer 1 is the leaf layer. Not a depth: ranks may skip values, so a layer's
nodes need not sit at one distance from the root.
_Avoid_: level

**Dummy item**:
The artificial least item of a tree (paper: dummy element ⊥): it carries the
tree's maximal rank and a key below every other key in its tree, so the
leftmost G-node of *every* layer — not just the leaf
layer — holds it as its first entry. Each dimension has its own dummy,
sorting before all outer dummies; dummies never count as real items.
_Avoid_: sentinel

**Inherited dummy**:
An outer dimension's dummy that became an ordinary member of a deeper tree
through conversion. It counts as an item and can serve as the deeper tree's
pivot, but never as a real item.

**Pivot item**:
The least item among those of maximal rank in a (sub)tree. It always sits in
that subtree's root G-node, as the first
item that is not a deeper dimension's dummy; the tree's own dummy —
inherited or not — qualifies. Every dummy is the pivot of the tree it
leads, but not vice versa: in G-nodes holding no dummy, the pivot is an
ordinary item (a replica, or a real item in a leaf).
_Avoid_: partition item (the paper's zip-tree analogue, unused for G-trees),
boundary item (the per-node least item, which coincides with the
pivot only in the root G-node)

**Real item**:
An item carrying a user key and value. Only real items count toward a tree's
size; dummies — own or inherited — never do.

**Replica**:
A value-less copy of a leaf item stored in internal G-nodes for routing.
_Avoid_: copy, duplicate

### Conversion

**Conversion**:
The representation switch of a node set at the conversion threshold — see
expansion and collapse.
_Avoid_: instantiation (reserved for the paper's sense: backing a G-tree with
a concrete set data structure)

**Conversion threshold**:
The item count k·l_factor of a node set: exceeding it triggers expansion,
falling to or below it triggers collapse. Deliberately hysteresis-free
(ADR-0002).

**Expansion**:
The conversion of a k-list into a Gᵏ⁺-tree of the next-deeper dimension when
its item count exceeds the conversion threshold.

**Collapse**:
The reverse conversion: a Gᵏ⁺-tree whose item count falls to the conversion
threshold becomes a k-list again.
_Avoid_: contraction, shrink

### Hashing

**Digest**:
The per-dimension hash of an item's key: dimension 1 hashes the key itself,
dimension d+1 hashes the dimension-d digest. Its trailing zero bits yield the
item's rank.
_Avoid_: hash (unqualified)

**Merkle hash**:
The structural hash of a G-node (rank, entries, subtrees), used to compare
and verify trees.
_Avoid_: digest (that is the rank-derivation hash)

### Invariants

**Compaction invariant**:
In a k-list, every segment except the last is full (the paper's k-list
well-formedness). Makes the segment layout — and hence the physical height —
a function of the item count alone.

**Packed invariant**:
Every internal G-node (node rank ≥ 2) holds more than one item; only leaf
G-nodes may be singletons.

**Replica invariant**:
Internal G-nodes carry no values: every item of node rank ≥ 2 is a replica
or a dummy; values live in the leaf layer only.

**Threshold invariant**:
A node set's representation matches its item count: a k-list holds at most
the conversion threshold, a nested Gᵏ⁺-tree holds more (ADR-0002).

### Properties & measurement

**History independence**:
The family's defining property: a tree's structure — and therefore its Merkle
hash — is a pure function of its current item set, independent of the
operation order that produced it.
_Avoid_: unique representation, structural unicity, anti-persistence,
confluent persistence (paper synonyms)

**Item count**:
The number of items in a tree or node set, dummies — own and inherited —
included; the count the conversion threshold compares against.
_Avoid_: size, count (unqualified)

**Real item count**:
The number of real items only, every dummy excluded. A tree's size always
means its real item count.
_Avoid_: item count (that includes dummies)

**G-node height**:
The height of a tree counted in G-nodes from the root to the leaf layer —
the paper's height. Unqualified "height" means this.
_Avoid_: depth, level

**Physical height**:
The height counted in concrete storage units, nested trees and k-list
segments included; a k-list's physical height is its segment count.
_Avoid_: height (unqualified — that is the G-node height)

**Adversarial keys**:
Keys crafted so their iterated digests yield rank 1 in every dimension up to
a target depth, colliding into single G-nodes and forcing maximal nesting —
the attack the Gᵏ⁺ conversion exists to bound.
_Avoid_: malicious keys, crafted datasets
