"""Structural metrics for the GK+-tree height probes.

Two observers over a built GK+-tree, kept together because they walk the
same skeleton/inner-tree structure and feed the same campaigns:

- :func:`analyze_structure` — the nesting-aware height model of L08 (a
  G-node costs 1; a converted node additionally costs its inner tree's
  nested height; ancestor-descendant converted nodes on one path *sum*,
  siblings *max*). Validated against L08 Part 4 (height 6) by the mixed-rank
  probe and reused verbatim here.
- :func:`analyze_chi` — the conversion multiplicity χ of L08's Setup, the
  guard for the bushy-pile-tree probe: max over instances of the number of
  converted nodes on one *within-dimension* skeleton path. χ=1 everywhere is
  the staircase / side-by-side-siblings regime; χ≥2 sustained is genuine
  nested branching (L08 Part 4's mechanism), the thing L15's candidate
  refutation needs.

Both live in the tracked package (not the machine-local runner) so the unit
tests can assert measured height / χ against L08's worked example.

Skeleton vs. inner tree (the one subtlety): a converted node's ``set`` is an
inner GK+-tree one dimension deeper. Iterating that set yields the *outer*
skeleton's gap subtrees (their ``left_subtree`` links survive the KList→tree
conversion), so descending ``entry.left_subtree`` / ``node.right_subtree``
stays inside the current dimension's instance, while ``inner_tree_of`` is the
only door into the next dimension. Both observers rely on this.
"""

from __future__ import annotations


def analyze_structure(tree):
    """Nesting-aware height, max dimension, and dummy count of a GK+-tree.

    Height model (L08 Part 1): visiting a G-node costs 1; if its set has been
    converted to an inner tree, searching the set costs the inner tree's
    nested height on top. KList search cost is constant per node and not
    counted. Returns ``(h_nested, dim_max, dummy_count)``.
    """
    max_dim = tree.DIM
    dummy_count = 0

    def h(t):
        nonlocal max_dim, dummy_count
        if t is None or t.is_empty():
            return 0
        max_dim = max(max_dim, t.DIM)
        node = t.node
        node_set = node.set
        inner = t.inner_tree_of(node_set)
        if inner is not None and not inner.is_empty():
            cost = 1 + h(inner)
        else:
            cost = 1
            for entry in node_set:
                if entry.item.key < 0:
                    dummy_count += 1
        deepest_child = 0
        if node.right_subtree is not None:
            for entry in node_set:
                if entry.left_subtree is not None:
                    deepest_child = max(deepest_child, h(entry.left_subtree))
            deepest_child = max(deepest_child, h(node.right_subtree))
        return cost + deepest_child

    height = h(tree)
    return height, max_dim, dummy_count


def analyze_chi(tree):
    """Conversion-multiplicity metrics of a GK+-tree (L08 Setup, χ).

    Returns ``(max_chi, max_prod_chi)``:

    - ``max_chi`` = max over all instances of χ(instance), where χ(instance)
      is the largest number of converted nodes on a single *within-dimension*
      skeleton path of that instance (never crossing into an inner tree).
      This is L08's ``max_j χ*_j``. χ=1 means every instance has at most one
      converted node per path (staircase / siblings); χ≥2 means a single
      skeleton path crosses ≥2 converted nodes whose inner heights both count
      (L08 Part 4). This is the direct guard against a bushy construction
      silently degenerating into side-by-side sibling piles.
    - ``max_prod_chi`` = max over root-to-leaf paths of the *instance tree*
      (an instance → the inner trees of its converted nodes) of the product
      of the per-instance χ (each floored at 1). This is L08 Part 3(iii)'s
      ``Π χ*_i`` along one expansion branch — the quantity whose growth
      decides whether the exact height envelope ``Σ_j g*_j·Π_{i<j} χ*_i`` is
      polylog (T11 holds) or polynomial (T11 refuted).
    """
    max_chi = 0

    def instance(root):
        # Returns (chi_of_root_instance, best_prod_over_this_instance_subtree).
        nonlocal max_chi
        children = []  # inner trees of the converted nodes in root's skeleton

        def skel(st):
            # Max converted-node count on a skeleton path from st downward,
            # WITHIN this instance (does not cross into inner trees). Collects
            # every converted node's inner tree as a child instance.
            if st is None or st.is_empty():
                return 0
            node = st.node
            node_set = node.set
            inner = st.inner_tree_of(node_set)
            converted = inner is not None and not inner.is_empty()
            if converted:
                children.append(inner)
            best_child = 0
            if node.right_subtree is not None:
                for entry in node_set:
                    if entry.left_subtree is not None:
                        best_child = max(best_child, skel(entry.left_subtree))
                best_child = max(best_child, skel(node.right_subtree))
            return (1 if converted else 0) + best_child

        chi = skel(root)
        max_chi = max(max_chi, chi)
        best_below = 1
        for child in children:
            _, prod_child = instance(child)
            best_below = max(best_below, prod_child)
        return chi, max(chi, 1) * best_below

    if tree is None or tree.is_empty():
        return 0, 1
    _, max_prod_chi = instance(tree)
    return max_chi, max_prod_chi
