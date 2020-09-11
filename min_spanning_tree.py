from disjoint_set_forest import DisjointSetForest


def kruskal(vertices, edges, weight):
    """
    Psaudo-code from Wikipedia:

    algorithm Kruskal(G) is
        F:= ∅
        for each v ∈ G.V do
            MAKE-SET(v)
        for each (u, v) in G.E ordered by weight(u, v), increasing do
            if FIND-SET(u) ≠ FIND-SET(v) then
               F:= F ∪ {(u, v)}
               UNION(FIND-SET(u), FIND-SET(v))
        return F
    """
    min_spanning_tree = []
    disjoint_set = DisjointSetForest()
    for v in vertices:
        disjoint_set.make_set(v)
    edges_sorted = sorted(edges, key=weight)
    for v, u in edges_sorted:
        v_tree = disjoint_set.find(v)
        u_tree = disjoint_set.find(u)
        if None in (u_tree, v_tree):
            continue
        if v_tree != u_tree:
            min_spanning_tree.append((v, u))
            disjoint_set.union(v_tree, u_tree)
    return min_spanning_tree
