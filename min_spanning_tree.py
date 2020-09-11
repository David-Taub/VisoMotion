
def kruskal(V, E, weight):
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
    for v in V:
        disjoint_set.make_set(v)
    E_sorted = sorted(E, key=weight)
    for v, u in E_sorted:
        v_tree = disjoint_set.find(v)
        u_tree = disjoint_set.find(u)
        if v_tree != u_tree:
            min_spanning_tree.append((v, u))
            disjoint_set.union(v_tree, u_tree)

    return min_spanning_tree


class DisjointSetNode:
    def __init__(self):
        self.parent = self
        self.size = 1


class DisjointSetForest:
    def __init__(self):
        self._value_to_tree = {}

    def make_set(self, value):
        if value not in self._value_to_tree:
            self._value_to_tree[value] = DisjointSetNode()

    def find(self, value):
        node = value if isinstance(value, DisjointSetNode) else self._value_to_tree[value]
        if node.parent == node:
            return node
        node.parent = self.find(node.parent)
        return node.parent

    def union(self, value1, value2):
        node1 = self.find(value1)
        node2 = self.find(value2)
        if node1 == node2:
            return
        if node1.size < node2.size:
            node1.parent = node2
            node2.size += node1.size
        else:
            node2.parent = node1
            node1.size += node2.size


def test():
    V = range(4)
    E = []
    for i in V:
        for j in V:
            if i != j:
                E.append((i, j))

    def weight(edge):
        return abs(edge[0] - edge[1])
    mst = kruskal(V, E, weight)
    for i, j in mst:
        assert(abs(i - j) == 1)


# TODO: move test to another file
test()
