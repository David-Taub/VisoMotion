
def get_min_spanning_tree(V, E, weight):
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
    V_trees = {}
    for v in V:
        tree = DisjointSetTree()
        V_trees[v] = tree
        disjoint_set.make_set(tree)
    E_sorted = sorted(E, key=weight)
    for v, u in E_sorted:
        v_tree = disjoint_set.find(V_trees[v])
        u_tree = disjoint_set.find(V_trees[u])
        if v_tree != u_tree:
            min_spanning_tree.append((v, u))
            disjoint_set.union(v_tree, u_tree)

    return min_spanning_tree


class DisjointSetTree:
    def __init__(self):
        self.parent = self
        self.size = 1


class DisjointSetForest:
    def __init__(self):
        self.forest = set()

    def make_set(self, x: DisjointSetTree):
        if x not in self.forest:
            self.forest.add(x)

    def find(self, x: DisjointSetTree):
        if x.parent == x:
            return x
        x.parent = self.find(x.parent)
        return x.parent

    def union(self, x: DisjointSetTree, y: DisjointSetTree):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if x.size < y.size:
            x.parent = y
            y.size += x.size
        else:
            y.parent = x
            x.size += y.size


def test():
    V = range(4)
    E = []
    for i in V:
        for j in V:
            if i != j:
                E.append((i, j))

    def weight(edge):
        return abs(edge[0] - edge[1])
    mst = get_min_spanning_tree(V, E, weight)
    for i, j in mst:
        assert(abs(i - j) == 1)


# TODO: move test to another file
test()
