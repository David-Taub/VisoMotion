from min_spanning_tree import kruskal
import itertools

N = 50


def test_clique():
    V = range(N)
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


def test_clique_self_pointing():
    V = range(N)
    E = itertools.product(V, V)

    def weight(edge):
        return abs(edge[0] - edge[1])
    mst = kruskal(V, E, weight)
    for i, j in mst:
        assert(abs(i - j) == 1)


# TODO: use unittest
test_clique()
test_clique_self_pointing()
