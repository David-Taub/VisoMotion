from min_spanning_tree import kruskal


def test():
    V = range(100)
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


# TODO: use unittest
test()
