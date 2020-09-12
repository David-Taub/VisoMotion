"""
Felzenszwalb and Huttenlocher, 2003
http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
"""
from min_spanning_tree import kruskal
from min_spanning_tree import DisjointSetForest
# import itertools
import numpy as np
K = 0.8


def selective_search(img):
    vertices, edges, weight = build_graph(img)
    edges_sorted = sorted(edges, key=lambda edge: weight(vertices[edge[0]], vertices[edge[1]]))
    disjoint_set = DisjointSetForest()
    for v in vertices:
        disjoint_set.make_set(v)
    for v, u in edges_sorted:
        component_v_tree = disjoint_set.find(v)
        component_u_tree = disjoint_set.find(u)
        if component_v_tree != component_u_tree and \
                weight((u, v)) <= min_internal_difference(component_v_tree.set, component_u_tree.set,
                                                          weight):
            # TODO: here we can make the components hold their edges, for faster min_internal_difference
            disjoint_set.union(component_v_tree, component_u_tree)
    return disjoint_set.get_forest_sets()


def build_graph(img):
    # TODO: implement
    # vertices = itertools.product(range(img.shape[0]), range(img.shape[1]))

    shape = img.shape[:2]
    positions = np.indices(shape)
    # positions = [tuple(row) for row in positions.reshape(2, -1).T]
    # 8 connected neighbors
    # horizontal
    edges = list(zip(np.ravel_multi_index(positions[:, :, :-1].reshape(2, -1), shape),
                     np.ravel_multi_index(positions[:, :, 1:].reshape(2, -1), shape)))
    # vertical
    edges += list(zip(np.ravel_multi_index(positions[:, 1:, :].reshape(2, -1), shape),
                      np.ravel_multi_index(positions[:, :-1, :].reshape(2, -1), shape)))
    # diagonal \
    edges += list(zip(np.ravel_multi_index(positions[:, :-1, :-1].reshape(2, -1), shape),
                      np.ravel_multi_index(positions[:, 1:, 1:].reshape(2, -1), shape)))
    # diagonal /
    edges += list(zip(np.ravel_multi_index(positions[:, 1:, :-1].reshape(2, -1), shape),
                      np.ravel_multi_index(positions[:, :-1, 1:].reshape(2, -1), shape)))
    # TODO: add optical flow to video version
    vertices = [(y, x) + tuple(img[y, x, :]) for y, x in positions.reshape(2, -1).T]

    def weight(vertex1, vertex2):
        return np.linalg.norm(np.array(vertex1) - np.array(vertex2))
    return vertices, edges, weight


def internal_difference(compoenent, edges, weight):
    mst = kruskal(compoenent, edges, weight)
    return min(mst, key=weight)


def min_internal_difference(component1, component2, edges_component1, edges_component2, weight):
    return min(internal_difference(component1, edges_component1, weight) + threshold_function(component1),
               internal_difference(component2, edges_component2, weight) + threshold_function(component2))


def threshold_function(component):
    return K / len(component)


def test():
    img = np.random.random((10, 10, 3))
    assert(sum([len(s) for s in selective_search(img)]) == 100)


test()
