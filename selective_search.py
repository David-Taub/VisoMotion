"""
Felzenszwalb and Huttenlocher, 2003
http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
"""
from functools import lru_cache
from min_spanning_tree import kruskal
from min_spanning_tree import DisjointSetForest
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

K = 1.2
COLOR_STENGTH = 1.0
CACHE_SIZE = 2 ** 16


def find_segments(img):
    vertices, edges, weight = build_graph(img)
    vertex_sets = selective_search(vertices, edges, weight)
    # position_sets = indices_to_positions(vertex_sets, vertices)
    return vertex_sets


def selective_search(vertices: list, edges: list, weight):
    edges_sorted = sorted(edges, key=lambda edge: weight(*edge))
    disjoint_set = DisjointSetForest()
    for v in tqdm(vertices):
        node = disjoint_set.make_set(v)
        node.edges = {edge for edge in edges if v in edge}
    for v, u in tqdm(edges_sorted):
        component_v_tree = disjoint_set.find(v)
        component_u_tree = disjoint_set.find(u)
        if component_v_tree != component_u_tree and \
                weight(v, u) <= min_internal_difference(tuple(component_v_tree.set), tuple(component_u_tree.set),
                                                        tuple(component_v_tree.edges), tuple(component_u_tree.edges), weight):
            # TODO: here we can make the components hold their edges, for faster min_internal_difference
            node = disjoint_set.union(component_v_tree, component_u_tree)
            node.edges = node.edges.union(component_v_tree.edges, component_u_tree.edges)
    return disjoint_set.get_forest_sets()


def indices_to_positions(index_sets, vertices):
    position_sets = []
    for index_set in index_sets:
        position_sets.append({(vertices[i][0], vertices[i][1]) for i in index_set})
    return position_sets


def build_graph(img):
    # vertices = itertools.product(range(img.shape[0]), range(img.shape[1]))

    shape = img.shape[:2]
    positions = np.indices(shape)
    # 8 connected neighbors
    # horizontal
    vertices = np.hstack((positions.reshape(2, -1).T, img.reshape([-1, 3])))
    indices = np.arange(shape[0] * shape[1])
    edges = list(zip(vertices[indices % shape[1] != shape[1] - 1], vertices[indices % shape[1] != 0]))
    # vertical
    edges += list(zip(vertices[:-shape[1]], vertices[shape[1]:]))
    # diagonal \
    edges += list(zip(vertices[(indices % shape[1] != shape[1] - 1) & (indices < shape[1] * (shape[0] - 1))],
                      vertices[(indices % shape[1] != 0) & (indices > shape[1])]))
    # diagonal /
    edges += list(zip(vertices[(indices % shape[1] != shape[1] - 1) & (indices > shape[1])],
                      vertices[(indices % shape[1] != 0) & (indices < shape[1] * (shape[0] - 1))]))
    # TODO: add optical flow to video version
    vertices = [tuple(v) for v in vertices]
    edges = [(tuple(v), tuple(u)) for v, u in edges]

    def weight(vertex1, vertex2):
        return (vertex1[2] - vertex2[2]) ** 2 + (vertex1[3] - vertex2[3]) ** 2 + (vertex1[4] - vertex2[4]) ** 2
    return vertices, edges, weight


@lru_cache(maxsize=CACHE_SIZE)
def internal_difference(component, edges, weight):
    mst = kruskal(component, edges, weight)
    if len(mst) == 0:
        return 0
    return min([weight(*edge) for edge in mst])


def min_internal_difference(component1, component2, edges_component1, edges_component2, weight):
    return min(internal_difference(component1, edges_component1, weight) + threshold_function(component1),
               internal_difference(component2, edges_component2, weight) + threshold_function(component2))


def threshold_function(component):
    return K / len(component)


def color_segments(img, segments):
    cmap = cm.tab20
    img = img.copy()
    for i, segment in enumerate(segments):
        segment_color = cmap(i % len(cmap.colors))
        for pixel in segment:
            y = int(pixel[0])
            x = int(pixel[1])
            img[y, x, :] = (1 - COLOR_STENGTH) * img[y, x, :] + COLOR_STENGTH * np.array(segment_color)[:3]
    return img


def main():
    image_path = sys.argv[1]
    img = imageio.imread(image_path) / 255.0
    segments = find_segments(img)
    img_colored = color_segments(img, segments)
    plt.subplot(121)
    plt.imshow(img_colored)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()


# test()
main()
