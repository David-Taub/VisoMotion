"""
Felzenszwalb and Huttenlocher, 2003
http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
"""
from min_spanning_tree import DisjointSetForest
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import cv2

BLUR_SIGMA = 2
KERNEL_SIZE = 11


def find_segments(img, k):
    vertices, edges, weight = build_graph(img)
    vertex_sets = selective_search(vertices, edges, weight, k)
    return vertex_sets


def selective_search(vertices: list, edges: list, weight, k=1):
    edges_sorted = sorted(edges, key=lambda edge: weight(*edge))
    disjoint_set = DisjointSetForest()
    for v in vertices:
        node = disjoint_set.make_set(v)
        node.internal_difference = 0
    for v, u in tqdm(edges_sorted):
        component_v_tree = disjoint_set.find(v)
        component_u_tree = disjoint_set.find(u)
        if component_v_tree != component_u_tree:
            w = weight(v, u)

            min_internal_difference = min(component_v_tree.internal_difference + k / component_v_tree.size,
                                          component_u_tree.internal_difference + k / component_u_tree.size)
            if w <= min_internal_difference:
                # new_internal_difference = (component_v_tree.size * component_v_tree.internal_difference +
                #                            component_u_tree.size * component_u_tree.internal_difference + w) / (
                #     component_v_tree.size + component_u_tree.size)
                node = disjoint_set.union(component_v_tree, component_u_tree)
                # node.internal_difference = new_internal_difference
                node.internal_difference = w
    return disjoint_set.get_forest_sets()


def build_graph(img):
    shape = img.shape[:2]
    positions = np.indices(shape)
    # 8 connected neighbors
    # horizontal
    vertices = np.hstack((positions.reshape(2, -1).T, img.reshape([-1, 1])))
    indices = np.arange(shape[0] * shape[1])
    edges = tuple(zip(vertices[indices % shape[1] != shape[1] - 1], vertices[indices % shape[1] != 0]))
    # vertical
    edges += tuple(zip(vertices[:-shape[1]], vertices[shape[1]:]))
    # diagonal \
    # edges += tuple(zip(vertices[(indices % shape[1] != shape[1] - 1) & (indices < shape[1] * (shape[0] - 1))],
    #                    vertices[(indices % shape[1] != 0) & (indices > shape[1])]))
    # # diagonal /
    # edges += tuple(zip(vertices[(indices % shape[1] != shape[1] - 1) & (indices > shape[1])],
    #                    vertices[(indices % shape[1] != 0) & (indices < shape[1] * (shape[0] - 1))]))
    # TODO: add optical flow to video version
    vertices = [tuple(v) for v in vertices]
    edges = [(tuple(v), tuple(u)) for v, u in edges]

    def weight(vertex1, vertex2):
        return abs(vertex1[2] - vertex2[2])
    return vertices, edges, weight


def color_segments(img, segments_colored):
    cmap = cm.tab20
    segmented_map = np.zeros(img.shape[:2])
    for segments in segments_colored:
        last_color_max = np.max(segmented_map)
        for i, segment in tqdm(enumerate(segments)):
            for pixel in segment:
                y = int(pixel[0])
                x = int(pixel[1])
                segmented_map[y, x] += i + last_color_max
    out_img = cmap(segmented_map % len(cmap.colors))[:, :, :3]
    import pdb
    pdb.set_trace()
    return out_img


def main():
    image_path = sys.argv[1]
    img = imageio.imread(image_path) / 255.0
    blur = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), BLUR_SIGMA)
    # segments = find_segments(blur, k=img.shape[0] / 70)
    segments_r = find_segments(blur[:, :, 0], k=1)
    segments_g = find_segments(blur[:, :, 1], k=1)
    segments_b = find_segments(blur[:, :, 2], k=1)
    img_colored = color_segments(img, [segments_r, segments_g, segments_b])

    plt.subplot(131)
    plt.imshow(img_colored)
    plt.subplot(132)
    plt.imshow(img)
    plt.subplot(133)
    plt.imshow(blur)
    plt.show()


if __name__ == '__main__':
    main()
