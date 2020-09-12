from selective_search import find_segments
import numpy as np


def test_random_data():
    N = 12
    img = np.random.random((N, N, 3))
    assert(sum([len(s) for s in find_segments(img)]) == N ** 2)
