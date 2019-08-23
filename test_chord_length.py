import numpy as np

import chord_length


def test_distance_array():
    sample_data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],[1, 1, 0], [1, 1, 1]])
    result = chord_length.build_distance_array(sample_data, [0.5, 0.5, 0.5], 2, 2, 2)
    assert np.allclose(result, np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]))
