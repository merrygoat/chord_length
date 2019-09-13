import numpy as np

import chord_length


def test_distance_array():
    lattice_points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).reshape((2, 2, 2, 3))
    result = chord_length.build_distance_array(lattice_points, [0.5, 0.5, 0.5], 2, 2, 2)
    assert np.allclose(result, np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]).reshape((2, 2, 2)))


def test_chord_length():
    # This should give an output with a 10 by 10 lattice of density maxima.
    chord_length.main("test_data/test_cube_1000.xyz", 0.5, 1, [10, 10, 10], "test_data/test_cube_1000_output.xyz")
