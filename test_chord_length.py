import numpy as np

import chord_length


def test_distance_array():
    lattice_points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]).reshape((2, 2, 2, 3))
    result = chord_length.build_distance_array(lattice_points, [0.5, 0.5, 0.5], 2, 2, 2)
    assert np.array_equal(result, np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]).reshape((2, 2, 2)))


def test_chord_length():
    # This should give an output with a 10 by 10 lattice of density maxima.
    chord_length.main("test_data/test_cube_1000.xyz", 0.5, 1, [10, 10, 10], "test_data/test_cube_1000_output.xyz")


class TestNegativeIndexing:
    @staticmethod
    def test_1d_wrap():
        x = np.arange(10)
        y = x.take(range(-2, 2), mode='wrap')
        assert np.array_equal(y, np.array((8, 9, 0, 1)))

    @staticmethod
    def test_3d_wrap():
        lattice_coords = np.array([(x, y, z) for x in range(10) for y in range(10) for z in range(10)])
        lattice_coords = np.reshape(lattice_coords, (10, 10, 10, 3))
        x = y = z = range(-1, 1)
        subset = lattice_coords.take(x, axis=0, mode='wrap').take(y, axis=1, mode='wrap').take(z, axis=2, mode='wrap')
        expected = np.array([[[[9, 9, 9], [9, 9, 0]],
                              [[9, 0, 9], [9, 0, 0]]],
                             [[[0, 9, 9], [0, 9, 0]],
                              [[0, 0, 9], [0, 0, 0]]]])
        assert np.array_equal(subset, expected)


def test_group_by():
    data = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    expected = np.array([5, 2, 3, 2])
    assert np.array_equal(chord_length.histogram_contiguous_lengths(data), expected)
