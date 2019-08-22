from typing import List

import numpy as np
from scipy.spatial.distance import cdist


def read_xyz(file_name: str) -> List[np.ndarray]:
    """
    Read particle coordinates from an XYZ file.
    :param file_name: The name of the file to read.
    """
    data = []
    with open(file_name, 'r') as input_file:

        num_particles = int(input_file.readline())
        frame = np.zeros((num_particles, 3))
        _ = input_file.readline()
        for particle_number in range(num_particles):
            line = input_file.readline().split()
            frame[particle_number, :] = line[1:4]
    data.append(frame)
    return data


def main(file_name: str):
    data = read_xyz(file_name)
    cell_size = 0.5
    half_cell = cell_size / 2
    for frame in data:
        x_cells = int(np.ceil(np.max(frame[:, 0]) - np.min(frame[:, 0]) / cell_size))
        y_cells = int(np.ceil(np.max(frame[:, 1]) - np.min(frame[:, 1]) / cell_size))
        z_cells = int(np.ceil(np.max(frame[:, 2]) - np.min(frame[:, 2]) / cell_size))

        for x in range(x_cells):
            for y in range(y_cells):
                for z in range(z_cells):
                    cell_coords = np.array([x, y, z]) + half_cell
                    distance = cdist(cell_coords[np.newaxis, :], frame)


main("sample_data/gel_frame.xyz")