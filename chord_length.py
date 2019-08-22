from typing import List

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import trange


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


def write_xyz(data: np.ndarray, file_name: str = "output.xyz"):
    """Write the density map to an XYZ file."""
    with open(file_name, 'w') as output_file:
        num_cells = data.shape[0] * data.shape[1] * data.shape[2]
        output_file.write("{}\n".format(num_cells))
        output_file.write("comment\n")
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    output_file.write("{}\t{}\t{}\t{}\n".format(x, y, z, data[x, y, z]))


def map_gel_density(cell_size: float, distance_cutoff_sq: float, frame: np.ndarray) -> np.ndarray:
    """Coarse grain the density of the system.
    :param cell_size: Side length of the cells used to grid the simulation box.
    :param distance_cutoff_sq: Distance over which to measure the local density.
    :param frame: A 3 by N frame of particle coordinates.
    """

    half_cell = cell_size / 2

    x_cells = int(np.ceil((np.max(frame[:, 0]) - np.min(frame[:, 0])) / cell_size))
    y_cells = int(np.ceil((np.max(frame[:, 1]) - np.min(frame[:, 1])) / cell_size))
    z_cells = int(np.ceil((np.max(frame[:, 2]) - np.min(frame[:, 2])) / cell_size))
    density = np.zeros((x_cells, y_cells, z_cells), dtype=int)
    for x in trange(x_cells):
        for y in range(y_cells):
            for z in range(z_cells):
                cell_coords = np.array([x, y, z]) * cell_size + half_cell
                distances = cdist(cell_coords[np.newaxis, :], frame, metric="sqeuclidean")
                density[x, y, z] = np.count_nonzero(distances < distance_cutoff_sq)
    density = density / np.max(density)
    density = 1 - density
    return density


def main(file_name: str, cell_size: float, distance_cutoff: float):
    data = read_xyz(file_name)
    distance_cutoff_sq = distance_cutoff * distance_cutoff
    for frame in data:
        density_map = map_gel_density(cell_size, distance_cutoff_sq, frame)
        write_xyz(density_map)


main("sample_data/gel_frame.xyz", 0.5, 1)