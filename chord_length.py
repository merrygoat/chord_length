from typing import List

import numpy as np
from tqdm import trange, tqdm
from numba import jit, float32


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


#@jit(nopython=True)
def build_distance_array(frame, cell_coord, x_len, y_len, z_len):
    # build a reduced distance array, taking into account PBCs
    dist_x = np.abs(frame[:, :, :, 0] - cell_coord[0])
    dist_y = np.abs(frame[:, :, :, 1] - cell_coord[1])
    dist_z = np.abs(frame[:, :, :, 2] - cell_coord[2])
    dist_x[dist_x > (x_len / 2)] = x_len - dist_x[dist_x > (x_len / 2)]
    dist_y[dist_y > (y_len / 2)] = y_len - dist_y[dist_y > (y_len / 2)]
    dist_z[dist_z > (z_len / 2)] = z_len - dist_z[dist_z > (z_len / 2)]
    dist_xyz = dist_x ** 2 + dist_y ** 2 + dist_z ** 2
    return dist_xyz


def map_gel_density(cell_size: float, distance_cutoff_sq: float, frame: np.ndarray, side_lengths: List[float]) -> np.ndarray:
    """Coarse grain the density of the system.
    :param cell_size: Side length of the cells used to grid the simulation box.
    :param distance_cutoff_sq: Distance over which to measure the local density.
    :param frame: A 3 by N frame of particle coordinates.
    :param side_lengths: The x, y, and z, box side lengths.
    """
    x_len, y_len, z_len = side_lengths
    num_x_cells = int(np.ceil(x_len / cell_size))
    num_y_cells = int(np.ceil(y_len / cell_size))
    num_z_cells = int(np.ceil(z_len / cell_size))
    cell_x = x_len / num_x_cells
    cell_y = y_len / num_y_cells
    cell_z = z_len / num_z_cells

    local_density = np.zeros((num_x_cells, num_y_cells, num_z_cells))
    cell_coords = np.zeros((num_x_cells, num_y_cells, num_z_cells, 3))
    for x in range(num_x_cells):
        for y in range(num_y_cells):
            for z in range(num_z_cells):
                cell_coords[x, y, z] = [x * cell_x + cell_x / 2, y * cell_y + cell_y / 2, z * cell_z + cell_z / 2]
    for particle in tqdm(frame):
        distances = build_distance_array(cell_coords, particle, x_len, y_len, z_len)
        local_density += np.count_nonzero(distances < distance_cutoff_sq)
    local_density = local_density / np.max(local_density)
    local_density = 1 - local_density
    return local_density


def main(file_name: str, cell_size: float, distance_cutoff: float, side_lengths: List[float]):
    data = read_xyz(file_name)
    distance_cutoff_sq = distance_cutoff * distance_cutoff
    for frame in data:
        density_map = map_gel_density(cell_size, distance_cutoff_sq, frame, side_lengths)
        write_xyz(density_map)


if __name__ == '__main__':
    main("sample_data/gel_frame.xyz", 0.5, 1, [36.840315, 36.840315, 36.840315])