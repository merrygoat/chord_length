from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def write_xyz(data: np.ndarray, lattice_size: np.ndarray, file_name: Union[str, None]):
    """Write the density map to an XYZ file."""
    if file_name is None:
        file_name = "output.xyz"

    with open(file_name, 'w') as output_file:
        num_cells = data.shape[0] * data.shape[1] * data.shape[2]
        output_file.write("{}\n".format(num_cells))
        output_file.write("comment\n")
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    output_file.write(f"{x * lattice_size[0]}\t{y * lattice_size[1]}\t{z * lattice_size[2]}\t{data[x, y, z]}\n")


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


def map_gel_density(lattice_spacing: float, distance_cutoff: float, frame: np.ndarray, side_lengths: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse grain the density of the system.
    :param lattice_spacing: Side length of the lattice used to grid the simulation box.
    :param distance_cutoff: Distance over which to measure the local density.
    :param frame: A 3 by N frame of particle coordinates.
    :param side_lengths: The x, y, and z, box side lengths.
    """
    tqdm.write("Calculating local density.")
    distance_cutoff_sq = distance_cutoff ** 2
    x_len, y_len, z_len = side_lengths
    num_x_cells = int(np.ceil(x_len / lattice_spacing))
    num_y_cells = int(np.ceil(y_len / lattice_spacing))
    num_z_cells = int(np.ceil(z_len / lattice_spacing))
    lattice_spacing = np.array([x_len / num_x_cells, y_len / num_y_cells, z_len / num_z_cells])

    local_density = np.zeros((num_x_cells, num_y_cells, num_z_cells))
    # Set up the lattice points where the density will be measured.
    lattice_coords = np.array([(x, y, z) for x in range(num_x_cells) for y in range(num_y_cells) for z in range(num_z_cells)]) * lattice_spacing
    lattice_coords = np.reshape(lattice_coords, (num_x_cells, num_y_cells, num_z_cells, 3))

    for particle_coords in tqdm(frame):
        min_coords = particle_coords - distance_cutoff
        min_indices = np.floor(min_coords / lattice_spacing).astype(int)
        max_coords = particle_coords + distance_cutoff
        max_indices = np.ceil(max_coords / lattice_spacing).astype(int)
        distances = build_distance_array(lattice_coords[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]],
                                         particle_coords, x_len, y_len, z_len)
        local_density[min_indices[0]:max_indices[0], min_indices[1]: max_indices[1], min_indices[2]: max_indices[2]] += distances < distance_cutoff_sq
    local_density = local_density * 3 / (4 * np.pi)
    plt.hist(local_density.flatten(), bins=100)
    plt.show()
    return local_density, lattice_spacing


def count_chords(density_map):
    tqdm.write("Counting chord lengths.")
    gel_sol_threshold = 0.3
    density_map = density_map > gel_sol_threshold
    x_chord_histogram = []
    y_chord_histogram = []
    z_chord_histogram = []
    for coord_1 in tqdm(range(density_map.shape[0])):
        for coord_2 in range(density_map.shape[1]):
            x_chord_histogram.extend(list(histogram_contiguous_lengths(density_map[:, coord_1, coord_2])))
            y_chord_histogram.extend(list(histogram_contiguous_lengths(density_map[coord_1, :, coord_2])))
            z_chord_histogram.extend(list(histogram_contiguous_lengths(density_map[coord_1, coord_2, :])))
    plt.hist(x_chord_histogram, bins=np.arange(max(z_chord_histogram)), alpha=0.3, label="x")
    plt.hist(y_chord_histogram, bins=np.arange(max(z_chord_histogram)), alpha=0.3, label="y")
    plt.hist(z_chord_histogram, bins=np.arange(max(z_chord_histogram)), alpha=0.3, label="z")
    plt.xlabel("Chord Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Non-periodic chords in each dimension.")
    plt.show()


def histogram_contiguous_lengths(data: np.ndarray):
    contiguous_lengths = np.diff(np.concatenate(([0], np.where(np.concatenate(([data[0]], data[:-1] != data[1:], [True])))[0])))
    contiguous_lengths = contiguous_lengths[contiguous_lengths.nonzero()]
    if len(contiguous_lengths) % 2 and len(contiguous_lengths) > 1:
        contiguous_lengths[0] += contiguous_lengths[-1]
        contiguous_lengths = contiguous_lengths[:-1]
    return contiguous_lengths


def density_smoothing(density_map: np.ndarray):
    """Smooths density map by averaging each point with its neighbours."""
    tqdm.write("Smoothing local density.")
    for x in tqdm(range(density_map.shape[0])):
        for y in range(density_map.shape[1]):
            for z in range(density_map.shape[2]):
                density_map[x, y, z] = 1 / 8 * (2 * density_map[x, y, z]
                                                + density_map[(x + 1) % density_map.shape[0], y, z]
                                                + density_map[x, (y + 1) % density_map.shape[1], z]
                                                + density_map[x, y, (z + 1) % density_map.shape[2]]
                                                + density_map[(x - 1) % density_map.shape[0], y, z]
                                                + density_map[x, (y - 1) % density_map.shape[1], z]
                                                + density_map[x, y, (z - 1) % density_map.shape[2]])
    plt.hist(density_map.flatten(), bins=30)
    plt.show()
    return density_map


def main(file_name: str, cell_size: float, distance_cutoff: float,
         side_lengths: List[float], output_path: str = None, output_xyz: bool = False):

    data = read_xyz(file_name)
    for frame in data:
        density_map, lattice_spacing = map_gel_density(cell_size, distance_cutoff,
                                                       frame, side_lengths)
        smoothed_density = density_smoothing(density_map)
        if output_xyz:
            write_xyz(smoothed_density, lattice_spacing, output_path)
        count_chords(smoothed_density)


if __name__ == '__main__':
    main("test_data/gel_frame.xyz", 0.5, 1, [36.840315, 36.840315, 36.840315])
