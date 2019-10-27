from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_density_histogram(density_data: np.ndarray, title: str):
    """Plot the cell density as a histogram.
    :param density_data: A flattened array of density data.
    :param title: The title of the plot."""
    plt.hist(density_data, bins=50, density=True)
    plt.xlabel("Density")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.ylim(0, 1)
    plt.show()


def plot_chord_length(chord_histogram: List[list]):
    """Plot the chord length in the x, y and z dimensions. At the moment this is the
     chord length in the gas and solid phases combined.
     :param chord_histogram: Chord lengths in the x, y and z dimensions."""
    labels = ["x", "y", "z"]
    for dimension, label in zip(chord_histogram, labels):
        plt.hist(dimension, bins=np.arange(max(dimension)), alpha=0.3, label=label)
    plt.xlabel("Chord Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Non-periodic chords in each dimension.")
    plt.show()
