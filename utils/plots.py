import matplotlib.pyplot as plt
import numpy as np


def plot_with_custom_grid(ax, x, y, x_distance, y_distance):
    """
    Plot data with custom grid intervals.

    Args:
    x (array-like): The x data points.
    y (array-like): The y data points.
    x_distance (float): The distance between vertical grid lines.
    y_distance (float): The distance between horizontal grid lines.
    """
    # Setting x-ticks for vertical grid lines
    ax.xticks(np.arange(start=min(x) - min(x) % x_distance,
                        stop=max(x) - max(x) % x_distance + x_distance,
                        step=x_distance))

    # Setting y-ticks for horizontal grid lines
    ax.yticks(np.arange(start=min(y) - min(y) % y_distance,
                        stop=max(y) - max(y) % y_distance + y_distance,
                        step=y_distance))

    ax.grid(True)


def add_colorized_raset_plot(ax, neuron_group, time_window, num_data):
    spike_events = neuron_group.behavior[351].variables['spikes']
    spike_times = spike_events[:, 0]
    neuron_ids = spike_events[:, 1]
    ax.scatter(spike_times, neuron_ids, c=spike_times//time_window % num_data, cmap='jet',s=1, label=f"")
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')
    ax.legend(loc='upper right')
    ax.set_title(f'Raster Plot: {neuron_group.tag}')
