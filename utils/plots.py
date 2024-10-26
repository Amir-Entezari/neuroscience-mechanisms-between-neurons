import matplotlib.pyplot as plt
import numpy as np
import torch


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


def add_colorized_raset_plot(ax, neuron_group, time_window, num_data, start_iteration=None, end_iteration=None, s=1, axhline=False, color_by='spike',
                             **kwargs):
    spike_events = neuron_group.behavior[351].variables['spikes']
    if start_iteration == None:
        start_iteration = spike_events[:, 0].min()
    if end_iteration == None:
        end_iteration = spike_events[:, 0].max()
    desired_spike_events = []
    for spike_event in spike_events:
        if start_iteration <= spike_event[0] <= end_iteration:
            desired_spike_events.append(spike_event)
    try:
        spike_events = torch.stack(desired_spike_events)
        spike_times = spike_events[:, 0]
        neuron_ids = spike_events[:, 1]
        c = spike_times if color_by=='spike' else neuron_ids
        ax.scatter(spike_times, neuron_ids, c=(c // time_window) % num_data, cmap='jet', s=s, **kwargs)
        # Add horizontal grid lines at each y-value
        if axhline:
            for neuron_id in neuron_ids:
                plt.axhline(y=neuron_id, color='gray', linewidth=0.5, alpha=0.1)
    except:
        ax.scatter([], [], cmap='jet', s=s, **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')
    ax.legend(loc='upper right')
    ax.set_title(f'Raster Plot: {neuron_group.tag}')
