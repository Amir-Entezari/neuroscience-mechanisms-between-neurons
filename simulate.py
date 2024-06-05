from pymonntorch import *
import torch
import matplotlib.pyplot as plt


def cosine_similarity(tensor_a, tensor_b):
    # Validate that the tensors are of same size
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Tensors are not of the same size")

    dot_product = torch.dot(tensor_a.flatten(), tensor_b.flatten())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)

    if norm_a.item() == 0 or norm_b.item() == 0:
        raise ValueError("One of the tensors has zero magnitude, cannot compute cosine similarity")

    similarity = dot_product / (norm_a * norm_b)
    return similarity


class Simulation:
    def __init__(self, net: Network = None):
        self.net: Network
        if net:
            self.net = net
        else:
            self.net = Network()

    def add_neuron_group(self, tag, **kwargs):
        if tag in [ng.tag for ng in self.net.NeuronGroups]:
            raise Exception("The neuron group's id already exist.")
        # NeuronGroup(net=self.net, tag=tag, **kwargs)
        return CustomNeuronGroup(net=self.net, tag=tag, **kwargs)

    def add_synapse_group(self, tag, **kwargs):
        # if tag in [sg.tag for sg in self.net.SynapseGroups]:
        #     raise Exception("The synapse group's id already exist.")
        return CustomSynapseGroup(net=self.net, tag=tag, **kwargs)

    def simulate(self, iterations=100):
        self.net.initialize()
        self.net.simulate_iterations(iterations=iterations)

    def initialize(self):
        self.net.initialize()

    def simulate_iterations(self, iterations):
        self.net.simulate_iterations(iterations=iterations)

    def plot_membrane_potential(self, title: str,
                                model_idx: int = 3,
                                record_idx=4,
                                save: bool = None,
                                filename: str = None):
        num_ng = len(self.net.NeuronGroups)
        legend_position = (0, -0.2) if num_ng < 2 else (1.05, 1)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_ng))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i, ng in enumerate(self.net.NeuronGroups):
            ax1.plot(ng.behavior[record_idx].variables["v"][:, :1], color=colors[i], label=f'{ng.tag} potential')
            ax2.plot(ng.behavior[record_idx].variables["I"][:, :1], color=colors[i], label=f"{ng.tag} current")

            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['threshold'], color='red', linestyle='--',
                        label=f'{ng.tag} Threshold')
            ax1.axhline(y=ng.behavior[model_idx].init_kwargs['v_reset'], color='black', linestyle='--',
                        label=f'{ng.tag} v_reset')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('v(t)')
        ax1.set_title(f'Membrane Potential')
        ax1.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')

        ax2.set_xlabel('Time')
        ax2.set_ylabel("I(t)")
        ax2.set_title('Current')
        ax2.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')
        fig.suptitle(title)
        plt.tight_layout()
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()

    def plot_w(self, title: str,
               record_idx: int = 4,
               save: bool = None,
               filename: str = None):
        num_ng = len(self.net.NeuronGroups)
        legend_position = (0, -0.2) if num_ng < 2 else (1.05, 1)
        # Generate colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_ng))
        for i, ng in enumerate(self.net.NeuronGroups):
            plt.plot(ng.behavior[record_idx].variables["w"][:, :1], color=colors[i], label=f'{ng.tag} adaptation')

        plt.xlabel('Time')
        plt.ylabel('w')
        plt.legend(loc='upper left', bbox_to_anchor=legend_position, fontsize='small')

        plt.title(title)
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()

    def plot_IF_curve(self, title: str = None,
                      label: str = None,
                      event_idx=5,
                      current_idx=2,
                      show=True,
                      save: bool = None,
                      filename: str = None):
        frequencies = []
        currents = []
        for i, ng in enumerate(self.net.NeuronGroups):
            spike_events = ng.behavior[event_idx].variables['spikes']
            frequencies.append(len(spike_events) / (self.net.network.dt * self.net.iteration))
            currents.append(ng.behavior[current_idx].init_kwargs['value'])
        plt.plot(currents, frequencies, label=label)
        plt.title(title)
        plt.xlabel('Current (I)')
        plt.ylabel('Frequency (f)')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(filename or title + '.pdf')
        if show:
            plt.show()
        else:
            return plt

    def add_raster_plot(self,
                        ax,
                        s=5):
        # Plot the raster plot
        last_id = 0
        for ng in self.net.NeuronGroups:
            spike_events = self.net[f"{ng.tag}_event", 0].variables["spikes"]
            spike_times = spike_events[:, 0]
            neuron_ids = spike_events[:, 1] + last_id
            ax.scatter(spike_times, neuron_ids, s=s, label=f"{ng.tag}")
            if neuron_ids.count_nonzero():
                last_id = neuron_ids.max()
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron ID')
        ax.legend(loc='upper right')
        ax.set_title('Raster Plot of total network')

    def add_activity_plot(self,
                          ax,
                          synapse_idx=4,
                          print_params=True,
                          text_x=0,
                          text_y=0.5):
        # Plot the activity
        total_activity = torch.zeros(self.net.iteration)
        total_size = 0
        for ng in self.net.NeuronGroups:
            # ax.plot(self.net[f"{ng.tag}_rec", 0].variables["activity"], label=f"{ng.tag}")
            total_activity += self.net[f"{ng.tag}_rec", 0].variables["activity"] * ng.size
            total_size += ng.size
        ax.plot(total_activity / total_size, label="total activity")
        ax.set_xlabel('Time')
        ax.set_ylabel('activity')
        ax.legend()
        ax.set_title('Activity of total network')

    def add_synapses_params_info(self, ax, synapse_idx=4, text_x=0.0, text_y=0.5):
        params_info = f"Synapses parameters:\n"
        for sg in self.net.SynapseGroups:
            params_info += f"Synapse {sg.tag} params:{sg.behavior[synapse_idx].init_kwargs}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
                fontsize=8)

    def get_synapses_params_info(self, synapse_idx=4):
        params_info = f"Synapses parameters:\n"
        for sg in self.net.SynapseGroups:
            params_info += f"Synapse {sg.tag} params:{sg.behavior[synapse_idx].init_kwargs}\n"
        return params_info


class CustomNeuronGroup(NeuronGroup):
    def get_behavior(self, behavior_class: type):
        for key, behavior in self.behavior.items():
            if behavior.__class__.__name__ == behavior_class.__name__:
                return behavior

    def add_current_params_info(self, ax, current_behavior_class: type, text_x=0.0, text_y=0.05):
        current_behavior = self.get_behavior(current_behavior_class)
        params_info = f"""{current_behavior.__class__.__name__} params: {current_behavior.init_kwargs}"""
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    def add_neuron_model_params_info(self, ax, model_idx, text_x=0.0, text_y=0.05):
        params_info = f"{self.behavior[model_idx].__class__.__name__} params:\n"
        for key, value in self.behavior[model_idx].init_kwargs.items():
            params_info += f"{key}: {value}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.4))

    def add_current_plot(self, ax):
        # Plot the current
        ax.plot(self.network[f"{self.tag}_rec", 0].variables["I"][:, :])
        ax.plot([], [], label="Other colors: Received I for each neuron")
        ax.plot(self.network[f"{self.tag}_rec", 0].variables["inp_I"][:, :1],
                label="input current",
                color='black')

        ax.set_xlabel('t')
        ax.set_ylabel('I(t)')
        ax.legend()
        ax.set_title(f'Current: {self.tag}')

    def add_raster_plot(self,
                        ax,
                        s=5):
        # Plot the raster plot
        spike_events = self.network[f"{self.tag}_event", 0].variables["spikes"]
        spike_times = spike_events[:, 0]
        neuron_ids = spike_events[:, 1]
        ax.scatter(spike_times, neuron_ids, s=s, label=f"{self.tag}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron ID')
        ax.legend(loc='upper right')
        ax.set_title(f'Raster Plot: {self.tag}')

    def add_activity_plot(self,
                          ax):
        # Plot the activity
        activities = self.network[f"{self.tag}_rec", 0].variables["activity"]
        x_range = np.arange(1, len(activities) + 1)
        ax.plot(x_range, activities, label="activity")
        ax.set_xlabel('Time')
        ax.set_ylabel('activity')
        ax.legend()
        ax.set_title(f'Activity {self.tag}')

    def add_membrane_potential_plot(self,
                                    ax,
                                    model_idx: int = 3,
                                    ):
        ax.plot(self.network[f"{self.tag}_rec", 0].variables["v"][:, :])

        # ax.axhline(y=self.behavior[model_idx].init_kwargs['threshold'], color='red', linestyle='--',
        #            label=f'{self.tag} Threshold')
        ax.axhline(y=self.behavior[model_idx].init_kwargs['v_reset'], color='black', linestyle='--',
                   label=f'{self.tag} v_reset')

        ax.set_xlabel('Time')
        ax.set_ylabel('v(t)')
        ax.set_title(f'Membrane Potential {self.tag}')
        ax.legend()

    def add_membrane_potential_distribution(self, ax):
        rotated_matrix = np.transpose(self.network[f"{self.tag}_rec", 0].variables["v"])
        # Plotting the rotated heatmap
        ax.imshow(rotated_matrix, aspect='auto', cmap='jet', origin='lower')
        # ax.colorbar(label='Membrane Potential')
        ax.set_ylabel('Neurons')
        ax.set_xlabel('Time (Iterations)')
        ax.set_title('Membrane Potentials Heatmap Distribution Over Time')

    def plot_w(self, title: str,
               record_idx: int = 4,
               save: bool = None,
               filename: str = None):
        # Generate colors for each neuron
        plt.plot(self.behavior[record_idx].variables["w"][:, :1], label=f'adaptation')

        plt.xlabel('Time')
        plt.ylabel('w')
        plt.legend(loc='upper left', fontsize='small')

        plt.title(title)
        if save:
            plt.savefig(filename or title + '.pdf')
        plt.show()


class CustomSynapseGroup(SynapseGroup):
    def get_behavior(self, behavior_class: type):
        for key, behavior in self.behavior.items():
            if behavior.__class__.__name__ == behavior_class.__name__:
                return behavior

    def add_current_plot(self, ax, recorder_behavior_class):
        recorder_behavior = self.get_behavior(recorder_behavior_class)
        # Plot the current
        ax.plot(recorder_behavior.variables["I"][:, :])

        ax.set_xlabel('t')
        ax.set_ylabel('I(t)')
        ax.legend()
        ax.set_title('Synapse Current')

    def add_synapses_params_info(self, ax, synapse_idx=4, text_x=0.0, text_y=0.5):
        params_info = f"Synapses parameters:\n"
        params_info += f"Synapse {self.tag} params:{self.behavior[synapse_idx].init_kwargs}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
                fontsize=8)

    def add_weights_plot(self, ax, neuron_id):
        ax.plot(self.network[f"{self.tag}_rec", 0].variables["weights"][:, :, neuron_id])
        ax.set_xlabel('t')
        ax.set_ylabel('Weights')
        ax.legend()
        ax.set_title(f'Synapse Weights for neuron {neuron_id}')

    def add_cosine_similarity_plot(self, ax, neuron_1, neuron_2):
        cosine_similarity_recorder = []
        for t in range(self.network.iteration):
            w_neuron_1 = self.network[f"{self.tag}_rec", 0].variables["weights"][t, :, neuron_1]
            w_neuron_2 = self.network[f"{self.tag}_rec", 0].variables["weights"][t, :, neuron_2]
            cosine_similarity_recorder.append(cosine_similarity(w_neuron_1, w_neuron_2))
        ax.plot(cosine_similarity_recorder)
        ax.set_xlabel('time')
        ax.set_ylabel('Cosine similarity')
        ax.legend()
        ax.set_title(f'Cosine similarity between neuron {neuron_1} and neuron {neuron_2}')

    def add_learning_params_info(self, ax, synapse_idx=8, text_x=0.0, text_y=0.05):
        params_info = f"{self.behavior[synapse_idx].__class__.__name__} params:\n"
        for key, value in self.behavior[synapse_idx].init_kwargs.items():
            params_info += f"{key}: {value}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.4))
