import torch.nn.functional as F
# from conex import BaseDendriticInput

from pymonntorch import *


class BaseDendriticInput(Behavior):
    """
    Base behavior for converting pre-synaptic spikes to post-synaptic currents.
    It accounts for excitatory/inhibitory nature of pre-synaptic neurons and adjusts coefficients accordingly.
    Weights must be initialized by other behaviors.

    Args:
        current_coef (float): Scalar coefficient that multiplies the weights.
    """

    def __init__(self, *args, current_coef=1, **kwargs):
        super().__init__(*args, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        """
        Initializes the synapse with a coefficient based on the neuron type (excitatory/inhibitory).

        Args:
            synapse: The synapse object to be initialized.
        """
        synapse.add_tag(self.__class__.__name__)
        self.current_coef = self.parameter("current_coef", 1)
        self.current_type = -1 if "inh" in synapse.src.tags else 1
        synapse.I = synapse.dst.vector(0)
        self.def_dtype = synapse.def_dtype

    def calculate_input(self, synapse):
        ...

    def forward(self, synapse):
        synapse.I += self.current_coef * self.current_type * self.calculate_input(synapse)


class LateralDendriticInput2D(BaseDendriticInput):
    """
    Lateral dendrite behavior specifically designed for handling 2D convolution.

    Note:
    - This class assumes weights are already initialized by other behaviors.
    - Weights shape must be [output_channels, input_channels, kernel_height, kernel_width].
    """

    def __init__(self, *args, current_coef=1, inhibitory=None, **kwargs):
        super().__init__(*args, current_coef=current_coef, inhibitory=inhibitory, **kwargs)

    def initialize(self, synapse):
        super().initialize(synapse)
        ctype = self.parameter("inhibitory", None)
        self.padding = (
            synapse.weights.shape[0] // 2, synapse.weights.shape[1] // 2)  # Assuming square kernels for simplicity

        if ctype is not None:
            self.current_type = ctype * -2 + 1

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(self.def_dtype)
        spikes = spikes.view(-1, *synapse.src_shape)  # Reshape to fit (batch, channels, height, width)

        I = F.conv2d(input=spikes, weight=synapse.weights.T.unsqueeze(0).unsqueeze(0), padding=(1, 0))
        return I.view((-1,))


class LateralInhibitionDendriticInput(Behavior):
    """
    Simple Lateral Inhibtion behavior.
    """

    def __init__(self, *args, current_coef=1, **kwargs):
        super().__init__(*args, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        """
        Initializes the synapse with a coefficient based on the neuron type (excitatory/inhibitory).

        Args:
            synapse: The synapse object to be initialized.
        """
        synapse.add_tag(self.__class__.__name__)
        self.current_coef = self.parameter("current_coef", 1)
        synapse.I = synapse.dst.vector(0)
        self.def_dtype = synapse.def_dtype

    def forward(self, synapse):
        synapse.I = self.current_coef * self.calculate_input(synapse)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(self.def_dtype)
        # synapse.weights.fill_diagonal_(0.)
        # I = -1 * spikes @ synapse.weights * spikes.sum()
        I = (spikes - 1) * spikes.sum()
        return I
