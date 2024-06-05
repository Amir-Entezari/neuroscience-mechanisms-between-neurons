from pymonntorch import *


class LIF(Behavior):
    """
    The neural dynamics of LIF is defined by:

    F(u) = v_rest - v,
    RI(u) = R*I.

    We assume that the input to the neuron is current-based.

    Note: at least one Input mechanism  should be added to the behaviors of the population.
          and Fire method should be called by other behaviors.

    Args:
        tau (float): time constant of voltage decay.
        R (float): the resistance of the membrane potential.
        threshold (float): the threshold of neurons to initiate spike.
        v_reset (float): immediate membrane potential after a spike.
        v_rest (float): neuron membrane potential in absent of input.
    """

    def __init__(
            self,
            R,
            threshold,
            tau,
            v_reset,
            v_rest,
            *args,
            init_v=None,
            init_s=None,
            **kwargs
    ):
        super().__init__(
            *args,
            R=R,
            tau=tau,
            threshold=threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            init_v=init_v,
            init_s=init_s,
            **kwargs
        )

    def initialize(self, neurons):
        """
        Set neuron attributes. and adds Fire function as attribute to population.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag(self.__class__.__name__)

        neurons.R = self.parameter("R", None, required=True)
        neurons.tau = self.parameter("tau", None, required=True)
        neurons.threshold = self.parameter("threshold", None, required=True)
        neurons.v_reset = self.parameter("v_reset", None, required=True)
        neurons.v_rest = self.parameter("v_rest", None, required=True)

        neurons.v = self.parameter("init_v", neurons.vector())
        neurons.spikes = self.parameter("init_s", neurons.v >= neurons.threshold)

        neurons.spiking_neuron = self

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return neurons.R * neurons.I

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return neurons.v_rest - neurons.v

    def Fire(self, neurons):
        """
        Basic firing behavior of spiking neurons:

        if v >= threshold then v = v_reset.
        """
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

    def forward(self, neurons):
        """
        Single step of dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.v += (
                (self._Fu(neurons) + self._RIu(neurons)) * neurons.network.dt / neurons.tau
        )
        self.Fire(neurons)
