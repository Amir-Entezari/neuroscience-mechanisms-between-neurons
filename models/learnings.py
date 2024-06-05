import numpy as np
import torch

from pymonntorch import Behavior


class BaseLearning(Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tag("weight_learning")

    def get_spike_and_trace(self, synapse):
        src_spike = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        dst_spike = synapse.dst.axon.get_spike(synapse.dst, synapse.dst_delay)

        src_spike_trace = synapse.src.axon.get_spike_trace(
            synapse.src, synapse.src_delay
        )
        dst_spike_trace = synapse.dst.axon.get_spike_trace(
            synapse.dst, synapse.dst_delay
        )

        return src_spike, dst_spike, src_spike_trace, dst_spike_trace

    def compute_dw(self, synapse):
        ...

    def forward(self, synapse):
        synapse.weights += self.compute_dw(synapse)
        self.reset_parameters(synapse)


class PairedSTDPLocalVar(BaseLearning):
    def initialize(self, synapse):
        self.w_max = self.parameter("w_max", 1.0)  # Maximum weight for hard bounds
        self.w_min = self.parameter("w_min", -1.0)
        # Trace parameters
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.learning_rate = self.parameter("learning_rate", None, required=True)
        self.normalization = self.parameter("normalization", True)

    def reset_parameters(self, synapse):
        if ((synapse.network.iteration - 1) % (synapse.network.instance_duration + synapse.network.sleep)) == 0:
            synapse.src.v = synapse.src.vector(synapse.src.v_reset)
            synapse.dst.v = synapse.dst.vector(synapse.dst.v_reset)
            synapse.src.trace = synapse.src.vector(0.0)
            synapse.dst.trace = synapse.dst.vector(0.0)

    def compute_dw(self, synapse):
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)
        # Update weights
        dw_minus = self.soft_bound_A_minus(synapse.weights) * torch.outer(src_spike, dst_spike_trace)

        dw_plus = self.soft_bound_A_plus(synapse.weights) * torch.outer(src_spike_trace, dst_spike)

        dW = self.learning_rate * (dw_plus - dw_minus)

        dW *= abs((self.w_max - synapse.weights) * (-1 - synapse.weights))
        if self.normalization:
            dW = dW - dW.sum(axis=0) / synapse.src.size
        dW = dW / (abs(dW).max() or 1)

        return dW * abs((self.w_max - synapse.weights) * (self.w_min - synapse.weights)) / (self.w_max * self.w_min)

    def soft_bound_A_plus(self, w):
        """ Calculate A+ for soft bounds for a matrix of weights """
        # return (w - self.w_min) * (self.w_max - w)
        return w * (self.w_max - w) ** self.eta

    def soft_bound_A_minus(self, w):
        """ Calculate A- for soft bounds for a matrix of weights """
        return np.abs(w) ** self.eta

    def hard_bound_A_plus(self, w):
        """ Calculate A+ for hard bounds for a matrix of weights """
        return np.heaviside(self.w_max - w, 0) * (self.w_max - w) ** self.eta

    def hard_bound_A_minus(self, w):
        """ Calculate A- for hard bounds for a matrix of weights """
        return np.heaviside(w, 0) * w ** self.eta


class PairedRSTDPLocalVar(BaseLearning):
    def initialize(self, synapse):
        # Trace parameters
        self.tau_c = self.parameter("tau_c", None, required=True)
        # Parameters of A- and A+
        self.w_max = self.parameter("w_max", None, required=True)  # Maximum weight for hard bounds
        self.w_min = self.parameter("w_min", None, required=True)  # Minimum weight for hard bounds
        # Learning parameters
        self.learning_rate = self.parameter("learning_rate", None, required=True)
        self.positive_dopamine = self.parameter("positive_dopamine", None, required=True)
        self.negative_dopamine = self.parameter("negative_dopamine", None, required=True)

        self.normalization = self.parameter("normalization", True)
        self.dopamine_method = self.parameter("dopamine_method", "hard")

        synapse.C = synapse.matrix(mode=0.0)
        self.spike_counter = synapse.dst.vector()
        self.dopamine_list = synapse.dst.vector()

    def forward(self, synapse):
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)
        self.spike_counter += dst_spike.byte()

        # Update weights
        dw_minus = torch.outer(src_spike, dst_spike_trace)
        dw_plus = torch.outer(src_spike_trace, dst_spike)
        dC = self.learning_rate * (dw_plus - 1 * dw_minus)

        dC -= dC.sum(axis=0) / synapse.src.size

        synapse.C += -synapse.C / self.tau_c + dC

        if ((synapse.network.iteration - 1) % (synapse.network.instance_duration + synapse.network.sleep)) == 0:
            winners = self.spike_counter.max() == self.spike_counter
            self.dopamine_list = synapse.dst.vector(self.negative_dopamine)
            self.dopamine_list[synapse.network.curr_data_idx] = self.positive_dopamine
            self.dopamine_list = self.dopamine_list * winners.byte()

            if self.dopamine_method == "hard":
                synapse.C *= abs((self.w_max - synapse.weights) * (-1 - synapse.weights))
                if self.normalization:
                    synapse.C -= synapse.C.sum(axis=0) / synapse.src.size
                synapse.C = synapse.C @ torch.diag(self.dopamine_list)
                synapse.C = synapse.C / (abs(synapse.C).max() or 1)

                synapse.weights += synapse.C * abs((self.w_max - synapse.weights) * (self.w_min - synapse.weights)) / (
                        self.w_max * self.w_min)

            if self.dopamine_method == "soft":
                # Normalization before C
                synapse.C *= abs((self.w_max - synapse.weights) * (self.w_min - synapse.weights)) / abs(
                    self.w_max * self.w_min)
                synapse.C = synapse.C / (abs(synapse.C).max() or 1)
                synapse.C = synapse.C @ torch.diag(self.dopamine_list)
                if self.normalization:
                    synapse.C -= synapse.C.sum(axis=0) / synapse.src.size

                synapse.weights += synapse.C
            # reset parameters
            synapse.C = synapse.matrix(mode=0.0)
            self.spike_counter = synapse.dst.vector()
            self.dopamine_list = synapse.dst.vector()
            synapse.src.v = synapse.src.vector(synapse.src.v_reset)
            synapse.dst.v = synapse.dst.vector(synapse.dst.v_reset)
            synapse.src.trace = synapse.src.vector(0.0)
            synapse.dst.trace = synapse.dst.vector(0.0)
