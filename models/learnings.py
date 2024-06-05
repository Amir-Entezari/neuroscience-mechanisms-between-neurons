import numpy as np
import torch

from pymonntorch import Behavior


def soft_bound(w, w_min, w_max):
    return (w - w_min) * (w_max - w)


def hard_bound(w, w_min, w_max):
    return (w > w_min) * (w < w_max)


def no_bound(w, w_min, w_max):
    return 1


BOUNDS = {"soft_bound": soft_bound, "hard_bound": hard_bound, "no_bound": no_bound}


class BaseLearning(Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tag("weight_learning")

    def get_spike_and_trace(self, synapse):
        src_spike = synapse.src.spikes
        dst_spike = synapse.dst.spikes

        src_spike_trace = synapse.x

        dst_spike_trace = synapse.y

        return src_spike, dst_spike, src_spike_trace, dst_spike_trace

    def compute_dw(self, synapse):
        ...

    def reset_parameters(self, synapse):
        ...

    def forward(self, synapse):
        synapse.weights += self.compute_dw(synapse)
        self.reset_parameters(synapse)


class SimpleSTDP(BaseLearning):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for simple connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        w_min (float): Minimum for weights. The default is 0.0.
        w_max (float): Maximum for weights. The default is 1.0.
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        positive_bound (str or function): Bounding mechanism for positive learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
        negative_bound (str or function): Bounding mechanism for negative learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
    """

    def __init__(
            self,
            a_plus,
            a_minus,
            *args,
            w_min=0.0,
            w_max=1.0,
            tau_pre=1.0,
            tau_post=1.0,
            positive_bound=None,
            negative_bound=None,
            **kwargs,
    ):
        super().__init__(
            *args,
            a_plus=a_plus,
            a_minus=a_minus,
            w_min=w_min,
            w_max=w_max,
            positive_bound=positive_bound,
            negative_bound=negative_bound,
            tau_pre=tau_pre,
            tau_post=tau_post,
            **kwargs,
        )

    def initialize(self, synapse):
        self.w_min = self.parameter("w_min", 0.0)
        self.w_max = self.parameter("w_max", 1.0)
        self.a_plus = self.parameter("a_plus", None, required=True)
        self.a_minus = self.parameter("a_minus", None, required=True)
        self.p_bound = self.parameter("positive_bound", "no_bound")
        self.n_bound = self.parameter("negative_bound", "no_bound")
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant

        self.p_bound = (
            BOUNDS[self.p_bound] if isinstance(self.p_bound, str) else self.p_bound
        )
        self.n_bound = (
            BOUNDS[self.n_bound] if isinstance(self.n_bound, str) else self.n_bound
        )

        self.def_dtype = (
            torch.float32
            if not hasattr(synapse.network, "def_dtype")
            else synapse.network.def_dtype
        )
        # initial value of x and y
        if not hasattr(synapse, 'x'):
            synapse.x = synapse.src.vector(0.0)  # Presynaptic trace
        if not hasattr(synapse, 'y'):
            synapse.y = synapse.dst.vector(0.0)  # Postsynaptic trace

    def compute_dw(self, synapse):
        # update traces:
        synapse.x += (-synapse.x / self.tau_pre + synapse.src.spikes.byte()) * synapse.network.dt
        synapse.y += (-synapse.y / self.tau_post + synapse.dst.spikes.byte()) * synapse.network.dt
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)

        dw_minus = (
                torch.outer(src_spike, dst_spike_trace)
                * self.a_minus
                * self.n_bound(synapse.weights, self.w_min, self.w_max)
        )
        dw_plus = (
                torch.outer(src_spike_trace, dst_spike)
                * self.a_plus
                * self.p_bound(synapse.weights, self.w_min, self.w_max)
        )

        return dw_plus - dw_minus


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
        # initial value of x and y
        if not hasattr(synapse, 'x'):
            synapse.x = synapse.src.vector(0.0)  # Presynaptic trace
        if not hasattr(synapse, 'y'):
            synapse.y = synapse.dst.vector(0.0)  # Postsynaptic trace

    def compute_dw(self, synapse):
        # update traces:
        synapse.x += (-synapse.x / self.tau_pre + synapse.src.spikes.byte()) * synapse.network.dt
        synapse.y += (-synapse.y / self.tau_post + synapse.dst.spikes.byte()) * synapse.network.dt

        # Update weights
        dw_minus = self.soft_bound_A_minus(synapse.weights) * torch.outer(synapse.src.spikes, synapse.y)

        dw_plus = self.soft_bound_A_plus(synapse.weights) * torch.outer(synapse.x, synapse.dst.spikes)

        dW = self.learning_rate * (dw_plus - dw_minus)

        dW *= abs((self.w_max - synapse.weights) * (-1 - synapse.weights))
        if self.normalization:
            dW -= dW.sum(axis=0) / synapse.src.size
        dW = dW / (abs(dW).max() or 1)

        return dW * abs((self.w_max - synapse.weights) * (self.w_min - synapse.weights)) / (self.w_max * self.w_min)

    def reset_parameters(self, synapse):
        if ((synapse.network.iteration - 1) % (synapse.network.instance_duration + synapse.network.sleep)) == 0:
            synapse.src.v = synapse.src.vector(synapse.src.v_reset)
            synapse.dst.v = synapse.dst.vector(synapse.dst.v_reset)
            synapse.x = synapse.src.vector(0.0)
            synapse.y = synapse.dst.vector(0.0)

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
        self.tau_pre = self.parameter("tau_pre", None, required=True)  # Presynaptic trace decay constant
        self.tau_post = self.parameter("tau_post", None, required=True)  # Postsynaptic trace decay constant
        self.tau_c = self.parameter("tau_c", None, required=True)
        # Parameters of A- and A+
        self.eta = self.parameter("eta", 1.0)  # Adding eta for weight change control
        self.w_max = self.parameter("w_max", None, required=True)  # Maximum weight for hard bounds
        self.w_min = self.parameter("w_min", None, required=True)  # Minimum weight for hard bounds
        # Learning parameters
        self.learning_rate = self.parameter("learning_rate", None, required=True)
        self.positive_dopamine = self.parameter("positive_dopamine", None, required=True)
        self.negative_dopamine = self.parameter("negative_dopamine", None, required=True)

        self.normalization = self.parameter("normalization", True)
        self.dopamine_method = self.parameter("dopamine_method", "hard")

        # initial value of x and y
        if not hasattr(synapse, 'x'):
            synapse.x = synapse.src.vector(0.0)  # Presynaptic trace
        if not hasattr(synapse, 'y'):
            synapse.y = synapse.dst.vector(0.0)  # Postsynaptic trace

        synapse.C = synapse.matrix(mode=0.0)
        self.spike_counter = synapse.dst.vector()
        self.dopamine_list = synapse.dst.vector()

    def forward(self, synapse):
        self.spike_counter += synapse.dst.spikes.byte()

        # update traces:
        synapse.x += (-synapse.x / self.tau_pre + synapse.src.spikes.byte()) * synapse.network.dt
        synapse.y += (-synapse.y / self.tau_post + synapse.dst.spikes.byte()) * synapse.network.dt

        # Update weights
        dw_minus = torch.outer(synapse.src.spikes, synapse.y)
        dw_plus = torch.outer(synapse.x, synapse.dst.spikes)
        dC = self.learning_rate * (dw_plus - 1*dw_minus)

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
            synapse.x = synapse.src.vector(0.0)
            synapse.y = synapse.dst.vector(0.0)