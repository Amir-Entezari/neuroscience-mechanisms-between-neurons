from pymonntorch import Behavior


class ClearSpikeTrace(Behavior):
    def forward(self, synapse):
        if ((synapse.network.iteration - 1) % (synapse.network.instance_duration + synapse.network.sleep)) == 0:
            synapse.src.v = synapse.src.vector(synapse.src.v_reset)
            synapse.dst.v = synapse.dst.vector(synapse.dst.v_reset)
            synapse.src.trace = synapse.src.vector(0.0)
            synapse.dst.trace = synapse.dst.vector(0.0)