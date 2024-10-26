from pymonntorch import Behavior


class ActivityRecorder(Behavior):
    # def initialize(self, neurons):
    #     num_spikes = neurons.spikes.sum()
    #     neurons.activity = num_spikes / neurons.size

    def forward(self, neurons):
        num_spikes = neurons.spikes.sum()
        neurons.activity = num_spikes / neurons.size
