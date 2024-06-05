import random

import torch
import numpy as np
from pymonntorch import Behavior


class FullyConnectedSynapse(Behavior):
    """
    Fully connected synapse class that connect all neurons in a source and destination.
    """

    def initialize(self, sg):
        # Weight parameters
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None)
        self.alpha = self.parameter("alpha", 1.0)

        self.N = sg.src.size
        mean = self.j0 / self.N

        if self.variance is None:
            self.variance = self.j0 / np.sqrt(self.N)
        else:
            self.variance = abs(mean) * self.variance

        sg.weights = sg.matrix(mode=f"normal({mean},{self.variance})")
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.weights.fill_diagonal_(0)
        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spikes
        # sg.I = torch.sum(sg.weights[pre_spike], axis=0)
        sg.I += torch.sum(sg.weights[pre_spike], axis=0) - sg.I * self.alpha
