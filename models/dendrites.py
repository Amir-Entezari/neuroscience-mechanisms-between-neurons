from pymonntorch import *


class Dendrite(Behavior):
    def forward(self, ng):
        ng.I += ng.inp_I
        for synapse in ng.afferent_synapses["All"]:
            ng.I += synapse.I # + ng.inp_I
