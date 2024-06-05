from pymonntorch import *


class SetCurrent(Behavior):
    def initialize(self, ng):
        self.offset = self.parameter("offset", 0.0)
        ng.I = ng.vector(mode=self.offset)
        # ng.Inp_I = ng.vector(mode=self.offset)

    def forward(self, ng):
        ng.I = ng.vector(mode=self.offset)
        # ng.I.fill_(self.offset)


class CurrentSum(Behavior):
    def forward(self, ng):
        ng.I += ng.inp_I  # Or add other currents


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        self.noise_range = self.parameter("noise_range", 0.0)
        ng.inp_I = ng.vector(self.value)

    def forward(self, ng):
        ng.inp_I = ng.vector(self.value)
        self.add_noise(ng)

    def add_noise(self, ng):
        # ng.inp_I += (ng.vector("uniform") - 0.5) * self.noise_range
        ng.inp_I += (ng.vector("normal(0,1)")) * self.noise_range


class StepCurrent(Behavior):
    def initialize(self, ng):
        self.offset = self.parameter("offset", 0.0, required=True)
        self.value = self.parameter("value", None, required=True)
        self.t_start = self.parameter("t_start", 0.0, required=True)
        self.t_end = self.parameter("t_end", None)
        self.noise_range = self.parameter("noise_range", 0.0)

        ng.inp_I = ng.vector()

    def forward(self, ng):
        t = ng.network.iteration * ng.network.dt
        ng.inp_I = ng.vector(self.offset + (self.t_start <= t < self.t_end) * self.value)

        self.add_noise(ng)

    def add_noise(self, ng):
        # ng.inp_I += (ng.vector("uniform") - 0.5) * self.noise_range
        ng.inp_I += (ng.vector("normal(0,1)")) * self.noise_range


class NoisyCurrent(Behavior):
    def initialize(self, ng):
        self.iterations = self.parameter("iterations", None, required=True)
        self.noise_type = self.parameter("noise_type", "white")
        self.mean = self.parameter("mean", 0.0)
        self.std = self.parameter("std", 1.0)
        self.seed = self.parameter("seed", None)

        ng.inp_I = ng.vector()
        self.noise_current = self.add_noise(mean=self.mean,
                                            std=self.std,
                                            size=self.iterations)

    def forward(self, ng):
        ng.inp_I += ng.vector(float(self.noise_current[ng.network.iteration]))

    def add_noise(self, mean, std, size):
        if self.seed is not None:
            self.set_seed()
        if self.noise_type == 'white':
            return self.white_noise(mean, std, size)
        elif self.noise_type == 'brownian':
            return self.brownian_noise(mean, std, size)
        else:
            raise ValueError("Unsupported noise type")

    def set_seed(self):
        np.random.seed(self.seed)

    def white_noise(self, mean, std, size):
        self.set_seed()
        return np.random.normal(mean, std, size)

    def brownian_noise(self, mean, std, size):
        # Generate white noise with mean=0 and std=1
        white_noise = np.random.normal(0, 1, size)

        # Generate cumulative sum to simulate Brownian motion
        brownian_motion = np.cumsum(white_noise)

        # Adjust mean and std
        adjusted_brownian_motion = (brownian_motion - np.mean(brownian_motion)) / np.std(brownian_motion)

        # Scale to desired mean and std
        scaled_brownian_noise = adjusted_brownian_motion * std + mean

        return scaled_brownian_noise


class RandomCurrent(Behavior):
    def initialize(self, ng):
        self.init_I = self.parameter("init_I", None)
        self.mean = self.parameter("mean", 10.0)
        self.std = self.parameter("std", 0.5)
        self.noise_range = self.parameter("noise_range", 0.0)

        ng.inp_I = ng.vector("uniform") * self.mean * 2
        # ng.inp_I = ng.vector(f"normal({self.mean}, {self.std})")
        if self.init_I is not None:
            ng.inp_I = ng.vector(self.init_I)

    def forward(self, ng):
        rand_I = (ng.vector("uniform") - (ng.inp_I / (self.mean * 2))) * self.std
        ng.inp_I += rand_I + (ng.vector("normal(-0,0.5)")) * self.noise_range
