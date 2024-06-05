import matplotlib.pyplot as plt
from pymonntorch import *
import torch


class CustomPoisson(torch.nn.Module):
    def __init__(self, time_window, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.time_window = time_window

    def __call__(self, img):
        if isinstance(img, tuple):
            return tuple([self(sub_inp) for sub_inp in img])

        if img.dim() > 1:
            print("Data must be converted to vector first.")

        # self.spikes = torch.zeros((self.duration,) + self.img.shape, dtype=torch.bool)
        encoded_spikes = torch.zeros((img.shape[0], self.time_window), dtype=torch.bool)

        # img = (img - img.min()) / (img.max() - img.min())
        # img = (img * (1 - self.epsilon)) + self.epsilon

        for i in range(img.shape[0]):
            spike_times = np.random.poisson(img[i], self.time_window)
            for j, t in enumerate(spike_times):
                if t > 0:
                    encoded_spikes[i, j: t + j] = 1
        encoded_spikes = encoded_spikes.T
        return encoded_spikes

    def add_encoder_info(self,
                         ax,
                         text_x=0,
                         text_y=0.05):
        info = {
            "duration": self.time_window,
            "epsilon": self.epsilon,
        }
        params_info = f"""{self.__class__.__name__} params:\n"""
        for key, value in info.items():
            params_info += f"{key}: {value}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


class Poisson(torch.nn.Module):
    """
    Poisson encoding.
    Input values should be between 0 and 1.
    The intervals between two spikes are picked using Poisson Distribution.

    Args:
        time_window (int): The interval of the coding.
        ratio (float): A scale factor for probability of spiking.
    """

    def __init__(self, time_window, ratio):
        super().__init__()
        self.time_window = time_window
        self.ratio = ratio

    def __call__(self, img):
        if isinstance(img, tuple):
            return tuple([self(sub_inp) for sub_inp in img])

        original_shape, original_size = img.shape, img.numel()
        flat_img = img.view((-1,)) * self.ratio
        non_zero_mask = flat_img != 0

        flat_img[non_zero_mask] = 1 / flat_img[non_zero_mask]

        dist = torch.distributions.Poisson(rate=flat_img, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([self.time_window]))
        intervals[:, non_zero_mask] += (intervals[:, non_zero_mask] == 0).float()

        times = torch.cumsum(intervals, dim=0).long()
        times[times >= self.time_window + 1] = 0

        spikes = torch.zeros(
            self.time_window + 1, original_size, device=img.device, dtype=torch.bool
        )
        spikes[times, torch.arange(original_size, device=img.device)] = True
        spikes = spikes[1:]

        return spikes.view(self.time_window, *original_shape)

    def add_encoder_info(self,
                         ax,
                         text_x=0,
                         text_y=0.05):
        info = {
            "time_window": self.time_window,
            "ratio": self.ratio,
        }
        params_info = f"""{self.__class__.__name__} params:\n"""
        for key, value in info.items():
            params_info += f"{key}: {value}\n"
        ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


class FeedDataset(Behavior):
    def initialize(self, ng):
        self.encoded_dataset = self.parameter("encoded_dataset", None, required=True)
        self.sleep = self.parameter("sleep", None, required=True)

        ng.network.duration = self.encoded_dataset[0].shape[0]
        ng.network.sleep = self.sleep

    def forward(self, ng):
        # TODO: rewrite the encoded_dataset to ignore multiple dots
        ng.network.curr_data_idx = (ng.network.iteration // (ng.network.duration + self.sleep)) % \
                                   self.encoded_dataset.shape[0]

        is_sleep = (ng.network.iteration - 1) % (
                ng.network.duration + self.sleep) < ng.network.duration
        ng.spikes = is_sleep * self.encoded_dataset[ng.network.curr_data_idx][
            (ng.network.iteration - 1) % ng.network.duration]
