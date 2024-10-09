"""
Module defines a class QuantizedDistribution which represents a continuous distribution
on interval (0,1) which has been discretized.
"""


from torch.distributions.categorical import Categorical
import torch


def discretize(value, num_buckets):
    return torch.clamp((value * num_buckets).floor(), 0, num_buckets - 1) / num_buckets + 1.0 / (num_buckets * 2)


class QuantizedDistribution:
    """A continuous distribution on interval (0,1) which has been discretized into num_buckets.

    Args:
        logits (Tensor): event log probabilities (unnormalized)

    Example:
        >>> qd = QuantizedDistribution(torch.zeros([10]))
        >>> qd.log_prob(torch.tensor(0.75)).shape
        torch.Size([])
        >>> qd = QuantizedDistribution(torch.zeros([7, 10]))
        >>> qd.batch_shape
        torch.Size([7])
        >>> qd.sample([3]).shape
        torch.Size([3, 7])
    """
    def __init__(self, logits, add_noise):
        self.logits = logits
        self.num_buckets = self.logits.shape[-1]
        self.event_shape = torch.Size([])  # pylint: disable=E1101
        self.batch_shape = self.logits.shape[:-1]
        self.log_buckets = torch.log(torch.tensor(self.num_buckets))  # pylint: disable=E1101
        self.add_noise = add_noise

    def log_prob(self, value):
        """returns log_prob of value under this distribution."""
        # pylint: disable=E1101
        quantized_samples = torch.clamp((value * self.num_buckets).floor(), 0, self.num_buckets - 1)
        return Categorical(logits=self.logits).log_prob(quantized_samples)

    def sample(self, sample_shape=torch.Size()):  # pylint: disable=E1101
        """sample from this distribution."""
        floor = Categorical(logits=self.logits).sample(sample_shape)/self.num_buckets
        if self.add_noise:
            samples = floor + torch.rand_like(floor)/self.num_buckets  # pylint: disable=E1101
        else:
            samples = floor + 1.0/(self.num_buckets*2)
        return samples


import doctest
doctest.testmod()
