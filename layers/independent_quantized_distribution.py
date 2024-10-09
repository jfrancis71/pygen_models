"""Defines IndependentQuantizedDistribution Layer"""


import math
import torch
import pygen_models.distributions.quantized_distribution as qd


class IndependentQuantizedDistribution(torch.nn.Module):
    """Layer module returning an independent quantized distribution.

    Args:
        event_shape (List): event shape of the returned distribution.
        num_buckets (Integer): Number of buckets to use in the quantized distribution.

    Example:
        >>> independent_qd_layer = IndependentQuantizedDistribution([3, 32, 32], num_buckets=10)
        >>> independent_qd_distribution = independent_qd_layer(torch.rand([7, 3*32*32*10]))
        >>> independent_qd_distribution.batch_shape
        torch.Size([7])
        >>> independent_qd_distribution.event_shape
        torch.Size([3, 32, 32])
        >>> independent_qd_distribution.sample([2]).shape
        torch.Size([2, 7, 3, 32, 32])
    """
    def __init__(self, event_shape, num_buckets=8, add_noise=True):
        super().__init__()
        self.event_shape = event_shape
        self.num_buckets = num_buckets
        self.add_noise = add_noise

    def params_size(self):
        """return number of parameters required to describe distribution"""
        return math.prod(self.event_shape)*self.num_buckets

    # pylint: disable=C0116
    def forward(self, logits):  # logits, e.g. B, Y, X, P where batch_shape would be B, Y, X
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [self.num_buckets])
        base_distribution = qd.QuantizedDistribution(logits=reshape_logits, add_noise=self.add_noise)
        return torch.distributions.independent.Independent(
            base_distribution,
            reinterpreted_batch_ndims=len(self.event_shape))


import doctest
doctest.testmod()
