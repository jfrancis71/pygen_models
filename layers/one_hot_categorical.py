"""
Defines OneHotCategorical layer.
"""


import torch


class OneHotCategorical(torch.nn.Module):
    """Layer which accepts a tensor and returns a OneHotCategorical probability distribution.

    Example::

        >>> one_hot_categorical_layer = OneHotCategorical()
        >>> distribution = one_hot_categorical_layer(torch.rand([7, 10]))
        >>> distribution.batch_shape
        torch.Size([7])
        >>> distribution.sample().shape
        torch.Size([7, 10])
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)


class IndependentOneHotCategorical(torch.nn.Module):
    """Layer which accepts a tensor and returns an IndependentOneHotCategorical probability distribution.

    Args:
        event_shape: eg [3, 10] is 3 independent rv's with event_shape [10].

    Example:
        >>> ind_one_hot_categorical_layer = IndependentOneHotCategorical([3, 10])
        >>> distribution = ind_one_hot_categorical_layer(torch.rand([7, 3, 10]))
        >>> distribution.batch_shape
        torch.Size([7])
        >>> distribution.sample().shape
        torch.Size([7, 3, 10])
    """

    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape

    def forward(self, logits):
        batch_shape = list(logits.shape[:-len(self.event_shape)])
        reshape_logits = logits.reshape(batch_shape + self.event_shape)
        base_distribution = torch.distributions.one_hot_categorical.OneHotCategorical(logits=reshape_logits)
        return torch.distributions.independent.Independent(
            base_distribution,
            reinterpreted_batch_ndims=len(batch_shape))


import doctest
doctest.testmod()
