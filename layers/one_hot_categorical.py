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

    # pylint: disable=C0116
    def forward(self, logits):
        return torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)


import doctest
doctest.testmod()
