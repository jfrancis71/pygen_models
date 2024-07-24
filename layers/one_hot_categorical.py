"""
Defines OneHotCategorical layer.
"""


import torch


class OneHotCategorical(torch.nn.Module):
    """Categorical is a layer which accepts a tensor as input and returns a probability
    distribution. Suitable for use directly in, eg Sequential, presumably as final layer.
    """
    def __init__(self):
        super().__init__()

    # pylint: disable=C0116
    def forward(self, logits):
        return torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
