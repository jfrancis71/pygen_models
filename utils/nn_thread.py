import torch
import torch.nn as nn


class NNThread(nn.Module):
    """Threads a batched tensor through a neural net.

    Some important PyTorch modules (eg Conv2d) support only 1 batched dimension.
    This module 'threads' a batched tensor (by flattening the batched dims) through a
    neural net and then reshapes the output to preserve the original batch shape.

    Args:
        nn_module (nn.Module): the module to thread through.
        batch_ndims (int): number of batched dimensions

    Example:
    >>> nnmod = nn.Conv2d(2, 3, kernel_size=1)
    >>> tens = torch.ones([5, 9, 2, 4, 4])
    >>> NNThread(nnmod, 2)(tens).shape
    torch.Size([5, 9, 3, 4, 4])
    """

    def __init__(self, nn_module, batch_ndims: int):
        super().__init__()
        self.nn_module = nn_module
        self.batch_ndims = batch_ndims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:self.batch_ndims]
        nn_output = self.nn_module(x.flatten(0, len(batch_shape)-1))
        return nn_output.reshape(batch_shape + nn_output.shape[1:])


import doctest
doctest.testmod()
