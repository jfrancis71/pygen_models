import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Module):
    """A 3x3 convolution where input from bottom, centre, and centre right are masked out."""
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.mask = nn.Parameter(torch.ones([out_channels, in_channels, 3, 3]).float(), requires_grad=False)
        self.mask[:, :, 2, 2] = 0.0
        self.mask[:, :, 1, 1:] = 0.0
        self.mask[:, :, 2] = 0.0

    def forward(self, x):
        weight = self.conv.weight * self.mask.detach()
        return nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.stride, padding=self.padding)


class SimplePixelCNNNet(nn.Module):
    """A simple PixelCNN style neural network.

    Example:

        >>> simple_pixel_cnn_net = SimplePixelCNNNet(3, 5, None)
        >>> simple_pixel_cnn_net(torch.rand([7, 3, 8, 8])).shape
        torch.Size([7, 5, 8, 8])
        >>> conditioned_net = SimplePixelCNNNet(3, 5, 9)
        >>> input_tensor = torch.rand([7, 3, 8, 8])
        >>> conditioned_tensor = torch.rand([7, 9, 8, 8])
        >>> conditioned_net(input_tensor, False, conditioned_tensor).shape
        torch.Size([7, 5, 8, 8])
    """

    def __init__(self, input_channels, output_channels, num_conditional):
        super().__init__()
        self.conv1 = MaskedConv2d(input_channels, 32, 1, 1)
        if num_conditional is not None:
            self.prj1 = nn.Conv2d(num_conditional, 32, kernel_size=1)
        self.conv2 = MaskedConv2d(32, output_channels, 1, 1)
        self.num_conditional = num_conditional

    def forward(self, x, sample=False, conditional=None):
        if self.num_conditional is None:
            if conditional is not None:
                raise RuntimeError("self.num_conditional is {self.num_conditional}), but passed in conditional is {conditional}")
        else:
            if conditional.shape[1] != self.num_conditional:
                raise RuntimeError("object num_conditional is {self.num_conditional}," +
                    " but conditional passed in {conditional.shape}")
        x = self.conv1(x)
        if conditional is not None:
            prj = self.prj1(conditional)
            x = x + prj
        x = F.relu(x)
        x = self.conv2(x)
        return x


import doctest
doctest.testmod()
