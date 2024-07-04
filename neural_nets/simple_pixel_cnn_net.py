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


class SimplePixelCNNNetwork(nn.Module):
    """A Simple PixelCNN style neural network."""
    def __init__(self, num_conditional):
        super().__init__()
        self.conv1 = MaskedConv2d(1, 32, 1, 1)
        if num_conditional is not None:
            self.prj1 = nn.Linear(num_conditional, 32*28*28)
        self.conv2 = MaskedConv2d(32, 1, 1, 1)

    def forward(self, x, sample=False, conditional=None):
        x = self.conv1(x)
        if conditional is not None:
            prj = self.prj1(conditional[:,:,0,0]).reshape([-1, 32, 28, 28])
            x = x + prj
        x = F.relu(x)
        x = self.conv2(x)
        return x
