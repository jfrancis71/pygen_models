import torch
import torch.nn as nn
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen.layers.independent_quantized_distribution as ql
from pygen_models.neural_nets import simple_pixelcnn_net


class PixelCNN(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, base_layer, num_conditional):
        super().__init__()
        self.pixelcnn_net = pixelcnn_net
        self.event_shape = event_shape
        self.base_layer = base_layer
        self.num_conditional = num_conditional

    def forward(self, x):
        batch_shape = x.shape[:-1]
        if x.shape[-3] != self.num_conditional:
            raise RuntimeError(
                f"input shape {x.shape}, but event_shape has shape {self.event_shape} with num_conditional" +
                    "{self.num_conditional}, " +
                "expecting [_,{self.num_conditional, self.event_shape[1], self.event_shape[2]}]")
        return pixelcnn_dist.PixelCNN(self.pixelcnn_net, self.event_shape, self.base_layer, x)


def make_simple_pixelcnn_net():
    def _fn(input_channels, output_channels, num_conditional):
        return simple_pixelcnn_net.SimplePixelCNNNet(input_channels, output_channels, num_conditional)
    return _fn


def make_pixelcnn_net(num_resnet):
    def _fn(input_channels, output_channels, num_conditional):
        return pixelcnn_model.PixelCNN(nr_resnet=num_resnet, nr_filters=160,
            input_channels=input_channels, nr_params=output_channels, nr_conditional=num_conditional)
    return _fn


def make_pixelcnn_layer(base_distribution, pixelcnn_net, event_shape, num_conditional):
    """makes a PixelCNN distribution

    >>> layer = make_pixelcnn_layer(pixelcnn_dist.make_bernoulli_base_distribution(), make_simple_pixelcnn_net(), [2, 12, 12], 10)
    >>> input_tensor = torch.rand([16, 10, 12, 12])
    >>> layer(input_tensor).sample().shape
    torch.Size([16, 2, 12, 12])
    """
    base_layer = base_distribution(event_shape[:1])
    return PixelCNN(pixelcnn_net(event_shape[0], base_layer.params_size(), num_conditional), event_shape,
        base_layer, num_conditional)


class SpatialExpand(nn.Module):
    """Expands a tensor from BxC to BxCxYxX.

    This is a learnable operation which allows for learning both channel and spatial mapping.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        spatial_dims [int,int]: the Y and X spatial size

    Example:
        >>> input_tensor = torch.zeros([7, 8])
        >>> spatial_expand = SpatialExpand(8, 12, [4,4])
        >>> spatial_expand(input_tensor).shape
        torch.Size([7, 12, 4, 4])
    """
    def __init__(self, in_channels, out_channels, spatial_dims):
        super().__init__()
        self.expand = nn.Linear(in_channels, out_channels*spatial_dims[0]*spatial_dims[1])
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = self.expand(x).reshape(batch_shape + torch.Size([self.out_channels, self.spatial_dims[0], self.spatial_dims[1]]))
        return x


import doctest
doctest.testmod()
