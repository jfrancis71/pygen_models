import torch
import torch.nn as nn
import pygen_models.distributions.pixelcnn as pixelcnn_dist


class PixelCNN(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, channel_layer, num_pixelcnn_params):
        super().__init__()
        self.pixelcnn_net = pixelcnn_net
        self.event_shape = event_shape
        self.channel_layer = channel_layer
        self.num_pixelcnn_params = num_pixelcnn_params

    def forward(self, x):
        if x.shape[-3] != self.num_pixelcnn_params:
            raise RuntimeError(
                f"input shape {x.shape}, but event_shape has shape {self.event_shape} with num_pixelcnn_params" +
                    "{self.num_pixelcnn_params}, " +
                "expecting [_,{self.num_pixelcnn_params, self.event_shape[1], self.event_shape[2]}]")
        return pixelcnn_dist.PixelCNN(self.pixelcnn_net, self.event_shape, self.channel_layer, x)


class SpatialExpand(nn.Module):
    """Expands a tensor from Bxin_channels to Bxout_channelsxYxX by projecting and reshaping.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        spatial_dims [int, int]: the Y and X spatial size

    Example:
        >>> input_tensor = torch.zeros([7, 8])
        >>> spatial_expand = SpatialExpand(8, 12, [4, 4])
        >>> spatial_expand(input_tensor).shape
        torch.Size([7, 12, 4, 4])
    """
    def __init__(self, in_channels, out_channels, spatial_dims):
        super().__init__()
        self.spatial_expand = nn.Sequential(
            nn.Linear(in_channels, out_channels*spatial_dims[0]*spatial_dims[1]),
            nn.Unflatten(dim=-1, unflattened_size=[out_channels, spatial_dims[0], spatial_dims[1]]))

    def forward(self, x):
        x = self.spatial_expand(x)
        return x


import doctest
doctest.testmod()
