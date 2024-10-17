import torch
from torch import nn


def channel_last(x: torch.Tensor):
    batch_dims = len(x.shape)-3
    return x.permute(list(range(batch_dims)) + [batch_dims+1, batch_dims+2, batch_dims])


def thread2(net, p1, p2, sample):
    batch_ndims = len(p1.shape[:-3])
    p1_flat = p1.flatten(0, batch_ndims-1)
    if p2 is None:
        p2_flat = None
    else:
        p2_flat = p2.flatten(0, batch_ndims-1)
    out_flat = net(p1_flat, sample, conditional=p2_flat)
    out = out_flat.unflatten(0, p1.shape[:-3])
    return out


class PixelCNN(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, channel_layer, pixelcnn_params):
        super().__init__()
        if len(event_shape) != 3:
            raise RuntimeError(f"event_shape should be list of 3 elements, but given {event_shape}")
        self.event_shape = torch.Size(event_shape)
        self.channel_layer = channel_layer
        if pixelcnn_params is None or len(pixelcnn_params.shape) == 3:
            self.batch_shape = torch.Size([])
        else:
            self.batch_shape = torch.Size(pixelcnn_params.shape[:-3])
        self.pixelcnn_params = pixelcnn_params
        self.output_channels = self.channel_layer.params_size()
        self.pixelcnn_net = pixelcnn_net

    def log_prob(self, value):
        logits = thread2(self.pixelcnn_net, (value*2)-1, self.pixelcnn_params, False)
        layer_logits = channel_last(logits)
        permute_samples = channel_last(value)
        return self.channel_layer(layer_logits).log_prob(permute_samples).sum(axis=[-1, -2])

    def sample(self, sample_shape=None):
        with torch.no_grad():
            sample = torch.zeros(torch.Size(sample_shape) + self.batch_shape + self.event_shape)
            params = None if self.pixelcnn_params is None else self.pixelcnn_params.expand(torch.Size(sample_shape) + self.pixelcnn_params.shape)
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    logits = thread2(self.pixelcnn_net, (sample * 2) - 1, params, True)
                    pixel_sample = self.channel_layer(logits[..., :, y, x]).sample()
                    sample[..., :, y, x] = pixel_sample
        return sample


import doctest
doctest.testmod()
