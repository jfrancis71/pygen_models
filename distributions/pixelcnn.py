import torch
from torch import nn


def channel_last(x: torch.Tensor):
    batch_dims = len(x.shape)-3
    return x.permute(list(range(batch_dims)) + [batch_dims+1, batch_dims+2, batch_dims])


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
        sample_shape = value.shape[:-len(self.batch_shape+self.event_shape)]
        nn_values = value.flatten(0, len(sample_shape+self.batch_shape)-1)
        nn_params = None
        if self.pixelcnn_params is not None:
            nn_params = self.pixelcnn_params.flatten(0, len(sample_shape+self.batch_shape)-1)
            if len(nn_params.shape) == 3:
                nn_params = nn_params.unsqueeze(0).repeat(sample_shape+torch.Size([1]*len(nn_params.shape)))
        logits = self.pixelcnn_net((nn_values*2.0)-1.0, conditional=nn_params)
        reshape_logits = logits.reshape(sample_shape + self.batch_shape + torch.Size([self.output_channels, self.event_shape[1], self.event_shape[2]]))
        layer_logits = channel_last(reshape_logits)
        permute_samples = channel_last(value)
        return self.channel_layer(layer_logits).log_prob(permute_samples).sum(axis=[-1, -2])

    def sample(self, sample_shape=None):
        with torch.no_grad():
            params = self.pixelcnn_params
            if params is not None:
                if len(params.shape) == 3:
                    if sample_shape is None:
                        params = params.unsqueeze(0)
                    else:
                        params = params.unsqueeze(0).repeat(sample_shape[0], 1, 1, 1)
                elif len(params.shape) == 4:
                    if sample_shape is None or sample_shape == torch.Size([]):
                        params = params
                    else:
                        params = params.repeat(sample_shape[0], 1, 1, 1)
            # pylint: disable=E1101
            if params is None:
                if sample_shape is None:
                    net_batch_shape = [1]
                else:
                    net_batch_shape = torch.Size(sample_shape)
            else:
                net_batch_shape = torch.Size([params.shape[0]])
            sample = torch.zeros(net_batch_shape+self.event_shape)
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    logits = self.pixelcnn_net((sample*2)-1, sample=True,
                        conditional=params)[:, :, y, x]
                    pixel_sample = self.channel_layer(logits).sample()
                    sample[:, :, y, x] = pixel_sample
        if sample_shape is None and (params is None or len(self.pixelcnn_params.shape)==3):
            return sample[0]
        if sample_shape is not None and params is not None:
            return torch.reshape(sample, torch.Size(sample_shape)+torch.Size([self.pixelcnn_params.shape[0]])+self.event_shape)
        return sample


import doctest
doctest.testmod()
