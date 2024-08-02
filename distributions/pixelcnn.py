import torch
from torch import nn
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen.layers.independent_quantized_distribution as ql
from pygen_models.neural_nets import simple_pixelcnn_net


def channel_last(x: torch.Tensor):
    batch_dims = len(x.shape)-3
    return x.permute(list(range(batch_dims)) + [batch_dims+2, batch_dims+1, batch_dims])

class PixelCNN(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, layer, params):
        super().__init__()
        if len(event_shape) != 3:
            raise RuntimeError(f"event_shape should be list of 3 elements, but given {event_shape}")
        self.event_shape = torch.Size(event_shape)
        self.layer = layer
        if params is None or len(params.shape) == 3:
            self.batch_shape = torch.Size([])
        else:
            self.batch_shape = torch.Size(params.shape[:-3])
        self.params = params
        self.pixelcnn_net = pixelcnn_net

    def log_prob(self, value):
        sample_shape = value.shape[:-len(self.batch_shape+self.event_shape)]
        nn_values = value.flatten(0, len(sample_shape+self.batch_shape)-1)
        nn_params = None
        if self.params is not None:
            nn_params = self.params.flatten(0, len(sample_shape+self.batch_shape)-1)
            if len(nn_params.shape) == 3:
                nn_params = nn_params.unsqueeze(0).repeat(sample_shape+torch.Size([1]*len(nn_params.shape)))
        logits = self.pixelcnn_net((nn_values*2.0)-1.0, conditional=nn_params)
        reshape_logits = logits.reshape(sample_shape + self.batch_shape + self.event_shape)
        layer_logits = channel_last(reshape_logits)
        permute_samples = channel_last(value)
        return self.layer(layer_logits).log_prob(permute_samples).sum(axis=[-1, -2])

    def sample(self, sample_shape=None):
        with torch.no_grad():
            params = self.params
            if params is not None:
                if len(params.shape) == 3:
                    if sample_shape is None:
                        params = params.unsqueeze(0)
                    else:
                        params = params.unsqueeze(0).repeat(sample_shape[0], 1, 1, 1)
                elif len(params.shape) == 4:
                    if sample_shape is None:
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
                    pixel_sample = self.layer(logits).sample()
                    sample[:, :, y, x] = pixel_sample
        if sample_shape is None and (params is None or len(self.params.shape)==3):
            return sample[0]
        if sample_shape is not None and params is not None:
            return torch.reshape(sample, torch.Size(sample_shape)+torch.Size([self.params.shape[0]])+self.event_shape)
        return sample


def make_bernoulli_base_distribution():
    return lambda event_shape: bernoulli_layer.IndependentBernoulli(event_shape=event_shape)


def make_quantized_base_distribution():
    return lambda event_shape: ql.IndependentQuantizedDistribution(event_shape=event_shape)


def make_pixelcnn_net(num_resnet=3):
    def _fn(input_channels, output_channels):
        return pixelcnn_model.PixelCNN(nr_resnet=num_resnet, nr_filters=160,
            input_channels=input_channels, nr_params=output_channels, nr_conditional=None)
    return _fn


def make_simple_pixelcnn_net():
    def _fn(input_channels, output_channels):
        return simple_pixelcnn_net.SimplePixelCNNNet(input_channels=input_channels, output_channels=output_channels,
            num_conditional=None)
    return _fn


def make_pixelcnn(base_distribution, pixelcnn_net, event_shape):
    """makes a PixelCNN distribution

    >>> dist = make_pixelcnn(make_bernoulli_base_distribution(), make_simple_pixelcnn_net(), event_shape=[2, 12, 12])
    >>> dist.sample().shape
    torch.Size([2, 12, 12])
    """
    base_layer = base_distribution(event_shape[:1])
    return PixelCNN(pixelcnn_net(input_channels=event_shape[0], output_channels=base_layer.params_size()), event_shape,
        base_layer, None)


import doctest
doctest.testmod()
