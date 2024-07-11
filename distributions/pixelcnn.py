import torch
from torch import nn
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen.layers.independent_quantized_distribution as ql


class _PixelCNN(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, layer, params):
        super().__init__()
        self.event_shape = event_shape
        self.layer = layer
        if params is None or len(params.shape)==3:
            self.batch_shape = []
        else:
            self.batch_shape = params.shape[0]
        self.params = params
        self.pixelcnn_net = pixelcnn_net

    def log_prob(self, samples):
        # pylint: disable=E1101
        if samples.size()[1:4] != torch.Size(self.event_shape):
            raise RuntimeError(f"sample shape {samples.shape[1:4]}, but event_shape has shape {self.event_shape}")
        params = self.params
        if params is not None:
            if len(params.shape) == 3:
                params = params.unsqueeze(0).repeat(samples.shape[0], 1, 1, 1)
        logits = self.pixelcnn_net((samples*2.0)-1.0, conditional=params)
        layer_logits = logits.permute(0, 2, 3, 1)  # B, Y, X, P where P for parameters
        permute_samples = samples.permute(0, 2, 3, 1)  # B, Y, X, C
        return self.layer(layer_logits).log_prob(permute_samples).sum(axis=[1, 2])

    def sample(self, sample_shape=None):
        if sample_shape is None:
            batch_shape = [1]
        else:
            batch_shape = sample_shape
        with torch.no_grad():
            params = self.params
            if params is not None:
                if len(params.shape) == 3:
                    params = params.unsqueeze(0).repeat(batch_shape[0], 1, 1, 1)
            # pylint: disable=E1101
            sample = torch.zeros(batch_shape+self.event_shape,
                device=next(self.parameters()).device)
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    logits = self.pixelcnn_net((sample*2)-1, sample=True,
                        conditional=params)[:, :, y, x]
                    pixel_sample = self.layer(logits).sample()
                    sample[:, :, y, x] = pixel_sample
        if sample_shape is None:
            return sample[0]
        return sample


class PixelCNNBernoulliDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1])
        super().__init__(
            pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                input_channels=event_shape[0],
                nr_params=base_layer.params_size(), nr_conditional=None),
            event_shape,
            base_layer,
            None)


class PixelCNNQuantizedDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        base_layer = ql.IndependentQuantizedDistribution(event_shape=event_shape[:1])
        super().__init__(
            pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                input_channels=event_shape[0],
                nr_params=base_layer.params_size(), nr_conditional=None),
            event_shape,
            base_layer,
            None)
