import torch
import torch.nn as nn
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen.layers.independent_quantized_distribution as ql


class _PixelCNNDistribution(nn.Module):
    def __init__(self, pixelcnn_net, event_shape, base_layer, num_conditional):
        super().__init__()
        self.event_shape = event_shape
        self.num_conditional = num_conditional
        self.base_layer = base_layer
        self.pixelcnn_net = pixelcnn_net

    def forward(self, x):
        if x.shape[1:4] != torch.Size([self.num_conditional, self.event_shape[1], self.event_shape[2]]):
            raise RuntimeError("input shape {}, but event_shape has shape {} with num_conditional {}, expecting {}".
                format(x.shape, self.event_shape, self.num_conditional, ["_", self.num_conditional, self.event_shape[1], self.event_shape[2]]))
        pixelcnn = pixelcnn_dist._PixelCNN(
            self.pixelcnn_net,
            self.event_shape,
            self.base_layer,
            x)
        return pixelcnn


class PixelCNNBernoulliDistribution(_PixelCNNDistribution):
    def __init__(self, event_shape, num_conditional, nr_resnet=3):
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1])
        pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                            input_channels=event_shape[0], nr_params=self.base_layer.params_size(), nr_conditional=num_conditional)
        super().__init__(pixelcnn_net, event_shape, base_layer, num_conditional)


class PixelCNNQuantizedDistribution(_PixelCNNDistribution):
    def __init__(self, event_shape, num_conditional, nr_resnet=3):
        base_layer = ql.IndependentQuantizedDistribution(event_shape=event_shape[:1])
        pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                            input_channels=event_shape[0], nr_params=self.base_layer.params_size(), nr_conditional=num_conditional)
        super().__init__(pixelcnn_net, event_shape, base_layer, num_conditional)
