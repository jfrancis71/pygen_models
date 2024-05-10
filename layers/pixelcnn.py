import torch.nn as nn
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen.layers.independent_quantized_distribution as ql


class _PixelCNNDistribution(nn.Module):
    def __init__(self, event_shape, layer, num_conditional, nr_resnet=3):
        super().__init__()
        self.event_shape = event_shape
        self.layer = layer
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
                            input_channels=event_shape[0], nr_params=self.layer.params_size(), nr_conditional=num_conditional)

    def forward(self, x):
        pixelcnn = pixelcnn_dist._PixelCNN(
            self.event_shape,
            self.layer,
            x)
        pixelcnn.pixelcnn_net = self.pixelcnn_net
        return pixelcnn


class PixelCNNBernoulliDistribution(_PixelCNNDistribution):
    def __init__(self, event_shape, num_conditional, nr_resnet=3):
        super().__init__(event_shape, bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1]), num_conditional, nr_resnet=3)


class PixelCNNQuantizedDistribution(_PixelCNNDistribution):
    def __init__(self, event_shape, num_conditional, nr_resnet=3):
        super().__init__(event_shape, ql.IndependentQuantizedDistribution(event_shape=event_shape[:1]), num_conditional, nr_resnet=3)
