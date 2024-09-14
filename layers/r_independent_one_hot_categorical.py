import torch.nn as nn
from pygen_models.distributions.r_independent_one_hot_categorical import RIndependentOneHotCategorical


class RIndependentOneHotCategoricalLayer(nn.Module):
    def __init__(self, event_shape, num_classes):
        super().__init__()
        self.event_shape = event_shape
        self.num_classes = num_classes

    def forward(self, logits):
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [self.num_classes])
        return RIndependentOneHotCategorical(event_shape=self.event_shape, logits=reshape_logits)
