import torch.nn as nn
import torch.distributions.one_hot_categorical
import torch.distributions.independent


class RIndependentOneHotCategorical(nn.Module):
    def __init__(self, event_shape, logits):
        super().__init__()
        base_dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
        self.dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=len(event_shape))

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, sample_shape=[]):
        return self.dist.sample(sample_shape)

    def rsample(self) -> torch.Tensor:
        z_logits = self.dist.base_dist.logits
        one_hot_z = nn.functional.gumbel_softmax(z_logits, hard=True)
        return one_hot_z
