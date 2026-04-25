import torch
import torch.distributions as td


class RIndependentBernoulliDistribution(td.independent.Independent):
    def __init__(self, logits):
        base_dist = td.bernoulli.Bernoulli(logits=logits)
        super().__init__(base_distribution=base_dist, reinterpreted_batch_ndims=1)

    def rsample(self, sample_shape=[]):
        z_like = torch.zeros_like(self.base_dist.logits)
        z_l = torch.stack([z_like, self.base_dist.logits], dim=-1)
        one_hot_z = torch.nn.functional.gumbel_softmax(z_l, hard=True)
        z = one_hot_z[..., -1]
        return z
