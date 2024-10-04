import torch


class RIndependentBernoulliDistribution():
    def __init__(self, logits):
        self.logits = logits

    def log_prob(self, x):
        base_dist = torch.distributions.bernoulli.Bernoulli(logits=self.logits)
        dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return dist.log_prob(x)

    def sample(self, sample_shape):
        base_dist = torch.distributions.bernoulli.Bernoulli(logits=self.logits)
        dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return dist.sample()

    def rsample(self, sample_shape):
        z_like = torch.zeros_like(self.logits)
        z_l = torch.stack([z_like, self.logits], dim=-1)
        one_hot_z = torch.nn.functional.gumbel_softmax(z_l, hard=True)
        z = one_hot_z[..., -1]
        return z

    @property
    def batch_shape(self):
        base_dist = torch.distributions.bernoulli.Bernoulli(logits=self.logits)
        dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return dist.batch_shape

    @property
    def event_shape(self):
        base_dist = torch.distributions.bernoulli.Bernoulli(logits=self.logits)
        dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return dist.event_shape
