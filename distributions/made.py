import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli as Bernoulli
import pyro.nn


class MadeBernoulli(nn.Module):
    def __init__(self, num_vars, hidden_layers=None):
        super().__init__()
        self.num_vars = num_vars
        if hidden_layers is None:
            hidden_layers = [num_vars*2, num_vars*2]
        self.arn = pyro.nn.AutoRegressiveNN(self.num_vars, hidden_layers, param_dims=[1], permutation=torch.arange(self.num_vars))

    def log_prob(self, value):
        net_output = self.arn(value.float())
        perm_net_output = net_output[:, self.arn.permutation]
        log_prob = Bernoulli(logits=perm_net_output).log_prob(value.float()).sum(axis=1)
        return log_prob

    def sample(self, sample_shape):
        sample = torch.zeros(sample_shape + [self.num_vars])
        for i in range(self.num_vars):
            net_output = self.arn(sample)
            perm_net_output = net_output[..., self.arn.permutation]
            sample_value = Bernoulli(logits=perm_net_output).sample()
            sample[..., i] = sample_value[..., i]
        return sample.long()


class MadeCategorical(nn.Module):
    """Simple MADE categorical implementation. Note this will only work for
    a two state categorical distribution.
    """
    def __init__(self, num_vars, hidden_layers=None):
        super().__init__()
        self.made_bernoulli = MadeBernoulli(num_vars, hidden_layers)

    def log_prob(self, value):
        return self.made_bernoulli.log_prob(value)

    def sample(self, sample_shape):
        return self.made_bernoulli.sample(sample_shape)
