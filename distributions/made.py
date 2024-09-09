import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli as Bernoulli


class MadeBernoulli(nn.Module):
    def __init__(self, net, num_vars, made_params=None):
        super().__init__()
        self.num_vars = num_vars
        self.net = net
        self.made_params = made_params

    def log_prob(self, value):
        if self.made_params is None:
            net_output = self.net(value.float())
        else:
            net_output = self.net(value.float(), self.made_params)
        perm_net_output = net_output[:, self.net.permutation]
        log_prob = Bernoulli(logits=perm_net_output).log_prob(value.float()).sum(axis=1)
        return log_prob

    def sample(self, sample_shape):
        if self.made_params is None:
            batch_shape = []
        else:
            batch_shape = [self.made_params.shape[0]]
        sample = torch.zeros(sample_shape + batch_shape + [self.num_vars])
        for i in range(self.num_vars):
            if self.made_params is None:
                net_output = self.net(sample)
            else:
                net_output = self.net(sample, self.made_params)
            perm_net_output = net_output[..., self.net.permutation]
            sample_value = Bernoulli(logits=perm_net_output).sample()
            sample[..., i] = sample_value[..., i]
        return sample.long()
