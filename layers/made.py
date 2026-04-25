import torch.nn as nn
import pygen_models.distributions.made as made


class Made(nn.Module):
    def __init__(self, net, num_vars):
        super().__init__()
        self.net = net
        self.num_vars = num_vars

    def forward(self, made_params):
        return made.MadeBernoulli(self.net, self.num_vars, made_params)
