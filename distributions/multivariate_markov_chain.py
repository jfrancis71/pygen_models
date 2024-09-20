import torch
import torch.nn as nn
import pyro.nn
import pygen_models.distributions.made as made
import pygen_models.layers.made as made_layer


class BernoulliMarkovChain(nn.Module):
    def __init__(self, num_steps, num_vars):
        super().__init__()
        self.num_steps = num_steps
        net_init = pyro.nn.AutoRegressiveNN(num_vars, [num_vars*4, num_vars*4, num_vars*4], param_dims=[1],
            permutation=torch.arange(num_vars), skip_connections=True)
        self.initial_state_distribution = made.MadeBernoulli(net_init, num_vars, None)

        net_transition = pyro.nn.ConditionalAutoRegressiveNN(num_vars, num_vars, [num_vars*2, num_vars*2], param_dims=[1],
            permutation=torch.arange(num_vars))
        self.transition_distribution = made_layer.Made(net_transition, num_vars)

    def log_prob(self, x):
        log_prob = self.initial_state_distribution.log_prob(x[:,0])
        for t in range(1, self.num_steps):
            log_prob += self.transition_distribution(x[:, t-1]).log_prob(x[:, t])
        return log_prob

    def sample(self, sample_shape=[]):
        state = self.initial_state_distribution.sample(sample_shape)
        state_seq = [state]
        for t in range(1, self.num_steps):
            state = self.transition_distribution(state).sample(sample_shape)
            state_seq.append(state)
        seq = torch.stack(state_seq, dim=0)
        return seq
