import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical


class MarkovChain(nn.Module):
    def __init__(self, num_steps, num_states):
        super().__init__()
        self.initial_state_vector = nn.Parameter(torch.randn([num_states]))
        self.state_transition_matrix = \
            nn.Parameter(torch.randn(num_states, num_states))  # (state, state')
        self.num_steps = num_steps
        self.num_states = num_states

    def log_prob(self, x):
        log_prob = torch.distributions.categorical.Categorical(logits=self.initial_state_vector).log_prob(x[:,0])
        for t in range(1, self.num_steps):
            log_prob += torch.distributions.categorical.Categorical(logits=self.state_transition_matrix[x[:,t-1]]).log_prob(x[:, t])
        return log_prob

    def sample(self, sample_shape=[]):
        state = self.initial_state_distribution().sample(sample_shape)
        state_seq = [state]
        for t in range(1, self.num_steps):
            state = Categorical(logits=self.state_transition_matrix[state]).sample()
            state_seq.append(state)
        seq = torch.stack(state_seq, dim=0)
        return seq

    def initial_state_distribution(self):
        return Categorical(logits=self.initial_state_vector)

    def state_transition_distribution(self):
        return Categorical(logits=self.state_transition_matrix)
