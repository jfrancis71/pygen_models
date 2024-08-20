import torch
from torch import nn
from torch.distributions.categorical import Categorical
import pygen_models.distributions.markov_chain as markov_chain


class HMM(nn.Module):
    """Defines a Hidden Markov Model.

    The observation model should be a layer type object which accepts a one hot state
    and returns a probability distribution over an observation.

    Args:
        num_steps (int): length of sequence
        num_states (int): number of hidden states
        observation_model: nn.Module with forward method (accepting one hot tensor) returning a probability distribution
    """
    def __init__(self, num_steps, num_states, observation_model):
        super().__init__()
        # pylint: disable=E1101
        self.markov_chain = markov_chain.MarkovChain(num_steps, num_states)
        self.observation_model = observation_model
        self.num_states = num_states
        self.num_steps = num_steps

    def sample(self):
        """Samples observation sequence."""
        return self.sample_variables()[1]

    def sample_variables(self):
        state_sequence = self.markov_chain.sample()
        one_hot_state_sequence = nn.functional.one_hot(state_sequence, self.num_states).float()
        observation_seq = self.observation_model(one_hot_state_sequence).sample()
        return state_sequence, observation_seq

    def forward(self, z):
        return self.observation_model(z)
