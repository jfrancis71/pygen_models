import torch
from torch import nn
from torch.distributions.categorical import Categorical


class HMM(nn.Module):
    """Defines a Hidden Markov Model.
    The observation model should be a layer type object which accepts a one hot state
    and returns a probability distribution over an observation.
    """
    def __init__(self, num_states, observation_model):
        super().__init__()
        # pylint: disable=E1101
        self.prior_states_vector = nn.Parameter(torch.randn([num_states]))  # (state)
        self.state_transitions_matrix = \
            nn.Parameter(torch.randn(num_states, num_states))  # (state, state')
        self.observation_model = observation_model
        self.num_states = num_states

    def sample(self, num_steps):
        """Samples observation sequence."""
        return self.sample_variables(num_steps)[1]

    def sample_variables(self, num_steps):
        """Samples state sequence and observation sequence and returns as a tuple."""
        state = self.prior_state_distribution().sample()
        state_sequence = [state]
        one_hot = nn.functional.one_hot(state, self.num_states).float()
        observation = self.observation_model(one_hot).sample()
        observation_sequence = [observation]
        for _ in range(num_steps-1):
            state = Categorical(logits=self.state_transitions_matrix[state]).sample()
            one_hot = nn.functional.one_hot(state, self.num_states).float()
            observation = self.observation_model(one_hot).sample()
            state_sequence.append(state)
            observation_sequence.append(observation)
        state_sequence = torch.stack(state_sequence, dim=0)
        observation_sequence = torch.stack(observation_sequence, dim=0)
        # pylint: disable=E1101
        return state_sequence, observation_sequence

    def prior_state_distribution(self):
        return Categorical(logits=self.prior_states_vector)

    def state_transition_distribution(self):
        return Categorical(logits=self.state_transitions_matrix)

    def forward(self, z):
        one_hot = nn.functional.one_hot(z, self.num_states).float()
        return self.observation_model(one_hot)

    def device(self):
        return next(self.parameters()).device


class HMMAnalytic(HMM):
    def __init__(self, num_states, observation_model):
        super().__init__(num_states, observation_model)

    def log_prob(self, value):
        # Below assumes it is of length 1 or more
        alpha = self.emission_logits(value[:, 0]) +\
            self.prior_state_distribution().logits
        for observation_idx in range(1, value.shape[1]):
            # pylint: disable=E1101
            alpha = self.emission_logits(value[:, observation_idx]) + \
                torch.logsumexp(
                    torch.transpose(self.state_transition_distribution().logits, 0, 1) +
                    alpha.unsqueeze(1), dim=2)
        return torch.logsumexp(alpha, dim=1)  # pylint: disable=E1101

    def emission_logits(self, observation):  # Batch + event_shape
        """returns vector of length num_states representing log p(observation | state)"""
        one_hot_states = torch.stack([nn.functional.one_hot(torch.tensor(s).to(self.device()), self.num_states).float()
                          for s in range(self.num_states)])
        batched_one_hot_states = one_hot_states.unsqueeze(0).repeat(observation.shape[0], 1, 1)
        emission_probs = torch.stack([self.observation_model(batched_one_hot_states[:, s]).log_prob(observation)
                                      for s in range(self.num_states)]).transpose(0, 1)
        return emission_probs
