import torch
from torch import nn
from torch.distributions.categorical import Categorical


class HMM(nn.Module):
    def __init__(self, num_states, observation_model):
        super().__init__()
        # pylint: disable=E1101
        self.prior_states_vector = nn.Parameter(torch.randn([num_states]))  # (state)
        self.state_transitions_matrix = \
            nn.Parameter(torch.randn(num_states, num_states))  # (state, state')
        self.observation_model = observation_model

    def log_prob(self, value):
        sample_shape = value.shape[:-1]
        # Below assumes it is of length 1 or more
        alpha = self.observation_model.emission_logits(value[:, 0]) +\
            self.prior_state_distribution().logits
        for observation_idx in range(1, value.shape[1]):
            # pylint: disable=E1101
            alpha = self.observation_model.emission_logits(value[:, observation_idx]) + \
                torch.logsumexp(
                    torch.transpose(self.state_transition_distribution().logits, 0, 1) + \
                    alpha.unsqueeze(1), dim=2)
        return torch.logsumexp(alpha, dim=1)  # pylint: disable=E1101

    def sample(self, num_steps):
        """Samples observation sequence."""
        return self.sample_variables(num_steps)[1]

    def sample_variables(self, num_steps):
        """Samples state sequence and observation sequence and returns as a tuple."""
        state = self.prior_state_distribution().sample()
        state_sequence = [state]
        observation = self.observation_model.state_emission_distribution(state).sample()
        observation_sequence = [observation]
        for step in range(num_steps):
            state = Categorical(logits=self.state_transitions_matrix[state]).sample()
            observation = self.observation_model.state_emission_distribution(state).sample()
            state_sequence.append(state)
            observation_sequence.append(observation)
        # pylint: disable=E1101
        return torch.tensor(state_sequence)[:-1], torch.tensor(observation_sequence)[:-1]

    def prior_state_distribution(self):
        return Categorical(logits=self.prior_states_vector)

    def state_transition_distribution(self):
        return Categorical(logits=self.state_transitions_matrix)


class MatrixObservationModel(nn.Module):
    def __init__(self, num_states, num_observations):
        super().__init__()
        # pylint: disable=E1101
        self.emission_logits_matrix = \
            nn.Parameter(torch.randn(num_states, num_observations))  # (state, observation)
        self.event_shape = []

    def emission_logits(self, observation):  # (Batch, observation)
        """returns vector of length num_states representing log p(states, observation)"""
        log_prob = self.emission_distribution().logits
        return log_prob[:, observation].transpose(0, 1)

    def emission_distribution(self):
        return Categorical(logits=self.emission_logits_matrix)

    def state_emission_distribution(self, state):
        return Categorical(logits=self.emission_logits_matrix[state])
