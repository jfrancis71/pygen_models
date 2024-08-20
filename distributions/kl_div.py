import torch
from torch.distributions.categorical import Categorical as Categorical
from pygen_models.distributions.markov_chain import MarkovChain as MarkovChain


def kl_div(p, q):
    if isinstance(p, torch.distributions.independent.Independent) and \
        isinstance(q, torch.distributions.independent.Independent):
            kl_div = kl_div_independent_categorical(p, q)
    else:
        if isinstance(p, torch.distributions.independent.Independent) and \
            isinstance(q, MarkovChain):
                kl_div = kl_div_independent_categorical_markov_chain(p, q)
        else:
            raise RuntimeError("KL_DIV Error, unknown distributions")
    return kl_div

def kl_div_independent_categorical(p, q):
    kl_div = torch.sum(p.base_dist.probs * (p.base_dist.logits - q.base_dist.logits), axis=-1)
    return kl_div.sum(axis=-1)

def kl_div_independent_categorical_markov_chain(p, q):
    kl_div = kl_div_categorical(Categorical(logits=p.base_dist.logits[:, 0]), q.initial_state_distribution())
    for t in range(1, q.num_steps):
        for s in range(q.num_states):
            kl_div += torch.exp(p.base_dist.logits[:, t - 1, s]) * kl_div_categorical(Categorical(logits=p.base_dist.logits[:, t]),
                Categorical(logits=q.state_transition_distribution().logits[s]))
    return kl_div

def kl_div_categorical(p, q):
    kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
    return kl_div
