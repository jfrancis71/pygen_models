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
    # This generates a gradient for the parameter of the expectation, but does not generate a gradient for the
    # expectation sample itself (due to the sampling step). Does this matter?
    sample_z = p.sample()
    kl_div = p.log_prob(sample_z).exp() * (p.log_prob(sample_z) - q.log_prob(sample_z))
    return kl_div

def kl_div_categorical(p, q):
    kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
    return kl_div
