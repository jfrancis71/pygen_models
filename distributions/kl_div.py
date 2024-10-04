import torch

from pygen_models.distributions.made import MadeBernoulli
from pygen_models.distributions.markov_chain import MarkovChain as MarkovChain
from pygen_models.distributions.r_independent_one_hot_categorical import RIndependentOneHotCategorical
import pygen_models.distributions.made as made
from pygen_models.distributions.multivariate_markov_chain import BernoulliMarkovChain as BernoulliMarkovChain
import pygen_models.distributions.r_independent_bernoulli as r_ind_bern

def kl_div(p, q):
    if isinstance(p, RIndependentOneHotCategorical) and \
        isinstance(q, torch.distributions.independent.Independent):
            kl_div = kl_div_independent_categorical(p, q)
    else:
        if isinstance(p, RIndependentOneHotCategorical) and \
            isinstance(q, made.MadeBernoulli):
                kl_div = kl_div_independent_categorical_made(p, q)
        else:
            if isinstance(p, RIndependentOneHotCategorical) and \
                    isinstance(q, BernoulliMarkovChain):
                kl_div = kl_div_independent_BernoulliMarkovChain(p, q)
            else:
                if isinstance(p, r_ind_bern.RIndependentBernoulliDistribution) and \
                        isinstance(q, MadeBernoulli):
                    kl_div = kl_div_r_ind_bern_made_bernoulli(p, q)
                else:
                    kl_div = kl_ind(p, q)
    return kl_div


def kl_ind(p, q):
    sample_z = p.sample().detach()
    kl_div = p.log_prob(sample_z).detach() - q.log_prob(sample_z)
    return kl_div
#    return sample_z[:,0,0] * 0.0


def kl_div_r_ind_bern_made_bernoulli(p, q):
    sample_z = p.rsample()
    kl_div = p.log_prob(sample_z) - q.log_prob(sample_z)
    return kl_div


def kl_div_independent_categorical(p, q):
    kl_div = torch.sum(p.dist.base_dist.probs * (p.dist.base_dist.logits - q.base_dist.logits), axis=-1)
    return kl_div.sum(axis=-1)

def kl_div_independent_categorical_made(p, q):
    # This generates a gradient for the parameter of the expectation, but does not generate a gradient for the
    # expectation sample itself (due to the sampling step). Does this matter?
    sample_z = p.sample()
#    sample_z_cat = torch.argmax(sample_z, dim=-1)
    kl_div = p.log_prob(sample_z) - q.log_prob(sample_z[...,1])
    return kl_div

def kl_div_independent_BernoulliMarkovChain(p, q):
    sample_z = p.sample().detach()
    #    sample_z_cat = torch.argmax(sample_z, dim=-1)
    kl_div = p.log_prob(sample_z).detach().sum(axis=-1) - q.log_prob(sample_z[...,1])
    return kl_div

def kl_div_categorical(p, q):
    kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
    return kl_div
