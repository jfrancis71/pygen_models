import torch
import torch.nn as nn
import pygen_models.distributions.kl_div as kl_div_mod


class DiscreteVAE(nn.Module):
    """Discrete VAE implements a multivariate discrete VAE.

    Args:
        q_z_given_x: a layer object, it accepts a tensor x and returns a probability distribution over latent variables.
                     The returned distribution is an independent categorical distribution.
        latent_model: a latent model, ie supporting p_z and p_x_given_z. p_x_given_z is a layer which accepts a tensor
                      of one hot sampled z and returns a probability distribution over x.

    Notes:
        p_x_given_z takes a one hot input as the output distribution must be differentiable wrt the one hot input.
    """
    def __init__(self, latent_model, q_z_given_x, beta=1.0):
        super().__init__()
        self.latent_model = latent_model
        self.q_z_given_x = q_z_given_x
        self.beta = beta

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        q_z_given_x_dist = self.q_z_given_x(x)
        log_prob_x_given_z = self.reconstruct_log_prob(q_z_given_x_dist, x)
        kl_div = kl_div_mod.kl_div(q_z_given_x_dist, self.latent_model.p_z())
        beta = self.beta
        return log_prob_x_given_z - beta*kl_div + beta*kl_div.detach() - kl_div.detach(), log_prob_x_given_z.detach(), kl_div.detach(), q_z_given_x_dist

    def sample(self, sample_shape=[]):
        z = self.latent_model.p_z().sample(sample_shape)
        decode = self.latent_model.p_x_given_z(z)
        return decode.sample()

    def reconstruct_log_prob(self, q_z_given_x_dist, x):
        z = q_z_given_x_dist.rsample()
        reconstruct_log_probs = self.latent_model.p_x_given_z(z).log_prob(x)
        return reconstruct_log_probs
