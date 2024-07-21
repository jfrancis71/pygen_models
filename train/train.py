import numpy as np


def distribution_trainer(distribution, batch):
    log_prob_mean = (distribution.log_prob(batch[0])).mean()
    return log_prob_mean, np.array((log_prob_mean.cpu().detach().numpy()), dtype=[('log_prob', 'float32')])

def vae_trainer(distribution, batch):
    log_prob, reconstruct_log_prob, kl_div = distribution.elbo(batch[0])
    log_prob_mean = log_prob.mean()
    reconstruct_log_prob_mean = reconstruct_log_prob.mean()
    kl_div_mean = kl_div.mean()
    return log_prob_mean, np.array(
        (log_prob_mean.cpu().detach().numpy(),
         reconstruct_log_prob_mean.cpu().detach().numpy(),
         kl_div_mean),
        dtype=[('log_prob', 'float32'), ('reconstruct', 'float32'), ('kl_div', 'float32')])
