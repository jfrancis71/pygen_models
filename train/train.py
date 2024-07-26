import numpy as np
from torch.distributions.categorical import Categorical


def distribution_objective(distribution, batch):
    log_prob_mean = (distribution.log_prob(batch[0])).mean()
    return log_prob_mean, np.array((log_prob_mean.cpu().detach().numpy()), dtype=[('log_prob', 'float32')])


def vae_objective(log_kl_gap):
    def _fn(distribution, batch):
        log_prob, reconstruct_log_prob, kl_div, q_z_given_x = distribution.elbo(batch[0])
        log_prob_mean = log_prob.mean()
        reconstruct_log_prob_mean = reconstruct_log_prob.mean()
        kl_div_mean = kl_div.mean()
        entropy_q_z_given_x = q_z_given_x.entropy().mean()
        entropy_q_z = Categorical(probs=q_z_given_x.probs.mean(axis=0)).entropy()
        metrics_data = (log_prob_mean.cpu().detach().numpy(), reconstruct_log_prob_mean.cpu().detach().numpy(),
            kl_div_mean, entropy_q_z_given_x, entropy_q_z)
        metrics_dtype = [('log_prob', 'float32'), ('reconstruct', 'float32'), ('kl_div', 'float32'),
            ('entropy_q_z_given_x', 'float32'), ('entropy_q_z', 'float32')]
        if log_kl_gap:
            kl_gap = distribution.kl_gap(batch[0]).mean()
            metrics_data, metrics_dtype = metrics_data + (kl_gap, ), metrics_dtype+[('kl_gap', 'float32')]
        metrics = np.array(metrics_data, metrics_dtype)
        return log_prob_mean, metrics
    return _fn
