# VAE Objective Decompositions

Taking the ELBO objective:

$$
E_{z \sim q_\phi(z)} [log(p_\theta(x|z))] - D_{KL}[q_\phi(z)||p_\theta(z)]
$$

This can be written as:

$$
= E_{z \sim q_\phi(z)} [log(p_\theta(x|z))] - E_{z \sim q_\phi(z)}[log(\frac{q_\phi(z)}{p_\theta(z)})]
$$

$$
= E_{z \sim q_\phi(z)} [log(p_\theta(x|z))] + E_{z \sim q_\phi(z)}[log(\frac{p_\theta(z)}{q_\phi(z)})]
$$

$$
= E_{z \sim q_\phi(z)} [log(\frac{p_\theta(x|z) p_\theta(z)}{q_\phi(z)})]
$$

$$
= E_{z \sim q_\phi(z)} [log(\frac{p_\theta(z|x) p_\theta(x)}{q_\phi(z)})]
$$
