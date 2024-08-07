# VAE Objective Decompositions

Taking the ELBO objective:

$$
E_{z \sim q_\phi(z)} [log(p_\theta(x|z))] - D_{KL}[q_\phi(z)||p_\theta(z)]
$$

We show this is equal to (assuming randomly distributed dataset according to p(x)):

$$
= -H(x) -D_{KL}[p(x)||p_\theta(x)] - E_{x \sim p(x)} [D_{KL}[q_\phi(z)||p_\theta(z|x)]]
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

$$
= E_{z \sim q_\phi(z)} [log(\frac{p_\theta(z|x) p_\theta(x)}{q_\phi(z)} \frac{p(x)}{p(x)})]
$$

$$
= E_{z \sim q_\phi(z)} [log(\frac{p_\theta(x)}{p(x)})] + E_{z \sim q_\phi(z)} [log(\frac{p_\theta(z|x)}{q_\phi(z)})] + E_{z \sim q_\phi(z)} [log(p(x))]
$$

$$
= log(p(x)) + log(\frac{p_\theta(x)}{p(x)}) - D_{KL}[q_\phi(z)||p_\theta(z|x)]
$$

Now this objective is for one data point randomly selected according to $p(x)$ so the objective for the dataset is:

$$
E_{x \sim p(x)} [log(p(x)) + log(\frac{p_\theta(x)}{p(x)}) - D_{KL}[q_\phi(z)||p_\theta(z|x)]]
$$

$$
= -H(x) -D_{KL}[p(x)||p_\theta(x)] - E_{x \sim p(x)} [D_{KL}[q_\phi(z)||p_\theta(z|x)]]
$$

## Discussion

If our objective is a good generative model then presumably we would want to minimize $D_{KL}[p(x)||p_\theta(x)]$, but our above result suggests our VAE objective may trade this off with minimizing $D_{KL}[q_\phi(z)||p_\theta(z|x)]$. So a poor q(z|x) may lead to a suboptimal generative model.

## References
Failure Modes of Variational Autoencoders and Their Effects on Downstream Tasks, Yacoby, Pan, Doshi-Velez (2022), [https://arxiv.org/pdf/2007.07124]
