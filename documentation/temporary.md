We are defining:

$$
p_\theta(x,z) = p_\theta(z) p_\theta(x|z)
$$

$$
q_\phi(x,z) = p(x) q_\phi(z|x)
$$


$$
log p_\theta(x) = E_{z \sim q(z|x)} [Log(p(x|z))] - D_{KL}[q(z|x)||p(z)] + D_{KL}[q(z|x)||p(z|x)]
$$

All under the expectation p(x) so:

$$
E_{x \sim p(x)}[log p_\theta(x)] = E_{x \sim p(x)}[E_{z \sim q(z)} [Log(p(x|z))]] - E_{x \sim p(x)}[D_{KL}[q(z)||p(z)]] + E_{x \sim p(x)}[D_{KL}[q(z)||p(z|x)]]
$$

For the LHS:

$$
E_{x \sim p(x)}[log p_\theta(x)] = E_{x \sim p(x)}[log p_\theta(x) \frac{p(x)}{p(x)}]
$$

$$
= E_{x \sim p(x)}[log p_\theta(x) p(x)] + E_{x \sim p(x)}[log \frac{1}{p(x)}]
$$

$$
= -E_{x \sim p(x)}[log \frac{p(x)}{p_\theta(x)}] + E_{x \sim p(x)}[log \frac{1}{p(x)}]
$$

$$
= H[X] - D_{KL}[p(x)||p_\theta(x)]
$$

For the RHS:

$$
E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z))]] - E_{x \sim p(x)}[D_{KL}[q(z|x)||p(z)]] + E_{x \sim p(x)}[D_{KL}[q(z|x)||p(z|x)]]
$$

Note the 1st term:

$$
E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z))]] = E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) \frac{q_\phi(x|z)}{q_\phi(x|z)})]]
$$

$$
 = E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) \frac{q_\phi(x|z) q(z)}{q(x) q(z|x)})]]
$$


CHECK...

$$
= H[X] - H_\phi[X|Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) \frac{q(z)}{q_\phi(z|x)})]]
$$

Combining into mutual information and adding in 2nd terms:

$$
= H[X] - H_\phi[X|Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) \frac{q(z)}{q_\phi(z|x)}) \frac{p(z)}{q(z|x)}]]
$$

Cancelling the q's

$$
= I_\phi[X,Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) q_\phi(x|z) q(z) \frac{p(z)}{p(z|x)})]]
$$

Applying Bayes Theorem to p(z|x)

$$
= I_\phi[X,Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x|z) q_\phi(x|z) q(z) \frac{p(z) p(x)}{p(z) p(x|z)})]]
$$

$$
= I_\phi[X,Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(q_\phi(x|z) q(z) p(x))]]
$$

$$
= I_\phi[X,Z] + H[X,Z] + E_{x \sim p(x)}[E_{z \sim q(z|x)} [Log(p(x))]]
$$

$$
= I_\phi[X,Z] + H[X,Z] + E_{x \sim p(x)}[[Log(p(x))]
$$
