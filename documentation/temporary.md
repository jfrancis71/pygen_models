$$
log p_\theta(x) = E_{z \sim q(z)} [Log(p(x|z))] - D_{KL}[q(z)||p(z)] + D_{KL}[q(z)||p(z|x)]
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
