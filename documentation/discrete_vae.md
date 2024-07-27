# Variational Approach to Discrete Mixture Model

We wish to optimise: (see [https://github.com/jfrancis71/pygen_models/documentation/elbo.md])

$$
E_{z \sim q(z | x)} [Log(p(x | z))] - D_{KL}[q(z | x)||p(z)]
$$

Strategy A:
We can expand the expectation. This is correct but will be slow for many states.

Strategy B:
We can do uniform sampling over q(z|x) and make an importance sampling correction.

Strategy C:
We can do a reinforce update. This is fast, but noisy.
