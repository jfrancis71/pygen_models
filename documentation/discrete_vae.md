# Variational Approach to Discrete Mixture Model

We wish to optimise: (see [https://github.com/jfrancis71/pygen_models/blob/documentation/documentation/elbo.md])

$$
E_{z \sim q_\phi(z | x)} [Log(p_\theta(x | z))] - D_{KL}[q_\phi(z | x)||p_\theta(z)]
$$

## Analytic
We can expand the expectation. This is correct but will be slow for many states.

$$
E_{z \sim q_\phi(z | x)} [Log(p_\theta(x | z))]  = \sum_z q_\phi(z|x) Log(p_\theta(x|z)
$$


## Uniform Sampling
We can do uniform sampling over q(z|x) and make an importance sampling correction.

$$
E_{z \sim q_\phi(z | x)} [Log(p_\theta(x | z))]  = E_{z \sim u(z)} [\frac{q_\phi(z|x)}{u(z)} Log(p_\theta(x | z)]
$$


## Reinforce
We can do a reinforce update. This is fast, but noisy.

$$
E_{z \sim q_\phi(z | x)} [Log(p_\theta(x | z))]  = E_{z \sim q_\phi(z|x)} [Log(p_\theta(x | z) \nabla_\phi Log(q_\phi(z|x))]
$$

see [https://github.com/jfrancis71/pygen_models/blob/documentation/documentation/reinforce.md]

## Reinforce with Baseline

$$
E_{z \sim q_\phi(z | x)} [Log(p_\theta(x | z))]  = E_{z \sim q_\phi(z|x)} [(Log(p_\theta(x | z)-Log(p_\eta(x))) \nabla_\phi Log(q_\phi(z|x))]
$$

where $Log(p_\eta(x))$ is some baseline distribution over p(x).



Gumbel Softmax:

