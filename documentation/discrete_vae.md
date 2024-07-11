# Variational Approach to Discrete Mixture Model

We wish to optimise: (see [https://github.com/jfrancis71/pygen_models/blob/documentation/documentation/elbo.md])

$$
E_{z \sim q(z | x)} [Log(p(x | z))] - D_{KL}[q(z | x)||p(z)]
$$

Analytic
We can expand the expectation. This is correct but will be slow for many states.

Uniform Sampling:
We can do uniform sampling over q(z|x) and make an importance sampling correction.

Reinforce:
We can do a reinforce update. This is fast, but noisy.

Reinforce with Baseline:

Gumbel Softmax:



### Multivariate Discrete VAE

Simple model where q factorises over N latent variables, ie:

$$
p(z) = \prod_{i=1}^N p(z_i)
$$

$$
q(z|x) = \prod_{i=1}^N q(z_i|x)
$$

For the KL divergence:
Taking the general multivariate DKL formula

$$
D_{KL}[p(X_{1..n} || q(X_{1..n})] = \sum_{i=1}^n E_{X_{1..i-1} \sim p(X_{1..i-1})} [D_{KL}[p(X_i|X_{1..i-1}||q(X_i|X_{1..i-1})] ]
$$

For a VAE:

$$
D_{KL}[q(Z_{1..n}|X) || p(Z_{1..n})] = \sum_{i=1}^n E_{Z_{1..i-1} \sim q(Z_{1..i-1},X)} [D_{KL}[q(Z_i|Z_{1..i-1},X)||P(Z_i|Z_{1..i-1})] ]
$$

Assuming independence:

$$
D_{KL}[q(Z_{1..n}|X) || p(Z_{1..n})] = \sum_{i=1}^n E_{Z_{1..i-1} \sim q(Z_{1..i-1},X)} [D_{KL}[q(Z_i|,X)||P(Z_i)] ]
$$

$$
D_{KL}[q(Z_{1..n}|X) || p(Z_{1..n})] = \sum_{i=1}^n [D_{KL}[q(Z_i|,X)||P(Z_i)] ]
$$
