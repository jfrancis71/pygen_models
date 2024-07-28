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


## Discussion

Some discussions around variational autoencoders seems confused as to the purpose. Is it to design a good generative model, or to provide a good latent model of the distribution? These purposes may be in conflict. At heart a variational autoencoder is just a tractable way to train a mixture model. The original purpose was if we have a poor model for p(x) then perhaps a mixture of p(x)'s ie p(x|z) would give rise to a better model. But in this sense the latent code is just patching up the inadequacies of the decoding model. For example if you have a very good decoder p(x|z) there is no need to learn a latent code at all (in fact for the VAE the KL penalty will penalise this).

My conclusions are that a VAE deisgned to be a good generative model may look very different to a VAE designed to produce good latent codes. Perhaps in the latter case alternative approaches that pursue this goal explicitly may be better.
