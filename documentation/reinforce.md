# Reinforce

We demonstrate:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]] = E_{x \sim p_\theta(x)}[f(x) \nabla_\theta[Log(p_\theta(x))]]
$$

We start by expanding the expectation:

$$
\nabla_\theta[E_{x \sim p_{\theta}(x)}[f(x)]] = \nabla_\theta[\sum_x p_\theta(x) f(x)]
$$

Swap the sum and derivative:

$$
= \sum_x \nabla_\theta[p_\theta(x) f(x)]
$$

f(x) does not depend on $\theta$ so pull out:

$$
= \sum_x f(x) \nabla_\theta[p_\theta(x)]
$$

Noting that:

$$
\nabla_x[Log(f(x))] = \frac{\nabla_x[f(x)]}{f(x)}
$$

Then we can write as:

$$
= \sum_x f(x) p_\theta(x) \nabla_\theta[Log(p_\theta(x))]
$$

Which can be written as an expectation:

$$
= E_{x \sim p_\theta(x)}[f(x) \nabla_\theta[Log(p_\theta(x))]]
$$

### Reinforce with Baseline

We show:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]] = E_{x \sim p_\theta(x)}[(f(x)-c) \nabla_\theta[Log(p_\theta(x))]]
$$

and this holds true for any expression c (which does not depend on x).

$$
= E_{x \sim p_\theta(x)}[f(x) \nabla_\theta[Log(p_\theta(x))]] + E_{x \sim p_\theta(x)}[c \nabla_\theta[Log(p_\theta(x))]]
$$

Now the 2nd term is zero by EPGL lemma:

$$
= c E_{x \sim p_\theta(x)}[\nabla_\theta[Log(p_\theta(x))]] = 0
$$

Application: computing the $\phi$ gradient on a VAE reconstruction term

$$
\nabla_\phi[E_{z \sim q_\phi(z)}[log p(x|z)]] = E_{z \sim q_\phi(z)}[(log p(x|z)- log p(x)) \nabla_\phi[Log(q_\phi(z))]]
$$

where log p(x) has taken on the c role.

### EPGL Lemma

States:

$$
E_{x \sim p_\theta(x)}[\nabla_\theta[Log(p_\theta(x))]] = 0
$$

Expand:

$$
= \sum_x p_\theta(x) \nabla_\theta[Log(p_\theta(x))]
$$

$$
= \sum_x p_\theta(x) \frac{1}{p_\theta(x)} \nabla_\theta[p_\theta(x)] = \sum_x \nabla_\theta[p_\theta(x)]
$$

$$
= \nabla_\theta \sum_x p_\theta(x) = \nabla_\theta [1] = 0
$$


### Reference:
CS229 Lecture notes, Part XV Policy Gradient (REINFORCE), Tengyu Ma, Stanford University.
(Note I have replaced integration with summation in my interpretation above)

OpenAI Spinning Up Notes: [https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#expected-grad-log-prob-lemma]
