# Reinforce Using Gradient Information


We would like to compute: ( where f(h) is a function mapping a one hot vector to a real number ).
x is a categorical random variable.

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(onehot(x))]]
$$

p(h) is a softmax over a gumbel distribution
H(h) "hardens" the softmax, ie sets the largest component to 1, and the other components to 0.
$\epsilon_i$ is a draw from the Gumbel (0,1) distribution.

$$
h(\epsilon, \theta) = softmax(\epsilon_i + \pi_i(\theta))
$$

We add f(h) as the control variate:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(onehot(x))]] = \nabla_\theta[E_{h \sim p_\theta(h)}[f(H(h)) - f(h)]] + \nabla_\theta[E_{h \sim p_\theta(h)}[f(h)]]
$$

We use reinforce for the first expectation:

$$
\nabla_\theta[E_{h \sim p_\theta(h)}[f(H(h)) - f(h)]] = E_{h \sim p_\theta(h)} [ (f(H(h)) - f(h)) \nabla_\theta [log(p_\theta(h))]]
$$

And path derivative for the second expectation:

$$
\nabla_\theta[E_{h \sim p_\theta(h)}[f(h)]] = E_{\epsilon \sim g(0,1)}[\nabla_\theta[f(h(\epsilon, \theta))]]
$$

So:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(onehot(x))]] = E_{h \sim p_\theta(h)} [ (f(H(h)) - f(h)) \nabla_\theta [log(p_\theta(h))]] + E_{\epsilon \sim g(0,1)}[\nabla_\theta[f(h(\epsilon, \theta))]]
$$

## References:

CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX, Eric Jang, Shixiang Gu, Ben Poole, ICLR 2017. [https://arxiv.org/pdf/1611.01144]

REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models, Tucker, Minh, Maddison, Lawson, Sohl-Dickstein, 2017 [https://arxiv.org/abs/1703.07370]

Variance Reduction, Art Owen, Section 8.9 Control Variates, [https://artowen.su.domains/mc/Ch-var-basic.pdf]
