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

We demonstrate:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)-f(y)]] = \nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]]
$$

Both the derivative and expectation are linear so:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)-f(y)]] = \nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]] - \nabla_\theta[E_{x \sim p_\theta(x)}[f(y)]]
$$

Expectation over a constant is a constant, and derivative of constant is zero:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(y)]] = \nabla_\theta[f(y)] = 0
$$

Hence it is shown.

### Reference:
CS229 Lecture notes, Part XV Policy Gradient (REINFORCE), Tengyu Ma, Stanford University.
(Note I have replaced integration with summation in my interpretation above)
