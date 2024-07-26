# Importance Sampling


We show:

$$
E_{x\sim p(x)}[f(x)] = E_{x\sim q(x)}[\frac{p(x)}{q(x)} f(x)]
$$

From definition:

$$
E_{x\sim p(x)}[f(x)] = \sum_x p(x) f(x)
$$

$$
= \sum_x p(x) \frac{q(x)}{q(x)} f(x)
$$

$$
= E_{x\sim q(x)}[\frac{p(x)}{q(x)} f(x)]
$$



## References:
p532, equ 11.19, Pattern Recognition and Machine Learning, Christopher Bishop (2006).
