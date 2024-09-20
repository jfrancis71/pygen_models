Deriving the ELBO:


$$
Log(p(x)) = E_{z \sim q(z)} [Log(p(x))]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)})]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)} \frac{q(z)}{q(z)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] + E_{z \sim q(z)}[Log(\frac{p(z)}{q(z)})] + E_{z \sim q(z)} [Log(\frac{q(z)}{p(z|x)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] - D_{KL}[q(z)||p(z)] + D_{KL}[q(z)||p(z|x
)]
$$


First two terms are the ELBO


# Appendix


Note the $D_{KL}$ term has an interpretation as the upper bound of the mutual information between X and Z where X and Z are jointly distributed as p(x) and q(z|x).

The derivation proceeds similarly to the upper bound derivation on p(x,y) in the mutual information page.

Define q(x,z) = p(x) q(z|x)

$$
I[X,Z] = E_{x,y \sim p(x) q(z|x)} [log \frac{p(x) q(z|x)}{p(x) q(z)}] = 
$$

The p(x) in numerator and denominator cancel.

$$
I[X,Z] = E_{x,z \sim p(x) q(z|x)} [log \frac{q(z|x) p(z)}{q(z) p(z)}]
$$

$$
I[X,Z] = E_{x,z \sim p(x) q(z|x)} [log \frac{q(z|x)}{p(z)}] + E_{x,z \sim p(x) q(z|x)} [log \frac{p(z)}{q(z)}]
$$

$$
I[X,Z] = E_{x \sim p(x)} D_{KL}[q(z|x)||p(z)] - E_{x \sim p(x)} D_{KL}[q(z)||p(z)]
$$

As the $D_{KL}$ is always positive the mutual information cannot be larger than this 1st $D_{KL}$ term.


# Reference:

YouTube: Lecture 13 Generative Models, Fei-Fei Li, Stanford University 
School of Engineering, published 17 Aug 2017. Time: 37:31
