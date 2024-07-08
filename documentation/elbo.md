Deriving the ELBO:


$$
Log(p(x)) = E_{x \sim q(z)} [Log(p(x))]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)})]
$$

$$
Log(p(x)) = E_{z \sim q(z)} [Log(\frac{p(x|z) p(z)}{p(z|x)} \frac{q(z)}
{q(z)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] + E_{z \sim q(z)}[Log(\frac{p(z)}{q(z)})]
 + E_{z \sim q(z)} [Log(\frac{q(z)}{p(z|x)})]
$$

$$
E_{z \sim q(z)} [Log(p(x|z))] - D_{KL}[q(z)||p(z)] + D_{KL}[q(z)||p(z|x
)]
$$


First two terms are the ELBO

Reference:
YouTube: Lecture 13 Generative Models, Fei-Fei Li, Stanford University 
School of Engineering, published 17 Aug 2017. Time: 37:31
