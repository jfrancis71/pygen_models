# Maximising Mututal Information For Learning

$$
I(X,Y) \equiv E_{(x,y) \sim p(x,y)}[log(\frac{p(x,y)}{p(x) p(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) p(x)}{p(x) p(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{p(y)})] (1)
$$

Also note (see p.139 Ref 2)

I[X,Y] = H[X] - H[X|Y]

and from p.138

$$
H[X|Y] = \sum_{x,y} p(x,y) log(\frac{1}{p(x|y)})
$$

## Upper Bound:

$$
I[X, Y] = E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) q(y)}{p(y) q(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] + E_{(x,y) \sim p(x,y)}[\frac{q(y)}{p(y)}]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] + E_{y \sim p(y)}[\frac{q(y)}{p(y)}]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] - D_{KL}[q(y) \parallel p(y)]
$$

$$
I[X, Y]  \le E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})]
$$

## Lower Bound:

$$
I[X, Y] = E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) q(y|x)}{p(y) q(y|x)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y|x)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{x \sim p(x)}[ E_{y \sim p(y|x)} [ log(\frac{p(y|x)}{q(y|x)})] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{x \sim p(x)}[ D_{KL} [p(y|x) \parallel q(y|x))] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(q(y|x))] + E_{(x,y) \sim p(x,y)}[log(\frac{1}{p(y)})] + E_{x \sim p(x)}[ D_{KL} [p(y|x) \parallel q(y|x))] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(q(y|x))] + H_{p(y)}[y] + E_{x \sim p(x)}[ D_{KL} [p(y|x) \parallel q(y|x))] ]
$$


## Strategy:

x = images, z = latent code

Relabelling the last equation:

$$
= E_{(x,z) \sim p(x,z)}[log(q(x|z))] + H_{p(x)}[x] + E_{z \sim p(z)}[ D_{KL} [p(x|z) \parallel q(x|z))] ]
$$

$$
= E_{x \sim p(x)}[ E_{z \sim p_\theta(z|x)}[log(q_\phi(x|z))] ] + H_{p(x)}[x] + E_{z \sim p_\theta(z)}[ D_{KL} [p_\theta(x|z) \parallel q_\phi(x|z))] ]
$$

We model $p_\theta(z|x)$ with the aim of maximizing mutual information between X and Z. So in this setup p(x) is given by nature and we do not have a model for it. $q_\phi(x|z)$ is our image reconstruction term. The middle term is an entropy term and the latter term is a KL term between the real (and unknown) reconstruction term and our model of it.

So the 1st term is a lower bound on mutual information and it is this term we aim to maximize.


## References:

On Variational Bounds of Mutual Information (2019), Poole, Ozair, Van den Oord, Alemi, Tucker
[https://arxiv.org/pdf/1905.06922]
Information Theory, Inference and Learning Algorithms, David Mackay, 2003.
