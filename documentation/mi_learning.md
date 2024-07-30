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
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) q(y)}{p(y) q(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] + E_{(x,y) \sim p(x,y)}[\frac{q(y)}{p(y)}]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] + E_{y \sim p(y)}[\frac{q(y)}{p(y)}]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y)})] - D_{KL}[q(y)||p(y)]
$$

## Lower Bound:

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) q(y|x)}{p(y) q(y|x)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{q(y|x)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{x \sim p(x)}[ E_{y \sim p(y|x)} [ log(\frac{p(y|x)}{q(y|x)})] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{q(y|x)}{p(y)})] + E_{x \sim p(x)}[ D_{KL} [p(y|x)||q(y|x))] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(q(y|x))] + E_{(x,y) \sim p(x,y)}[log(\frac{1}{p(y)})] + E_{x \sim p(x)}[ D_{KL} [p(y|x)||q(y|x))] ]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(q(y|x))] + H_{p(y)}[y] + E_{x \sim p(x)}[ D_{KL} [p(y|x)||q(y|x))] ]
$$


## Strategy:

x = images, z = latent code

From equation 1:

$$
I(x,z) = E_{(x,z) \sim p(x,z)}[log(\frac{p(z|x)}{p(z)})]
$$


## References:

On Variational Bounds of Mutual Information (2019), Poole, Ozair, Van den Oord, Alemi, Tucker
[https://arxiv.org/pdf/1905.06922]
Information Theory, Inference and Learning Algorithms, David Mackay, 2003.
