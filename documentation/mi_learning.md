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

## Lower Bound:

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

## Upper Bound:

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



## Strategy:

x = images, z = latent code

From equation 1:

$$
I(x,z) = E_{(x,z) \sim p(x,z)}[log(\frac{p(z|x)}{p(z)})]
$$


## References:

On Variational Bounds of Mutual Information (2019), Poole, Ozair, Van den Oord, Alemi, Tucker
[https://arxiv.org/pdf/1905.06922]
