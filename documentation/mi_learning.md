# Maximising Mututal Information For Learning

$$
I(X,Y) \equiv E_{(x,y) \sim p(x,y)}[log(\frac{p(x,y)}{p(x) p(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x) p(x)}{p(x) p(y)})]
$$

$$
= E_{(x,y) \sim p(x,y)}[log(\frac{p(y|x)}{p(y)})]
$$

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


# References:

On Variational Bounds of Mutual Information (2019), Poole, Ozair, Van den Oord, Alemi, Tucker
[https://arxiv.org/pdf/1905.06922]
