## Modelling a probability distribution

Our setup is we wish to model a probability distribution p(x). My assumptions are that p(x) is both unknown and unknowable. It comes from nature. However we can sample from p(x).

We wish to model $p_\theta(x)$ such that $p_\theta(x)$ is as close as possible to p(x).

I shall formulate our objective as minimising:

$$
D_{KL}[p(x)||p_\theta(x)]
$$

This can be expanded as:

$$
= \sum_x{p(x) log \frac{p(x)}{p_\theta(x)}} = -H[x] - \sum{p(x) log p_\theta(x)}
$$

We can do nothing about H[X] it is given to us by nature. But minimizing the above corresponds to maximising:

$$
E_{x \sim p(x)}[log p_\theta(x)]
$$
