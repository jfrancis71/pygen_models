## Multivariate Kullback Leibler Divergence

We demonstrate:

$$
D_{KL}[p(X_{1..n} || q(X_{1..n})] = \sum_{i=1}^n E_{X_{1..i-1} \sim p(X_{1..i-1})} [D_{KL}[p(X_i|X_{1..i-1}||q(X_i|X_{1..i-1}] ]
$$

Using the definition:

$$
D_{KL}[p(X_{1..n} || q(X_{1..n})] = \sum_{X_{1..n}} p(X_{1..n}) log \frac{p(X_{1..n})}{q(X_{1..n})}
$$

$$
 = \sum_{X_{1..n}} p(X_{1..n}) log \prod_{i=1}^n \frac{p(X_i | X_{1..i-1})}{q(X_i | X_{1..i-1})}
$$

$$
= \sum_{i=1}^n \sum_{X_{1..n}} p(X_{1..n}) log \frac{p(X_i | X_{1..i-1})}{q(X_i | X_{1..i-1})}
$$

Seperating the sum over variables into before i, i, and after i:

$$
= \sum_{i=1}^n \sum_{X_{1..i-1}} \sum_{X_i} \sum_{X_{i+1..n}} p(X_{1..i-1}) p(X_i | X_{1..i-1}) p(X_{i+1..n}) log \frac{p(X_i | X_{1..i-1})}{q(X_i | X_{1..i-1})}
$$

Using associative rules to pull terms out of factors:

$$
= \sum_{i=1}^n \sum_{X_{1..i-1}} p(X_{1..i-1}) \sum_{X_i} p(X_i | X_{1..i-1}) log \frac{p(X_i | X_{1..i-1})}{q(X_i | X_{1..i-1})} \sum_{X_{i+1..n}}   p(X_{i+1..n}) 
$$

Last term is just 1, probabilities sum to 1. Previous term is KL on a conditional distribution, and term before is the expectation over this. So:

$$
= \sum_{i=1}^n E_{X_{1..i-1} \sim p(X_{1..i-1})} [D_{KL}[p(X_i|X_{1..i-1}||q(X_i|X_{1..i-1}] ]
$$

Reference:
The definition of the Kullback-Leibler divergence came from Wikipedia, Kullback-Leiber divergence, section Definition, accessed on 07 July 2024.
