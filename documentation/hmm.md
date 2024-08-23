# Variational Approach to Hidden Markov Model

We are modelling p(Z) as a Markov Chain:

Utilising $Z_i \perp Z_{1..i-2}   | Z_{i-1}$

$$
p(Z) = p(Z_1) \prod_{i=2}^T p(Z_i | Z_{1..i-1}) = p(Z_1) \prod_{i=2}^T p(Z_i | Z_{i-1})
$$

and

$$
q(Z|X) = \prod_{t=1}^T q(Z_t|X_t)
$$

From the result in multivariate KL [https://github.com/jfrancis71/pygen_models/edit/documentation/documentation/multivariate_kl.md] (we have swapped p and q and conditioned q on X):

$$
D_{KL}[q(Z_{1..T}|X) \parallel p(Z_{1..T})] = \sum_{t=1}^n E_{Z_{1..t-1} \sim q(Z_{1..t-1}|X)} [D_{KL}[q(Z_t|Z_{1..t-1}, X) \parallel p(Z_t|Z_{1..t-1})] ]
$$

Using independence assumptions:

$$
D_{KL}[q(Z_{1..T}|X) \parallel p(Z_{1..T})] = \sum_{t=1}^n E_{Z_{t-1} \sim q(Z_{t-1}|X)} [D_{KL}[q(Z_t|X) \parallel p(Z_t|Z_{t-1})] ]
$$

If the length of the sequence is T and $Z_i$ is a categorical variable with N states, this calculation is of order $O(T N^2)$
