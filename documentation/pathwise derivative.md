# Pathwise Derivative

We wish to calculate:

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]]
$$

Define $y(\epsilon,\theta)$ such that $p_\theta(x) = p(y)$

$$
\nabla_\theta[E_{x \sim p_\theta(x)}[f(x)]] = \nabla_\theta[E_{\epsilon \sim p(\epsilon)} [f(y(\epsilon,\theta))] ]
$$

by exchanging derivative and expectation:

$$
= E_{\epsilon \sim p(\epsilon)} [\nabla_\theta[ f(y(\epsilon,\theta))] ]
$$

Reference:
slide 10, STA 4273: Minimizing Expectations, Lecture 3 - Gradient Estimation 1, Chris Maddison, University of Toronto
[https://www.cs.toronto.edu/~cmaddis/courses/sta4273_w21/slides/lec03.pdf]
