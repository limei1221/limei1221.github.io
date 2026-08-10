---
layout: post
comments: true
mathjax: true
title: "Optimizers in Deep Learning"
excerpt: ""
date: 2026-08-02
---

Assume a vector of parameters `x` and the gradient `dx`.

## SGD
```
# Vanilla update
x += -learning_rate * dx
```

## SGD + momentum
```
# Momentum update
v = mu * v - learning_rate * dx  # integrated velocity
x += v  # integrated position
```
* `mu` is in optimization referred to as *momentum*, its typical value is about 0.9.
* With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient.

## Adagrad
[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
](https://jmlr.org/papers/v12/duchi11a.html)

```
cache += dx**2
x += -learning_rate * dx / (np.sqrt(cache) + eps)
```
* The weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased.
* The smoothing term `eps` (usually set somewhere in range from 1e-4 to 1e-8) avoids division by zero.
* `cache` grows bigger and bigger, updates become very small after certain updates.

## RMSProp
[slide 29 of Lecture 6](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

```
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```
* `decay_rate` is a hyperparameter and typical values are [0.9, 0.99, 0.999].

## Adam
[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

```
# t is your iteration counter going from 1 to infinity
m = beta1 * m + (1 - beta1) * dx
mt = m / (1 - beta1**t)  # bias correction
v = beta2 * v + (1 - beta2) * dx**2
vt = v / (1 - beta2**t)  # bias correction
x += -learning_rate * mt / (np.sqrt(vt) + eps)
```
* Recommended values in the paper are `eps = 1e-8`, `beta1 = 0.9`, `beta2 = 0.999`.
* Bias correction: corrects relatively small `m` and `v` at the beginning of the training.

## AdamW
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

AdamW separate weight decay from Adam.
```
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:  # for every group of parameters
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:  # for every parameter in the group
                if p.grad is None:
                    continue
                state = self.state[p]

                # state initialization with 0s
                t = state.get("t", 0)
                m, v = state.get("m", torch.zeros_like(p.data)), state.get("v", torch.zeros_like(p.data))

                # weight decay
                p.data -= lr * weight_decay * p.data

                # Adam update
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))
                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                # update optimizer state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
```
* No L2-term in loss function.
* Usually we don’t want weight decay on biases and LayerNorm params:
```
torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
])
```
* In practice it’s better to apply weight decay before the Adam update because weight decay depends on the parameter.

## Muon
[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)

Muon is an optimizer for 2D parameters of neural network hidden layers. It is defined as follows:
![Muon Algorithm](/images/post_2026_08_02_optimizers/muon_algo.png)
When training a neural network with Muon, scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW.

### Newton-Schulz

Ordinary SGD momentum gives you some $B_t \in \mathbb{R}^{m \times n}$. Apply the (thin) SVD (Singular Value Decomposition) to $B_t$ results
$$
B_t = U \Sigma V^{\mathrm{T}},
$$
where, with $r = \min(m, n)$, the matrices $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ have orthonormal columns and $\Sigma \in \mathbb{R}^{r \times r}$ is a diagonal matrix.

The matrix might have singular values like $\Sigma = diag(100, 20, 3, 0.1, 0.001, ...)$. So some directions dominate the update enormously more than others.

Muon approximately replaces $B_t$ with
$$
Ortho(B_t) = UV^{\mathrm{T}} \in \mathbb{R}^{m \times n},
$$
whose singular values are $diag(1,1,1,1,1,...)$. So instead of spending almost the entire update on one dominant direction, Muon makes a substantial update along every useful singular direction simultaneously.

And Newton-Schulz is just an efficient way of approximately computing this $UV^{\mathrm{T}}$ without explicitly performing an SVD. In practice Muon runs the iteration on the normalized momentum $B_t / \lVert B_t \rVert_F$, and the quintic coefficients are deliberately tuned *not* to converge exactly — the resulting singular values land roughly in $[0.7, 1.3]$ rather than exactly $1$, which turns out to be good enough.
