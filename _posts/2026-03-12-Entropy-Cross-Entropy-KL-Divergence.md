---
layout: post
comments: true
mathjax: true
title: "Entropy, Cross-Entropy, and KL Divergence Explained"
excerpt: "A practical explanation of entropy, cross-entropy, and KL divergence: what each quantity means, how to write the formulas, and why they show up together so often in machine learning."
date: 2026-04-18
---

Three related but distinct quantities keep appearing together in information theory and machine learning:

* **Entropy** — uncertainty in a distribution.
* **Cross-entropy** — cost of describing data from $P$ using $Q$.
* **KL divergence** — the gap between the two.

---

## Setup

Let $X$ be discrete with values $x \in \mathcal{X}$, true distribution $P(x)$, and model $Q(x)$.

The **surprisal** of an outcome is

$$
I(x) = -\log P(x),
$$

low for likely events, high for rare ones. All three quantities below are expectations of log-probabilities under $P$.

Log base sets units: $\log_2$ gives bits, $\ln$ gives nats.

---

## 1. Entropy

$$
H(P) = - \sum_{x} P(x) \log P(x) = \mathbb{E}_{x \sim P}[-\log P(x)]
$$

Entropy is the expected surprise of samples from $P$ — a property of **one** distribution.

* Concentrated $P$ → low entropy (a coin that always shows heads has $H=0$).
* Spread-out $P$ → high entropy. A fair coin gives

$$
H = -\tfrac{1}{2}\log_2\tfrac{1}{2} - \tfrac{1}{2}\log_2\tfrac{1}{2} = 1 \text{ bit},
$$

which is the maximum over two outcomes. More generally, entropy over $n$ outcomes is maximized by the uniform distribution at $\log_2 n$ bits — uniform means maximally uncertain.

**Coding interpretation:** $H(P)$ is the optimal average code length for outcomes from $P$, i.e. the compression limit.

---

## 2. Cross-Entropy

$$
H(P, Q) = - \sum_{x} P(x) \log Q(x) = \mathbb{E}_{x \sim P}[-\log Q(x)]
$$

Samples are drawn from $P$ but scored under $Q$. If $Q$ matches $P$, cost is low; if $Q$ underweights outcomes that $P$ favors, cost blows up.

**In classification** with a one-hot target,

$$
H(P, Q) = - \sum_i y_i \log q_i = -\log q_{\text{correct}}
$$

so minimizing it pushes probability mass onto the correct class.

---

## 3. KL Divergence

$$
D_{\mathrm{KL}}(P \,\|\, Q)
= \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
= \mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]
$$

With $r(x) = Q(x)/P(x)$, an equivalent form is

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \mathbb{E}_{x \sim P}[r(x) - \log r(x) - 1],
$$

since $\mathbb{E}_{x \sim P}[r(x)] = \sum_x Q(x) = 1$.

**Key properties:**

* $D_{\mathrm{KL}}(P \,\|\, Q) \ge 0$, with equality if and only if $P(x) = Q(x)$ for all $x$. This follows from Jensen's inequality: $-\mathbb{E}_P[\log(Q/P)] \ge -\log \mathbb{E}_P[Q/P] = -\log 1 = 0$.
* Not symmetric: $D_{\mathrm{KL}}(P \,\|\, Q) \ne D_{\mathrm{KL}}(Q \,\|\, P)$ in general — it is not a metric.
* If $P(x) > 0$ but $Q(x) = 0$ for some $x$, both $D_{\mathrm{KL}}$ and $H(P,Q)$ diverge.

---

## 4. The Core Identity

$$
H(P, Q) = H(P) + D_{\mathrm{KL}}(P \,\|\, Q)
$$

Derivation:

$$
\begin{aligned}
H(P, Q)
&= - \sum_x P(x)\log Q(x) \\
&= - \sum_x P(x)\log P(x) + \sum_x P(x)\log \frac{P(x)}{Q(x)} \\
&= H(P) + D_{\mathrm{KL}}(P \,\|\, Q).
\end{aligned}
$$

Reading it: cross-entropy = intrinsic uncertainty + mismatch penalty.

**Consequences:**

1. $H(P, Q) \ge H(P)$, since KL is nonnegative.
2. $H(P, Q) = H(P)$ if and only if $Q = P$.
3. With $P$ fixed, $H(P)$ is constant, so

$$
\arg\min_Q H(P, Q) = \arg\min_Q D_{\mathrm{KL}}(P \,\|\, Q).
$$

That is why cross-entropy is the natural classification loss: minimizing it drives $Q$ toward $P$.

---

## 5. Where They Show Up

* **Classification:** cross-entropy loss penalizes low probability on the true class.
* **Maximum likelihood:** minimizing empirical cross-entropy = maximizing log-likelihood.
* **Distribution matching:** variational inference, knowledge distillation, RL regularization, generative modeling — all use KL, and the direction ($P\|Q$ vs $Q\|P$) encodes different behaviors.

---

## 6. Summary

$$
H(P) = -\sum_x P(x)\log P(x)
$$

$$
H(P, Q) = -\sum_x P(x)\log Q(x)
$$

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

$$
H(P, Q) = H(P) + D_{\mathrm{KL}}(P \,\|\, Q)
$$

> **Cross-entropy is entropy plus the cost of being wrong.**
