---
layout: post
title: Understanding Speculative Decoding
---

Speculative Decoding is a latency reduction technique for large language model (LLM) inference.
Instead of letting the big model (the target model) generate each token sequentially, we first use a smaller, faster draft model to propose multiple candidate tokens ahead of time. Then the large model verifies these proposals in parallel, significantly reducing the number of expensive forward passes.


## Background
Inference from large autoregressive models like Transformers is slow — decoding $K$ tokens takes $K$ serial runs of the model.

Key observations:
* Some inference steps are "harder" and some are "easier".
<figure>
  <img src="/images/post_2025_8_18_speculative_decoding/speculative_decoding_easy_hard.png" alt="Easy vs hard decoding steps in speculative decoding">
  <figcaption>Image source: <a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">Speculative Decoding — Make LLM Inference Faster</a></figcaption>
</figure>
* Inference from large models is often not bottlenecked by arithmetic operations, but rather by memory bandwidth and communication.


## Speculative Decoding
Let $M_p$ be the target model, inference from which we're trying to accelerate, $M_q$ be a more efficient draft model.

The core idea is to:
1. use the more efficient model $M_q$ to generate $\gamma \in \mathbb{Z}^+$ completions,
2. use the target model $M_p$ to evaluate all of the guesses and their respective probabilities from $M_q$ in parallel, accepting all those that can lead to an identical distribution,
3. sample an additional token from an adjusted distribution to fix the first one that was rejected, or to add an additional one if they are all accepted.

This way, each parallel run of the target model $M_p$ will produce at least one new token, but it can potentially generate many new tokens, up to $\gamma + 1$, depending on how well $M_q$ approximates $M_p$.

## Speculative Sampling
To sample $x \sim p(x)$, we instead sample $x \sim q(x)$, keeping it if $q(x) \leq p(x)$, and in case $q(x) > p(x)$ we reject the sample with probability $1 - \frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p'(x) = \mathrm{norm}(\max(0, p(x) - q(x)))$.

It can be proven (see Appendix A.1 of the paper <a href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</a>) that for any distributions $p(x)$ and $q(x)$, and $x$ sampled in this way, indeed $x \\sim p(x)$.

<figure>
  <img src="/images/post_2025_8_18_speculative_decoding/speculative_decoding_algo.png" alt="Speculative decoding algorithm diagram">
  <figcaption>Algorithm 1 of the paper <a href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</a></figcaption>
</figure>


## References
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Speculative Decoding — Make LLM Inference Faster](https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120)
