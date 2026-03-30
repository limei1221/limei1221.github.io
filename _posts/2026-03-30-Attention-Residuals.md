---
layout: post
comments: true
mathjax: true
title: "Attention Residuals: Letting Transformers Retrieve Across Depth"
excerpt:
date: 2026-03-30
---

Residual connections are one of those ideas that became so standard that we stopped looking at them. In a modern Transformer, each block takes the current hidden state, computes something useful, and adds it back. We usually explain residuals as a gradient highway, and that is right. But they also do something else: they decide how information is aggregated across depth. Unrolled over many layers, a standard residual stack does not really give layer $l$ "the previous layer." It gives it the token embedding plus the sum of everything all earlier layers have contributed so far. That mixing rule is fixed, blind, and uniform.

The [Attention Residuals](https://arxiv.org/abs/2603.15031) paper asks a simple but powerful question: what if the model could *choose* which earlier layers to build on, instead of inheriting a hard-coded sum? The authors argue that in PreNorm Transformers, standard residual accumulation causes hidden-state magnitudes to grow with depth, which progressively dilutes each individual layer's contribution. Early information is still technically present, but it is buried inside a large running total.

That framing makes the idea click. Residual connections over depth look a lot like recurrence over time: both compress a long history into a single running state. Self-attention replaced that compression over time with content-dependent retrieval over previous tokens. Attention Residuals applies the same move over depth: replace fixed accumulation with learned retrieval over earlier layer outputs.

One notation detail from the paper is worth stating upfront because it affects the cost discussion later: the authors count each attention sublayer and each MLP or MoE sublayer as a separate "layer." So a decoder with 32 Transformer blocks is treated as roughly $L=64$ in their formulas.

---

## 1. Introduction & Background

The usual residual update is

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1}).
$$

If you expand that recurrence, you get

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i),
$$

where $h_1$ is the token embedding. This is the hidden design choice inside residual networks: all earlier contributions are merged with coefficient 1, regardless of whether the current layer wants them, needs them, or would rather ignore them. Standard residuals are therefore not just an optimization trick. They are a very specific policy for mixing information across depth.

The paper's motivation is that this policy is too rigid. Different kinds of layers may want different mixtures of earlier computations. An attention sublayer might want a representation that still preserves lexical identity. An MLP might benefit more from a later semantic abstraction. Standard residuals force both to consume the same compressed running sum.

That is the real conceptual shift behind Attention Residuals: residual streams should be thought of as a form of memory. The question is not only whether gradients can pass through them, but whether later layers can *retrieve the right earlier computation*.

---

## 2. A cheap alternative first: the `parameter-golf` approach

Before jumping to full attention over depth, it helps to look at a cheaper family of fixes. If the main issue is that useful early information gets washed out, maybe we do not need every layer to attend over the entire history of previous layers. Maybe it is enough to keep two privileged paths alive: the original token embedding, and a small number of explicit long skip connections. That is very close to what OpenAI's [parameter-golf](https://github.com/openai/parameter-golf/blob/main/train_gpt.py) training code does.

The first trick is simple and clever. Each block receives both the current hidden state $x$ and the original normalized token embedding $x_0$. It then applies a learned per-channel mixture:

$$
x \leftarrow a \odot x + b \odot x_0.
$$

In the code, this is implemented with a parameter called `resid_mix`, initialized so the model starts as "mostly current state, almost no embedding reinjection," and can learn to change that over time. In plain English: every block gets to decide how much of the original embedding should be reintroduced before doing attention and MLP work.

The second trick is U-Net-ish. The first half of the network stores hidden states in a `skips` list. The second half reuses them in reverse order, scaled by learned `skip_weights`. So the architecture is no longer just a straight stack. It becomes a depth-wise encoder/decoder shape with mirrored long-range skip connections. Early representations are not merely hoped to survive through repeated addition; they are explicitly brought back later.

This is a very practical design. It is cheap, easy to implement, and gives the model two things standard residuals do not: a permanent lexical anchor via the embedding, and a handful of explicit highways through depth. But it is still limited. The skip topology is fixed in advance. A later layer cannot decide, token by token, that for this example it wants layer 2 and for that example it wants layer 11. It only gets the paths the architecture designer preselected. That is exactly where Attention Residuals goes further.

---

## 3. Full Attention Residuals

Full Attention Residuals turns the residual stream into a retrieval problem. Instead of forcing layer $l$ to consume the uniform sum of all earlier outputs, the model lets that layer form a weighted combination of previous depth sources: the embedding plus every prior layer output. Conceptually, the hidden state is no longer "whatever survived repeated addition." It becomes "the result of querying the model's own computation history."

The mechanism is lightweight. Let

$$
v_0 = h_1, \qquad v_i = f_i(h_i)\ \text{for}\ i \ge 1.
$$

Then layer $l$ computes depth-wise attention weights

$$
\alpha_{i \to l}
=
\frac{\exp\!\left(q_l^\top \mathrm{RMSNorm}(k_i)\right)}
{\sum_{j=0}^{l-1}\exp\!\left(q_l^\top \mathrm{RMSNorm}(k_j)\right)},
$$

with a learned pseudo-query $q_l = w_l \in \mathbb{R}^d$, keys $k_i = v_i$, and output

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} v_i.
$$

The elegant part is that the query is just one learned vector per layer. There is no large extra attention module over depth. The keys are RMS-normalized so the softmax reflects useful structure rather than just raw activation magnitude.

The intuition is strong: standard residuals are the depth analogue of recurrence, while Full AttnRes is the depth analogue of self-attention. The paper explicitly makes this analogy and shows that standard residuals and related recurrence-style variants can be viewed as forms of depth-wise linear attention, while AttnRes upgrades that to depth-wise softmax attention.

Empirically, the learned depth attention stays mostly local, which is good news. The model does not abandon ordinary layer-to-layer composition. Most of the mass remains near the diagonal, but the embedding keeps nontrivial weight and some layers learn meaningful jumps to much earlier sources. In other words, AttnRes preserves the familiar local path while enabling selective long-range retrieval across depth.

The price is memory and communication. Full AttnRes requires keeping all previous depth sources available. The paper characterizes that as $O(L^2 d)$ depth-mixing arithmetic per token and $O(Ld)$ depth-state memory. In standard training this can be manageable because activations are already retained for backpropagation, but at large scale, with activation recomputation and pipeline parallelism, the storage and communication overhead become much more significant.

---

## 4. Block Attention Residuals

Block Attention Residuals is the practical version. Instead of keeping every earlier layer output, the model groups layers into $N$ depth blocks and compresses each block into a single summary. A layer now attends over three things: the embedding, the completed summaries of earlier blocks, and the running partial sum inside its current block.


This is a surprisingly natural compromise. Old history is compressed. Recent local history stays sharp. The embedding remains a first-class source throughout. That is exactly the shape you would want if you believe that layers mostly care about nearby computation, but still benefit from occasional access to older regions of the network.

It also creates a clean continuum. If you use one block per layer, you recover Full AttnRes. If you use a single block, you collapse back toward standard residual accumulation, except that the embedding is still separated out. The paper reports that relatively small numbers of blocks already recover most of the gains, which suggests that perfect per-layer granularity is not necessary.

The systems work is what makes Block AttnRes especially compelling. The paper adds cross-stage caching for pipeline parallelism and a two-phase computation strategy for inference: first batch the inter-block attention work, then handle the sequential intra-block part and merge them with online softmax. This is how the idea stops being "architecturally appealing but impractical" and becomes something close to a drop-in replacement for standard residuals. And the reported overhead is small.

---

## 5. Compute and memory comparison

To compare these approaches fairly, it helps to separate two different costs:

1. the usual token-attention cost of the decoder itself, and
2. the extra depth-mixing cost introduced by the residual scheme.

Let:

- $T$: sequence length
- $d$: model width
- $L$: number of sublayers in the paper's sense
- $H$: number of query heads
- $G$: number of KV heads or KV groups in GQA
- $N$: number of depth blocks in Block AttnRes

For a standard decoder with GQA, the attention backbone is unchanged by AttnRes. During training or prefilling, the dominant token-attention cost remains the usual quadratic attention term, while during autoregressive decoding the per-token attention work becomes linear in context length thanks to the KV cache. GQA mainly reduces KV-cache size by sharing keys and values across groups of query heads; it changes token attention efficiency, not the residual-mixing rule across depth.

![Overview of Attention Residuals](/images/post_2026_03_30_attention_residuals/attention-residuals.png)

### Comparison table

| Method | What each layer can read from depth | Extra depth-mixing compute | Extra depth-state memory | Typical structure |
|---|---|---:|---:|---|
| Standard residuals + GQA | only the current running hidden state | $O(Ld)$ | $O(d)$ | ordinary decoder stack |
| Full Attention Residuals | embedding + all previous sublayer outputs | $O(L^2 d)$ | $O(Ld)$ | 32 decoder blocks $\approx L=64$ sublayers |
| Block Attention Residuals | embedding + previous block summaries + current block partial sum | $O(LNd)$ | $O(Nd)$ | same decoder, grouped into $N$ depth blocks |
| `parameter-golf` approach | current hidden state + original embedding + fixed long skips | $O(Ld)$ | $O(Ld)$ | embedding reinjection + mirrored U-Net skips |


### A concrete way to think about the models

A standard GQA decoder is still a chain. Each layer inherits one running state and pushes it forward.

The `parameter-golf` variant is still mostly a chain, but with two architectural biases: every block can look back at the original embedding, and the second half of the network gets explicit long skips from the first half.

Full AttnRes changes the abstraction entirely. The model is no longer forced to inherit one compressed depth state. Every layer can retrieve from the whole earlier computation history.

Block AttnRes says: most of that flexibility is useful, but we can compress older history into block summaries and keep almost all the benefit for much lower systems cost.

---

## 6. Why this idea matters

My favorite way to summarize the paper is this: standard residuals treat depth as a pipe, while Attention Residuals treats depth as memory.

That sounds like a small wording change, but it is not. In the pipe view, every layer inherits one running state and adds to it. In the memory view, every layer can ask which earlier computations it actually wants to build on. The cheap alternatives, like embedding anchoring and mirrored skips, are valuable because they preserve a few important highways through the network. Full AttnRes is more ambitious because it turns those highways into retrieval. Block AttnRes is what makes that retrieval practical.

If this line of work continues, I think the lasting shift will be conceptual as much as empirical. Residual streams should not be viewed only as optimization scaffolding. They are also the model's internal memory system over depth. Once you see them that way, Attention Residuals feels less like a trick and more like the obvious generalization.

---

## References

- [Attention Residuals (Kimi Team, 2025)](https://arxiv.org/abs/2603.15031)
- [parameter-golf/train_gpt.py (OpenAI)](https://github.com/openai/parameter-golf/blob/main/train_gpt.py)
