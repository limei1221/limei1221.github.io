---
layout: post
comments: true
mathjax: true
title: "Attention Residuals: Turning Depth from a Pipe into Memory"
excerpt: "Attention Residuals reframes residual streams as a retrieval mechanism over depth, replacing blind accumulation with learned access to earlier computations."
date: 2026-03-30
---

Residual connections are so standard in Transformers that it is easy to stop seeing the design choice hidden inside them.

We usually describe residuals as a gradient highway, and that is true. But residuals also define how information is aggregated across depth. In a standard Transformer stack, layer $l$ does not really receive “the previous layer” in a clean sense. It receives the token embedding plus a running sum of everything earlier layers have added so far.

That mixing rule is fixed, blind, and uniform.

The [Attention Residuals](https://arxiv.org/abs/2603.15031) paper asks a simple question: what if depth worked more like retrieval and less like accumulation? Instead of forcing each layer to inherit one hard-coded running state, let it choose which earlier computations it wants to build on.

That is the idea of the paper in one line: **standard residuals treat depth like a pipe; Attention Residuals treats depth like memory.**

![Overview of Attention Residuals](/images/post_2026_03_30_attention_residuals/attention-residuals.png)

One notation detail matters for the rest of the discussion. The paper counts each attention sublayer and each MLP or MoE sublayer as a separate “layer.” So a decoder with 32 Transformer blocks is treated as roughly $L = 64$ in the formulas.

---

## TL;DR

**Standard residuals** implicitly merge all earlier computations with fixed, uniform coefficients. **Attention Residuals** replaces that fixed rule with learned retrieval over earlier layer outputs. The full version lets each layer attend over the entire depth history; the block version compresses old history into summaries and keeps the idea practical.

---

## 1. What standard residuals are really doing

The usual residual update is

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1}).
$$

If you unroll it, you get

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i),
$$

where $h_l \in \mathbb{R}^d$ is the hidden state entering layer $l$, $h_1$ is the token embedding.

This is the hidden policy inside a residual stack: every earlier contribution is merged with coefficient 1. The current layer does not get to say “give me more of the early lexical signal,” or “ignore layers 5 through 10,” or “jump back to an older abstraction that is useful for this computation.” It just inherits the running total.

That is why residuals are not only an optimization trick. They are also a specific rule for mixing information across depth.

The paper’s motivation is that this rule is too rigid. Different sublayers may want different mixtures of earlier computation. An attention sublayer may benefit from access to a representation that still preserves token identity. An MLP may prefer a later semantic abstraction. Standard residuals force both to consume the same compressed state.

A useful mental model is the analogy to sequence modeling. Recurrence compresses token history into one running state; self-attention replaces that compression with content-dependent retrieval over earlier tokens. Attention Residuals makes the same conceptual move over **depth** rather than **time**.

---

## 2. The main idea: retrieve across depth instead of accumulating across depth

The paper treats earlier layer outputs as a memory bank. Later layers do not have to inherit the whole running sum. They can retrieve from earlier depth locations.

This is the core shift.

Instead of saying:

> Here is the single hidden state that survived repeated addition.

it says:

> Here are the earlier computations. Decide which ones matter now.

That framing is what makes the method feel natural rather than exotic. Once you view residual streams as a memory system over depth, attention over depth looks like the obvious generalization.

---

## 3. A cheaper fix first: `parameter-golf`

Before jumping to full attention over depth, it is worth looking at a cheaper family of fixes. If the problem is that useful early information gets washed out, maybe it is enough to keep a few privileged paths alive rather than letting every layer attend over the full history.

That is close to what OpenAI's [parameter-golf](https://github.com/openai/parameter-golf/blob/main/train_gpt.py) code does.

### Embedding reinjection

Each block receives both the current hidden state $x$ and the original normalized token embedding $x_0$, and applies a learned channel-wise mixture:

$$
x \leftarrow a \odot x + b \odot x_0.
$$

In plain English, every block gets to decide how much of the original embedding should be reintroduced before it does more work.

That gives the model a permanent lexical anchor.

### Mirrored long skips

The architecture also stores hidden states from the first half of the network in a `skips` list, then reuses them in reverse order in the second half, scaled by learned `skip_weights`.

So instead of a pure chain, the network gets a U-Net-like depth structure with explicit long skip paths.

### Why this is useful but limited

This is cheap and practical. It helps preserve early information without requiring every layer to attend over the full history of depth.

But the skip topology is fixed in advance. The architecture designer decides which long-range paths exist. A later layer cannot dynamically choose which earlier layers to rely on more or less for different parts of the input.

That is where Attention Residuals goes further: it turns fixed skip paths into learned retrieval.

---

## 4. Full Attention Residuals

Full Attention Residuals is the direct version of the idea.

Define the depth sources as

$$
v_0 = h_1, \qquad v_i = f_i(h_i) \quad \text{for } i \ge 1.
$$

So $v_0$ is the embedding, and each later $v_i$ is the contribution produced by layer $i$.

Now let layer $l$ compute attention weights over all earlier depth sources:

$$
\alpha_{i \to l}
=
\frac{\exp\!\left(q_l^\top \operatorname{RMSNorm}(k_i)\right)}
{\sum_{j=0}^{l-1} \exp\!\left(q_l^\top \operatorname{RMSNorm}(k_j)\right)},
$$

with

$$
q_l = w_l \in \mathbb{R}^d, \qquad k_i = v_i,
$$

and output

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} v_i.
$$

So instead of inheriting a uniform sum, layer $l$ forms a learned weighted combination of the embedding and all previous layer outputs.

A nice detail is that the query is just one learned vector per layer. This is not a giant new self-attention module over depth. It is a lightweight depth-mixing mechanism, with RMS normalization on the keys so the softmax is not dominated by raw activation scale.

Conceptually, Full AttnRes is the depth analogue of softmax attention. Standard residual accumulation behaves more like a fixed recurrence over depth; Full AttnRes upgrades that recurrence into retrieval.

### Why this is appealing

The point is not that every layer should ignore locality and jump wildly across the network. In fact, the paper reports that the learned attention remains mostly local. Most mass stays near the diagonal, meaning ordinary layer-by-layer composition is still the default behavior.

But the model is no longer trapped inside that local path.

It can keep nontrivial access to the embedding, and it can occasionally jump to older layers when those earlier computations are useful. That is the important flexibility standard residuals do not have.

![Depth-wise attention weight distribution](/images/post_2026_03_30_attention_residuals/depth-wise_attention_weight_distribution.png)

### The cost of the full version

The downside is straightforward: you must keep all earlier depth sources available.

That leads to extra depth-mixing cost on the order of $O(L^2 d)$ per token, and depth-state storage on the order of $O(Ld)$. In ordinary training this may be acceptable because activations are already retained for backpropagation, but it becomes more painful when activation recomputation, pipeline parallelism, or large-scale systems constraints matter.

That is what motivates the block version.

---

## 5. Block Attention Residuals

Block Attention Residuals is the practical version of the idea.

Instead of storing every earlier layer output separately, the network groups layers into $N$ depth blocks. Older history is compressed into one summary per block, while recent history inside the current block stays uncompressed.

So when a layer computes its residual input, it attends over three types of sources:

1. the embedding,
2. the summaries of completed earlier blocks, and
3. the running partial sum inside the current block.

This is a very natural compromise.

Recent local computation stays sharp. Old history is compressed. The embedding remains a first-class source throughout. That matches the intuition that most layers mainly care about nearby computation, but still benefit from occasional access to much older regions of the network.

![PyTorch-style pseudo code for Block Attention Residuals](/images/post_2026_03_30_attention_residuals/pseudo_code_block_attention_residuals.png)

### A toy example

Suppose the network has 8 sublayers and we divide them into 2 depth blocks:

- Block 1: layers 1–4
- Block 2: layers 5–8

When layer 7 runs, it does **not** need to read the outputs of layers 1, 2, 3, and 4 individually. Instead, it can read:

- the embedding,
- a single summary representing Block 1, and
- the partial block summary accumulated so far within Block 2 (up to layer 6).

So the model keeps detailed access to recent depth and compressed access to older depth.

That is exactly the systems tradeoff you would want.

### Why the block version matters

Block Attention Residuals gives a clean continuum:

- If each block contains exactly one layer, you recover Full AttnRes.
- If the whole network is one block, you collapse back toward ordinary residual accumulation, except that the embedding is still treated as a separate source.

The paper reports that relatively small numbers of blocks recover most of the gains. That suggests full per-layer depth resolution is not necessary to get most of the benefit.

This is also the version that makes the idea practical in a real system. The paper describes implementation work for pipeline parallelism, including cross-stage caching, and a two-phase inference strategy that separates inter-block attention from sequential intra-block accumulation before merging them with online softmax. That is the difference between “nice architectural idea” and “something you might actually deploy.”

---

## 6. Compute and memory: what actually changes

It helps to separate two different costs:

1. the ordinary token-attention cost of the decoder, and
2. the **extra** depth-mixing cost introduced by the residual scheme.

Let:

- $T$: sequence length
- $d$: model width
- $L$: number of sublayers in the paper’s sense
- $N$: number of depth blocks in Block AttnRes

For a standard decoder with GQA, the token-attention backbone is unchanged by AttnRes. During training or prefilling, the dominant token-attention cost is still the usual quadratic attention term $O(T^2 d)$. During autoregressive decoding, per-token attention work is still linear in context length because of the KV cache. GQA changes token-attention efficiency by reducing KV storage and bandwidth; it does not change how residual information is mixed across depth.

### Extra overhead from the residual scheme

| Method | What each layer can read from depth | Extra depth-mixing compute | Extra depth-state memory | Typical structure |
|---|---|---:|---:|---|
| Standard residuals + GQA | only the current running hidden state | $O(Ld)$ | $O(d)$ | ordinary decoder stack |
| Full Attention Residuals | embedding + all previous sublayer outputs | $O(L^2 d)$ | $O(Ld)$ | 32 decoder blocks $\approx L=64$ sublayers |
| Block Attention Residuals | embedding + previous block summaries + current block partial sum | $O(LNd)$ | $O(Nd)$ | same decoder, grouped into $N$ depth blocks |
| `parameter-golf`-style skips | current hidden state + original embedding + fixed long skips | $O(Ld)$ | $O(Ld)$ | embedding reinjection + mirrored skips |

### A simple way to think about the design space

A standard decoder is a chain. Each layer inherits one running state.

The `parameter-golf`-style design is still mostly a chain, but with two strong architectural biases: every block can look back at the embedding, and the second half of the network gets explicit long skips from the first half.

Full AttnRes changes the abstraction. A layer is no longer forced to inherit one compressed depth state; it can retrieve from the full earlier computation history.

Block AttnRes says that most of this flexibility can be kept while compressing older depth into summaries, which dramatically improves the systems story.

---

## 7. Why this matters

My main takeaway from the paper is not just that it proposes a new residual variant. It proposes a better way to think about residual streams.

Residuals are usually treated as optimization scaffolding: they stabilize training, improve gradient flow, and make deep networks easier to optimize. All of that is true.

But they are also the model’s internal memory system over depth.

Once you see that, the usual residual stack starts to look like a very rigid memory design. It stores everything in one running sum and gives later layers no real choice about which earlier computation to use. Attention Residuals relaxes that constraint. The full version does it directly with retrieval over all earlier layers; the block version does it in a way that looks much more deployable.

That is why I find the paper interesting. Even if the final architecture that wins in practice is not exactly this one, the framing feels durable: **depth should not only be something information flows through; it can also be something the model reads from.**

---

## References

- [Attention Residuals (Kimi Team, 2025)](https://arxiv.org/abs/2603.15031)
- [parameter-golf (OpenAI)](https://github.com/openai/parameter-golf)
