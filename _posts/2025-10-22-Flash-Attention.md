---
layout: post
title: Understanding Flash Attention
---

## 1. Background
### 1.1 GPU Memory Hierarchy
The GPU memory hierarchy comprises multiple types of memory with different sizes and speeds, where smaller memories are faster.

![GPU memory hierarchy](/images/post_2025_10_22_flashattn/GPU_memory_hierarchy.png)

**HBM (High Bandwidth Memory)**: high‑capacity DRAM used as the GPU’s main memory (e.g., 40 GB on an A100).

**SRAM (Static RAM)**: smaller, faster on‑chip memory used within the GPU’s processing units.

On‑chip SRAM is roughly an order of magnitude faster than HBM but many orders of magnitude smaller. As compute has outpaced memory bandwidth, operations are increasingly bottlenecked by HBM access. Exploiting fast SRAM is therefore increasingly important.

### 1.2 Execution Model
GPUs execute operations (kernels) with massive thread parallelism. Each kernel loads inputs from HBM into registers and SRAM, performs computation, then writes outputs back to HBM.

Reducing HBM access is the most common approach to accelerating memory‑bound operations.

### 1.3 Performance Characteristics
**Compute-bound**: operations where the total time is dominated by arithmetic computations, e.g., matrix multiply.

**Memory-bound**: operations where the qtotal time is dominated by memory access (reading or writing data), e.g., most elementwise and reduction operations.

### 1.4 Kernel Fusion
Kernel fusion combines multiple operations that use the same inputs into a single GPU kernel, so the data is loaded once and intermediates stay on‑chip, reducing HBM traffic and launch overhead.

![Before kernel fusion](/images/post_2025_10_22_flashattn/before_kernel_fusion.png)

![After kernel fusion](/images/post_2025_10_22_flashattn/after_kernel_fusion.png)

## 2. Standard Attention

### 2.1 Standard Attention Forward Pass
Given input sequences $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ where $N$ is the sequence length and $d$ is the head dimension, the attention output $\mathbf{O} \in \mathbb{R}^{N \times d}$ is calculated as:
$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{N \times N}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d},
$$
where softmax is applied row‑wise.

![Algorithm 0 Standard Attention Implementation](/images/post_2025_10_22_flashattn/algo_0_standard_attention.png)


This requires $O(N^2)$ memory because $\mathbf{S}$ and $\mathbf{P}$ are materialized. $\mathbf{P}$ must be saved for the backward pass to compute gradients.

### 2.2 Standard Attention Backward Pass
Let $d\mathbf{O}$ be the gradient of $\mathbf{O}$ with respect to a loss. By the chain rule (backpropagation):

$$
\begin{aligned}
d\mathbf{V} &= \mathbf{P}^{\top}d\mathbf{O} \in \mathbb{R}^{N \times d} \\
d\mathbf{P} &= d\mathbf{O}\mathbf{V}^{\top} \in \mathbb{R}^{N \times N} \\
d\mathbf{S} &= \text{dsoftmax}(d\mathbf{P}) \in \mathbb{R}^{N \times N} \\
d\mathbf{Q} &= d\mathbf{S}\mathbf{K} \in \mathbb{R}^{N \times d} \\
d\mathbf{K} &= d\mathbf{S}^{\top}\mathbf{Q} \in \mathbb{R}^{N \times d}
\end{aligned}
$$

where $\text{dsoftmax}$ is the row‑wise softmax Jacobian. If $p = \text{softmax}(s)$, then with output gradient $dp$, the input gradient is $ds=(\text{diag}(p) - pp^{\top})dp$.

![Algorithm 3 Standard Attention Backward Pass](/images/post_2025_10_22_flashattn/algo_3_standard_attention_backward_pass.png)


## 3. FlashAttention
### 3.1 Online Softmax
"Online softmax" is a numerically stable, memory‑efficient way to compute softmax in a single pass—crucial on GPUs where memory access is a major bottleneck.

Here's the idea:

#### 3.1.1 The Standard Softmax Function

The softmax function converts a vector of numbers (called logits) into a probability distribution. The formula for a value $z_i$ in a vector $z$ is:

$$
\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_{j} e^{s_j}}
$$

* **The Problem (Numerical Overflow):** If any of the $s_i$ values are large (e.g., 1000), the term $e^{1000}$ becomes an astronomically large number that computers cannot store, leading to overflow (`inf`).

#### 3.1.2 The "Safe Softmax" Trick
To solve the overflow problem, a common trick is to subtract the maximum value of the vector from every element *before* exponentiating. Let $m = \max(s)$.

$$
\text{softmax}(s)_i = \frac{e^{s_i - m}}{\sum_{j} e^{s_j - m}}
$$

This is mathematically identical to the original formula but is "safe" because the largest exponent is now 0 ($e^0 = 1$), preventing any overflow.

* **The New Problem (Memory Access):** This "safe" method is a **multi‑pass** algorithm. It requires:
    1.  **Pass 1:** Iterate through the entire vector $s$ to find the maximum value, $m$.
    2.  **Pass 2:** Iterate through the vector $s$ again to calculate the denominator (the sum $\sum_{j} e^{s_j - m}$).
    3.  **Pass 3:** Iterate a third time to divide each $e^{s_i - m}$ by the denominator.

On modern GPUs, these repeated passes over memory are slow and inefficient.

#### 3.1.3 Online Softmax
**Online softmax** computes the same result while minimizing memory access by effectively fusing the passes.

For simplicity, consider just one row block of the attention matrix $S$, of the form $[\mathbf{S}^{(1)} \quad \mathbf{S}^{(2)}]$ for some matrices $\mathbf{S}^{(1)}, \mathbf{S}^{(2)} \in \mathbb{R}^{B_r \times B_c}$, where $B_r$ and $B_c$ are the row and column block sizes. We want to compute the softmax of this row block and multiply by the values, of the form $\begin{bmatrix}
\mathbf{V}^{(1)} \\
\mathbf{V}^{(2)}
\end{bmatrix}$ for some matrices $\mathbf{V}^{(1)}, \mathbf{V}^{(2)} \in \mathbb{R}^{B_c \times d}$. Standard softmax would compute:

$$
\begin{aligned}
m &= \text{max}(\text{rowmax}(\mathbf{S}^{(1)}), \text{rowmax}(\mathbf{S}^{(2)})) \in \mathbb{R}^{B_r} \\
\ell &= \text{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \text{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R}^{B_r} \\
\mathbf{P} &= [\mathbf{P}^{(1)} \quad \mathbf{P}^{(2)}] = \text{diag}(\ell)^{-1} [e^{\mathbf{S}^{(1)}-m} \quad e^{\mathbf{S}^{(2)}-m}] \in \mathbb{R}^{B_r \times 2B_c} \\
\mathbf{O} &= [\mathbf{P}^{(1)} \quad \mathbf{P}^{(2)}] \begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix} = \text{diag}(\ell)^{-1} (e^{\mathbf{S}^{(1)}-m}\mathbf{V}^{(1)} + e^{\mathbf{S}^{(2)}-m}\mathbf{V}^{(2)}) \in \mathbb{R}^{B_r \times d}
\end{aligned}
$$

Online softmax instead computes a local softmax for each block and rescales to obtain the correct final output:

$$
\begin{aligned}
m^{(1)} &= \text{rowmax}(\mathbf{S}^{(1)}) \in \mathbb{R} ^{B_r} \\
\ell^{(1)} &= \text{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R} ^{B_r} \\
\tilde{\mathbf{P}}^{(1)} &= \text{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \in \mathbb{R} ^{B_r \times B_c} \\
\mathbf{O}^{(1)} &= \tilde{\mathbf{P}}^{(1)} \mathbf{V}^{(1)} = \text{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R} ^{B_r \times d} \\
m^{(2)} &= \text{max}(m^{(1)}, \text{rowmax}(\mathbf{S}^{(2)})) = m \\
\ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \text{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \text{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \text{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R} ^{B_r} = \ell \\
\tilde{\mathbf{P}}^{(2)} &= \text{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\
\mathbf{O}^{(2)} &= \text{diag}(\ell^{(2)}/\ell^{(1)})^{-1} e^{m^{(1)} - m^{(2)}} \mathbf{O}^{(1)} + \tilde{\mathbf{P}}^{(2)}\mathbf{V}^{(2)} = \text{diag}(\ell^{(2)})^{-1}  e^{\mathbf{S}^{(1)} - m} \mathbf{V}^{(1)} + \text{diag}(\ell^{(2)})^{-1}  e^{\mathbf{S}^{(2)} - m} \mathbf{V}^{(2)} = \mathbf{O}
\end{aligned}
$$


### 3.2 FlashAttention Forward Pass
![Algorithm 1 FLASHATTENTION](/images/post_2025_10_22_flashattn/algo_1_flashattention.png)

<!-- Algorithm 1 returns $\mathbf{O}=\text{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ with $O(N^2d)$ FLOPs and requires $O(N)$ additional memory beyond input and output.

Standard attention (Algorithm 0) requires $\Theta(Nd + N^2)$ HBM accesses, while FLASHATTENTION (Algorithm 1) requires $\Theta(N^2 d^2 M^{-1})$ HBM accesses. -->

<!-- ![Algorithm 2 FLASHATTENTION Forward Pass](/images/post_2025_10_22_flashattn/algo_2_flashattention_forward_pass.png) -->

<!-- ### 3.3 FlashAttention Backward Pass

![Algorithm 4 FLASHATTENTION Backward Pass](/images/post_2025_10_22_flashattn/algo_4_flashattention_backward_pass.png)

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d \le M \le Nd$. Standard attention backward pass (Algorithm 3) requires $\Theta(Nd + N^2)$ HBM accesses, while FLASHATTENTION backward pass (Algorithm 4) requires $\Theta(N^2 d^2 M^{-1})$ HBM accesses. -->


## 4. FlashAttention-2
We tweak the FlashAttention algorithm to reduce non‑matmul FLOPs, because modern GPUs have specialized compute units (e.g., NVIDIA Tensor Cores) that make matmul much faster.

FlashAttention-2 makes two minor tweaks to the online softmax trick (Section 3.1.3) to reduce non‑matmul FLOPs:
1. We do not need to rescale both terms of the output by $\text{diag}(\ell^{(2)})^{-1}$:

    $$
    \mathbf{O}^{(2)} = \text{diag}(\ell^{(2)}/\ell^{(1)})^{-1} e^{m^{(1)} - m^{(2)}} \mathbf{O}^{(1)} + \text{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}
    $$

    Instead, we can maintain an unscaled version of $\mathbf{O}^{(2)}$:

    $$
    \tilde{\mathbf{O}}^{(2)} = \text{diag}(\ell^{(1)}) e^{m^{(1)} - m^{(2)}} \mathbf{O}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}
    $$

    Only at the end of the loop do we scale the final $\tilde{\mathbf{O}}^{(last)}$ by $\text{diag}(\ell^{(last)})^{-1}$ to obtain the correct output.

2. We do not need to save both the max $m^{(j)}$ and the sum of exponentials $\ell^{(j)}$ for the backward pass; we only need the log‑sum‑exp $L^{(j)} = m^{(j)} + \log(\ell^{(j)})$.

In the simple case of two blocks in Section 3.1.3, the online softmax trick now becomes:

$$
\begin{aligned}
m^{(1)} &= \text{rowmax}(\mathbf{S}^{(1)}) \in \mathbb{R} ^{B_r} \\
\ell^{(1)} &= \text{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R} ^{B_r} \\
\tilde{\mathbf{O}}^{(1)} &= e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R} ^{B_r \times d} \\
m^{(2)} &= \text{max}(m^{(1)}, \text{rowmax}(\mathbf{S}^{(2)})) = m \in \mathbb{R} ^{B_r} \\
\ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \text{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \text{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \text{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R} ^{B_r} = \ell \\
\tilde{\mathbf{P}}^{(2)} &= e^{\mathbf{S}^{(2)} - m^{(2)}} \\
\tilde{\mathbf{O}}^{(2)} &= e^{m^{(1)} - m^{(2)}} \tilde{\mathbf{O}}^{(1)} + \tilde{\mathbf{P}}^{(2)} \mathbf{V}^{(2)} = e^{m^{(1)} - m^{(2)}} \tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)} = e^{\mathbf{S}^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf{S}^{(2)} - m} \mathbf{V}^{(2)} \\
\mathbf{O}^{(2)} &= \text{diag}(\ell^{(2)})^{-1} \tilde{\mathbf{O}}^{(2)}
\end{aligned}
$$


### 4.1 FlashAttention-2 Forward Pass
![Algorithm 1 FLASHATTENTION-2 Forward Pass](/images/post_2025_10_22_flashattn/algo1_flashattention2_forward_pass.png)

<!-- As with FLASHATTENTION, Algorithm 1 FLASHATTENTION-2 Forward Pass returns the correct output $\mathbf{O}=\text{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ (with no approximation), using $O(N^2d)$ FLOPs and requires $O(N)$ additional memory beyond input and output (to store the logsumexp L). -->

### 4.2 FlashAttention-2 Backward Pass
![Algorithm 2 FLASHATTENTION-2 Backward Pass](/images/post_2025_10_22_flashattn/algo2_flashattention2_backward_pass.png)


## Comparison
Let $N$ be the sequence length, $d$ the head dimension, and $M$ the SRAM size with $d \le M \le Nd$.  
We report FLOPs for full attention; causal attention uses roughly half.  

| Method             | Pass | FLOPs    | Extra memory | HBM accesses           |
| ---                | ---  | ---      | ---          | ---                    |
| Standard Attention | FWD  | $4N^2d$  | $O(N^2)$     | $\Theta(Nd + N^2)$     |
| Standard Attention | BWD  | $8N^2d$  | $O(N^2)$     | $\Theta(Nd + N^2)$     |
| FlashAttention-1   | FWD  | $4N^2d$  | $O(N)$       | $\Theta(N^2d^2M^{-1})$ |
| FlashAttention-1   | BWD  | $10N^2d$ | $O(N)$       | $\Theta(N^2d^2M^{-1})$ |
| FlashAttention-2   | FWD  | $4N^2d$  | $O(N)$       | $\Theta(N^2d^2M^{-1})$ |
| FlashAttention-2   | BWD  | $10N^2d$ | $O(N)$       | $\Theta(N^2d^2M^{-1})$ |

FlashAttention-2 has fewer non-matmul ops. Also, its HBM access gains come from better parallelization (across sequence length, batch, and heads) and from fewer shared‑memory transfers, not different I/O asymptotics.


## References
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Flash Attention derived and coded from first principles with Triton (Python)](https://www.youtube.com/watch?v=zy8ChVd_oTM)
