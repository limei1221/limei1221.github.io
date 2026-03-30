---
layout: post
comments: true
mathjax: true
title: "From PPO to GRPO: The Evolution of Alignment Algorithms in Large Language Models"
excerpt: "An exploration of LLM alignment techniques — from RLHF with PPO to the more efficient DPO and GRPO — explaining the mathematical intuition behind each method and why the field moved beyond PPO-based pipelines."
date: 2026-03-16
---

If you've been following the rapid progress of large language models (LLMs), you know that pretraining is only part of the story. Predicting the next token can produce a very capable base model, but it does not automatically produce a model that is helpful, safe, or aligned with human preferences.

That second part happens during **alignment**: the stage where a raw language model is adapted into an assistant that behaves the way people actually want.

For a long time, this stage was dominated by **Reinforcement Learning from Human Feedback (RLHF)** built on top of **PPO**. But PPO-based pipelines turned out to be expensive, hard to tune, and operationally complex at scale. As a result, the field began looking for simpler and more efficient alternatives. That search led to methods like **DPO** and **GRPO**, which take very different approaches to the same underlying problem.

In this post, we'll first explain the basic RLHF pipeline, then look at three important post-training methods used in modern alignment: **PPO**, **DPO**, and **GRPO**. The goal is to keep the discussion self-contained while still giving enough mathematical intuition to make the differences between these methods clear.

---

## Primer: What is RLHF?

Before comparing algorithms, it helps to understand the larger framework they fit into: **RLHF**.

RLHF usually refers to a post-training pipeline for aligning a pretrained language model with human preferences. In its classic form, it has three stages:

* **Supervised Fine-Tuning (SFT):** train on demonstration prompt-response pairs so the model learns assistant-style behavior.
* **Reward Modeling (RM):** train a reward model on human preference data so it can score candidate outputs.
* **Reinforcement Learning (RL):** optimize the policy (the language model we want to align) so it produces outputs that achieve high reward while remaining close to a reference behavior.

This is the broad pipeline popularized by **InstructGPT** [[Ouyang et al., 2022]](https://arxiv.org/abs/2203.02155). The figure below (Figure 2 from the InstructGPT paper) illustrates the three steps concretely:

![Figure 2 from the InstructGPT paper, showing the three-step RLHF pipeline: (1) supervised fine-tuning, (2) reward model training, and (3) PPO-based reinforcement learning.](/images/post_2026_03_16_llm_alignment/instructgpt_fig2.png)
*Figure 2 from Ouyang et al. (2022): A diagram illustrating the three steps of InstructGPT — (1) supervised fine-tuning (SFT), (2) reward model training, and (3) reinforcement learning via PPO.*

In practice, some people use the term "RLHF" more narrowly to refer only to the reward-modeling and RL stages. In this post, we will use the broader definition, because it makes it easier to compare PPO, DPO, and GRPO within one shared framework, even though DPO skips parts of the classic pipeline.

At a high level, all three methods are trying to answer the same question:

> Once we already have a model that can behave like an assistant, how do we improve it further using preferences or reward signals?

The three answers are quite different:

* **PPO** uses full reinforcement learning.
* **DPO** skips explicit reward modeling and RL, and instead directly optimizes on preference pairs.
* **GRPO** brings back RL-style exploration, but removes one of PPO's most expensive components.

---

## 1. PPO: Proximal Policy Optimization

*Original Paper: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)*

PPO was originally developed for standard reinforcement learning tasks such as robotics and game-playing. It later became the foundation of early large-scale RLHF systems for language models.

### The Core Idea

PPO is built around a simple principle: **improve the policy, but do so cautiously**.

In RLHF, the policy is the language model we want to train. We sample outputs from it, score them with a reward signal, and then update the policy so that higher-reward behaviors become more likely. But if we push the model too aggressively in a single update, the policy can drift too far, become unstable, or collapse into degenerate behavior.

PPO addresses this by making policy updates **conservative**. It does not only ask, "Was this action good?" It also asks, "Are we changing the model too much at once?"

### The Four Model Roles

In PPO-based RLHF, there are often **four model roles** involved:

1. **Actor / Policy Model ($\pi_\theta$):** the language model being optimized.
2. **Reference Model ($\pi_{ref}$):** a frozen copy of the initial policy used to penalize excessive drift.
3. **Reward Model:** a model that scores completed outputs.
4. **Critic / Value Model:** a model that estimates expected future reward at intermediate generation steps.

A useful nuance: these are four **roles**, not necessarily four full-size models. In practical systems, the reward model and value model may be smaller than the policy, and some implementations initialize the value model from the reward model.

### Reward Model vs. Critic Model

This is one of the most common points of confusion, because both models output scalar values.

* The **reward model** evaluates the **final response**. It looks at the completed output and returns a score representing how preferred, helpful, or safe that answer is.
* The **critic model** estimates the **expected future return during generation**. It predicts a value function $V(s_t)$ at intermediate states so PPO can tell whether a particular token choice was better or worse than expected.

A helpful analogy:

* The **reward model** is the final grader.
* The **critic** is an estimate of what grade you're likely to get while you're still writing.

### The Math

Before looking at the objective, one important term: a **rollout** is a trajectory sampled from the current policy. For language models, that usually means a full generated response for a given prompt. PPO, and later GRPO, repeatedly generate rollouts, score them, and update the policy using those results. DPO does not use rollouts, which is one of its main simplifications.

PPO uses the **advantage** to measure whether an action turned out better or worse than expected:

$$
\hat{A}_t \approx R_t - V(s_t)
$$

where $R_t$ is the discounted return starting from time step $t$, meaning the total future reward the agent actually received after state $s_t$.

In practice, many implementations use multi-step returns or Generalized Advantage Estimation (GAE), but this expression captures the core idea: compare what actually happened with what the critic predicted.

PPO then optimizes, more precisely **maximizes**, the clipped surrogate objective $J^{\text{CLIP}}(\theta)$:

$$
J^{\text{CLIP}}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

where

$$
\begin{aligned}
r_t(\theta)
&= \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \\
&= \frac{\pi_\theta(o_{t} \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_{t} \mid q, o_{<t})}
\end{aligned}
$$

is the probability ratio between the new policy and the old one.

Here, in the generic RL notation, $s_t$ is the state at time step $t$ and $a_t$ is the action taken; for language models, this corresponds to $(q, o_{<t})$ as the state (prompt plus all tokens generated so far) and $o_t$ as the token chosen at that step.
The policy $\pi_{\theta_{\text{old}}}$ is the policy that generated the current batch of trajectories, and we keep it fixed while we optimize the new policy $\pi_\theta$.


If an action has positive advantage, PPO wants to make it more likely. But the clipping term prevents the probability ratio from moving too far in a single update. That clipping is the main stabilizing mechanism.

### Where the KL Penalty Comes In

RLHF almost always includes a penalty to keep the trained policy close to a frozen reference model. At a high level, you can think of the total reward as something like

$$
r_{\text{total}}(q,o) \approx r_{\text{RM}}(q,o) - \beta \cdot D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{ref})
$$

Here, $q$ is the prompt, $o$ is the model's full generated response, and $r_{\text{RM}}(q,o)$ is the reward model score. The scalar reward used for RL is that score minus a KL penalty weighted by $\beta$.

This is best understood as a **schematic** view rather than an exact implementation. In practice, many RLHF systems apply the KL penalty more locally, often as a **per-token penalty** relative to the reference or SFT model.

Conceptually, the pipeline works like this:

1. For each prompt-response pair $(q,o)$, construct a scalar reward $r_{\text{total}}(q,o)$.
2. Turn that scalar reward into returns $R_t$ and advantages $\hat{A}_t$ along the trajectory.
3. Use those advantages inside the PPO objective to update the policy.

### Why PPO Was So Important — and Why It’s Expensive

PPO mattered because it gave practitioners a workable way to train language models with reward signals rather than pure imitation. That was a major step beyond supervised fine-tuning.

But PPO is also expensive and operationally heavy. It typically involves policy, reference, reward, and value components, plus an RL training loop that can be delicate to tune. In large-model settings, that complexity becomes a real systems bottleneck.

---

## 2. DPO: Direct Preference Optimization

*Original Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)*

DPO became popular because it asked a deceptively simple question:

> What if we could get the benefits of preference optimization without explicitly training a reward model and then running PPO on top of it?

The answer turned out to be yes.

### The Core Idea

In standard RLHF after SFT, we usually do two things:

1. train a reward model from preference data,
2. optimize the policy to maximize that reward while staying close to a reference model.

DPO shows that, under a particular formulation, this two-stage setup can be rewritten as a direct objective over preference pairs. Instead of explicitly training a reward model and then doing reinforcement learning, we can optimize the language model itself so that it assigns higher probability to preferred responses than to rejected ones.

This does **not** make DPO ordinary supervised fine-tuning. It is still preference optimization, but it is done through a simpler **pairwise objective**.

### The Math

Suppose that for prompt $q$, humans prefer response $o_w$ over response $o_l$. DPO **maximizes** the following preference objective:

$$
J_{\text{DPO}}(\pi_\theta;\pi_{ref})
=
\mathbb{E}_{(q,o_w,o_l)\sim\mathcal{D}}
\left[
\log \sigma \left(
\beta \log \frac{\pi_\theta(o_w|q)}{\pi_{ref}(o_w|q)}
-
\beta \log \frac{\pi_\theta(o_l|q)}{\pi_{ref}(o_l|q)}
\right)
\right]
$$

where:

* $q$ is the prompt,
* $o_w$ is the preferred response,
* $o_l$ is the dispreferred response,
* $(q, o_w, o_l) \sim \mathcal{D}$ is sampled from a dataset of human preference triples,
* $\sigma$ is the sigmoid function,
* $\beta$ controls the strength of regularization toward the reference model.

The interpretation is straightforward: the model is pushed to assign relatively higher probability to the preferred answer than to the rejected answer, measured against the frozen reference model.

In practice, many implementations write this as **minimizing a loss** $L_{\text{DPO}} = -J_{\text{DPO}}$, but that is mathematically equivalent to the maximization form above.

### Why DPO Took Off

DPO is attractive because it removes much of PPO's infrastructure burden. There is no explicit value model, no rollout-based RL loop, and no need to first train a separate reward model for this stage.

That makes it:

* simpler to implement,
* cheaper to run,
* and often more stable in practice.

This is a big reason DPO became so popular in open-source post-training pipelines.

One important caveat: DPO skips explicit reward model training, but it still requires a **preference dataset**. In other words, it still depends on the same kind of human comparison data that would otherwise have been used to train a reward model. The saving is in infrastructure and training stages, not in data collection.

### The Tradeoff

The limitation of DPO is that it is fundamentally tied to the **offline preference dataset** it trains on. It can only learn from examples that already exist in the data. It does not naturally support the same kind of online exploration that reinforcement learning does.

That tradeoff is perfectly acceptable for many chat-assistant settings. But in domains where the model may need to discover better strategies through trial and error, reinforcement learning becomes attractive again.

---

## 3. GRPO: Group Relative Policy Optimization

*Original Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)*

GRPO emerged from the observation that reinforcement learning is still extremely useful in settings where answers can be checked, scored, or verified — especially in domains like mathematics and coding.

So the question was not whether RL still mattered. The question was whether we could keep much of PPO's benefit **without** carrying around a separate critic.

GRPO's answer is: yes.

### The Core Idea

The central idea behind GRPO is simple:

> Remove the critic, and estimate advantage by comparing multiple sampled answers to one another.

Instead of training a value model to estimate expected return, GRPO samples a **group** of responses for the same prompt and scores them all. Each response is then evaluated relative to the others in its group.

So for a prompt $q$, the policy samples

$$
\{o_1, o_2, \dots, o_G\}
$$

and each output receives a reward.

That reward can come from a learned reward model or from another task-specific evaluation signal. In DeepSeekMath, the discussion is framed in terms of outcome- and process-level supervision for mathematical reasoning. In broader reasoning systems, people also use rule-based verifiers such as math checkers or code execution.

This is an important clarification: GRPO is **critic-free**, but it is **not reward-free**.

### Outcome Supervision: the Simplest Case

In the simplest version, each sampled output gets a single scalar reward:

$$
\{r_1, r_2, \dots, r_G\}
$$

GRPO then converts those rewards into normalized group-relative advantages:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

The interpretation is intuitive:

* if a sampled answer is better than the average answer in its group, it gets positive advantage;
* if it is worse, it gets negative advantage.

So GRPO no longer needs a separate learned value function to estimate what was "better than expected." The group itself provides the baseline.

### The Math

Even when the reward is assigned at the sequence level, the model is still autoregressive, so optimization still happens token by token.

Define the token-level policy ratio as

$$
r_{i,t}(\theta)
=
\frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}
$$

where $q$ is the prompt, $o_i$ is the $i$-th sampled output sequence for that prompt, and $o_{i,t}$ is the token at position $t$.

A GRPO-style objective can then be written as

$$
J_{\text{GRPO}}(\theta)
=
\mathbb{E}\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\min\left(
r_{i,t}(\theta)\hat{A}_i,\;
\mathrm{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_i
\right)
-
\beta \, D_{\mathrm{KL}}
\right)
\right]
$$

and, as with PPO, we **maximize** this objective with respect to the policy parameters $\theta$.

The KL term is often estimated as

$$
D_{\mathrm{KL}}
=
\frac{\pi_{ref}}{\pi_\theta}
-
\log \frac{\pi_{ref}}{\pi_\theta}
-
1
$$

A useful detail here: in InstructGPT-style PPO, the KL penalty is often folded into the reward during rollout as a **per-token penalty**. In the original DeepSeekMath formulation of GRPO, the KL regularizer is instead added **directly to the objective**. So the presentation above is closer to the GRPO paper than a reward-penalty interpretation would be.

It is also important not to confuse $\pi_{ref}$ with $\pi_{\theta_{\text{old}}}$:

* $\pi_{\theta_{\text{old}}}$ is the behavior policy that generated the sampled data and appears in the importance-sampling ratio.
* $\pi_{ref}$ is usually a **fixed reference model**, often a frozen SFT checkpoint, used only for KL regularization.

There are two additional details worth emphasizing.

First, the factor

$$
\frac{1}{|o_i|}
$$

means the objective includes **length normalization** over each sampled sequence. That is part of the original GRPO formulation and is easy to overlook.

Second, the expression above corresponds to the simplest **outcome-supervision** setting, where each whole response gets one reward and therefore one sequence-level advantage.

### Process Supervision: the More General View

In the full GRPO framework, the advantage does not always need to be the same for every token in a sequence. Under **process supervision**, intermediate reasoning steps can also receive rewards, which leads to token-dependent advantages such as $\hat{A}_{i,t}$.

So the sequence-level normalized reward above is best understood as the simplest and most intuitive version of GRPO, not the most general one.

### Why GRPO Matters

GRPO occupies an interesting middle ground.

Like PPO, it still allows **online sampling and exploration**. The model can generate multiple candidate solutions, receive rewards, and improve based on those rollouts.

Like DPO, it avoids some of PPO's heaviest machinery — most notably the separate critic.

That makes GRPO especially attractive for tasks where rewards can be computed automatically, such as:

* mathematical problem solving,
* coding problems,
* theorem-like reasoning tasks,
* or any setting with a reliable verifier.

It would be too strong to say that RL is universally superior for reasoning. But RL has often been **particularly effective** in settings with verifiable rewards, because it allows the model to search for strategies that may not already appear in a static preference dataset.

---

## Putting the Three Together

At a high level, these methods reflect three different philosophies of post-training.

### PPO: Full RL with Explicit Value Estimation

PPO is the classic reinforcement-learning approach. It gives you exploration, reward optimization, and explicit advantage estimation through a critic. But it is also the heaviest and most operationally complex.

### DPO: Preference Optimization Without the RL Loop

DPO says: if your main resource is a good offline preference dataset, maybe you do not need a full RL system at all. You can optimize directly on pairwise comparisons and get a much simpler pipeline.

### GRPO: RL Without the Critic

GRPO says: if you still want exploration and rollout-based learning, you may be able to remove the value model and replace it with a group-relative baseline. That preserves much of the RL flavor while reducing some of PPO's cost.

---

## Summary

* **RLHF** is the broad post-training framework for aligning language models using human preferences or related reward signals.
* **PPO** is the classic RLHF workhorse: powerful, but heavy because it often involves policy, reference, reward, and value components.
* **DPO** turns preference optimization into a direct pairwise objective, avoiding an explicit RL loop.
* **GRPO** is a critic-free RL method that estimates advantage relative to a group of sampled outputs, while still relying on reward signals such as reward models or verifiers.

The shift from PPO toward methods like DPO and GRPO reflects a broader pattern in AI: not abandoning alignment or reinforcement learning, but **re-engineering them to be cheaper, more stable, and better matched to the structure of the task**.

As language models move from general chat toward deeper reasoning, tool use, and verifiable problem solving, post-training algorithms are becoming more specialized. That trend is likely to continue. The future of alignment may not belong to one universal algorithm, but to a family of methods tuned to different kinds of feedback, supervision, and capability.

<!-- <p style="text-align: center; font-style: italic;">
  ⭐️ If you find this post helpful, please consider starring the
  <a href="https://github.com/limei1221/limei1221.github.io">source code for this blog</a>. ⭐️
</p> -->

## References
- [InstructGPT: Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
