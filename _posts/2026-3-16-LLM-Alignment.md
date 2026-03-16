---
layout: post
title: From PPO to GRPO - The Evolution of Alignment Algorithms in Large Language Models
---

If you've been following the rapid progress of Large Language Models (LLMs), you know that pretraining is only part of the story. Predicting the next token gives us a powerful base model, but it does not automatically give us a model that is helpful, safe, or aligned with human preferences.

That second part happens during **alignment**: the stage where a raw language model is adapted into an assistant that behaves the way people actually want.

For a long time, this space was dominated by **Reinforcement Learning from Human Feedback (RLHF)** using **PPO**. But as practitioners ran into the engineering complexity, instability, and memory overhead of PPO-based pipelines, the field began searching for simpler and more efficient alternatives. That search produced methods like **DPO** and **GRPO**, which take very different approaches and serve different use cases.

In this post, we'll walk through the core ideas behind RLHF and then examine three important optimization approaches used in modern post-training: **PPO**, **DPO**, and **GRPO**. The goal is to keep the discussion self-contained while still giving enough mathematical intuition to understand why these methods differ, and why the field has gradually moved from one to the next.

---

## Primer: What is RLHF?

Before comparing algorithms, we need to understand the larger framework they fit into: **RLHF**.

RLHF usually refers to a post-training pipeline for aligning a pretrained language model with human preferences. In its classic form, it has three stages:

* **Supervised Fine-Tuning (SFT):** train on demonstration prompt-response pairs so the model learns assistant-style behavior.
* **Reward Modeling (RM):** train a reward model on human preference data so it can score candidate outputs.
* **Reinforcement Learning (RL):** optimize the policy so it produces outputs that achieve high reward while remaining close to a reference behavior.

This is the broad pipeline popularized by **InstructGPT**. In practice, some people use the term "RLHF" more narrowly to mean only the reward-modeling and RL stages, but in this post we will use it for the full three-stage setup.

The key point is that **PPO**, **DPO**, and **GRPO** are all different answers to the same question:

> Once we have a model that can already behave like an assistant, how do we improve it further using preferences or reward signals?

PPO answers that question with full reinforcement learning. DPO replaces the RL step with a direct preference objective. GRPO brings back RL-style exploration, but removes one of PPO's most expensive pieces.

---

## 1. PPO: Proximal Policy Optimization

*Original Paper: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)*

PPO was originally developed for standard reinforcement learning problems such as robotics and game-playing. It later became the foundation of early large-scale RLHF systems for language models.

### The Core Idea

PPO is built around a simple principle: **improve the policy, but do so cautiously**.

In RLHF, the policy is the language model we want to train. We sample outputs from it, score them using a reward signal, and then update the policy so that higher-reward behaviors become more likely. But if we push the model too hard in one step, the policy can drift, become unstable, or collapse into degenerate behavior.

PPO solves this by making policy updates **conservative**. It does not just ask, "Was this action good?" It also asks, "Are we changing the model too much all at once?"

### The Four Model Roles

In PPO-based RLHF, there are often **four model roles** involved:

1. **Actor / Policy Model ($\pi_\theta$):** the language model being optimized.
2. **Reference Model ($\pi_{ref}$):** a frozen copy of the initial policy used to penalize excessive drift.
3. **Reward Model:** a model that scores completed outputs.
4. **Critic / Value Model:** a model that estimates expected future reward at intermediate generation steps.

A small but important nuance: these are four **roles**, not always four separate full-size models. In practical systems, the reward model and value model may be smaller than the policy, and some implementations initialize the value model from the reward model.

### Reward Model vs. Critic Model

This is one of the easiest places to get confused, because both models output scalar values.

* **The Reward Model** evaluates the **final response**. It looks at the completed output and returns a score representing how preferred, helpful, or safe that answer is.
* **The Critic Model** estimates the **expected future return during generation**. It predicts a value function $V(s_t)$ at intermediate states so PPO can determine whether a particular token choice was better or worse than expected.

A useful analogy is this:

* The **reward model** is like the final grader.
* The **critic** is like a running estimate of what grade you're on track to get while you're still writing.

### The Math

PPO uses the **advantage** to measure whether an action turned out better than expected:

$$
\hat{A}_t \approx R_t - V(s_t)
$$

In practice, many implementations use multi-step returns or Generalized Advantage Estimation (GAE), but this expression captures the main intuition: compare what actually happened to what the critic predicted.

PPO then optimizes the clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

where

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

is the probability ratio between the new policy and the old one.

If an action has positive advantage, PPO wants to make it more likely. But the clipping term prevents the probability ratio from moving too far in one update. This is the core stabilizing mechanism.

### Where the KL Penalty Comes In

RLHF almost always includes a penalty to keep the trained policy close to a frozen reference model. At a high level, you can think of the reward as something like

$$
r_{\text{total}}(x,y) \approx r_{\text{RM}}(x,y) - \beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{ref})
$$

This is a useful **schematic** view, but it is not always the exact literal implementation. In practice, many RLHF systems apply the KL penalty more locally, often as a **per-token penalty** relative to the reference or SFT model.

### Why PPO Was So Important — and Why It’s Expensive

PPO mattered because it provided a workable way to train language models with reward signals rather than pure imitation. That was a major step beyond supervised fine-tuning.

But PPO is also expensive and operationally complex. It typically involves policy, reference, reward, and value components, plus an RL training loop that can be delicate to tune. In large-model settings, that complexity becomes a real systems bottleneck.

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

DPO shows that, under a particular formulation, this two-stage setup can be rewritten into a direct objective over preference pairs. Instead of explicitly training a reward model and then doing reinforcement learning, we can optimize the language model itself to favor preferred responses over dispreferred ones.

This does **not** mean DPO is just ordinary supervised fine-tuning. It is still doing preference optimization, but through a much simpler **pairwise objective**.

### The Math

Suppose that for prompt $x$, humans prefer response $y_w$ over response $y_l$. DPO optimizes

$$
L_{\text{DPO}}(\pi_\theta;\pi_{ref})
=
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma \left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
\right]
$$

where:

* $x$ is the prompt,
* $y_w$ is the preferred response,
* $y_l$ is the dispreferred response,
* $\sigma$ is the sigmoid function,
* $\beta$ controls the strength of regularization toward the reference model.

The interpretation is straightforward: the model is pushed to assign relatively higher probability to the preferred answer than to the rejected answer, measured against the frozen reference model.

### Why DPO Took Off

DPO is appealing because it removes much of the infrastructure burden of PPO. There is no explicit value model, no rollout-based RL loop, and no need to first train a separate reward model for this stage.

That makes it:

* simpler to implement,
* cheaper to run,
* and often more stable in practice.

This is a big reason DPO became so popular in open-source post-training pipelines.

It is worth noting: DPO skips explicit reward model training, but it still requires a **preference dataset** — the same kind of human comparison data that would have been used to train that reward model. The saving is in infrastructure and training stages, not in data collection.

### The Tradeoff

The limitation is that DPO is fundamentally tied to the **offline preference dataset** it trains on. It can only learn from examples that already exist in the data. It does not naturally support the same kind of online exploration that reinforcement learning does.

That tradeoff is acceptable for many chat-assistant settings. But once we care about domains where the model may need to discover better strategies through trial and error, RL starts to become attractive again.

---

## 3. GRPO: Group Relative Policy Optimization

*Original Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)*

GRPO emerged from the observation that reinforcement learning is still extremely useful in settings where answers can be checked, scored, or verified — especially in domains like mathematics and coding.

The question was not whether RL had value. The question was whether we could recover much of PPO's benefit **without** carrying around a separate critic.

GRPO's answer is: yes.

### The Core Idea

The main systems innovation of GRPO is simple:

> Remove the critic, and estimate advantage by comparing multiple sampled answers to one another.

Instead of training a value model to estimate expected return, GRPO samples a **group** of responses for the same prompt and scores them all. Each response is then judged relative to the others in its group.

So for prompt $q$, the policy samples

$$
\{o_1, o_2, \dots, o_G\}
$$

and each output receives a reward.

That reward can come from:

* a learned reward model,
* a rule-based verifier,
* a code execution result,
* a math checker,
* or some combination of these.

This point matters because GRPO is often described too loosely. It is **critic-free**, but it is **not reward-free**.

### Outcome Supervision: the Simplest Case

In the simplest version, each sampled output gets a single scalar reward:

$$
\{r_1, r_2, \dots, r_G\}
$$

GRPO then converts those rewards into normalized group-relative advantages:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

This gives a simple interpretation:

* if a sampled answer is better than the other answers in its group, it gets positive advantage;
* if it is worse, it gets negative advantage.

That means GRPO does not need a separate learned value function to say what was "better than expected." The group itself provides the baseline.

### The Math

Even when the reward is assigned at the sequence level, the model is still autoregressive, so optimization happens token by token.

Define the token-level policy ratio as

$$
r_{i,t}(\theta)
=
\frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}
$$

Then a GRPO-style objective can be written as

$$
L_{\text{GRPO}}(\theta)
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
\beta \,\mathbb{D}_{KL}
\right)
\right]
$$

where the KL term is often estimated as

$$
\mathbb{D}_{KL}
=
\frac{\pi_{ref}}{\pi_\theta}
-
\log \frac{\pi_{ref}}{\pi_\theta}
-
1
$$

A note on the KL term: in the original DeepSeekMath formulation, the KL penalty is incorporated as a **per-token penalty into the reward signal** during rollout, and then the penalized reward flows into the policy gradient update. The presentation above — subtracting it directly inside the objective — is a common equivalent way to write it, but the precise location of the KL in the implementation matters for understanding how the penalty is computed in practice.

There are two more details worth emphasizing.

First, the factor

$$
\frac{1}{|o_i|}
$$

means the objective includes **length normalization** over each sampled sequence. That is part of the original GRPO formulation and is easy to miss.

Second, the expression above corresponds to the simplest **outcome-supervision** setting, where each whole response gets one reward and therefore one sequence-level advantage.

### Process Supervision: the More General View

In the full GRPO framework, the advantage does not always need to be the same for every token in a sequence. Under **process supervision**, intermediate reasoning steps can receive rewards too, which leads to token-dependent advantages such as $\hat{A}_{i,t}$.

So the sequence-level normalized reward above is best understood as the simplest and most intuitive version of GRPO, not the most general one.

### Why GRPO Matters

GRPO sits in an interesting middle ground.

Like PPO, it still allows **online sampling and exploration**. The model can generate multiple candidate solutions, receive rewards, and improve based on those rollouts.

Like DPO, it avoids some of PPO's heavy machinery — most notably the separate critic.

That makes GRPO especially attractive in tasks where rewards can be computed automatically, such as:

* mathematical problem solving,
* coding problems,
* theorem-like reasoning tasks,
* or any setting with a reliable verifier.

It would be too strong to claim that RL is universally superior for reasoning. But RL has often been **particularly effective** in settings with verifiable rewards, because it lets the model search for strategies that may not already be present in a static preference dataset.

---

## Putting the Three Together

At a high level, these methods reflect three different philosophies of post-training.

### PPO: Full RL with Explicit Value Estimation

PPO is the classic reinforcement-learning answer. It gives you exploration, reward optimization, and explicit advantage estimation through a critic. But it is also the heaviest and most operationally complex.

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

The evolution from PPO to DPO to GRPO reflects a broader pattern in AI: not abandoning alignment or reinforcement learning, but **re-engineering them to be cheaper, more stable, and better matched to the structure of the task**.

As language models move from general chat toward deeper reasoning, tool use, and verifiable problem solving, the algorithms used in post-training are becoming more specialized. And that shift is likely to continue: the future of alignment may not belong to one universal algorithm, but to a family of methods tuned to different kinds of feedback, different kinds of supervision, and different kinds of capability.
