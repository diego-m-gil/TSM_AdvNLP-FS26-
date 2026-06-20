# Alignment: RLHF + DPO (Lectures 7 & 8) — THE highest-yield exam topic

Puccinelli's own exam hints are almost entirely about this. Master this file and the classification/parsing metrics and you clear the 3.5 you need.

---

## Part A — Why align at all
- Pretraining (next-token prediction) gives **ability, not alignment**: data mixes truth/error, helpful/harmful. We want accurate, instruction-following, helpful, safe behavior → a **post-training** stage.
- **Instruction Fine-Tuning (IFT / SFT):** supervised fine-tuning on (prompt, good-response) pairs. Gives instruction following + zero-shot generalization. Provides the **starting policy** for RLHF. Can also teach safety (with refusals), but too much → over-refusal.
- **Self-Instruct:** bootstrap instruction data from a small seed set using the model itself (input-first for normal tasks; **label-first** for classification tasks to avoid label imbalance).
- **Why IFT isn't enough:** alignment questions are **comparative** (which answer is more helpful/safe/honest?). Demonstrations don't capture that → learn from **preferences** → RLHF.

---

## Part B — RLHF (Lecture 7)

### The pipeline (5 steps)
1. Gather prompt data. 2. **SFT** baseline. 3. Collect **human preference** comparisons. 4. Train a **reward model**. 5. Optimize policy with **RL (PPO)** using the reward model.

### Reward model
- Separate model. Input (prompt x, response y) → scalar `r(x,y)`. Can be **smaller** than the LM (only scores, doesn't generate).
- Trained from **pairwise** human comparisons (humans compare two responses, pick winner `y_w` over loser `y_l`, written `y_w ≻ y_l | x`).

### Bradley-Terry preference model (⭐)
- Assume a latent reward `r*(x,y)`. Probability that y1 beats y2:
  `p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / [exp(r*(x,y1)) + exp(r*(x,y2))]`
- Equivalent sigmoid form: `p* = σ(r*(x,y1) − r*(x,y2))`, with `σ(z)=1/(1+e^{−z})`.
- BT = **softmax over two items**. Preference depends only on the **reward difference**.

### Reward-model loss (NLL) (⭐ "Loss" question)
`L_R(r_φ, D) = − E_(x,y_w,y_l)~D [ log σ( r_φ(x,y_w) − r_φ(x,y_l) ) ]`
- **Describe in a paragraph (exam asks this):** We have preference triples (prompt, winner, loser). The reward model scores both; the score **difference** is squashed by the sigmoid into the probability that the winner beats the loser. Training = **maximum likelihood** = make the observed human choices as probable as possible; adding a minus turns it into a loss to minimize. The loss is **small when the winner's reward is much higher** than the loser's, and **large when the model ranks them the wrong way**. So minimizing NLL pushes the model to assign higher reward to human-preferred responses.
- Worked numbers: `r(x,y_w)=2.0, r(x,y_l)=1.2` → `σ(0.8)≈0.69` → loss `−log0.69≈0.37` (good ranking, low loss). Wrong way `r_w=1.0, r_l=1.5` → `σ(−0.5)≈0.38` → loss `≈0.97` (high).
- **Why not regression/MSE?** Humans give **relative** judgments, not reliable absolute scores; pairwise is the cleaner, more consistent signal.

### Policy optimization with PPO (KL-regularized) (⭐ role of β)
Per-response reward used by RL:
`r(x,y) = r_φ(x,y) − β·D_KL[ π_θ(y|x) ‖ π_SFT(y|x) ]`
Objective: `max_θ E_{x~D, y~π_θ}[ r_φ(x,y) ] − β D_KL[π_θ ‖ π_SFT]`.
- `π_θ` = policy being trained (starts at π_SFT). `π_SFT` = frozen reference / anchor.
- **Role of β:** balances "improve reward" vs "stay close to the reference."
  - **High β** → strong KL penalty → policy stays near SFT (safe, less drift).
  - **Low β** → policy free to drift → risk of **reward hacking** / degeneration.
- Analogy: improve the essay, but don't rewrite it so much it stops being the same assignment.

### KL divergence (⭐ "Distribution / Ground Truth / Measured Distribution" — weeks 7–8 calc)
`D_KL(P‖Q) = Σ_x P(x) log( P(x)/Q(x) )`. Properties: ≥0; =0 iff P=Q; **not symmetric**.
- P = true/ground-truth distribution, Q = model/measured distribution.
- **Worked example (travel times to Lagerstrasse, in bits = log₂):**
  - P=[0.5,0.3,0.2], Q=[0.4,0.4,0.2]:
    `0.5·log2(0.5/0.4) + 0.3·log2(0.3/0.4) + 0.2·log2(1)` = `0.5(0.3219)+0.3(−0.4150)+0` ≈ `0.1609 − 0.1245 = 0.0364 bits` (small → close).
  - P=[0.5,0.3,0.2], Q=[0.85,0.10,0.05]:
    `0.5·log2(0.5/0.85)+0.3·log2(0.3/0.10)+0.2·log2(0.2/0.05)` ≈ `−0.3828 + 0.4755 + 0.4 = 0.4927 bits` (bigger → worse model).
- **Method:** for each outcome, `P·log2(P/Q)`, sum. log2(a/b)=log2 a − log2 b. (On a calculator: `log2(z)=ln z/ln 2`.)

---

## Part C — DPO (Lecture 8): the 13-step derivation (⭐⭐ any of the 13 may be asked)

**One-sentence DPO (memorize):** *DPO makes it possible to shift probability mass toward completions humans like and away from completions humans dislike — directly, by training on preference pairs, without a separate reward model or RL loop.*

**Why DPO over RLHF?** RLHF has many moving parts (train a separate reward model, run a PPO loop, instability). DPO learns **directly from preference pairs** with a simple **classification-style loss** → easier, more stable, cheaper.

### The derivation, step by step
- **Start (KL-regularized RLHF objective):**
  `max_π E_{x~D, y~π}[ r(x,y) ] − β D_KL[π(y|x) ‖ π_ref(y|x)]`.
- **Step 2 — expand KL:** becomes `max_π E[ r(x,y) − β log(π/π_ref) ]`.
- **Step 3 — to a min:** `min_π E[ log(π/π_ref) − (1/β) r(x,y) ]` (just algebra: max A−B = min (B−A)/β).
- **Step 4 — introduce the partition function:** rewrite `−(1/β)r = −log exp((1/β)r)`, define
  `Z(x) = Σ_y π_ref(y|x) · exp( (1/β) r(x,y) )`.  ← sum over **every possible completion** for prompt x.
- **Step 5 — target distribution:** `π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp((1/β) r(x,y))`. Valid distribution (≥0, sums to 1). It's the reference policy **reward-tilted**: higher reward → more mass.
- **Step 6 — rewrite objective:** `min_π E_x[ D_KL(π ‖ π*) − log Z(x) ]`.
- **Step 7 — Z(x) doesn't affect the optimum** (it doesn't depend on π — it's an additive constant in π) → minimize `D_KL(π‖π*)`. KL ≥ 0, =0 iff equal → **optimal `π = π*`**.
- **Closed-form optimal policy:** `π*(y|x) = (1/Z(x)) π_ref(y|x) exp((1/β) r(x,y))`. (Known before DPO; not usable because Z(x) is **intractable** — sum over all completions.)
- **Step 8 — invert it:** take logs and rearrange →
  `r(x,y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x)`.
  (If π* puts more mass on y than π_ref, the reward is higher.)
- **Step 9 — Bradley-Terry:** `p*(y1≻y2|x) = σ(r*(x,y1) − r*(x,y2))` — only reward **differences** matter.
- **Step 10 — the partition function cancels (⭐):** plug the reward expression into the difference; the `β log Z(x)` term is the **same for y1 and y2** (depends only on x, not y) → **cancels**. This is what removes the need to compute Z(x).
- **Step 11 — preference in terms of policy:**
  `p*(y_w≻y_l|x) = σ( β log(π*(y_w|x)/π_ref(y_w|x)) − β log(π*(y_l|x)/π_ref(y_l|x)) )`. No explicit reward model needed.
- **Step 12 — parameterize** the unknown π* by trainable `π_θ`.
- **Step 13 — the DPO loss (final objective):**
  `L_DPO(π_θ; π_ref) = − E_(x,y_w,y_l)~D [ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`.
  - Minimizing loss = maximizing the gap → policy puts **higher** prob on preferred (vs ref) and **lower** on rejected (vs ref). The reference-relative comparison plays the role of the **KL regularization**.

### Puccinelli's exact Q&A (have these one-liners ready)
- **Role of β in DPO/Step 13?** Same as in RLHF: scales how strongly preferences move the policy vs. how tightly it's tied to the reference (implicit KL strength). Bigger β = sharper tilt / stronger pull per unit reward difference.
- **Role of the winning ("win") term?** The `log(π_θ(y_w|x)/π_ref(y_w|x))` term **increases the relative probability of the human-preferred (winning) response** vs the reference; the loser term decreases the rejected one. Loss is small when the winner's log-ratio greatly exceeds the loser's.
- **What happened to the partition function? Why does it disappear (Step 11/10)?** It cancels in the **difference** of two rewards for the same prompt, because `Z(x)` depends only on the prompt `x` (not the response `y`) — the prompt stays the same, so the term is identical for win and lose and subtracts out.
- **Why do we NEED it to disappear?** Because `Z(x)` sums `π_ref·exp(r/β)` over **all possible completions** — that's **impossible/intractable to compute**. Cancellation lets us train without ever computing it.
- **Why was the partition function introduced (Step 4)?** Because we're **shifting probability mass** (reward-tilting π_ref); `Z(x)` **normalizes** so π* is a proper probability distribution (sums to 1).
- **Why is Z(x) called a "normalization constant"?** It rescales the tilted weights so they sum to 1. It's "constant" w.r.t. the response y, but it **changes if you change the prompt x** (hence prompt-dependent `Z(x)`).
- **Why separate / why DPO?** See one-sentence above: shift mass toward liked / away from disliked completions, directly, no reward model + no RL.

---

## ⭐ Alignment formulas to have cold
1. Sigmoid `σ(z)=1/(1+e^{−z})`.
2. Bradley-Terry `p*(y1≻y2|x)=σ(r(x,y1)−r(x,y2))`.
3. Reward-model NLL `L_R = −E[ log σ(r(x,y_w)−r(x,y_l)) ]`.
4. RLHF/PPO reward `r = r_φ(x,y) − β D_KL[π_θ‖π_SFT]`.
5. KL `D_KL(P‖Q)=Σ P log(P/Q)`.
6. Optimal policy `π*(y|x) = (1/Z(x)) π_ref(y|x) exp((1/β)r(x,y))`, `Z(x)=Σ_y π_ref exp((1/β)r)`.
7. Reward from policy `r(x,y)=β log(π*/π_ref) + β log Z(x)`.
8. **DPO loss** `L_DPO = −E[ log σ( β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)) ) ]`.
