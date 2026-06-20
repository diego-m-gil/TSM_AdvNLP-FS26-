# Practice Problems with Solutions

Cover them up and try first. These mirror the exam's "compute scores" style. Calculator tip: `log2(x) = ln(x)/ln(2)`.

---

## §1 — Classification metrics (P / R / F1 / macro / weighted)

**Q1.1** A 3-class classifier (CH, FR, DE) is evaluated on 5 sentences:

| # | GT | Predicted |
|---|----|-----------|
| 1 | CH | CH |
| 2 | FR | DE |
| 3 | DE | CH |
| 4 | CH | FR |
| 5 | FR | FR |

Compute per-class precision, recall, F1, then macro-F1 and weighted-F1.

<details><summary>Solution</summary>

Count per class (TP / FP / FN):
- **CH:** predicted CH on #1(✓),#3(✗). TP=1, FP=1 (#3). GT CH = #1,#4; found #1 only → FN=1. P=1/2, R=1/2, **F1=1/2**.
- **FR:** predicted FR on #4(✗),#5(✓). TP=1, FP=1 (#4). GT FR=#2,#5; found #5 → FN=1. P=1/2, R=1/2, **F1=1/2**.
- **DE:** predicted DE on #2(✗). TP=0, FP=1. GT DE=#3 → FN=1. P=0, R=0, **F1=0**.

Supports: CH=2, FR=2, DE=1 (total 5).
- **Macro-F1** = (½ + ½ + 0)/3 = **1/3 ≈ 0.333**.
- **Weighted-F1** = (2·½ + 2·½ + 1·0)/5 = (1+1+0)/5 = **0.4**.
- Accuracy = correct(#1,#5)/5 = **0.4**.
</details>

**Q1.2** A lazy classifier always predicts CH. Test set: 300 CH, 300 FR, 400 DE (1000 total). Give accuracy, and precision/recall/F1 for each class.

<details><summary>Solution</summary>

- Accuracy = 300/1000 = **0.30**.
- **CH:** TP=300, FP=700, FN=0 → P=300/1000=**0.30**, R=**1.0**, F1=2·0.3·1/(1.3)=**0.4615**.
- **FR:** no FR predictions → TP+FP=0 → **Precision = N/A (0/0)**; R = 0/300 = 0; **F1 = N/A**.
- **DE:** same → **Precision N/A**, R=0, **F1 N/A**.
- Lesson: recall for CH is perfect but precision terrible; accuracy alone hides this.
</details>

---

## §2 — Dependency parsing: UAS / LAS

**Q2.1** Gold tree vs system output for a 6-token sentence (token: gold head, gold label | pred head, pred label):

| tok | gold head | gold label | pred head | pred label |
|-----|-----------|------------|-----------|------------|
| 1 | 2 | det | 2 | det |
| 2 | 3 | nsubj | 3 | nsubj |
| 3 | 0 (root) | root | 0 | root |
| 4 | 3 | dobj | 3 | iobj |
| 5 | 6 | det | 4 | det |
| 6 | 4 | nmod | 4 | nmod |

Compute UAS and LAS (score all 6 tokens).

<details><summary>Solution</summary>

Check each token:
- t1 head 2=2 ✓, label det=det ✓ → UAS✓ LAS✓
- t2 head 3=3 ✓, nsubj=nsubj ✓ → UAS✓ LAS✓
- t3 head 0=0 ✓, root=root ✓ → UAS✓ LAS✓
- t4 head 3=3 ✓, label dobj≠iobj ✗ → **UAS✓ LAS✗**
- t5 head 6≠4 ✗ → UAS✗ LAS✗
- t6 head 4=4 ✓, nmod=nmod ✓ → UAS✓ LAS✓

Correct heads (UAS): t1,t2,t3,t4,t6 = 5/6 → **UAS = 5/6 ≈ 0.833**.
Correct head+label (LAS): t1,t2,t3,t6 = 4/6 → **LAS = 4/6 ≈ 0.667**.
Note LAS ≤ UAS. (If root excluded: UAS=4/5=0.8, LAS=3/5=0.6.)
</details>

**Q2.2** In arc-standard parsing, stack=[…, A, B] (B on top), buffer=[C,…]. What does LEFT-ARC do? RIGHT-ARC? SHIFT?

<details><summary>Solution</summary>

- **SHIFT:** move C from buffer onto stack → stack=[…,A,B,C].
- **LEFT-ARC(r):** add arc B→A (B is head of A, the second-top), label r; **pop A**. Stack=[…,B].
- **RIGHT-ARC(r):** add arc A→B (A head of B, the top), label r; **pop B**. Stack=[…,A].
</details>

---

## §3 — KL divergence (the "distribution / ground truth / measured" calc)

**Q3.1** True distribution P=[0.5, 0.3, 0.2], model Q=[0.4, 0.4, 0.2]. Compute D_KL(P‖Q) in **bits**.

<details><summary>Solution</summary>

`D_KL = Σ P·log2(P/Q)`
= 0.5·log2(0.5/0.4) + 0.3·log2(0.3/0.4) + 0.2·log2(0.2/0.2)
= 0.5·log2(1.25) + 0.3·log2(0.75) + 0
= 0.5·(0.3219) + 0.3·(−0.4150) + 0
= 0.1609 − 0.1245 = **0.0364 bits**. (Small → model close to truth.)
</details>

**Q3.2** Same P, but Q=[0.85, 0.10, 0.05]. Compute D_KL(P‖Q) in bits.

<details><summary>Solution</summary>

= 0.5·log2(0.5/0.85) + 0.3·log2(0.3/0.10) + 0.2·log2(0.2/0.05)
= 0.5·log2(0.588) + 0.3·log2(3) + 0.2·log2(4)
= 0.5·(−0.7655) + 0.3·(1.585) + 0.2·(2)
= −0.3828 + 0.4755 + 0.4 = **0.4927 bits**. (Bigger → worse model.)
Note D_KL is **not symmetric**; ≥0; =0 iff P=Q.
</details>

---

## §4 — RLHF / DPO

**Q4.1** Reward model gives r(x,y_w)=2.0, r(x,y_l)=1.2. Compute P(y_w ≻ y_l) and the NLL loss.

<details><summary>Solution</summary>

P = σ(2.0−1.2) = σ(0.8) = 1/(1+e^−0.8) = 1/(1+0.4493) = **0.690**.
NLL loss = −log(0.690) = **0.371**. Correct ranking → low loss.
(If instead r_w=1.0, r_l=1.5: σ(−0.5)=0.378, loss=−log0.378=**0.973** → wrong ranking, high loss.)
</details>

**Q4.2 (conceptual — likely text answers).** Answer in one or two sentences each:
(a) Why was the partition function Z(x) introduced? (b) Why does it disappear in DPO? (c) Why do we *need* it to disappear? (d) Why is it called a normalization constant? (e) Role of β? (f) Role of the "win" term? (g) DPO in one sentence?

<details><summary>Solution</summary>

(a) To **normalize** the reward-tilted reference policy π*(y|x) ∝ π_ref·exp(r/β) so it's a valid probability distribution (sums to 1) — we're shifting probability mass, so we must renormalize.
(b) In the Bradley-Terry **difference** of two rewards for the **same prompt**, the `β log Z(x)` term is identical for winner and loser (Z depends only on x, not y) → it **cancels**.
(c) Because Z(x) sums over **all possible completions** of x → **intractable** to compute; cancellation lets us train without it.
(d) It rescales the tilted weights to sum to 1; it's constant in y but **changes with the prompt x** (prompt-dependent), hence "normalization constant."
(e) β balances reward maximization vs staying close to the reference (implicit KL strength): high β → stay near reference; low β → free to drift (risk reward hacking).
(f) `log(π_θ(y_w|x)/π_ref(y_w|x))` raises the preferred ("winning") response's probability relative to the reference; the loser term lowers the rejected one.
(g) DPO shifts probability mass toward completions humans prefer and away from those they reject, directly from preference pairs — no separate reward model, no RL loop.
</details>

**Q4.3** Write the DPO loss from memory.

<details><summary>Solution</summary>

`L_DPO(π_θ;π_ref) = − E_(x,y_w,y_l)~D [ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`.
</details>

---

## §5 — Benchmarking (MCC / Spearman)

**Q5.1** 100 sentences: 90 acceptable, 10 not. Classifier always says "acceptable." Compute accuracy and MCC.

<details><summary>Solution</summary>

Let "acceptable" = positive. TP=90, FP=10, TN=0, FN=0.
Accuracy = (TP+TN)/100 = 90/100 = **0.90**.
MCC numerator = TP·TN − FP·FN = 90·0 − 10·0 = 0 → **MCC = 0**.
Lesson: high accuracy, zero real skill — why CoLA uses MCC (uses all 4 cells; F1 ignores TN).
</details>

**Q5.2** Human ranks R=[1,2,3,4,5], model ranks S=[5,4,3,2,1]. Compute Spearman ρ.

<details><summary>Solution</summary>

d = R−S = [−4,−2,0,2,4]; d² = [16,4,0,4,16]; Σd² = 40. n=5.
ρ = 1 − 6·Σd²/[n(n²−1)] = 1 − 240/[5·24] = 1 − 240/120 = **−1** (perfect reverse correlation).
</details>

---

## §6 — Quick Perruchoud-side calcs (easy points)

**Q6.1** WER: reference has 10 words; transcript has 1 substitution, 1 deletion, 2 insertions. WER?
<details><summary>Solution</summary>WER = (S+D+I)/N = (1+1+2)/10 = **0.4 = 40%**.</details>

**Q6.2** Two unit vectors with dot product 0.6. Squared Euclidean distance?
<details><summary>Solution</summary>‖x−y‖² = 2 − 2cos = 2 − 2(0.6) = **0.8**.</details>

**Q6.3** Trigram model has lower perplexity than bigram. Which models the language better?
<details><summary>Solution</summary>The **trigram** — lower perplexity = better predictive model (PP = P(W)^(−1/N)).</details>

**Q6.4** LoRA on a 4096×4096 weight with rank r=4: how many trainable params vs full?
<details><summary>Solution</summary>Full = 4096·4096 = 16,777,216. LoRA = r(d+k) = 4·(4096+4096) = **32,768** (A and B). ~512× fewer.</details>
