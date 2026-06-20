# Puccinelli Core Exam Notes — Advanced NLP

*Source: Lectures 03, 04, 05, 10, 14 (D. Puccinelli) + the two Worked-Example PDFs.*
*These are the HIGHEST-yield topics. The study guide gives explicit slide ranges (below).*

> **Study-guide scope (Puccinelli):**
> - **L3 Encoders:** slides 2–15, 17–36 (skip 16, 37–40)
> - **L4 Classification:** slides 18–26 (esp. perf-eval example) + 36–41; skip 2–17 and 27–35 (LogReg/SVM)
> - **L5 Tagging & Parsing:** whole deck; **slide 33 = UAS/LAS** is critical
> - **L7 RLHF:** whole deck; quantitative examples on 31–32 and 47–48
> - **L8 DPO:** whole deck; slides 20–21, 29–31 especially
> - **L10 Benchmarking:** whole deck except 4–8, 31–33, 35–36
> - **L14 Text-to-Speech:** 49–51 relevant; skipped 9, 11, 18, 42–48

---

## Lecture 3 — Encoders / BERT

### Core ideas
- **Foundation model:** large net, **self-supervised** pretraining on huge corpus, reused as a backbone; train once, adapt many times; mostly Transformer-based.
- **Embedding evolution:** one-hot (no similarity) → **word2vec** static dense → **contextual** (BERT): the vector for "bank" depends on the sentence.
- **Encoder** = function `f_θ : R^(n×d) → R^(n×d)`. Pipeline: text → tokens → embeddings → contextual vectors. One vector per token (+ optional global vector).
- **Self-attention (DB analogy):** each token emits Query, Key, Value.
  - similarity `QKᵀ`; weights `softmax(QKᵀ/√d_k)`; output = weighted sum of Values.
  - `Attention(Q,K,V) = softmax(QKᵀ/√d_k) V`. The `√d_k` is for **numerical stability** (keeps softmax from saturating).
- **Multi-head attention:** `Q=XW_Q, K=XW_K, V=XW_V`; split into `A` heads of size `d_h=d/A`; concat + project `W_O`. BERT-Base: `d=768, A=12, d_h=64`.
- **FFN:** `FFN(x)=W₂·σ(W₁x+b₁)+b₂`, expands `768→3072→768` (d→4d→d).
- **Residual** `y = x + Sublayer(x)`; **LayerNorm** `γ (h−μ)/σ + β`.
- **BERT:** encoder-only, deep **bidirectional** context.
  - **MLM pretraining:** mask **15%** of tokens, predict them.
  - **NSP:** `[CLS] A [SEP] B [SEP]` → predict if B follows A (RoBERTa later showed NSP adds little).
- **Fine-tuning for classification:**
  1. tokenize, prepend `[CLS]`; 2. run BERT; 3. take final `[CLS]` vector `h_[CLS]`;
  4. head `ŷ = softmax(W·h_[CLS] + b)`, `W∈R^(K×d)`; 5. predict `argmax_k ŷ_k`.
- **Pooling alternatives** to [CLS]: mean `s=(1/n)Σ h_i`, max `s_j=max_i h_{i,j}`.
- **Problem with BERT for similarity:** sentence pairs are processed jointly → comparing N sentences needs `N(N−1)/2` forward passes (O(N²)); also **anisotropy** (embeddings clustered in a cone → cosine unreliable).
- **Sentence-BERT (SBERT):** bi-encoder, maps each sentence to a vector independently; similarity = `cos(f(s₁), f(s₂))`; embeddings **precomputable** → fast search. Trained with e.g. **triplet loss** `L=max(0, d(a,p)−d(a,n)+m)`.
- **Cross-encoder vs bi-encoder:** cross-encoder `[CLS] s1 [SEP] s2` = deep interaction, accurate, **slow** (rerun every pair); bi-encoder = encode separately, **fast** cosine search.

### Exam focus
- static vs contextual; encoder vs decoder; cross- vs bi-encoder.
- attention formula + why `√d_k`. MLM rate 15%. BERT-Base 768/12/64.
- BERT fine-tuning steps & role of `[CLS]`. Why SBERT fixes O(N²).

---

## Lecture 4 — Text Classification (⭐ very high yield: compute metrics)

### Definitions (memorize)
For class `c`:
- **Precision** `P_c = TP/(TP+FP)` — of what I predicted as c, how much was right.
- **Recall** `R_c = TP/(TP+FN)` — of the true c, how much I found.
- **F1** `F1_c = 2·P_c·R_c/(P_c+R_c)` (harmonic mean; only defined when both P,R defined).
- **Accuracy** `= (#correct)/(N) = Σ_c TP_c / N`.
- **Confusion matrix:** rows = true (GT), cols = predicted; diagonal = correct.

**Undefined cases (important on this exam):**
- No predictions for c → `TP+FP=0` → **Precision = N/A (0/0)**.
- No true instances of c → `TP+FN=0` → **Recall = N/A**.
- If P or R is N/A → **F1 = N/A**.

**Averaging:**
- **Macro-F1** `= (1/C) Σ_c F1_c` (every class equal — penalizes bad rare-class performance).
- **Weighted-F1** `= Σ_c n_c·F1_c / Σ_c n_c`, `n_c` = #true instances (support) of c (follows class frequency).
- **Micro-F1** = pool all TP/FP/FN, then one P,R,F1 (≈ accuracy in multiclass; favors frequent classes).

### Worked example (the "Swiss cities" style — exactly the exam format)
From `Worked_Examples/Classification_Evaluation_Example.pdf`. Classes {CH, FR, DE}. Columns: GT (ground truth) vs CLA (classifier).

**Example 1** (5 sentences): correct on CH#1, FR#5; wrong on #2 (FR→DE), #3 (DE→CH), #4 (CH→FR).
- Per class: `P_CH=1/2, P_FR=1/2, P_DE=0`; `R_CH=1/2, R_FR=1/2, R_DE=0`.
- `F1_CH=1/2, F1_FR=1/2, F1_DE=0`.
- **Macro-F1** `=(1/2+1/2+0)/3 = 1/3`.
- **Weighted-F1** `=(½·2 + ½·2 + 0·1)/5 = 0.4` (supports: CH=2, FR=2, DE=1).

**Example 2** (classifier predicts CH for almost everything):
- `F1_CH=1/3, F1_FR=0, F1_DE=0` → **Macro-F1 = 1/9**.
- **Weighted-F1** `=(⅓·2 + 0·2 + 0·1)/5 = 2/15`.
- `P_CH=1/4, P_FR=0, P_DE=0/0 (N/A)`; `R_CH=1/2, R_FR=0, R_DE=0`. Since P_DE undefined, F1_DE technically undefined too.

**Method to memorize for the exam:**
1. Build the confusion matrix (or just count TP/FP/FN per class from the table).
2. Per class compute P, R, then F1.
3. Macro = simple average of F1s; Weighted = average weighted by support `n_c`.
4. Watch for 0/0 → write **N/A** and say why.

### Slides 36–41
- Progression legacy-ML (hand features) → RNN (sequential, long-range issues) → Transformers (learned representations + attention). Sentiment analysis = classification with pos/neg/neutral labels. Binary vs multiclass vs **multilabel**.

---

## Lecture 5 — Tagging & Parsing (⭐ UAS/LAS + transition parsing)

### POS tagging
- Why: ambiguity ("light" = adj/noun/verb), context, bridge to syntax.
- Baselines: dictionary lookup ≈ **85%** (types); most-frequent-tag ≈ **90%** (tokens); SOTA ≈ **97%**; human ceiling ≈ **98%**.
- Sequence classification: find most probable tag sequence for the words.

### Hidden Markov Model (HMM)
- States = tags (hidden), observations = words. Two assumptions:
  1. **Markov** (transition): `P(t_i | t_1..t_{i-1}) ≈ P(t_i | t_{i-1})`.
  2. **Emission:** `P(w_i | all) = P(w_i | t_i)`.
- Tagging via Bayes: `t̂ = argmax_t P(t|w) = argmax_t P(w|t)P(t)` (drop `P(w)`, constant).
- **Viterbi** = dynamic programming to find best tag path. Recurrence:
  `V_{i,t} = max_{t'} V_{i-1,t'} · P(t|t') · P(w_i|t)`. Complexity `O(n·|T|²)`.

### Dependency parsing (arc-standard, transition-based)
- Dependencies = labeled, **asymmetric** head→dependent relations. Root = main verb usually.
- **Configuration:** Stack, Buffer, set of Arcs. Three actions:
  - **SHIFT:** move buffer front onto stack.
  - **LEFT-ARC(r):** top of stack is head of the **second-top**; add arc head→2nd-top labeled r; remove the second-top (dependent). (i.e. second-top depends on top)
  - **RIGHT-ARC(r):** second-top is head of **top**; add arc; remove top.
- Worked example in `Worked_Examples/Dependency_Parsing_Example.pdf`: "Economic news had little effect on financial markets." → SHIFT/LEFT-ARC/RIGHT-ARC sequence producing labels ATT, SBJ, ATT, ATT, POBJ, PREP, OBJ, PU, PRED. **Be able to step through stack/buffer.**
- Neural parser: classify each configuration → which transition (+label). Loss = cross-entropy over actions.

### UAS / LAS (⭐ slide 33 — guaranteed-style question)
- **UAS (Unlabeled Attachment Score)** = (# tokens with the **correct head**) / (# tokens). Label ignored.
- **LAS (Labeled Attachment Score)** = (# tokens with **correct head AND correct dependency label**) / (# tokens).
- Always **LAS ≤ UAS**.
- Example: gold (head=3, nsubj), pred (head=3, dobj) → counts for **UAS** but **not LAS**.
- To compute: for each token compare predicted head (and label) to gold; count matches; divide by number of tokens (root usually excluded; punctuation sometimes excluded).

---

## Lecture 10 — Benchmarking

### Concepts
- **Benchmark** = dataset(s) + **metric** for a task/ability; lets us compare systems.
- **GLUE (2018):** multi-task NLU. Task→metric:
  - CoLA → **Matthews corr**; SST-2 → accuracy; MRPC → acc/F1; STS-B → **Pearson/Spearman**; QQP → acc/F1; MNLI/QNLI/RTE/WNLI → accuracy.
- **NLI:** premise+hypothesis → entailment / contradiction / neutral. RTE = binary entailment. WNLI = Winograd pronoun resolution.

### Matthews Correlation Coefficient (MCC) — compute
`MCC = (TP·TN − FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]`
- Range −1…+1; uses **all four** confusion cells (F1 ignores TN).
- Example: 90 acceptable / 10 not, classifier always says "acceptable": TP=90,TN=0,FP=10,FN=0 → **Accuracy=0.90 but MCC=0** (no real skill). Great "accuracy is misleading" example.

### Spearman rank correlation — compute
`ρ = 1 − 6·Σ d_i² / [n(n²−1)]`, where `d_i = R_i − S_i` (rank differences).
- Example: n=5, ranks reversed R=[1..5], S=[5..1] → d²=[16,4,0,4,16], Σ=40 → `ρ = 1 − 240/120 = −1`.

### Benchmark evolution & issues
- GLUE → **SuperGLUE** (harder) → **HellaSwag** (adversarial commonsense) → **MMLU** (57 subjects, knowledge+reasoning, zero/few-shot) → **MMMU** (multimodal expert) → **HLE**.
- **Test-set contamination:** model may have seen benchmark data online → inflated scores.
- **LiveBench (2025):** frequently updated, **objective ground-truth** scoring (no LLM judge), rolling.
- **Hallucination types:** **Factuality** (false w.r.t. world) vs **Faithfulness** (inconsistent with prompt/context/its own reasoning).

---

## Lecture 14 — Text-to-Speech (focus 49–51)

### Pipeline (two stages, trained separately)
1. **Text → mel-spectrogram** (e.g. **Tacotron 2**: char embeddings → conv → biLSTM encoder → attention → autoregressive decoder outputs 80-dim log-mel frames + stop token).
2. **Mel-spectrogram → waveform** = **vocoder** (e.g. **WaveNet**: autoregressive per-sample `P(x_t|x_<t)`, dilated causal convs, μ-law → 256 classes).
- TTS usually **speaker-dependent** (single voice); ASR is speaker-independent. Needs **text normalization** (numbers/dates → words).
- Useful constants: mel `m=2595·log10(1+f/700)`; μ-law `F(x)=sgn(x)·ln(1+μ|x|)/ln(1+μ)`, μ=255.

### Evaluation (⭐ slides 49–51)
- **MOS (Mean Opinion Score):** humans rate naturalness **1 (bad) → 5 (excellent)**; MOS = mean of ratings. Compare systems on same sentences + significance test.
- **A/B testing:** same sentence, system A vs B, listeners pick better, tally over many sentences.
- **No widely accepted automatic TTS metric** → evaluation is human-centered.

---

## ⭐ Top formulas to have cold
1. `P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2PR/(P+R)`; macro vs weighted vs micro; N/A when 0/0.
2. `UAS = correct heads / tokens`; `LAS = correct (head+label) / tokens`; LAS ≤ UAS.
3. `Attention = softmax(QKᵀ/√d_k)V`; BERT head `softmax(W h_[CLS]+b)`.
4. `MCC = (TP·TN−FP·FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]`.
5. `Spearman ρ = 1 − 6Σd²/[n(n²−1)]`.
6. Viterbi `V_{i,t}=max_{t'} V_{i-1,t'}·P(t|t')·P(w_i|t)`.
7. MOS = mean human naturalness (1–5).
