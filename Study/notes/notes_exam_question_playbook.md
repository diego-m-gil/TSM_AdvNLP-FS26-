# Exam Question Playbook (synthesized from sample exams + class quizzes)

> Built from official **sample exam questions (2023 & 2024)**, a student Q&A compilation, and the **in-class Kahoot quizzes**. The earlier course (then called *AnTeDe*) leaned on classic NLP; this year adds RAG/agents/alignment. Topics below recur almost identically year to year — **learn the patterns, not single answers.** Everything here is also condensed into the cheat sheet.

---

## A. Recurring written questions (and how to answer them)

### 1. Preprocessing (almost always Q1)
- **Tokenizer / lemmatizer output**: list tokens; lemmatizer needs PoS (`ate→eat`, `smaller→small`, `better→good`).
- **Tokens vs types**: tokens = every occurrence; types = distinct tokens. *"The big dog ate my exam questions and the smaller dog ate my homework questions"* → **15 tokens, 10 types**.
- **Stopword removal**: drop function words → keeps content words. **Trap**: removing the negation *"not"* flips sentiment to positive. Lowercasing *"May"→"may"* loses the proper-noun/month (hurts PoS & NER).

### 2. Text classification + smoothing
- `P(Spreitenbach | Aargau) > P(Spreitenbach | Thurgau)` because the token appears in Aargau (>0) and never in Thurgau (=0).
- **Laplace / additive smoothing**: add `α` to each count and `α|V|` to the denominator → no zero probabilities, non-zero probs barely change. `α=1` ⇒ Laplace. One zero otherwise kills the whole Naive-Bayes product. Use **log probs** to avoid underflow.

### 3. Word embeddings
- **Distributional semantics**: *"You shall know a word by the company it keeps"* (Firth) — meaning from context.
- **Skip-gram**: logistic/sigmoid turns the dot product of (center, context) embeddings into a probability; objective maximizes log-prob of context given center. Small window → syntactic; large → semantic. **Negative sampling** updates a few negatives instead of full-vocab softmax.
- **Word2Vec vs GloVe vs fastText** (frequent true/false):
  - Word2Vec = local context window, predictive, single static vector, **fails OOV**.
  - GloVe = **count-based**, global co-occurrence matrix factorization (NOT predictive, NO negative sampling).
  - fastText = **sum of char n-grams** → handles OOV / typos / morphology.
  - All three are **static** (one vector per word) → can't disambiguate polysemy (*bank*); contextual models (BERT) can.
- **WSD with WordNet + embeddings**: average the embeddings of each sense's definition words, average the sentence's words, compare by **cosine**, pick the max.

### 4. PoS tagging / HMM / CRF
- HMM↔PoS: **hidden states = tags**, **transitions = P(tᵢ|tᵢ₋₁)**, **emissions = P(wᵢ|tᵢ)**, **observations = words**.
- Assumptions: **Markov** (tag depends only on previous tag) + **output independence** (word depends only on its tag).
- **Viterbi** = dynamic programming for the most likely tag sequence.
- Unseen word in test ⇒ emission probability 0 (needs smoothing).
- *"that"* vs *"dolphin"*: **that** is harder — many possible tags; dolphin is only a noun.
- **CRF**: globally normalized, models features over the whole sequence, **less affected by label bias** than MEMM.

### 5. Language models / metrics
- **n-gram LM**: predict a word from the previous n−1 words (Markov); chain rule of probabilities. Limits: no long-range dependencies, sparsity → smoothing/backoff. Log probs for numerical stability.
- **BLEU** = n-gram **precision** (clipped) + brevity penalty → fluency/exact phrasing (MT). **ROUGE** = **recall** → content coverage (summarization).

### 6. RNN / LSTM
- **Vanishing gradient**: gradients shrink exponentially back through time (repeated small Jacobians from sigmoid/tanh) → can't learn long-range dependencies. Exploding = opposite (unstable).
- **LSTM/GRU** fix it with gates (LSTM: forget/input/output + cell state; GRU merges input+forget into update gate). Also: ReLU, gradient clipping, residual connections.

### 7. Seq2seq + attention
- **Encoder** processes source → **context** (its final hidden state) → **decoder** generates target.
- **Attention** lets the decoder see *all* encoder states (weighted sum = context vector per step) → fixes the single-vector bottleneck on long sentences. **Query** (decoder) vs **Keys** (encoder) → scores → softmax weights → weighted sum of **Values**. Global vs local attention.

### 8. Syntax
- **Constituency** = phrase-structure tree (S→NP VP; NP/VP/PP); **dependency** = head→dependent arcs.
- **PCFG**: `P(S→NP VP)=0.8` ⇒ 80% of sentences expand as NP+VP; `P(PP→P NP)=1.0` ⇒ only rule for PP.
- **Attachment ambiguity**: *"counts whales from space"* / *"saw the man with the binoculars"* — the PP attaches to the verb, but a parser may attach it to the noun.
- **UAS/LAS**: 5-token sentence, all heads correct but 1 mislabeled ⇒ **UAS = 100%**, **LAS = 4/5 = 80%**. Always **LAS ≤ UAS**.

### 9. Information extraction / NER
- **IO scheme problem**: adjacent same-type entities merge (*"I gave Nancy Ken Follett's novel"* → Nancy Ken Follett looks like one PER). **IOB fix**: `B-` starts each entity, `I-` continues → Nancy=B-PER, Ken=B-PER, Follett=I-PER.
- **CRF scores → probabilities** via **softmax**: `[1000,1,2,-30] → ≈[1,0,0,0]`.
- **Label bias** = locally-normalized models favor low-entropy states (few outgoing transitions) regardless of the input. *Select-all*: **lower-entropy states** and **states with fewer outgoing transitions**.

### 10. Transformers / BERT (heavy emphasis — see also §B)
- **Winograd schema** (coreference): *"councilors refused the demonstrators a permit because they feared/advocated violence"* — "they" depends on the verb; needs **attention / contextual models**.
- **Positional encoding**: attention is **permutation-invariant** (inner products, no order) → must inject position (sine/cosine). RNNs don't need it (order is in the hidden state).
- **BERT's two pretraining tasks**: **Masked LM** (mask 15%, predict via softmax) + **Next-Sentence Prediction**.
- Why BERT is great at coreference / grammar / WSD: **transformer + deep bidirectional attention** (left & right context).
- Param count ≈ `(V×H) + 12·L·H²` (token embeddings + transformer blocks).

### 11. Speech (ASR/TTS)
- **Pitch** ≈ log(frequency Hz); **loudness** ≈ log(intensity), dB. **Spectrogram** = |STFT| over time.
- **MFCC**: apply a **mel filterbank** to the power spectrum → **log** energy per bin → **DCT** → spectral envelope.
- **TTS** = (1) text → mel-spectrogram (**Tacotron 2**: char emb→conv→biLSTM→attention→AR decoder, 80-dim log-mel + stop token) → (2) mel → waveform (**vocoder**, e.g. WaveNet). TTS is speaker-dependent; ASR speaker-independent. Eval = **MOS (1–5)** + A/B tests.

### 12. RAG / alignment (this year's additions)
- **RAG**: retrieve relevant chunks from a vector DB → augment the prompt → generate. Fixes stale/parametric knowledge & hallucination without retraining. Components: embedder, retriever, (reranker), generator. Retriever metrics: precision/recall/F1, MAP, **NDCG = DCG/IDCG**.
- **RLHF**: prompts → SFT → human preferences → reward model (Bradley-Terry) → PPO. **DPO** does the same directly from preference pairs (no reward model, no RL); partition function `Z(x)` cancels because Bradley-Terry uses reward *differences*. (Full 13-step derivation + Puccinelli Q&A in `notes_alignment_RLHF_DPO.md` and the cheat sheet.)
- **KL divergence** `D_KL(P‖Q)=Σ P log(P/Q)` ≥ 0, not symmetric. **Cross-entropy** `H(p,q) = -Σ p log q = H(p) + D_KL(p‖q)`; CE loss with one-hot truth = `-log q_true`.

---

## B. In-class Kahoot quiz — encoders / BERT (correct answers ticked)

These 15 came up in a live class quiz this year and map directly to the Encoders/BERT block of the cheat sheet.

| # | Question | Correct answer |
|---|----------|----------------|
| 1 | Core mechanism inside transformer encoders | **Self-attention** |
| 2 | What the feed-forward network does | **Processes each token independently with nonlinear transformations** |
| 3 | Why divide attention scores by √d | **To stabilize the magnitude of dot products** |
| 4 | Main goal of a foundation model | **Learn general language representations from large data** |
| 5 | To make sense of the word *interest* … | **Contextual embeddings must be used** |
| 6 | What an encoder's output represents | **One contextualized vector for each token** |
| 7 | Purpose of multi-head attention | **To allow the model to learn different relationships between tokens** |
| 8 | Why ML models need word embeddings | **Neural networks require numerical inputs** |
| 9 | Why transformers need positional embeddings | **Because attention is permutation-invariant** |
| 10 | What determines how strongly one token attends to another | **The dot product between queries and keys** |
| 11 | Role of the `[CLS]` token in BERT classification | **It provides a summary representation of the input sequence** |
| 12 | What the Query vector represents | **Information the token wants to retrieve** |
| 13 | Key limitation of one-hot vectors | **They do not encode similarity between words** |
| 14 | What softmax does in attention | **Normalizes attention scores into weights** |
| 15 | How BERT is primarily trained during pretraining | **Predict masked words in a sentence** |

---

## C. Strategy takeaways

- **Q1 is reliably preprocessing** (tokens/types, lemmatization, stopword traps) — free points, nail it.
- **Always-computable**: P/R/F1 (macro & weighted), UAS/LAS, KL in bits, reward/DPO sigmoid-NLL, MCC, Spearman, WER, perplexity, cosine, NDCG. Practice these (`02_practice_problems.md`).
- **Definition/short-answer favorites**: Laplace smoothing, label bias, attachment ambiguity, IO vs IOB, positional encoding, BERT's two tasks, vanishing gradient, attention Q/K/V.
- **Multiple choice** clusters on transformers/BERT — the Kahoot table above is your answer key.
