# Advanced NLP — Lab Guide (what each notebook does)

Quick index for studying. **Exam priority** is low for most labs — but a few reinforce high-yield exam topics (classification metrics, DPO, RAG, PoS/parsing intuition).

| Lab | Folder | Graded? | Exam relevance |
|-----|--------|---------|----------------|
| 1 Tokenization | `Lab01_Tokenization/` | Yes | Low — BPE, Zipf (L1) |
| 4 Classification | `Lab04_Classification/` | Yes (quiz) | **Medium** — P/R/F1, TF-IDF vs BERT |
| 5 PoS (A+B) | `Lab05_PoS/` | No | Low — tagsets, tagger comparison |
| 6 LoRA / emotion | `Lab06_LoRA_Emotion/` | No | Low — LoRA params (L6) |
| 7 BERT MLM | `Lab07_BERT_MLM/` | No | Low — MLM, contextual embeddings |
| 8 DPO geography | `Lab08_DPO_Geography/` | Quiz prep | **High** — DPO loss, policy vs reference |
| 11 RAG | `Lab11_RAG/` | Yes | Medium — Recall@N, bi- vs cross-encoder |
| 12 Agentic AI | `Lab12_Agentic/` | No | Low — ReAct, tools, planner |
| 14 TTS | `Lab14_TTS/` | No | Low — Tacotron + vocoder pipeline |
| — | `Lectures/Worked_Examples/Classification_Evaluation_Notebook.ipynb` | — | **High** — Swiss-cities P/R/F1 |

Each notebook has a **“Study takeaway”** cell at the top with a short walkthrough.

---

## Lab 1 — Text Tokenization (`Lab01_Tokenization/MSE_AdvNLP_Lab1_Kandiah_Gil.ipynb`)

**What it is:** Hands-on tokenization with NLTK and BPE (SentencePiece). Compare a normal book (Pride and Prejudice from Gutenberg) with the Voynich manuscript.

**Walkthrough:**
1. Download & clean text (strip Gutenberg header/footer).
2. NLTK: sentence tokenization → word tokenization; count **types** (unique) vs **tokens** (total); **TTR** = types/tokens.
3. Plot **token frequency** → Zipf's law (log-log straight line for natural language).
4. Repeat for Voynich — does it look like natural language?
5. Train **BPE** on book sentences and Voynich; inspect learned merges/vocab.
6. Optional: step through Karpathy-style BPE implementation.

**Key takeaways:**
- **Type** = unique word form; **token** = every occurrence. TTR drops as corpus grows.
- **Zipf's law:** few words very frequent, long tail of rare words — natural text follows this; Voynich may not.
- **BPE:** iteratively merge most frequent character pairs → subword vocab handles OOV.
- Connects to **Lecture 1** (BPE, WordPiece, subword tokenization).

**Time:** ~1–2 h if you skim executed outputs; longer if you rerun BPE training.

---

## Lab 4 — Text Classification (`Lab04_Classification/AdvNLP_Classification_public.ipynb`)

**What it is:** Two-part lab: (A) classical ML classifiers on a text dataset with **TF-IDF** features; (B) **BERT fine-tuning** on IMDB sentiment with HuggingFace `Trainer`.

**Walkthrough:**
1. Load dataset → train/test split → **TF-IDF** vectorization.
2. Train several sklearn classifiers (LogReg, SVM, …) → `classification_report` (precision, recall, F1 per class).
3. Switch to **BERT**: tokenize with `bert-base-uncased`, fine-tune `bert-large` for 2-class sentiment.
4. `compute_metrics` returns weighted precision/recall/F1 + accuracy.
5. Optional: Sentence-BERT adaptation mentioned at end.

**Key takeaways:**
- See **precision / recall / F1** in sklearn's `classification_report` — same metrics as the exam Swiss-cities example.
- **TF-IDF + LogReg** = strong baseline; **BERT fine-tune** = encoder + classification head on `[CLS]`.
- **Weighted F1** in `compute_metrics` — know macro vs weighted (exam!).
- Skim only if short on time; do the **Worked Example notebook** instead for exam calcs.

**Time:** Part A ~30 min read; Part B needs GPU (~30 min+ training).

---

## Lab 5A — Understanding PoS Tags (`Lab05_PoS/AdvNLP_Lab_a_PoS_tags.ipynb`)

**What it is:** Explore Penn Treebank / Brown corpus tag distributions — how ambiguous is PoS tagging?

**Walkthrough:**
1. Load Brown + Treebank corpora with NLTK.
2. Implement `get_ground_truth_distribution(token)` — for a word, count which tags it gets across the corpus.
3. See that many words have **multiple possible tags** (e.g. "light" = noun/verb/adj) → tagging is a **sequence classification** problem, not lookup.

**Key takeaways:**
- Dictionary lookup ≈ 85% accuracy; context matters.
- **Tag ambiguity** is why we need sequence models (HMM, neural taggers).
- Connects to **Lecture 5** (HMM, Viterbi, baselines 85/90/97/98%).

**Time:** ~20 min.

---

## Lab 5B — Using PoS Taggers (`Lab05_PoS/AdvNLP_Lab_b_PoS_basics.ipynb`)

**What it is:** Compare three taggers on the same sentences: **NLTK Perceptron**, **Stanza**, **spaCy**.

**Walkthrough:**
1. Install/load all three taggers.
2. Tag the same example sentences with each.
3. Compare outputs — where do they disagree? (e.g. "Book" as verb vs noun depending on context.)

**Key takeaways:**
- Different taggers use different tagsets and models; agreement is not 100%.
- Modern taggers (Stanza, spaCy) use neural models; NLTK perceptron is a classic baseline.
- Low exam priority — skim for intuition.

**Time:** ~20 min.

---

## Lab 6 — LoRA Emotion Classification (`Lab06_LoRA_Emotion/MSE_AdvNLP_Lab6.ipynb`)

**What it is:** Fine-tune **GPT-2 (small)** for 6-way **emotion classification** using **LoRA** (PEFT library). Not graded — template lab you fill in yourself.

**Walkthrough:**
1. Load `dair-ai/emotion` dataset (sadness, joy, love, anger, fear, surprise); balance classes if subsampling.
2. Load GPT-2 for sequence classification; wrap with **LoRA** (`r`, `alpha`, target modules).
3. Train classifier head + LoRA adapters; evaluate on test set.
4. Export model.

**Key takeaways:**
- **LoRA:** freeze full weights `W`, train low-rank `A,B` → `h = x(W + AB)`; params `r(d+k)` vs `d·k`.
- **PEFT** = Parameter-Efficient Fine-Tuning; same idea as QLoRA in lectures.
- GPT-2 is **decoder-only** used here as classifier (not typical today, but illustrates fine-tuning).
- Connects to **Lecture 6** (LoRA, instruction tuning).

**Time:** Read structure ~20 min; running needs GPU.

---

## Lab 7 — BERT Masked Language Model (`Lab07_BERT_MLM/AdvNLP_BERTlab.ipynb`)

**What it is:** Play with pre-trained **BERT** (no training). Predict masked words in English and German; run a **cloze test**.

**Walkthrough:**
1. Load `bert-large-uncased` (English) or multilingual BERT (German).
2. `ask_BERT(sentence_with_[MASK])` → top-k predicted tokens with probabilities.
3. English examples: syntactic/semantic plausibility of fill-ins.
4. Switch to **multilingual BERT** for German gender/case agreement examples.
5. Cloze test: multiple-choice blanks using BERT scores.

**Key takeaways:**
- BERT is trained with **MLM** — predict masked token using **bidirectional context**.
- Same word, different contexts → different predictions = **contextual embeddings**.
- Multilingual BERT shares subword vocab across languages.
- Connects to **Lecture 3** (Encoders, MLM 15%, fine-tuning). Slides in `BertLab.pdf`.

**Time:** ~30 min interactive; no GPU strictly needed.

---

## Lab 8 — DPO Geography (`Lab08_DPO_Geography/ADV_NLP_DPO_Geo(2).ipynb`) ⭐

**What it is:** End-to-end **DPO training** demo. Teach an instruction-tuned model to prefer **concise** geography answers (`Paris`) over verbose ones (`The capital of France is Paris.`). **Look at this before the alignment quiz.**

**Walkthrough:**
1. Build synthetic **preference dataset** from country facts (capital, continent, yes/no, multiple choice).
2. Each example: `(prompt, chosen=concise, rejected=verbose)`.
3. Create **policy** and frozen **reference** model (same base checkpoint).
4. Implement **DPO loss:** `-log σ(β log(π_θ(y_w)/π_ref(y_w)) − β log(π_θ(y_l)/π_ref(y_l)))`.
5. Mask prompt tokens — loss only on **answer** tokens.
6. Train; compare generations **before vs after** on held-out geography questions.

**Key takeaways:**
- DPO needs **no separate reward model** — directly optimizes policy from preference pairs.
- **Reference model** stays frozen; policy learns to increase relative prob of preferred vs rejected.
- **β** controls strength of preference signal (here ~0.1).
- This is the **practical counterpart** to the 13-step derivation in **Lecture 8** — read `Study/notes/notes_alignment_RLHF_DPO.md` alongside.
- GPU strongly recommended.

**Time:** 1–2 h to read; training adds time on GPU.

---

## Lab 11 — RAG Retrieval (`Lab11_RAG/MSE_AdvNLP_Lab11_students.ipynb`)

**What it is:** Build and evaluate the **retrieval** part of RAG. Compare **FAISS bi-encoder** search vs **cross-encoder reranking**. Solution in `solution/MSE_AdvNLP_Lab11_Kandiah_Gil_V2.ipynb`.

**Walkthrough:**
1. Load `docs.csv` (knowledge chunks) and `queries.csv` (questions + ground-truth doc IDs).
2. Implement **Recall@N** = (# relevant docs in top-N) / (# total relevant docs).
3. Embed documents with **HuggingFace embeddings** → **FAISS** vector store.
4. For each query, retrieve top-N; compute mean Recall@N for N ∈ {1,3,5,25}.
5. Add **cross-encoder reranker** (`BAAI/bge-reranker-base`) on top-25 candidates; recompute Recall@N.
6. Write two **generation prompts** (with/without context) and test on an LLM.

**Key takeaways:**
- **Recall@25** same before/after rerank (same candidate set); **Recall@1/3/5** improve — reranker puts correct doc higher.
- **Bi-encoder** = fast retrieval (independent embeddings); **cross-encoder** = slow but accurate (joint query-doc encoding).
- RAG pipeline: chunk → embed → index → retrieve → augment prompt → generate.
- Connects to **Lecture 11** (RAG, cosine, MRR, NDCG).

**Time:** ~1 h read + implementation; use solution if stuck.

---

## Lab 12 — Agentic AI (`Lab12_Agentic/`)

Two notebooks:

### `M5_UGL_1.ipynb` — Customer service multi-agent workflow
**Walkthrough:** Retail scenario (returns, purchases). **Planner** breaks request into steps → **Reflection Agent** validates → **Executor** calls tools (inventory lookup, stock update). Validation hooks prevent negative stock.

**Takeaways:** Multi-agent orchestration; **tools-only registry**; validation before DB writes; evaluate **full transcript** not just final answer.

### `Demo Agentic AI.ipynb` — ReAct with Gemini
**Walkthrough:**
1. Phase 1: Plain LLM fails at multi-step task (currency conversion for share purchase).
2. Phase 2: Define **tools** (get share price, convert currency).
3. Phase 3: Manual **ReAct loop** — Thought → Action → Observation until done.
4. Phase 4: Same with native SDK orchestration.

**Takeaways:** **ReAct** = Thought → Action → Observation; agents **act** not just answer; needs API key (Gemini). Connects to **Lectures 9 & 12**.

**Time:** ~45 min read; M5 lab needs OpenAI API key.

---

## Lab 14 — Text-to-Speech (`Lab14_TTS/AdvNLP_TTS.ipynb`)

**What it is:** Generate speech with **Coqui TTS** — loads a **Tacotron2** acoustic model + **HiFi-GAN vocoder**, outputs `output.wav`.

**Walkthrough:**
1. Install `coqui-tts`.
2. List available TTS/vocoder models.
3. Run TTS CLI: text → mel spectrogram (Tacotron2) → waveform (vocoder).
4. Listen to `output.wav`.

**Key takeaways:**
- TTS = **two stages:** text→**mel spectrogram** (acoustic model) → mel→**waveform** (**vocoder**).
- Tacotron2 ≈ lecture example; HiFi-GAN / WaveGrad = vocoder family.
- Evaluation in lecture = **MOS** (human 1–5) + **A/B testing** — no good automatic metric.
- Connects to **Lecture 14** (Puccinelli, slides 49–51).

**Time:** ~20 min if you just read; Colab GPU helps for first model download.

---

## Worked Example — Classification Evaluation (not in Labs/)

**Path:** `Lectures/Worked_Examples/Classification_Evaluation_Notebook.ipynb`

**What it is:** The **Swiss-cities** style example — computes precision, recall, F1, macro/weighted from GT vs classifier columns. Plots confusion matrices.

**Why do this:** Direct exam prep for Puccinelli's "compute scores" questions. **Do this before the exam** even if you skip other labs.

---

## Suggested lab order for exam prep (minimal time)

1. **Classification Evaluation notebook** (30 min) — exam calcs
2. **Lab 8 DPO** (read takeaway + skim loss/training cells) — aligns with biggest exam topic
3. **Lab 11 RAG** (read Recall@N + reranker comparison) — medium yield
4. **Lab 1** (Zipf + BPE section only) — quick L1 refresh
5. Skip or skim: PoS, BERT MLM, LoRA, TTS, Agentic — unless you have extra time

---

## Folder reference

```
Labs/
├── 00_LAB_GUIDE.md          ← this file
├── Lab01_Tokenization/
├── Lab04_Classification/
├── Lab05_PoS/
├── Lab06_LoRA_Emotion/
├── Lab07_BERT_MLM/
├── Lab08_DPO_Geography/
├── Lab11_RAG/               (+ solution/)
├── Lab12_Agentic/
├── Lab14_TTS/
└── Reference/               (installation PDF)
```
