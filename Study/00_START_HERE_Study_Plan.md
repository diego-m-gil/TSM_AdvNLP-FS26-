# Advanced NLP — Final Exam Study Plan & Guide

**Exam:** Thu **02.07.2026, 14:00–16:00** (120 min), HWZ Sihlhof, Lagerstrasse 5 (rooms 402/403).
**Format:** Online **Moodle quiz** on your laptop. Questions in random order, different weights. Answer types: **text**, **numerical results of simple calculations**, **multiple choice**. You can navigate/skip/return.
**Allowed:** 2 A4 sheets (front+back = **4 pages**) of personal notes (handwritten or typed), blank scratch paper, **non-programmable calculator**, laptop+charger, student ID. Internet only for Moodle.

---

## Your goal & the math (read this first — it's motivating)
- You have a **5.5 pregrade** that counts **25%**. You need a combined **4.0** to pass.
- Required final-exam grade X:  `0.25·5.5 + 0.75·X ≥ 4.0` → **X ≥ 3.5**.
- **You only need 3.5/6 on the final exam to pass.** (Assumes pregrade=25%, final=75%, Swiss 1–6 scale. Double-check the exact weighting in the module description — if pregrade weight is higher, you need even less.)
- **Strategy:** don't chase a top grade. Lock in the **high-yield, computable Puccinelli topics** (they're worth the most and are predictable), get the easy multiple-choice points everywhere, and you clear 3.5 comfortably.

---

## What's in this Study folder
- `00_START_HERE_Study_Plan.md` — this file (plan + logistics + priorities).
- `notes/notes_alignment_RLHF_DPO.md` — ⭐⭐ the single most important file (RLHF + DPO + the 13 steps + Puccinelli's exact Q&A).
- `notes/notes_puccinelli_core.md` — ⭐ classification metrics, parsing/UAS-LAS, encoders, benchmarking, TTS.
- `notes/notes_perruchoud_other.md` — breadth for the other lecturer's lectures.
- `notes/notes_exam_question_playbook.md` — ⭐ **NEW**: synthesized from past **sample exams** + the in-class **Kahoot quizzes**. Recurring question patterns, model answers, and the quiz answer key. Read this the day before.
- `02_practice_problems.md` — worked practice questions with solutions (do these!).
- `Cheatsheet/cheatsheet.tex` — your printable 4-page exam cheat sheet (edit + compile in Overleaf).
- `Cheatsheet/README_cheatsheet.md` — how to compile/print.

Lecture PDFs live in `../Lectures/` (numbered 01–14), exam docs in `../Lectures/Exam_Info/`, worked examples in `../Lectures/Worked_Examples/`.

---

## Exam priority map (what to actually study)
Ranked by expected return. Puccinelli gave explicit hints → treat the ⭐⭐ items as near-certain.

| Priority | Topic | Where | Type |
|---|---|---|---|
| ⭐⭐ | **Classification evaluation** (P/R/F1, macro/weighted, confusion matrix, N/A cases) | L4 + Worked Example | **Compute** |
| ⭐⭐ | **Dependency parsing + UAS/LAS** (arc-standard transitions; compute UAS/LAS) | L5 + Worked Example | **Compute** |
| ⭐⭐ | **RLHF** (reward model, Bradley-Terry, NLL loss, KL, β, PPO) | L7 | Compute + explain |
| ⭐⭐ | **DPO** (13-step derivation, partition function, β, win term, DPO loss) | L8 | **Explain + derive** |
| ⭐⭐ | **KL divergence calc** (the "distribution / ground truth / measured" travel-times example) | L7 | **Compute** |
| ⭐ | **Benchmarking** (MCC, Spearman, GLUE metrics, contamination, hallucination types) | L10 | Compute + MC |
| ⭐ | **Encoders/BERT** (attention, fine-tuning, SBERT, cross vs bi-encoder) | L3 | Explain + MC |
| ⭐ | **POS tagging / HMM / Viterbi** | L5 | Explain |
| ○ | **Text-to-Speech** (MOS, A/B, Tacotron+WaveNet) | L14 | MC / short text |
| ○ | Tokenization, perplexity, BLEU, word2vec | L1 | Compute / MC |
| ○ | MT / Transformers / attention details | L2 | MC |
| ○ | Decoding (greedy/beam/temp/top-k/top-p), LoRA | L6 | MC |
| ○ | ICL / prompting / CoT / ReAct | L9 | MC |
| ○ | RAG (cosine, RRF, NDCG, MRR, bi vs cross-encoder) | L11 | Compute / MC |
| ○ | Agentic AI (ReAct, pass@k) | L12 | MC |
| ○ | ASR (CTC, WER, wav2vec, Whisper) | L13 | Compute / MC |

**The four "you must be able to compute" skills:** (1) precision/recall/F1 + macro/weighted, (2) UAS/LAS, (3) KL divergence in bits, (4) reward-model/DPO sigmoid+NLL loss values. Puccinelli said so almost verbatim.

> **Insight from past sample exams (2023/2024):** the written part historically opens with **classic-NLP** questions — preprocessing (tokens vs types, lemmatization, stopword/negation traps), Naive-Bayes + **Laplace smoothing**, **Word2Vec/GloVe/fastText**, HMM/Viterbi, constituency/**PCFG**, **IO vs IOB** + label bias, and a big **transformers/BERT** block (positional encoding, MLM+NSP, attention Q/K/V). These are now all in the cheat sheet and in `notes_exam_question_playbook.md`. We don't know if this year matches, but the patterns are stable and cheap to learn — bank them.

---

## 5–6 day plan (≈4–5 focused hours/day)

### Day 1 — Alignment core (highest value)
- Read `notes/notes_alignment_RLHF_DPO.md` end to end. Open `Lectures/07_Alignment_RLHF` and `08_Alignment_DPO` PDFs alongside.
- Memorize: sigmoid, Bradley-Terry, reward NLL loss, KL formula, PPO reward with β.
- **Do:** the KL travel-times calculation by hand twice (P,Q given → bits). Practice problems §3.
- Write the one-line answers to Puccinelli's Q&A (β role, win term, partition function why-introduced / why-cancels / why-intractable / why-normalization-constant, one-sentence DPO).

### Day 2 — DPO derivation + Classification metrics
- Re-derive the 13 DPO steps from memory on scratch paper until you can sketch: KL-objective → tilt π_ref by exp(r/β) → Z(x) normalizes → KL≥0 gives closed form → invert to reward → BT difference cancels Z → DPO loss. (Practice problems §4.)
- Classification: read `notes_puccinelli_core.md` L4. Re-do the Swiss-cities example yourself (confusion matrix → P,R,F1 → macro vs weighted). Practice problems §1.

### Day 3 — Parsing + Benchmarking
- L5: step through the dependency-parsing example (SHIFT/LEFT-ARC/RIGHT-ARC). Nail **UAS vs LAS** and compute both from a small tree. Practice problems §2. Skim HMM/Viterbi.
- L10: MCC formula + the "accuracy 0.90 but MCC 0" example; Spearman ρ on a 5-item list; GLUE task→metric table; contamination; factuality vs faithfulness. Practice problems §5.

### Day 4 — Encoders + finalize cheat sheet
- L3: attention formula, BERT fine-tuning with [CLS], MLM 15%, SBERT, cross vs bi-encoder.
- L14: MOS, A/B testing, two-stage TTS (Tacotron 2 + WaveNet). Quick.
- **Compile and print the cheat sheet** (`Cheatsheet/`). Edit anything you want to phrase your own way — building it cements memory. Make sure it's exactly 4 pages.

### Day 5 — Perruchoud breadth + multiple-choice farming
- Read `notes_perruchoud_other.md`. You're hunting for **easy MC and short-calc points**: perplexity (lower=better), BPE vs WordPiece, temperature/top-k/top-p, BLEU (precision+BP), cosine sim, WER=(S+D+I)/N, NDCG/MRR, LoRA params, ReAct, RAG bi- vs cross-encoder.
- Re-do all practice problems under time pressure.

### Day 6 (buffer/optional) — Mock + weak spots
- Full pass of all practice problems + redo any KL/F1/UAS-LAS/Spearman calc cold.
- Glance at the lab notebooks (`../Labs/`): Lab 1 tokenization (BPE), classification eval notebook, the DPO/alignment notebook, RAG lab — just enough to recognize concepts. Labs are low priority for you; don't over-invest.
- Skim the Zoom recordings (`../Lectures/Recordings_Links/`) only for a topic that still feels fuzzy.

---

## Exam-day tactics
- **Calculator:** bring a non-programmable one. Pre-practice `log2(x)=ln x / ln 2` on it (needed for KL in bits).
- Moodle lets you skip/return: **bank all the multiple-choice and quick-calc points first**, then spend time on the longer "explain the loss / derivation" text answers.
- For "explain in a paragraph" questions (loss, β, partition function): there's a ready one-liner/paragraph for each in the alignment notes — reproduce the intuition, not just the formula.
- If a question is ambiguous, the rules say: **make an assumption and state it** — you still get credit.
- Numeric answers: show the formula + plugged-in numbers on scratch paper; for Moodle, enter the number (watch units: bits vs nats for KL → use log2 unless told otherwise).

---

## Notes / assumptions
- Grade-weight (25% pregrade) is from your message; confirm in the official module sheet. Even at higher final-exam weight the required X stays low given your 5.5.
- The UAS/LAS exact wording is from slide 33 (image-only in the PDF); the definitions here follow the standard CoNLL convention used in the course and match the worked parsing example — verify the slide visually once.
