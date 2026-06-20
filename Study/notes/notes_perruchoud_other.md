# Perruchoud / Milosevic Notes — Advanced NLP (Lectures 1,2,6,9,11,12,13)

Lower exam priority than the Puccinelli half (no study guide for these), but they can appear. Focus on the **bold** facts and the computable bits (perplexity, BLEU, cosine, top-k/top-p, NDCG, WER, CTC).

---

## L1 — Intro, Tokenization & Embeddings
- **Why subword tokenization:** word-level → huge vocab + OOV; char-level → no OOV but long seqs, weak semantics; **subword** balances both.
- **BPE:** start from chars, iteratively **merge the most frequent adjacent pair**; vocab = chars + #merges. Apply same merges at test time.
- **WordPiece:** merge the pair with best **score = P(xy)/(P(x)P(y))**; `##` marks continuations.
- **Unigram LM:** start big, **remove** tokens with least likelihood impact. **SentencePiece:** works on raw text, whitespace→`▁`.
- **n-gram LM (Markov order m):** `P(w_k|context) ≈ P(w_k | w_{k−m..k−1})`, MLE = relative counts. Zero counts → need **smoothing/backoff** (Laplace, Kneser-Ney, interpolation).
- **Perplexity (compute):** `PP(W) = P(w_1..w_N)^(−1/N) = 2^{H(W)}`. **Lower = better.** Drops with higher n-gram order (e.g. WSJ: unigram 962 → bigram 170 → trigram 109).
- **word2vec:** CBOW (predict center from context, faster) vs **Skip-gram** (predict context from center, more samples). **SGNS** = skip-gram + negative sampling (sample negatives ∝ freq^0.75). **fastText** adds char n-grams (handles rare words). Static (one vector/word).
- **BLEU (appears here & in MT):** `BLEU = BP · exp(Σ_n w_n log p_n)`, brevity penalty `BP = min(1, e^{1−r/c})`, `p_n` = **clipped** n-gram precision. Precision-based; ROUGE is recall-based.

## L2 — Machine Translation, RNNs & Transformers
- **RNN:** `h_t = g(U h_{t−1} + W x_t)`, `ŷ_t = softmax(V h_t)`; shared weights; BPTT; vanishing/exploding gradients. **LSTM:** cell state + forget/input/output gates.
- **Seq2seq bottleneck:** encoder's final state must hold the whole source → quality drops with length.
- **Attention (fixes bottleneck):** scores `e_t = [s_tᵀh_1,…,s_tᵀh_N]`, `α_t = softmax(e_t)`, context `a_t = Σ_i α_i h_i`.
- **Transformer:** no recurrence; self-attention `Z = softmax(QKᵀ/√d_k)V`; **multi-head** (8 heads, d_model=512, d_k=64); masked decoder self-attention (`−∞` above diagonal); **cross-attention** (Q from decoder, K,V from encoder); **positional encodings** (sinusoidal / RoPE) because attention is permutation-invariant; FFN + residual + LayerNorm.
- Trains in **parallel** (teacher forcing); inference is **autoregressive** (sequential). KV-cache avoids recomputing past K/V. GQA/MQA share K/V heads to shrink cache.
- **Exam trap:** attention weights for token i use **query q_i with all keys** (not key i with all queries). `√d_k` scaling = numerical stability.

## L6 — Transformer Decoders & Text Generation
- Decoder-only loop: logits → softmax → next token until EOS/max len.
- **Decoding strategies:**
  - **Greedy:** argmax each step.
  - **Beam search:** keep top-k partial hypotheses; better for **directed** tasks (MT, summarization).
  - **Temperature:** `P(x_l) = exp(u_l/t)/Σ exp(u_/t)`. `t<1` sharpens (more deterministic), `t>1` flattens (more random).
  - **Top-k:** sample among k highest-prob tokens (fixed k).
  - **Top-p (nucleus):** smallest set with cumulative prob ≥ p (adaptive size). Sampling better for **open-ended/creative**.
- **Instruction tuning:** supervised CE on (instruction, response); mask instruction tokens (ignore_index −100).
- **PEFT / LoRA:** freeze W, learn low-rank `h = x(W + AB)`, rank r≪d → `r(d+k)` params instead of `d·k`. **QLoRA** = LoRA on a 4-bit quantized frozen base. Huge memory savings.

## L9 — In-Context Learning & Prompt Engineering
- **ICL:** condition with instructions+examples, **no weight update**; emergent in large models. Zero-shot / one-shot / few-shot (2–5 examples). Improves with **model size** and **#examples** (but few-shot can bias toward examples).
- **Prompt roles:** System/Developer (persona, rules), User (task+data+format), Assistant (prior/few-shot outputs).
- **Chain-of-Thought (CoT):** "let's think step by step" → better math/logic (pattern matching, not true reasoning). **Chain-of-Draft** = minimal steps (fewer tokens). **Self-consistency** = sample multiple, majority vote. **ReAct** = Thought → Action → Observation loop (reason + tool use).
- Prompt engineering: iterate, keep simple, prefer instructions over many constraints, document (model, temp, top-p, template).

## L11 — Retrieval-Augmented Generation (RAG)
- **RAG:** retrieve external chunks → put in prompt → generate. Fixes **stale/parametric** knowledge, domain specificity, confidentiality — **without retraining**.
- Pipeline: parse → **chunk** → embed → index (vector DB) → embed query → similarity search → prompt (instruction+query+context) → generate.
- **Chunking:** fixed / recursive / structure-based / **semantic** (merge adjacent sentences while cosine sim stays high).
- **Embeddings for retrieval:** **bi-encoder / SBERT** (precompute, fast) for retrieval; **cross-encoder** for **reranking** (accurate, slow — e.g. 65h vs 5s for 10k pairs). Cosine `cos(x,y)=xᵀy/(‖x‖‖y‖)`; for normalized vectors cosine ≡ dot-product ranking ≡ L2 ranking (`‖x−y‖²=2−2cos`).
- **Sparse retrieval:** BM25 (TF-IDF + length norm). **Hybrid** = BM25 + dense fused by **RRF** `Σ_i 1/(k + rank_i(d))`.
- **ANN indexes:** brute force O(n); **IVF** (k-means cells); **HNSW** (graph, high recall).
- **Retriever metrics:** hit rate, P/R/F1, **MRR** `=(1/|Q|)Σ 1/rank_i`, **NDCG@K** `=DCG/IDCG`, `DCG@K=Σ (2^{rel_i}−1)/log2(i+1)`.
- **RAGAS:** Context Relevancy, **Faithfulness** = (#answer claims supported by context)/(#claims), Answer Relevancy (cosine of query embeddings).

## L12 — Agentic AI (Milosevic)
- **Agentic AI:** sets goals, plans, and **takes real-world actions** (vs chatbot that just responds). Single agent vs orchestrated multi-agent (Manager → Researcher/Writer/Reviewer).
- Capabilities: perception, **memory** (short = context window; long = vector DB/RAG), planning, **tool use** (search, code, APIs), reflection.
- **ReAct** = Thought → Action → Observation loop. Code execution often improves math/data tasks.
- **Evaluate on full transcripts** (tool calls + reasoning), not just final answer. **pass@k** = success over k stochastic trials. Risks: cascading failures, data leakage/over-privilege, reward hacking.

## L13 — Speech-to-Text / ASR (Perruchoud)
- ASR: audio → text; long input vs short output → **alignment problem**.
- Features: waveform → FFT → spectrogram → **log-mel** (~80 channels, 25 ms windows). `dB(I)=10·log10(I/I_min)`, I_min=1e−12 W/m².
- **wav2vec 2.0:** CNN encoder + **Transformer**, **self-supervised** contrastive learning + **product quantization** (discrete units), masked spans; fine-tune with **CTC**. Strong with little labeled data.
- **CTC:** introduces a **blank** token; many alignments → one output (collapse repeats, then remove blanks). Loss = `−log Σ_{valid alignments} Π_t p(a_t|h_t)` (forward-backward DP). Doesn't learn an LM → add LM + length penalty at decode: `score = log P_CTC(Y|X) + λ1 log P_LM(Y) + λ2 L(Y)`.
- **Whisper:** encoder-decoder Transformer, 80-ch log-mel input, **weak supervision** at scale (680k h, 99 langs), multitask (lang ID, transcribe, translate, timestamps); robust but can hallucinate/loop.
- **WER = (S + D + I)/N** (substitutions+deletions+insertions over reference words). 0 = perfect; can exceed 100%.

---

## ⭐ Computable items most likely to be quizzed (Perruchoud side)
- **Perplexity** `PP=P(W)^{−1/N}`; lower better.
- **BLEU** with clipped n-gram precision + brevity penalty.
- **Cosine similarity** / normalized → dot product.
- **Temperature / top-k / top-p** mechanics (which set, sharpen vs flatten).
- **NDCG@K**, **MRR** for retrieval.
- **WER = (S+D+I)/N**.
- **LoRA** parameter count `r(d+k)` vs `d·k`.
