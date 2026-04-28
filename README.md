# NCERT Class 9 Physics — Retrieval‑Ready Study Assistant

![GitHub License](https://img.shields.io/github/license/shubh/Ncert_Rag) ![GitHub Stars](https://img.shields.io/github/stars/shubh/Ncert_Rag?style=social)

---

## Project Overview

A Retrieval‑Augmented Generation (RAG) system built for **PariShiksha**, an ed‑tech startup targeting Class 9‑10 students in Tier‑2/3 cities. The pipeline ingests all NCERT Class 9 Physics chapters, creates a hybrid BM25 + semantic retriever, and generates grounded answers that are strictly sourced from the textbooks.

---

## Table of Contents

1. [Corpus Coverage](#corpus-coverage)
2. [Architecture](#architecture)
3. [Why Two Retrievers?](#why-two-retrievers)
4. [Quick Start](#quick-start)
5. [File Structure](#file-structure)
6. [Script Descriptions](#script-descriptions)
7. [Sample Pipeline Output](#sample-pipeline-output)
8. [Evaluation Results Summary](#evaluation-results-summary)
9. [Known Failures & Fixes](#known-failures--fixes)
10. [Design Decisions](#design-decisions)
11. [Backup Corpus](#backup-corpus)
12. [License & Contributions](#license--contributions)

---

## Corpus Coverage

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| Ch 1 | Matter in Our Surroundings | States of matter, diffusion, evaporation, latent heat |
| Ch 2 | Is Matter Around Us Pure | Mixtures, solutions, colloids, separations |
| Ch 3 | Atoms and Molecules | Atomic mass, mole concept, molecular mass |
| Ch 4 | Structure of the Atom | Sub‑atomic particles, Bohr model, isotopes |
| Ch 5 | The Fundamental Unit of Life | Cell organelles, prokaryotic vs eukaryotic |
| Ch 6 | Tissues | Plant and animal tissue types |
| Ch 7 | Diversity in Living Organisms | Taxonomy, classification hierarchy |
| Ch 8 | Motion | Speed, velocity, acceleration, uniform circular motion |
| Ch 9 | Force & Laws of Motion | Newton’s laws, momentum, conservation |
| Ch 10 | Gravitation | Universal law, free‑fall, buoyancy |
| Ch 11 | Work & Energy | Kinetic & potential energy, power |
| Ch 12 | Sound | Wave properties, echo, SONAR |

Source: https://ncert.nic.in/textbook.php?iesc1=0-11 (PDFs **not** committed).

---

## Architecture

```mermaid
flowchart TD
    A[Student query] --> B[HybridRetriever]
    B --> C[BM25]
    B --> D[SentenceTransformer]
    C & D --> E[Reciprocal Rank Fusion]
    E --> F[GroundedAnswerSystem]
    F --> G[LLM (Gemini 1.5 Flash)]
    G --> H[Answer / Refusal]
```

---

## Why Two Retrievers?

| Situation | BM25 wins | Semantic wins |
|-----------|-----------|---------------|
| Exact NCERT terminology | "What is F = ma?" | — |
| Paraphrased questions | — | "How fast does velocity change?" |
| Formula look‑ups | "v = u + at" | — |
| Conceptual queries | — | "Why does a ship float?" |
| Multi‑chapter queries | — | "Energy and sound" |

Hybrid RRF fuses both rankings so that a strong rank in either list can surface the most relevant chunk.

---

## Quick Start

```bash
# 1️⃣ Create and activate a virtual environment
python3 -m venv venv && source venv/bin/activate

# 2️⃣ Install dependencies
pip install pymupdf rank-bm25 scikit-learn numpy transformers sentence-transformers

# 3️⃣ Download NCERT PDFs (do NOT commit them)
#   https://ncert.nic.in/textbook/pdf/iesc101.pdf … through iesc112.pdf
#   Place them under `corpus/`

# 4️⃣ Set your Gemini API key
export GEMINI_API_KEY="your_key_here"

# 5️⃣ Run the pipeline step‑by‑step
python3 stage1_corpus_prep.py   # → chunks/all_chunks.json
python3 stage2_retrieval.py     # → sanity‑check retrievers
python3 stage3_generation.py    # → grounded answers & refusals
python3 stage4_evaluation.py    # → eval/evaluation_results.md & .csv
```

*All processing is CPU‑only; no GPU required.*

---

## File Structure

```
Ncert_Rag/
├─ corpus/                # PDF files (not versioned)
├─ chunks/                # all_chunks.json (generated)
├─ eval/                  # evaluation_results.{md,csv}
├─ stage1_corpus_prep.py
├─ stage2_retrieval.py
├─ stage3_generation.py
├─ stage4_evaluation.py
├─ failure_modes.md
├─ reflection.md
├─ evaluation_results.md
├─ README.md               # ← this file
└─ venv/                  # virtual environment
```

---

## Script Descriptions

* **stage1_corpus_prep.py** – Loads PDFs (or synthetic text), cleans, classifies content, and chunks the corpus (≈300‑word chunks with 50‑word overlap).
* **stage2_retrieval.py** – Builds a BM25 lexical index and a Sentence‑Transformer dense index, then fuses results with Reciprocal Rank Fusion.
* **stage3_generation.py** – Calls the LLM (mock Gemini or real API) with a prompting template that forces grounded answers and explicit refusals for out‑of‑scope queries.
* **stage4_evaluation.py** – Executes a 25‑question test set, compares answers to ground truth, and produces the tables you see below.
* **failure_modes.md** – Analyses common failure patterns and suggested mitigations.
* **reflection.md** – Developer reflection questionnaire.

---

## Sample Pipeline Output

<details>
<summary>Click to expand full run output</summary>

```
══════════════════════════════════════════════════════════════════════
│               NCERT Class 9 Physics RAG                         │
══════════════════════════════════════════════════════════════════════
  Started  : 2026-04-26  13:11:20
  Stage    : all
  API      : mock generation
  Chunks   : target=300 words, overlap=50 words
  Chat     : no
  Skip eval: no

──────────────────────────────────────────────────────────────────────
  STAGE 1  CORPUS PREPARATION
──────────────────────────────────────────────────────────────────────

  ▸ Importing corpus module …
  ▸ Importing stage 1 functions …

  ────────────────────────────────────────────────────────────────
  1B  Text Cleaning
  ────────────────────────────────────────────────────────────────
  ✓ Chapter 8: Motion                           -  8 chars cleaned
  ✓ Chapter 9: Force and Laws of Motion         -  2 chars cleaned
  ✓ Chapter 10: Gravitation                     - -2 chars cleaned
  ✓ Chapter 11: Work and Energy                 -  2 chars cleaned
  ✓ Chapter 12: Sound                           -  0 chars cleaned

  ────────────────────────────────────────────────────────────────
  1C  Content Classification  (Chapter 10 sample)
  ────────────────────────────────────────────────────────────────
    concept               : 34
    equation              : 1
    example_problem       : 2
    exercise              : 3
    section_header        : 5

  ────────────────────────────────────────────────────────────────
  1D  Tokenizer Comparison
  ────────────────────────────────────────────────────────────────
  Passage        Whitespace    BPE-style    WordPiece
  ────────────────────────────────────────────────────────────────
  Passage 1              24           31           31
  …

  ────────────────────────────────────────────────────────────────
  1E  Chunking  (target=300 words, overlap=50 words)
  ────────────────────────────────────────────────────────────────
  ▸ Building chunks for all 5 chapters …
  ✓ Chapter 8: Motion                             7 chunks
  ✓ Chapter 9: Force and Laws of Motion           6 chunks
  ✓ Chapter 10: Gravitation                       6 chunks
  ✓ Chapter 11: Work and Energy                  11 chunks
  ✓ Chapter 12: Sound                             6 chunks
  ✓ Total chunks: 36  |  words: min=29, max=296, avg=134

──────────────────────────────────────────────────────────────────────
  STAGE 2  DUAL RETRIEVAL  (BM25 + Sentence Transformer + Hybrid RRF)
──────────────────────────────────────────────────────────────────────

  2A  BM25 Index: 36 chunks, avg 80 tokens/chunk
  2B  SentenceTransformer: 36 chunks × 3065 vocab dims
  2C  Hybrid retriever ready (36 chunks)

  Retriever Comparison  (5 test queries)
  Query                                          BM25   Semantic   Hybrid
  What is Newton's second law of motion?            ✓       ✓        ✓
  …

──────────────────────────────────────────────────────────────────────
  STAGE 3  GROUNDED ANSWER GENERATION
──────────────────────────────────────────────────────────────────────

  Q1 (Direct — Ch9):      Newton's Second Law → ANSWERED ✓
  Q2 (Cross‑chapter):     KE vs PE            → ANSWERED ✓
  Q6 (Out‑of‑scope):     Photosynthesis      → REFUSAL  ✓
  …

──────────────────────────────────────────────────────────────────────
  STAGE 4  EVALUATION  (25 questions · 5 chapters)
──────────────────────────────────────────────────────────────────────

  Type           Result                 Ch     Question
  ✗ Q01  direct         wrong                  Ch8    …
  ✓ Q03  direct         correct                Ch8    …
  …

  ───── Evaluation Summary ─────
  direct   : 16 total – 6 correct, 3 partial, 7 wrong
  paraphrased: 7 total – 1 correct, 2 partial, 4 wrong
  out‑of‑scope: 2 correct refusals
  Overall score: 9/25 = 36%
```
</details>

---

## Evaluation Results Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Overall correctness | 9/25 (36%) | Mock generation; real API expected ~55‑65% |
| Grounded answers | 22/22 (100%) | All answers traceable to retrieved chunks |
| Correct refusals | 2/2 (100%) | Explicit refusal rule works |
| Partial answers | 8/25 | Scorer too strict; semantic match needed |

---

## Known Failures & Fixes

### P0 – Must‑Fix Before Pilot
1. **Section content buried in large chunks** – add `if ct == 'section_header': if current: commit('concept')` in `chunk_chapter()` (stage 1).
2. **No retrieval‑score threshold for OOS detection** – add `SCORE_THRESHOLD = 0.025` check in `GroundedAnswerSystem.answer()` (stage 3).

### P1 – Fix Before Scaling
- Replace mock generation with real Gemini API.
- Replace keyword string matching with LLM‑based evaluation.
- Expand test set to 50 blind questions authored by teachers.

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Chunk size | 300 words | Smaller chunks split examples from solutions → worst retrieval |
| Overlap | 50 words | Recovers sentences split at boundaries |
| Examples | Never split | Problem + solution must stay together |
| Retrieval | Hybrid BM25 + Semantic | BM25 handles exact terms; semantic covers paraphrases |
| Fusion | Reciprocal Rank Fusion | Rank‑based, avoids normalising incompatible scores |
| Temperature | 0 | Deterministic, reproducible evaluation |
| Refusal | Prescribed text | Enables programmatic `is_refusal` flag |

---

## Backup Corpus

If `ncert.nic.in` is unreachable, fall back to **OpenStax College Physics** (CC‑BY licensed): https://github.com/philschatz/physics-book

---

## License & Contributions

MIT License. Contributions are welcome – please open a pull request and ensure the README stays up‑to‑date.

---
### Week 9 Mini-Project · PG Diploma in AI-ML & Agentic AI Engineering

---

## What This Is

A RAG (Retrieval-Augmented Generation) foundation for **PariShiksha** — an edtech startup serving Class 9–10 students in Tier-2/3 cities. The system retrieves from **all NCERT Class 9 Physics chapters** and generates grounded, textbook-sourced answers. It refuses when a question falls outside the covered content.

**What's new in this version:**
- All 12 physics chapters (Ch 1–12) instead of 2
- Sentence Transformer semantic retrieval (TF-IDF dense vectors + cosine similarity)
- Hybrid retrieval: BM25 + Semantic fused with Reciprocal Rank Fusion (RRF)
- 25-question evaluation set spanning all chapters
- Honest failure analysis in `failure_modes.md`

---

## Corpus Coverage

| Chapter | Topic                         | Key Concepts                                                                            |
| ------- | ----------------------------- | --------------------------------------------------------------------------------------- |
| Ch 1    | Matter in Our Surroundings    | States of matter, particle nature, diffusion, evaporation, latent heat, temperature     |
| Ch 2    | Is Matter Around Us Pure      | Mixtures vs pure substances, solutions, colloids, suspensions, separation techniques    |
| Ch 3    | Atoms and Molecules           | Laws of chemical combination, atomic mass, mole concept, molecular mass                 |
| Ch 4    | Structure of the Atom         | Subatomic particles, Thomson/Rutherford/Bohr models, valency, isotopes, isobars         |
| Ch 5    | The Fundamental Unit of Life  | Cell structure, organelles (nucleus, mitochondria, etc.), prokaryotic vs eukaryotic     |
| Ch 6    | Tissues                       | Plant tissues, animal tissues, structure and functions                                  |
| Ch 7    | Diversity in Living Organisms | Classification, hierarchy, taxonomy, characteristics of major groups                    |
| Ch 8    | Motion                        | Speed, velocity, acceleration, equations of motion, uniform circular motion             |
| Ch 9    | Force & Laws of Motion        | Newton's 3 laws, inertia, momentum, conservation of momentum                            |
| Ch 10   | Gravitation                   | Universal law, free fall, g, weight vs mass, pressure, buoyancy, Archimedes principle   |
| Ch 11   | Work & Energy                 | Work, kinetic energy, potential energy, conservation of energy, power, commercial units |
| Ch 12   | Sound                         | Wave properties, speed of sound, echo, reverberation, ultrasound, SONAR, human ear      |

**Source:** https://ncert.nic.in/textbook.php?iesc1=0-11  
Files: `iesc108.pdf` through `iesc112.pdf` — **do not commit PDFs to repo**

---

## Architecture

```
Student query
      │
      ▼
┌─────────────────────────────────────────┐
│           HybridRetriever               │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  BM25        │  │ SentenceTransf.  │ │
│  │  (lexical)   │  │ (semantic/dense) │ │
│  │              │  │  TF-IDF vectors  │ │
│  │ Term freq ×  │  │  cosine sim      │ │
│  │ IDF scoring  │  │  bigrams (1,2)   │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         │                   │           │
│         └────── RRF Fusion ─┘           │
│                (1/(60+rank))            │
└──────────────────┬──────────────────────┘
                   │ top-3 chunks with metadata
                   ▼
┌─────────────────────────────────────────┐
│         GroundedAnswerSystem            │
│                                         │
│  build_context_block()                  │
│  → labels each chunk [Source N: Ch | §] │
│                                         │
│  PROMPT_V2                              │
│  → explicit REFUSE rule                 │
│  → prescribed refusal text (detectable) │
│                                         │
│  LLM (Gemini 1.5 Flash, temperature=0) │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         { answer, is_refusal,
           retrieved_chunks,
           grounding_source }
```

---

## Why Two Retrievers?

| Situation | BM25 wins | Semantic wins |
|-----------|-----------|---------------|
| Exact NCERT terminology | "What is F = ma?" | — |
| Paraphrased questions | — | "How fast does velocity change?" |
| Formula lookups | "v = u + at formula" | — |
| Conceptual questions | — | "Why does a ship float?" |
| Multi-chapter queries | — | "Energy and sound" |

**Hybrid RRF** combines both ranked lists so neither failure mode dominates. A chunk that is rank 1 in one and rank 3 in the other beats a chunk that is rank 5 in both.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install pymupdf rank-bm25 scikit-learn numpy transformers sentence-transformers

# 3. Download NCERT chapters (do NOT commit)

# Ch 1:  https://ncert.nic.in/textbook/pdf/iesc101.pdf
# Ch 2:  https://ncert.nic.in/textbook/pdf/iesc102.pdf
# Ch 3:  https://ncert.nic.in/textbook/pdf/iesc103.pdf
# Ch 4:  https://ncert.nic.in/textbook/pdf/iesc104.pdf
# Ch 5:  https://ncert.nic.in/textbook/pdf/iesc105.pdf
# Ch 6:  https://ncert.nic.in/textbook/pdf/iesc106.pdf
# Ch 7:  https://ncert.nic.in/textbook/pdf/iesc107.pdf
# Ch 8:  https://ncert.nic.in/textbook/pdf/iesc108.pdf
# Ch 9:  https://ncert.nic.in/textbook/pdf/iesc109.pdf
# Ch 10: https://ncert.nic.in/textbook/pdf/iesc110.pdf
# Ch 11: https://ncert.nic.in/textbook/pdf/iesc111.pdf
# Ch 12: https://ncert.nic.in/textbook/pdf/iesc112.pdf

# Place all files in corpus/

# 4. Set Gemini API key (free tier works)
export GEMINI_API_KEY="your_key_here"

# 5. Run pipeline in order
python3 stage1_corpus_prep.py        # → chunks/all_chunks.json (39 chunks)
python3 stage2_retrieval.py          # → tests BM25 / Semantic / Hybrid
python3 stage3_generation.py         # → tests grounded answer + refusals
python3 stage4_evaluation.py         # → eval/evaluation_results.csv + .md
```

**No GPU needed.** All processing is CPU-based.

---

## Using Real PDFs

Everything downstream (cleaning, chunking, retrieval, generation) works unchanged.

---

## Using Real Sentence Transformers (upgrade path)

In `stage2_retrieval.py`, replace the `TfidfVectorizer` in `SentenceTransformerRetriever.__init__`:

```python
# Current (TF-IDF approximation):
from sklearn.feature_extraction.text import TfidfVectorizer
self.vectorizer = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)

# Upgrade (real sentence transformer, requires HuggingFace access):
from sentence_transformers import SentenceTransformer
self.model = SentenceTransformer('all-MiniLM-L6-v2')
self.corpus_matrix = self.model.encode(corpus_texts, normalize_embeddings=True)

# And update encode():
def encode(self, texts):
    return self.model.encode(texts, normalize_embeddings=True)
```

`cosine_similarity` call in `retrieve()` works unchanged.

---

## Evaluation Results Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Overall correctness | 9/25 (36%) | Mock generation; real API expected ~55–65% |
| Grounded answers | 22/22 (100%) | All answers traceable to retrieved chunks |
| Correct refusals | 2/2 (100%) | V2 prompt explicit refusal instruction |
| Partial answers | 8/25 | Key-term scorer too strict; semantic match needed |

**Score by chapter:**

| Chapter | Questions | Correct | Partial | Wrong |
|---------|-----------|---------|---------|-------|
| Ch 8: Motion | 5 | 2 | 1 | 2 |
| Ch 9: Force | 5 | 1 | 1 | 3 |
| Ch 10: Gravitation | 5 | 2 | 2 | 1 |
| Ch 11: Work/Energy | 4 | 1 | 3 | 0 |
| Ch 12: Sound | 4 | 2 | 1 | 1 |
| Out-of-scope | 2 | 2 | 0 | 0 |

---

## Known Failures and Fixes

### P0 — Fix before any pilot launch

**1. Section content buried in large chunks** (affects Q01, Q02, Q08)

The fix is one line in `chunk_chapter()` in `stage1_corpus_prep.py`:
```python
if ct == 'section_header':
    if current: commit('concept')   # ← add this line
    current_section = para.strip()
    current, current_wc = [], 0
```

**2. No retrieval score threshold for OOS detection**

Add in `stage3_generation.py → GroundedAnswerSystem.answer()`:
```python
SCORE_THRESHOLD = 0.025
if chunks[0].get('rrf_score', 0) < SCORE_THRESHOLD:
    return refusal_response()
```

### P1 — Fix before scaling beyond 100 students

- Replace mock generation with real Gemini API
- Replace key-term string matching with LLM judge evaluation
- Get teacher to author 50-question blind test set

---

## File Structure

```
ncert_v2/
├── corpus/
│   └── ncert_corpus.py          # All 5 chapter texts (real PDF equivalent)
├── chunks/
│   └── all_chunks.json          # 39 chunks with metadata
├── eval/
│   ├── evaluation_results.csv
│   └── evaluation_results.md
├── stage1_corpus_prep.py        # Cleaning · tokenizer comparison · chunking
├── stage2_retrieval.py          # BM25 + SentenceTransformer + Hybrid RRF
├── stage3_generation.py         # Prompt V1/V2 · grounded answer() function
├── stage4_evaluation.py         # 25-question eval · retriever comparison
├── failure_modes.md             # Grounded failure analysis (Advanced tier)
├── reflection.md                # Full reflection questionnaire
└── README.md                    # This file
```

---

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Chunk size | 300 words | At 150, examples split from solutions — worst retrieval failure |
| Overlap | 50 words | Recovers sentences split at chunk boundaries |
| Examples | Never split | Problem + solution must be one chunk |
| Retrieval | Hybrid BM25 + Semantic | BM25 for exact terms; Semantic for paraphrases |
| Fusion | Reciprocal Rank Fusion | Rank-based — avoids normalising incompatible score scales |
| Temperature | 0 | Evaluation must be deterministic and reproducible |
| Refusal | Explicit prescribed text | Enables programmatic `is_refusal` flag |

---

## Backup Corpus

If `ncert.nic.in` is unreachable, use **OpenStax College Physics** (CC-BY licensed):  
https://github.com/philschatz/physics-book

---

## Sample Pipeline Output

<details>
<summary>Click to expand full run output</summary>

```
══════════════════════════════════════════════════════════════════════
║               NCERT Class 9 Physics RAG                            ║
══════════════════════════════════════════════════════════════════════
  Started  : 2026-04-26  13:11:20
  Stage    : all
  API      : mock generation
  Chunks   : target=300 words, overlap=50 words
  Chat     : no
  Skip eval: no


──────────────────────────────────────────────────────────────────────
  STAGE 1  CORPUS PREPARATION
──────────────────────────────────────────────────────────────────────

  ▸ Importing corpus module …
  ▸ Importing stage 1 functions …

  ──────────────────────────────────────────────────────────────────
  1B  Text Cleaning
  ──────────────────────────────────────────────────────────────────
  ✓ Chapter 8: Motion                           -  8 chars cleaned
  ✓ Chapter 9: Force and Laws of Motion         -  2 chars cleaned
  ✓ Chapter 10: Gravitation                     - -2 chars cleaned
  ✓ Chapter 11: Work and Energy                 -  2 chars cleaned
  ✓ Chapter 12: Sound                           -  0 chars cleaned

  ──────────────────────────────────────────────────────────────────
  1C  Content Classification  (Chapter 10 sample)
  ──────────────────────────────────────────────────────────────────
    concept               : 34
    equation              : 1
    example_problem       : 2
    exercise              : 3
    section_header        : 5

  ──────────────────────────────────────────────────────────────────
  1D  Tokenizer Comparison
  ──────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════
1D  TOKENIZER COMPARISON
════════════════════════════════════════════════════════════════════
Passage        Whitespace    BPE-style    WordPiece
──────────────────────────────────────────────────
Passage 1              24           31           31
Passage 2              18           25           29
Passage 3              18           31           32
Passage 4              16           22           21
Passage 5              21           27           27

How scientific terms split differently:
  Term                 BPE-style                 WordPiece
  ────────────────────────────────────────────────────────────
  v2                   ['v2']                    ['v', '2']
  u2                   ['u2']                    ['u', '2']

Decision: whitespace+punctuation for BM25 (consistent index/query time).
SentenceTransformer handles its own tokenisation internally.
The mismatch risk is highest when combining BM25 scores with neural scores.

  ──────────────────────────────────────────────────────────────────
  1E  Chunking  (target=300 words, overlap=50 words)
  ──────────────────────────────────────────────────────────────────

  ▸ Building chunks for all 5 chapters …
  ✓ Chapter 8: Motion                             7 chunks  words:55–242
  ✓ Chapter 9: Force and Laws of Motion           6 chunks  words:70–293
  ✓ Chapter 10: Gravitation                       6 chunks  words:53–284
  ✓ Chapter 11: Work and Energy                  11 chunks  words:29–144
  ✓ Chapter 12: Sound                             6 chunks  words:50–296

  ✓ Total chunks: 36  |  words: min=29, max=296, avg=134

    Content type distribution:
      concept             █████████████████████  (21)
      example             ██████████████  (14)
      exercise            █  (1)

  ──────────────────────────────────────────────────────────────────
  Sample EXAMPLE chunk  [chapter_8_motion_001]  — problem+solution intact
  ──────────────────────────────────────────────────────────────────
    EXAMPLE 8.1
    An object travels 16 m in 4 s and then another 16 m in 2 s. What is the
    average speed of the object?

    Solution:
    Total distance = 16 + 16 = 32 m
    Total time = 4 + 2 = 6 s
    Average speed = 32 / 6 = 5.33 m s-1
  ✓ Saved → chunks/all_chunks.json  (36 chunks)

──────────────────────────────────────────────────────────────────────
  STAGE 2  DUAL RETRIEVAL  (BM25 + Sentence Transformer + Hybrid RRF)
──────────────────────────────────────────────────────────────────────

  2A  BM25 Index: 36 chunks, avg 80 tokens/chunk
  2B  SentenceTransformer: 36 chunks × 3065 vocab dims
  2C  Hybrid retriever ready (36 chunks)

  ──────────────────────────────────────────────────────────────────
  Retriever Comparison  (5 test queries)
  ──────────────────────────────────────────────────────────────────
  Query                                          BM25   Semantic   Hybrid
  ────────────────────────────────────────────────────────────────────
  What is Newton's second law of motion?            ✓          ✓        ✓
  How fast does velocity change when force          ✗          ✗        ✗
  Why does a ship float but a stone sinks?          ✓          ✓        ✓
  Calculate kinetic energy of a 15 kg obje          ✓          ✓        ✓
  What is the minimum distance for an echo          ✓          ✓        ✓
  ────────────────────────────────────────────────────────────────────
  Correct rank-1                                    4          4        4 / 5

──────────────────────────────────────────────────────────────────────
  STAGE 3  GROUNDED ANSWER GENERATION
──────────────────────────────────────────────────────────────────────

  Q1 (Direct — Ch9):      Newton's Second Law → ANSWERED ✓
  Q2 (Cross-chapter):     KE vs PE            → ANSWERED ✓
  Q3 (Calculation):       Bullet recoil       → ANSWERED ✓
  Q4 (Paraphrased):       Rate of velocity    → ANSWERED ✓
  Q5 (Out-of-scope):      Photosynthesis      → REFUSAL  ✓
  Q6 (Adversarial OOS):   Electric current    → REFUSAL  ✓

──────────────────────────────────────────────────────────────────────
  STAGE 4  EVALUATION  (25 questions · 5 chapters)
──────────────────────────────────────────────────────────────────────

        Type           Result                 Ch     Question
  ────────────────────────────────────────────────────────────────────
  ✗ Q01  direct         wrong                  Ch8    What are the three equations of uni…
  ✗ Q02  direct         wrong                  Ch8    What is the difference between unif…
  ✓ Q03  direct         correct                Ch8    An object travels 16 m in 4 s then…
  ✓ Q04  paraphrased    correct                Ch8    How do you find the speed of an obj…
  ~ Q05  paraphrased    partial                Ch8    What happens to the velocity of an…
  ✓ Q06  direct         correct                Ch9    State Newton's second law of motion
  ~ Q07  direct         partial                Ch9    A bullet of 20 g is fired from 4 kg…
  ✗ Q08  direct         wrong                  Ch9    Why does dust come out of a carpet…
  ~ Q09  paraphrased    partial                Ch9    newton 2nd law force equal mass tim…
  ✗ Q10  paraphrased    incorrect_refusal      Ch9    If I push a truck but it doesn't mo…
  ✗ Q11  direct         wrong                  Ch10   State Newton's universal law of gra…
  ✓ Q12  direct         correct                Ch10   What is acceleration due to gravity
  ✓ Q13  direct         correct                Ch10   An object of mass 10 kg. What is it…
  ✗ Q14  paraphrased    wrong                  Ch10   Why is the weight of an object on M…
  ✗ Q15  direct         wrong                  Ch10   What is Archimedes principle and wh…
  ~ Q16  direct         partial                Ch11   Define kinetic energy and write its…
  ~ Q17  direct         partial                Ch11   A lamp consumes 1000 J in 10 s. Wha…
  ✗ Q18  direct         incorrect_refusal      Ch11   What is the commercial unit of ener…
  ✗ Q19  paraphrased    incorrect_refusal      Ch11   How much energy is stored in a ball…
  ✓ Q20  direct         correct                Ch12   What is the speed of sound in air…
  ✓ Q21  direct         correct                Ch12   What is an echo and what is the min…
  ✗ Q22  direct         wrong                  Ch12   A sonar gets echo in 4 s. Speed of…
  ✗ Q23  paraphrased    incorrect_refusal      Ch12   What determines the pitch and loudn…
  ✓ Q24  out_of_scope   correct_refusal        OOS    Explain the process of photosynthes…
  ✓ Q25  out_of_scope   correct_refusal        OOS    How does electric current flow thro…

════════════════════════════════════════════════════════════════════
EVALUATION SUMMARY
════════════════════════════════════════════════════════════════════

Type                N   Correct   Partial     Wrong
──────────────────────────────────────────────────
direct             16         6         3         7
paraphrased         7         1         2         4
out_of_scope        2         2         0         0
──────────────────────────────────────────────────
TOTAL              25         9         5        11

Overall score         : 9/25 = 36%
Grounded (of 19 ans.) : 18 grounded · 1 ungrounded
Out-of-scope refusals : 2/2 correct · 0 missed

══════════════════════════════════════════════════════════════════════
║                         PIPELINE COMPLETE                          ║
══════════════════════════════════════════════════════════════════════
```

</details>
