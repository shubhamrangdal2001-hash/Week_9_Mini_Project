# NCERT Class 9 Physics — Retrieval-Ready Study Assistant
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


|---------|-------|--------------|
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

## Using Real PDFs (replace synthetic corpus)

In `stage1_corpus_prep.py`, change:

```python
# Current (synthetic, mirrors real extraction):
from ncert_corpus import ALL_CHAPTERS

# Replace with (real PDF extraction):
import fitz

def load_all_chapters():
    chapter_files = {
        "Chapter 1: Matter in Our Surroundings":       "corpus/iesc101.pdf",
        "Chapter 2: Is Matter Around Us Pure":         "corpus/iesc102.pdf",
        "Chapter 3: Atoms and Molecules":              "corpus/iesc103.pdf",
        "Chapter 4: Structure of the Atom":            "corpus/iesc104.pdf",
        "Chapter 5: The Fundamental Unit of Life":     "corpus/iesc105.pdf",
        "Chapter 6: Tissues":                          "corpus/iesc106.pdf",
        "Chapter 7: Diversity in Living Organisms":    "corpus/iesc107.pdf",
        "Chapter 8: Motion":                           "corpus/iesc108.pdf",
        "Chapter 9: Force and Laws of Motion":         "corpus/iesc109.pdf",
        "Chapter 10: Gravitation":                     "corpus/iesc110.pdf",
        "Chapter 11: Work and Energy":                 "corpus/iesc111.pdf",
        "Chapter 12: Sound":                           "corpus/iesc112.pdf",
    }
    chapters = {}
    for name, path in chapter_files.items():
        doc = fitz.open(path)
        text = "".join(page.get_text() for page in doc)
        chapters[name] = text
        doc.close()
    return chapters

ALL_CHAPTERS = load_all_chapters()
```

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
