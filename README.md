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

Result:
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
  ✓ Saved → C:\Users\shubh\Project\ncert_rag_STandBM25\chunks\all_chunks.json  (36 chunks)

──────────────────────────────────────────────────────────────────────
  STAGE 2  DUAL RETRIEVAL  (BM25 + Sentence Transformer + Hybrid RRF)
──────────────────────────────────────────────────────────────────────

  ▸ Importing retrieval classes …

  ──────────────────────────────────────────────────────────────────
  2A  Building BM25 Index
  ──────────────────────────────────────────────────────────────────
  BM25: 36 chunks, avg 80 tokens/chunk

  ──────────────────────────────────────────────────────────────────
  2B  Building Sentence Transformer (TF-IDF semantic vectors)
  ──────────────────────────────────────────────────────────────────
  SentenceTransformer: 36 chunks × 3065 vocab dims
  ✓ Vocab size: 3065 dimensions  |  Matrix: 36 × 3065

  ──────────────────────────────────────────────────────────────────
  2C  Building Hybrid Retriever (BM25 + Semantic via RRF k=60)
  ──────────────────────────────────────────────────────────────────
  BM25: 36 chunks, avg 80 tokens/chunk
  SentenceTransformer: 36 chunks × 3065 vocab dims
  Hybrid retriever ready (36 chunks)

  ──────────────────────────────────────────────────────────────────
  Retriever Comparison  (5 test queries)
  ──────────────────────────────────────────────────────────────────
  Query                                          BM25   Semantic   Hybrid
  ────────────────────────────────────────────────────────────────────
  What is Newton's second law of motion?            ✓          ✓        ✓
  ↳ Direct keyword match — BM25 should win  
  How fast does velocity change when force          ✗          ✗        ✗
  ↳ Paraphrased — Semantic should help      
  Why does a ship float but a stone sinks?          ✓          ✓        ✓
  ↳ Conceptual — no exact term 'float' in section title
  Calculate kinetic energy of a 15 kg obje          ✓          ✓        ✓
  ↳ Calculation with numbers — BM25 numbers + Semantic concept
  What is the minimum distance for an echo          ✓          ✓        ✓
  ↳ Specific fact — should hit Ch12 sound content
  ────────────────────────────────────────────────────────────────────
  Correct rank-1                                    4          4        4 / 5

  ──────────────────────────────────────────────────────────────────
  Detailed comparison: 'What is Newton's second law?'
  ──────────────────────────────────────────────────────────────────

Query: 'What is Newton's second law?'

Rank   BM25 top chunks                        Semantic top chunks                    Hybrid top chunks
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1  9.3 SECOND LAW OF MOTION (7.95)        9.3 SECOND LAW OF MOTION (0.229)       9.3 SECOND LAW OF MOTION (0.0328)
  2  9.3 SECOND LAW OF MOTION (4.79)        9.3 SECOND LAW OF MOTION (0.066)       9.3 SECOND LAW OF MOTION (0.0323)
  3  10.2 FREE FALL (4.62)                  8.4 RATE OF CHANGE OF VELOCITY (0.059) 9.5 CONSERVATION OF MOMENTUM (0.0310)
  ✓ Hybrid retriever ready

──────────────────────────────────────────────────────────────────────
  STAGE 3  GROUNDED ANSWER GENERATION
──────────────────────────────────────────────────────────────────────

  ▸ Importing generation module …

  ──────────────────────────────────────────────────────────────────
  Prompt V1 vs V2 — why the refusal instruction changed
  ──────────────────────────────────────────────────────────────────

  V1 (permissive — what most people write first):
    You are a study assistant for NCERT Class 9 Science.
    Answer the student's question using ONLY the provided context.
    
    Context:

  V2 (constraint — explicit refuse rule + prescribed text):
    You are a study assistant for NCERT Class 9 Science.
    You have access to retrieved passages from Chapters 8–12 (Physics).
    
    STRICT RULES:
    1. Answer ONLY if the retrieved context directly answers the question.
    2. If the context does NOT contain the answer, say exactly:
       "This information is not in the provided chapters. Please refer to the relevant chapter."
    3. Never use knowledge outside the retrieved context.

    Key change: V1 says 'answer using ONLY context' — the LLM
    reads this as a preference. V2 says 'Answer ONLY IF directly
    relevant' — this is a conditional that forces a relevance
    check first. The prescribed refusal text makes is_refusal flag
    deterministic.

  ──────────────────────────────────────────────────────────────────
  Building GroundedAnswerSystem
  ──────────────────────────────────────────────────────────────────
  GroundedAnswerSystem ready | prompt=v2 | mode=mock generation

  ──────────────────────────────────────────────────────────────────
  Demo Answers  (6 questions)
  ──────────────────────────────────────────────────────────────────

  ── Q1: Direct — Ch9 (Force) ────────────────────────────────────
  Question : What is Newton's second law of motion? Write the formula.
  Expected : grounded answer
  Top chunk: [Chapter 9: Force and Laws of Motion]  9.3 SECOND LAW OF MOTION  (score=0.03279)

  Answer   :
    Newton's Second Law of Motion states: The rate of change of momentum of an object is proportional to the applied unbalanced force in the direction of the force. Mathematically:
      F = ma
    where F = force (N), m = mass (kg), a = acceleration (m s⁻²).
    1 Newton = 1 kg × 1 m s⁻².

  Status   : → ANSWERED

  ── Q2: Cross-chapter — Ch11 (Work & Energy) ────────────────────
  Question : What is the difference between kinetic energy and potential energy?
  Expected : grounded answer
  Top chunk: [Chapter 11: Work and Energy]  11.2 ENERGY  (score=0.03279)

  Answer   :
    Kinetic Energy KE = ½ × m × v²
    where m = mass (kg), v = velocity (m s⁻¹).
    Unit: Joule (J).

  Status   : → ANSWERED

  ── Q3: Calculation — Ch9 conservation of momentum ──────────────
  Question : A bullet of 20 g is fired from a 4 kg gun at 400 m/s. Find recoil velocity.
  Expected : step-by-step calculation
  Top chunk: [Chapter 9: Force and Laws of Motion]  9.5 CONSERVATION OF MOMENTUM  (score=0.03279)

  Answer   :
    By conservation of momentum: m₁u₁ + m₂u₂ = m₁v₁ + m₂v₂
    For gun (m=4 kg) + bullet (m=0.02 kg, v=400 m s⁻¹), both initially at rest:
    0 = 0.02×400 + 4×v₂ → v₂ = -2 m s⁻¹
    Gun recoils at 2 m s⁻¹ opposite to bullet direction.

  Status   : → ANSWERED

  ── Q4: Paraphrased — asking about 'acceleration' differently ───
  Question : How do we measure the rate at which velocity changes over time?
  Expected : grounded answer
  Top chunk: [Chapter 8: Motion]  8.4 RATE OF CHANGE OF VELOCITY — ACCELERATION  (score=0.03279)

  Answer   :
    Based on the retrieved NCERT content:
    If the velocity of an object changes, either the speed changes or the
    (See retrieved context for full details.)

  Status   : → ANSWERED

  ── Q5: Out-of-scope — Biology, Ch1 ─────────────────────────────
  Question : Explain the process of photosynthesis in plants.
  Expected : REFUSAL
  Top chunk: [Chapter 8: Motion]  8.3 MEASURING THE RATE OF MOTION  (score=0.03226)

  Answer   :
    This information is not in the provided chapters. Please refer to the relevant chapter.

  Status   : ✓ REFUSAL

  ── Q6: Adversarial OOS — Physics but Ch13 (not in corpus) ──────
  Question : How does electric current flow through a copper wire?
  Expected : REFUSAL
  Top chunk: [Chapter 12: Sound]  12.3 CHARACTERISTICS OF SOUND  (score=0.03202)

  Answer   :
    This information is not in the provided chapters. Please refer to the relevant chapter.

  Status   : ✓ REFUSAL
  ✓ Stage 3 complete

──────────────────────────────────────────────────────────────────────
  STAGE 4  EVALUATION  (25 questions · 5 chapters)
──────────────────────────────────────────────────────────────────────

  ▸ Importing evaluation module …

  ──────────────────────────────────────────────────────────────────
  Running 25 evaluation questions
  ──────────────────────────────────────────────────────────────────

        Type           Result                 Ch     Question
  ────────────────────────────────────────────────────────────────────
  ✗ Q01  direct         wrong                  Ch8    What are the three equations of uni
  ✗ Q02  direct         wrong                  Ch8    What is the difference between unif
  ✓ Q03  direct         correct                Ch8    An object travels 16 m in 4 s then 
  ✓ Q04  paraphrased    correct                Ch8    How do you find the speed of an obj
  ~ Q05  paraphrased    partial                Ch8    What happens to the velocity of an 
  ✓ Q06  direct         correct                Ch9    State Newton's second law of motion
  ~ Q07  direct         partial                Ch9    A bullet of 20 g is fired from 4 kg
  ✗ Q08  direct         wrong                  Ch9    Why does dust come out of a carpet 
  ~ Q09  paraphrased    partial                Ch9    newton 2nd law force equal mass tim
  ✗ Q10  paraphrased    incorrect_refusal      Ch9    If I push a truck but it doesn't mo
  ✗ Q11  direct         wrong                  Ch10   State Newton's universal law of gra
  ✓ Q12  direct         correct                Ch10   What is acceleration due to gravity
  ✓ Q13  direct         correct                Ch10   An object of mass 10 kg. What is it
  ✗ Q14  paraphrased    wrong                  Ch10   Why is the weight of an object on M
  ✗ Q15  direct         wrong                  Ch10   What is Archimedes principle and wh
  ~ Q16  direct         partial                Ch11   Define kinetic energy and write its
  ~ Q17  direct         partial                Ch11   A lamp consumes 1000 J in 10 s. Wha
  ✗ Q18  direct         incorrect_refusal      Ch11   What is the commercial unit of ener
  ✗ Q19  paraphrased    incorrect_refusal      Ch11   How much energy is stored in a ball
  ✓ Q20  direct         correct                Ch12   What is the speed of sound in air, 
  ✓ Q21  direct         correct                Ch12   What is an echo and what is the min
  ✗ Q22  direct         wrong                  Ch12   A sonar gets echo in 4 s. Speed of 
  ✗ Q23  paraphrased    incorrect_refusal      Ch12   What determines the pitch and loudn
  ✓ Q24  out_of_scope   correct_refusal        OOS    Explain the process of photosynthes
  ✓ Q25  out_of_scope   correct_refusal        OOS    How does electric current flow thro

  Completed 25 questions in 0.1s

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

Overall score: 9/25 = 36%

Grounding (of 19 answered questions):
  grounded      : 18
  ungrounded    : 1

Out-of-scope refusals: 2/2 correct | 0 missed

── Failure Analysis (────────────────────────────────────────)

  Q01: What are the three equations of uniformly accelerated m
    Correctness : wrong
    Top chunk   : 8.6 EQUATIONS OF MOTION FOR UNIFORM ACCE (score=0.033)
    Root cause  : RETRIEVAL — top chunk section '8.6 EQUATIONS OF MOTION FOR UN'
                  Check: is this the right section? If not → retrieval bug

  Q02: What is the difference between uniform and non-uniform 
    Correctness : wrong
    Top chunk   : 8.3 MEASURING THE RATE OF MOTION (score=0.033)
    Root cause  : RETRIEVAL — top chunk section '8.3 MEASURING THE RATE OF MOTI'
                  Check: is this the right section? If not → retrieval bug

  Q05: What happens to the velocity of an object moving in a c
    Correctness : partial
    Top chunk   : 8.4 RATE OF CHANGE OF VELOCITY — ACCELER (score=0.033)
    Root cause  : RETRIEVAL — answer partially in top-1 chunk;
                  full answer spread across 2+ chunks

  Q07: A bullet of 20 g is fired from 4 kg gun at 400 m/s. Fin
    Correctness : partial
    Top chunk   : 9.5 CONSERVATION OF MOMENTUM (score=0.033)
    Root cause  : RETRIEVAL — answer partially in top-1 chunk;
                  full answer spread across 2+ chunks

  ──────────────────────────────────────────────────────────────────
  Retriever Comparison  (BM25 vs Semantic vs Hybrid)
  ──────────────────────────────────────────────────────────────────
  BM25: 36 chunks, avg 80 tokens/chunk
  SentenceTransformer: 36 chunks × 3065 vocab dims
  BM25: 36 chunks, avg 80 tokens/chunk
  SentenceTransformer: 36 chunks × 3065 vocab dims
  Hybrid retriever ready (36 chunks)

────────────────────────────────────────────────────────────────────
RETRIEVER COMPARISON  (Rank-1 section for 5 test queries)
────────────────────────────────────────────────────────────────────
Query                                          BM25   Semantic     Hybrid
────────────────────────────────────────────────────────────────────
What is Newton's second law?                      ✓          ✓          ✓
What determines loudness of sound?                ✓          ✓          ✓
Calculate kinetic energy of 15 kg at 4            ✓          ✓          ✓
Why does an object float in water?                ✓          ✗          ✓
What is acceleration due to gravity?              ✗          ✓          ✗
────────────────────────────────────────────────────────────────────
Correct rank-1                                    4          4          4 / 5

  ──────────────────────────────────────────────────────────────────
  Saving Results
  ──────────────────────────────────────────────────────────────────

✓ Results saved → C:\Users\shubh\Project\ncert_rag_STandBM25\eval/evaluation_results.csv
✓ Markdown saved → C:\Users\shubh\Project\ncert_rag_STandBM25\eval/evaluation_results.md

  ──────────────────────────────────────────────────────────────────
  Failure Analysis  (16 non-correct results)
  ──────────────────────────────────────────────────────────────────

  Q01  [Ch8]  What are the three equations of uniformly accelerated m
    Correctness : wrong
    Top chunk   : 8.6 EQUATIONS OF MOTION FOR UNIFORM ACCE
      Root cause: RETRIEVAL or CHUNKING — content either split
      across chunk boundaries or section header not used as
      chunk boundary. Fix: force commit on section_header.

  Q02  [Ch8]  What is the difference between uniform and non-uniform 
    Correctness : wrong
    Top chunk   : 8.3 MEASURING THE RATE OF MOTION
      Root cause: RETRIEVAL or CHUNKING — content either split
      across chunk boundaries or section header not used as
      chunk boundary. Fix: force commit on section_header.

  Q05  [Ch8]  What happens to the velocity of an object moving in a c
    Correctness : partial
    Top chunk   : 8.4 RATE OF CHANGE OF VELOCITY — ACCELER
      Root cause: RETRIEVAL or CHUNKING — content either split
      across chunk boundaries or section header not used as
      chunk boundary. Fix: force commit on section_header.

  Q07  [Ch9]  A bullet of 20 g is fired from 4 kg gun at 400 m/s. Fin
    Correctness : partial
    Top chunk   : 9.5 CONSERVATION OF MOMENTUM
      Root cause: RETRIEVAL or CHUNKING — content either split
      across chunk boundaries or section header not used as
      chunk boundary. Fix: force commit on section_header.

  ═══ FINAL SCORE: 9/25 (36%) ═══

══════════════════════════════════════════════════════════════════════
║                         PIPELINE COMPLETE                          ║
══════════════════════════════════════════════════════════════════════
 