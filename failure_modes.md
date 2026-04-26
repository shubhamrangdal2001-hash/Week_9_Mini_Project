# failure_modes.md — PariShiksha RAG System

**Grounded in actual evaluation results (25 questions, 5 chapters, 9/25 = 36% correct)**

---

## Failure Mode 1: Over-Inclusive Chunks Bury Section-Specific Content

**Observed in:** Q01, Q02, Q08, Q14, Q22 (5 of 8 "wrong" answers)

**What happens:**

Our chunker targets 300 words per chunk. When a chapter introduction is 290 words and contains section 8.1 *and* section 8.1.2 content, both land in chunk 0. A query for "uniform vs non-uniform motion" (section 8.1.2 content) has to compete against all the section 8.1 vocabulary in the same chunk. BM25 score is diluted. The retriever returns a different chunk whose section *title* matches the query better, even though its text doesn't contain the answer.

**Concrete example:**

```
Query: "What is the difference between uniform and non-uniform motion?"
Top-1 retrieved: "8.2 MEASURING THE RATE OF MOTION"  (score 0.033)
Correct section: "8.1.2 Uniform Motion and Non-uniform Motion"  (buried in chunk 0)
```

The correct content is in the corpus — the retriever just can't find it because it shares a chunk with 200 words of unrelated introduction.

**Root cause:** Chunker uses word-count threshold only. It does not force chunk boundaries at section headers.

**Fix:**

```python
# In chunk_chapter(), add this at the section_header check:
if ct == 'section_header':
    if current:
        commit('concept')          # flush current chunk
    current_section = para.strip()
    current, current_wc = [], 0   # start fresh
```

This one-line change forces each named NCERT section to begin a new chunk. Estimated impact: fixes Q01, Q02, Q08 → correctness from 36% to ~52%.

**Production risk level:** HIGH. Affects the most common query type — students asking about a specific concept that is mixed into a larger chunk.

---

## Failure Mode 2: Mock Generation Key-Term Mismatch

**Observed in:** Q07, Q11, Q15, Q16, Q17, Q18, Q23 (7 "partial" answers)

**What happens:**

The retriever returns the correct chunk (right section, right content). But our mock generation uses slightly different formatting than what the key-term scorer expects. For example:

- Q07: Key term `'v2'` (from ground truth `v² = u² + 2as`) — answer contains `v₂` (subscript form) which fails `'v2' in answer.lower()`
- Q17: Key term `'watt'` — answer says `100 W` (abbreviation), not the word "watt"
- Q18: Key term `'3.6'` — answer says `3.6 × 10⁶ J` — the scoring finds `3.6` but requires the full number `3600000` for "commercial unit"

**Root cause:** Two separate issues:
1. Mock generation doesn't align with key term format
2. Key term selection is brittle — should use semantic matching, not exact substring

**Fix (scoring):**
```python
# Normalise answer before matching
ans_norm = answer.lower().replace('×','x').replace('⁻','').replace('²','2')
found = [t for t in key_terms if t.lower() in ans_norm or
         t.lower().replace(' ','') in ans_norm.replace(' ','')]
```

**Fix (evaluation):** Replace with real Gemini API + LLM judge scoring. An LLM judge ("does this answer correctly address the question? yes/no/partial") is far more reliable than string matching.

**Production risk level:** MEDIUM. With a real LLM (not mock), this failure mode mostly disappears — Gemini will use the textbook's exact terminology. But automated evaluation scoring must still handle format variation.

---

## Failure Mode 3: Adversarial Out-of-Scope in Adjacent Domains

**Observed in:** Q25 ("How does electric current flow through a conductor?")

**What happens:**

The query contains physics vocabulary that appears in our corpus — "flow," "conductor" (sounds like "conservation"), "current." BM25 tokenises and finds weak term matches scattered across Ch8–9 chunks. The retriever returns a result with score ~0.032. The V2 prompt's explicit refusal instruction catches this in mock generation, but with a real LLM there is no guarantee — the plausible-looking context combined with a V1-style prompt would generate a confident wrong answer.

**Concrete risk:**

A student in Class 9 studying Ch8–12 asks about electricity (Ch13, studied later). Our system returns motion/force chunks. A weak-prompt LLM says "current flows through a conductor because of the force applied" — incorrect physics, stitched from Newton's laws vocabulary. The student is now more confused than before.

**Root cause:** No minimum score threshold on retrieval. Any non-zero score passes to generation.

**Fix:**

```python
def answer(self, question, k=3):
    chunks = self.retriever.retrieve(question, k=k)
    
    # Score threshold: if best score is below this, treat as out-of-scope
    SCORE_THRESHOLD = 0.025   # tuned on eval set
    if chunks[0].get('rrf_score', 0) < SCORE_THRESHOLD:
        return {
            'answer': "This information is not in the provided chapters. "
                      "Please refer to the relevant chapter.",
            'is_refusal': True,
            ...
        }
    # ... continue to generation
```

The threshold `0.025` comes from observing that all in-scope correct retrievals scored ≥ 0.033 while the adversarial OOS scores were 0.032 (borderline). In production, tune this threshold on a held-out OOS validation set.

**Production risk level:** HIGH. This is the failure mode that gets PariShiksha parents to escalate. A confidently wrong answer about electricity that sounds like physics is worse than a correct refusal.

---

## Summary Table

| Failure Mode | Affected Qs | Impact | Fix Complexity | Priority |
|---|---|---|---|---|
| Section content buried in large chunks | Q01,Q02,Q08,Q14,Q22 | 5 wrong → correct | 1 line | **P0** |
| Mock/scorer key-term mismatch | Q07,Q11,Q15–18,Q23 | 7 partial → correct | Real API | **P1** |
| Adversarial OOS leaks through | Q25 (borderline) | Hallucination risk | Score threshold | **P0** |

**P0 items to fix before any pilot launch:**
1. Section-boundary chunk commit (1 line of code)
2. Retrieval score threshold for OOS detection (~5 lines)

**P1 items before scaling beyond 100 students:**
1. Replace mock generation with real Gemini API at temperature=0
2. Replace key-term scoring with LLM judge evaluation
3. Build a teacher-authored 50-question evaluation set
