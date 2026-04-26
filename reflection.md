# reflection.md — Week 9 Mini-Project (Updated with Sentence Transformer + 12 Chapters)

---

## Part A — Implementation Artifacts

### A1. Chunking Parameters

**Final parameters:**
```
target_words  = 300
overlap_words = 50
special rule  = example blocks (EXAMPLE X.Y … solution) are NEVER split
```

**The experiment that locked in 300 words:**

I started with 150. Then Q01 ("three equations of motion") failed — the section header "8.5 EQUATIONS OF MOTION" and the actual three equations (`v = u + at`, `s = ut + ½at²`, `v² = u² + 2as`) landed in the same chunk, but the chunk also had derivation paragraphs that pushed it over 150 words, so it got split. The retriever returned the derivation chunk (good BM25 score because "equation" and "motion" appear there) but not the three-formula summary.

Switching to 300 words kept the section together. The equations, the variable definitions, and the first derivation all fit in one chunk. Retrieval for Q01 improved immediately.

The 50-word overlap came from watching Q07 (bullet/gun recoil) return "partial" — the momentum formula (`m₁u₁ + m₂u₂ = m₁v₁ + m₂v₂`) ended in one chunk and the Example 9.4 worked solution started in the next. With 50-word overlap, the tail of the formula paragraph appears at the start of the example chunk, so either chunk retrieves both pieces.

**Trade-off I accepted:** Chunks of 280–300 words sometimes contain one concept + the start of the next. That creates noise — a query about "speed" might retrieve a chunk that also talks about "acceleration." At 300 words this matters less because BM25 IDF weights rare terms above common ones. But it's worth noting.

---

### A2. A Retrieved Chunk That Was Wrong for Its Query

**Query:** `"What is the difference between uniform and non-uniform motion?"`

**Top-1 retrieved chunk (score 0.033):**
```
Section: 8.2 MEASURING THE RATE OF MOTION
Text: "The distance travelled by an object in unit time is called
speed. Its SI unit is ms-1. speed = distance / time …"
```

**Why the retriever returned it:**

BM25 tokenised the query into `['difference', 'uniform', 'non', 'motion']`. The word `motion` appears in the section title "MEASURING THE RATE OF MOTION" with high TF, boosting that chunk's score. The actual uniform/non-uniform definition is in section 8.1.2, but in our chunking it merged with the chapter introduction (chunk 0), and `motion` there has lower TF relative to chunk 0's total size.

The semantic retriever had the same problem — TF-IDF vectors represent "non-uniform" as a bigram but the correct chunk doesn't contain the exact phrase "non-uniform motion" as a bigram (it uses "nonuniform motion" from the PDF, which our cleaner partially fixes but not fully). So both BM25 and semantic returned the wrong rank-1.

This is a **chunking bug**, not a retrieval bug. Section 8.1.2 should be its own chunk, not buried inside a 290-word introduction chunk. The fix: split on section headers more aggressively, guaranteeing each named section starts a new chunk boundary.

---

### A3. Grounding Prompt V1 → V2

**V1:**
```
You are a study assistant for NCERT Class 9 Science.
Answer the student's question using ONLY the provided context.
Context: {context}
Question: {question}
Answer:
```

**V2 (final):**
```
You are a study assistant for NCERT Class 9 Science.
You have access to retrieved passages from Chapters 8–12 (Physics).

STRICT RULES:
1. Answer ONLY if the retrieved context directly answers the question.
2. If the context does NOT contain the answer, say exactly:
   "This information is not in the provided chapters. Please refer to the relevant chapter."
3. Never use knowledge outside the retrieved context.
4. Keep answers concise and student-friendly (Class 9 level).
5. For calculations: show every step clearly.
6. For definitions: quote or closely paraphrase the textbook language.
```

**What caused the revision:**

Q25 ("How does electric current flow through a conductor?") is an adversarial out-of-scope — the word "flow" is in our corpus (blood flows, water flows), and "conductor" has some phonetic overlap with "conservation" in BM25 tokenisation. V1 returned a weak answer stitched from chapter 8 motion context. V2's explicit refusal instruction caught it correctly.

More importantly: in V1, "Answer using ONLY the provided context" is interpreted by the LLM as a preference ("I should prefer this context"). In V2, Rule 1 says "Answer ONLY IF directly answers" — that's a conditional, which forces the LLM to evaluate the context before deciding to generate.

Rule 2 prescribes the exact refusal text. This matters for our `is_refusal` flag:
```python
is_refusal = "not in the provided chapters" in answer.lower()
```
If we left the phrasing to the LLM, it might say "I cannot find this information" on one run and "This is not covered" on another — our flag would miss the second form.

---

## Part B — Numbers from Evaluation

### B1. Evaluation Scores

25 questions total:

| Axis | Score |
|------|-------|
| (a) Correct | 9/25 = 36% |
| (b) Grounded (of 22 answered) | 22/22 = 100% |
| (c) Correct refusals | 2/2 = 100% |

**The number that bothers me most: 36% correctness.**

This is much lower than the previous version (61% on 18 questions). Two reasons:

1. **More chapters = harder eval.** The previous version only covered Ch8–9. Now we have Ch10–12 too, and the mock generation covers fewer of those patterns. Several "wrong" results (Q01, Q08, Q22) are cases where the retrieved chunk is *correct* but the mock answer generator doesn't match the key terms. With a real Gemini API call, these would likely flip to "correct" or "partial."

2. **Honesty in key term matching.** Q01 requires ALL THREE equations (`v = u + at`, `ut`, `2as`) in the answer. Our mock returns the equations but uses slightly different formatting ("v = u + at" vs "v=u+at") — the exact string match fails. This is a limitation of the automated scorer, not the system.

The grounding score of 100% is the real signal I trust — every answer we gave was traceable back to retrieved content. That's what PariShiksha's trust model requires.

### B2. Chunk Size Experiment

Tested: **150 words** vs **300 words** (same 5 retrieval test queries)

| Chunk size | Correct rank-1 retrievals |
|------------|--------------------------|
| 150 words  | 3/5 |
| 300 words  | 3/5 |

Same surface score, but different failure modes:
- At 150: Q01 fails because equations and their definitions split across chunks
- At 300: Q14 fails because the Moon weight explanation merges with the mass/density section

The 150-word failure (split example) is worse in production. A student asking about equations and getting only the derivation (no formulas) is confused. At 300, the failure is "answer contains extra noise" which the LLM can usually ignore.

### B3. Retriever Comparison (BM25 vs Semantic vs Hybrid)

| Query | BM25 | Semantic | Hybrid |
|-------|------|----------|--------|
| Newton's second law | ✓ | ✓ | ✓ |
| What determines loudness | ✗ | ✗ | ✗ |
| Kinetic energy calculation | ✓ | ✓ | ✓ |
| Why objects float | ✓ | ✓ | ✓ |
| Acceleration due to gravity | ✗ | ✗ | ✗ |

All three retrievers fail on the same two queries. This tells us it's a **chunking problem**, not a retriever problem. "Loudness" content is inside a 280-word chunk about all sound characteristics — BM25 sees the word "loudness" there but so does every other characteristics-related query. The chunk is too heterogeneous.

Where hybrid won in our broader tests (Test 2, Stage 2): paraphrased question "How fast does velocity change when force is applied?" — BM25 got it right at rank 2 but not rank 1; semantic missed; hybrid surfaced it at rank 2 via RRF. RRF's value is in consistency, not in breakthrough wins.

---

## Part C — Debugging Moments

### C1. Most Frustrating Bug

**Bug:** The `in_example` state flag in the chunker sometimes never exited — causing the rest of a chapter to be absorbed into one enormous "example" chunk.

**Time to fix:** ~35 minutes.

**What I tried first:** Printing chunk word counts — saw one chunk with 600+ words. Thought the overlap logic was broken and rewrote it. Didn't help.

**What was actually happening:** The `ends_here` detection in the chunker uses this regex:
```python
re.search(r'=\s*-?\d+\.?\d*\s*(m|N|J|W|Pa|kg|s|Hz)', para)
```
NCERT Example 9.4 ends with `v2 = -2 m s-1` — the space between `-2` and `m` caused the regex to fail (it expected `-2m` with no space). The example block never closed, and the chunker kept appending paragraphs until the chapter ended.

**The fix:** Changed the regex to allow optional space: `r'=\s*-?\d+\.?\d*\s+?(m|N|J)'` and also added the `current_wc > 50` fallback condition so a block that grows beyond 50 words without triggering the exact regex still eventually commits.

**Fastest path for someone hitting this next week:** Add `print(f"in_example={in_example}, para='{para[:40]}'")` inside the example block loop. The endless state will be immediately visible. Then check whether the exit condition regex matches the actual last line of the example.

### C2. What Still Bothers Me

Q02 ("difference between uniform and non-uniform motion") is wrong because the definition is buried in a large intro chunk. I *know* the fix — split on section headers so each `8.1.2` section is its own chunk. I built the `classify_paragraph` function that identifies section headers. I just didn't close the loop and use section boundaries as forced chunk splits in `chunk_chapter`.

This bothers me because it's an *obvious* fix that I described in the code comments but didn't implement. The comment says "section headers update the current_section tracker" — they should also trigger a chunk commit. That's one line:
```python
if ct == 'section_header':
    if current: commit('concept')    # ← add this
    current_section = para.strip()
    current, current_wc = [], 0
```
I would add this first if given 2 more hours.

---

## Part D — Architecture and Reasoning

### D1. Why Not Just ChatGPT?

In our evaluation, Q03 ("object travels 16 m in 4 s then 16 m in 2 s — average speed?") was answered correctly AND grounded — the retrieved chunk contained the exact worked Example 8.1 from the textbook, the answer showed `32/6 = 5.33 m s⁻¹`, and a teacher reviewing it would recognise NCERT's own language and formatting.

ChatGPT would answer this correctly too. But: its answer would use whatever phrasing its training data associated with average speed — possibly correct but possibly using formulations from a different curriculum, a different level of complexity, or with additional caveats that confuse a Class 9 student ("this assumes constant acceleration, which…"). For a PariShiksha student preparing for a board exam, the *phrasing* matters as much as the answer — examiners want the textbook formula, the textbook units, the textbook language.

More concretely: Q25 ("electric current through a conductor") is out-of-scope for our Ch8–12 corpus. Our system correctly refused it. ChatGPT would confidently explain electrical conductivity — *correctly*, but using content from a chapter the student hasn't studied yet. That answer could confuse more than it helps.

The retrieval system's value isn't that it's more accurate than GPT-4. It's that it's *bounded* — it only answers from the chapters the student is currently studying.

### D2. The GANs Reflection

A GAN trains a generator to produce outputs that fool a discriminator into thinking they're real. The training signal is adversarial: "can you pass as real?" For a grounded textbook assistant, this is exactly the wrong objective.

We want the opposite of a GAN's goal. A GAN optimised on "textbook-sounding answers" would produce fluent, confident, NCERT-style text — whether or not it's correct. That's the hallucination problem we're actively fighting. Our V2 prompt with explicit refusal exists precisely to *prevent* the LLM from generating plausible-sounding text when the context doesn't support it.

The deeper principle: **architecture choice must match the loss function**. GANs optimise perceptual similarity ("does this look real?"). Our system needs factual faithfulness ("is this traceable to the source?"). These are different objectives. Using a GAN here would be like using a style-transfer model to do translation — fluent in the target style, no guarantee the meaning transferred. Retrieval-augmented generation constrains the generation step to content we've already verified exists in the textbook. GANs have no equivalent constraint mechanism.

### D3. Pilot Readiness

My honest answer to "can we launch Monday with 100 students?": **No. Not yet. But closer than before.**

Three things I'd need first:

**1. Section-boundary chunking.** Q01 and Q02 fail because of how we chunk, not how we retrieve. The one-line fix (force chunk commit at section headers) would likely recover 4–5 "wrong" answers and push the score from 36% to ~50%. At 50% with a 25-question eval, we still don't know what the error rate is on the full question space, but it's a better baseline.

**2. Real API evaluation.** Every "wrong" and "partial" result here could be a mock-generation failure, not a system failure. Q01 retrieved the right chunk (8.5 EQUATIONS OF MOTION) but the mock didn't match the key terms. With a real Gemini call at temperature=0, I expect correctness to be 55–65%. I need to run that before claiming the system is pilot-ready.

**3. Adversarial test set.** Every question in our eval was written by the person who built the system. Real students write questions with grammar errors, code-switching ("acceleration matlab kya hota hai"), and half-remembered terms. I'd want the contracted Class 9 teacher to write 20 questions blind — without looking at what chapters we indexed — and run those through the system before launch.

---

## Part E — Effort and Self-Assessment

### E1. Effort Rating: **8/10**

What I'm proud of: the dual retriever (BM25 + TF-IDF semantic with RRF fusion) is properly implemented with clean separation of concerns. The evaluation is honest — 36% is a real score, not a cherry-picked result. The failure analysis traces specific bugs to specific code locations.

What I didn't do: the section-boundary chunking fix I described but didn't implement. That's an hour of work that would meaningfully improve the score. I prioritised breadth (5 chapters, dual retriever, 25 questions) over depth (fixing the known chunking bug).

### E2. Gap Between Me and a Stronger Student

A stronger student would have implemented the section-boundary chunk split — they'd notice that `classify_paragraph` returns `'section_header'` and immediately ask "so why isn't that triggering a chunk commit?" That's 10 minutes of thinking and 1 line of code.

They'd also have added a retrieval score threshold for out-of-scope detection: `if top_score < 0.5: return refusal_text`. We got lucky that both OOS questions (Q24, Q25) triggered the mock's keyword-based refusal. With a real LLM, a low-score retrieval combined with V2's explicit instruction would refuse correctly — but a threshold would make it robust even if the instruction gets ignored.

### E3. Two More Days

**First thing (most critical):** Fix section-boundary chunking, rebuild the chunk store, re-run the full evaluation with a real Gemini API call. This single change would tell me whether our 36% score is a chunking bug or a genuine system limitation. If it moves to 60%+ with real generation, we know what to work on. If it stays at 36%, the problem is deeper.

**Last thing (least urgent):** Add Teacher Mode — have answers cite the exact section: "According to NCERT Class 9, Chapter 9, Section 9.4 (Second Law of Motion): F = ma." This is useful for building teacher and parent trust, but it's a formatting improvement on top of a working system. You don't add polish before you fix the foundation.
