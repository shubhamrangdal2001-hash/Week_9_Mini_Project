"""
main.py
───────
PariShiksha — NCERT Class 9 Physics Study Assistant
Week 9 Mini-Project · PG Diploma AI-ML · Cohort 2

Run the full pipeline stage by stage, or jump to any stage directly.

USAGE
─────
  python main.py                     # full pipeline (all 4 stages)
  python main.py --stage 1           # only Stage 1 (corpus prep)
  python main.py --stage 2           # only Stage 2 (build retrievers)
  python main.py --stage 3           # only Stage 3 (generation demo)
  python main.py --stage 4           # only Stage 4 (evaluation)
  python main.py --chat              # interactive Q&A after pipeline
  python main.py --stage 4 --chat    # run eval then drop into chat
  python main.py --api-key KEY       # pass Gemini API key directly

STAGES
──────
  Stage 1 · Corpus Preparation
    → cleans 5 NCERT chapters, compares tokenisers,
      chunks into 300-word overlapping segments
    → output: chunks/all_chunks.json

  Stage 2 · Dual Retrieval
    → builds BM25 (lexical) + Sentence Transformer (semantic)
    → fuses them with Reciprocal Rank Fusion (RRF)
    → runs 5 comparison tests showing where each retriever wins

  Stage 3 · Grounded Generation
    → shows prompt V1 vs V2 difference
    → runs 5 demo answers (correct answers + refusals)
    → uses Gemini API if key is set, mock generation otherwise

  Stage 4 · Evaluation
    → runs 25-question eval across all 5 physics chapters
    → scores correctness, grounding, refusal-appropriateness
    → compares BM25-only vs Semantic-only vs Hybrid retrieval
    → saves eval/evaluation_results.csv + .md
"""

import sys
import os
import json
import time
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# ── ensure our project modules are importable ─────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# Every print goes through these so the output looks consistent.
# ══════════════════════════════════════════════════════════════

W = 70   # terminal width

def banner(title: str) -> None:
    """Print a full-width banner."""
    print("\n" + "═" * W)
    pad  = (W - 2 - len(title)) // 2
    print("║" + " " * pad + title + " " * (W - 2 - pad - len(title)) + "║")
    print("═" * W)

def stage_header(n: int, title: str) -> None:
    """Print a stage divider."""
    label = f"  STAGE {n}  {title}"
    print("\n" + "─" * W)
    print(label)
    print("─" * W)

def step(msg: str) -> None:
    """Print an in-progress step."""
    print(f"\n  ▸ {msg}")

def ok(msg: str) -> None:
    """Print a success line."""
    print(f"  ✓ {msg}")

def warn(msg: str) -> None:
    """Print a warning (non-fatal)."""
    print(f"  ⚠ {msg}")

def fail(msg: str) -> None:
    """Print an error and exit."""
    print(f"\n  ✗ ERROR: {msg}")
    sys.exit(1)

def section(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  {'─' * (W - 4)}")
    print(f"  {title}")
    print(f"  {'─' * (W - 4)}")

def wrap(text: str, indent: int = 4) -> str:
    """Word-wrap text at terminal width with indentation."""
    prefix = " " * indent
    return textwrap.fill(text, width=W - indent,
                         initial_indent=prefix,
                         subsequent_indent=prefix)

def progress(items: list, label: str = "") -> None:
    """Print a simple inline progress bar."""
    total = len(items)
    for i, item in enumerate(items, 1):
        bar_len = 30
        filled  = int(bar_len * i / total)
        bar     = "█" * filled + "░" * (bar_len - filled)
        pct     = int(100 * i / total)
        print(f"\r  [{bar}] {pct:3d}%  {label}", end="", flush=True)
        yield item
    print()   # newline after bar completes


# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    argparse.ArgumentParser  — standard library argument parser.

    add_argument():
      '--stage'    → optional int, choices 1-4; if omitted runs all
      '--chat'     → store_true flag; drops into interactive Q&A after pipeline
      '--api-key'  → string; Gemini API key (overrides env variable)
      '--skip-eval'→ skip the heavy eval loop (useful during dev)

    parse_args() reads from sys.argv automatically.
    """
    p = argparse.ArgumentParser(
        description="PariShiksha NCERT RAG — stepwise pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python main.py                       run all 4 stages
          python main.py --stage 1             corpus prep only
          python main.py --stage 2             retriever build only
          python main.py --stage 3             generation demo only
          python main.py --stage 4             evaluation only
          python main.py --chat                full pipeline + interactive chat
          python main.py --stage 3 --chat      generation demo + chat
          python main.py --api-key YOUR_KEY    use real Gemini API
        """)
    )
    p.add_argument(
        '--stage', type=int, choices=[1, 2, 3, 4],
        help='Run a single stage (1-4). Omit to run all stages in order.'
    )
    p.add_argument(
        '--chat', action='store_true',
        help='After pipeline, enter interactive Q&A mode.'
    )
    p.add_argument(
        '--api-key', type=str, default='',
        help='Gemini API key. Falls back to GEMINI_API_KEY environment variable.'
    )
    p.add_argument(
        '--skip-eval', action='store_true',
        help='Skip the 25-question evaluation loop (faster dev iteration).'
    )
    p.add_argument(
        '--chunk-size', type=int, default=300,
        help='Target words per chunk (default: 300). Try 150 or 500 to experiment.'
    )
    p.add_argument(
        '--overlap', type=int, default=50,
        help='Overlap words between chunks (default: 50).'
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# STAGE 1 — CORPUS PREPARATION
# ══════════════════════════════════════════════════════════════

def run_stage1(chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Imports and runs stage1_corpus_prep.py functions.

    Returns:
        list of chunk dicts, each with:
          id, text, chapter, section, content_type, word_count

    What happens inside:
      1. ALL_CHAPTERS  → dict of {chapter_name: raw_text}
         (synthetic data that mirrors real PyMuPDF extraction)
      2. clean_text()  → removes fused words, page headers, normalises whitespace
      3. classify_paragraph() → labels each paragraph: concept/example/exercise/etc
      4. compare_tokenizers() → shows BPE vs WordPiece token counts on 5 passages
      5. chunk_chapter() → state-machine chunker with example-intact rule
      6. build_full_chunk_store() → runs all chapters through 1→5
    """
    stage_header(1, "CORPUS PREPARATION")

    step("Importing stage 1 functions …")
    import inspect
    import importlib

    # ── Import the module first so we can inspect it ──────────
    try:
        _s1 = importlib.import_module("stage1_corpus_prep")
    except ModuleNotFoundError:
        fail("stage1_corpus_prep.py not found. Make sure it is in the "
             "same directory as main.py.")

    # ── Locate the corpus-loader (callable or plain dict) ─────
    # Try every reasonable name the function might have been exported as.
    _CORPUS_FN_NAMES = (
        "stage1_input_corpus",   # original expected name
        "get_input_corpus",
        "input_corpus",
        "load_corpus",
        "get_corpus",
        "get_all_chapters",
        "load_all_chapters",
        "get_chapters",
        "load_chapters",
    )
    _CORPUS_DICT_NAMES = (
        "ALL_CHAPTERS", "CHAPTERS", "all_chapters", "chapters",
    )

    _corpus_loader = None

    # 1) Check for a callable under any known function name
    for _fname in _CORPUS_FN_NAMES:
        if callable(getattr(_s1, _fname, None)):
            _corpus_loader = getattr(_s1, _fname)
            ok(f"Corpus loader found: stage1_corpus_prep.{_fname}()")
            break

    # 2) Fall back to a module-level dict variable
    if _corpus_loader is None:
        for _dname in _CORPUS_DICT_NAMES:
            _val = getattr(_s1, _dname, None)
            if isinstance(_val, dict):
                _corpus_loader = lambda _v=_val: _v   # wrap in callable
                ok(f"Corpus dict found: stage1_corpus_prep.{_dname}")
                break

    # 3) Nothing matched — report what IS available and stop
    if _corpus_loader is None:
        _available_fns = [
            n for n, _ in inspect.getmembers(_s1, inspect.isfunction)
        ]
        _available_dicts = [
            n for n in dir(_s1)
            if isinstance(getattr(_s1, n, None), dict)
        ]
        fail(
            f"Cannot find a corpus-loader in stage1_corpus_prep.\n"
            f"    Functions found : {_available_fns}\n"
            f"    Dicts found     : {_available_dicts}\n"
            f"    Expected one of : {list(_CORPUS_FN_NAMES)}\n"
            f"    Or a dict named : {list(_CORPUS_DICT_NAMES)}\n"
            f"    → Rename your corpus function/variable in "
            f"stage1_corpus_prep.py to match one of the names above."
        )

    # ── Import the remaining stage-1 helpers normally ─────────
    try:
        from stage1_corpus_prep import (
            clean_text,
            classify_paragraph,
            compare_tokenizers,
            split_into_paragraphs,
            chunk_chapter,
            build_full_chunk_store,
        )
    except ImportError as e:
        fail(f"Missing function in stage1_corpus_prep: {e}\n"
             f"    Ensure clean_text, classify_paragraph, "
             f"compare_tokenizers, split_into_paragraphs, "
             f"chunk_chapter, and build_full_chunk_store are all defined.")

    step("Loading Stage 1 corpus source …")
    try:
        ALL_CHAPTERS = _corpus_loader()
    except Exception as e:
        fail(f"Cannot load Stage 1 corpus: {e}")

    # ── 1B: Clean all chapters ────────────────────────────────
    section("1B  Text Cleaning")
    cleaned = {}
    for name, raw in ALL_CHAPTERS.items():
        c = clean_text(raw)
        cleaned[name] = c
        removed = len(raw) - len(c)
        ok(f"{name:<42}  -{removed:>3} chars cleaned")

    # ── 1C: Content type breakdown ────────────────────────────
    section("1C  Content Classification  (Chapter 10 sample)")
    ch10_paras = split_into_paragraphs(cleaned.get("Chapter 10: Gravitation", ""))
    counts: dict = {}
    for p in ch10_paras:
        t = classify_paragraph(p)
        counts[t] = counts.get(t, 0) + 1
    for t, n in sorted(counts.items()):
        print(f"    {t:<22}: {n}")

    # ── 1D: Tokenizer comparison ──────────────────────────────
    section("1D  Tokenizer Comparison")
    sample_passages = [
        "The rate of change of velocity of an object is called its acceleration. a = (v - u) / t. SI unit is ms-2.",
        "F = G × m1 × m2 / d2 where G = 6.673 × 10-11 N m2 kg-2.",
        "KE = (1/2) × m × v2. For m=15 kg and v=4 m s-1: KE = 120 J.",
        "v = f × λ. Speed of sound in air at 25°C = 346 m s-1.",
        "W = F × s = 5 × 2 = 10 J. Power P = W/t = 1000/10 = 100 W.",
    ]
    compare_tokenizers(sample_passages)

    # ── 1E: Chunking ──────────────────────────────────────────
    section(f"1E  Chunking  (target={chunk_size} words, overlap={overlap} words)")
    step("Building chunks for all 5 chapters …")
    all_chunks = []
    for name, raw in ALL_CHAPTERS.items():
        c_text  = clean_text(raw)
        c_chunks = chunk_chapter(c_text, name,
                                 target_words=chunk_size,
                                 overlap_words=overlap)
        all_chunks.extend(c_chunks)
        wcs = [c['word_count'] for c in c_chunks]
        ok(f"{name:<42}  {len(c_chunks):>3} chunks  "
           f"words:{min(wcs)}–{max(wcs)}")

    print()
    wcs_all = [c['word_count'] for c in all_chunks]
    ok(f"Total chunks: {len(all_chunks)}  |  "
       f"words: min={min(wcs_all)}, max={max(wcs_all)}, "
       f"avg={sum(wcs_all)//len(wcs_all)}")

    # Content type distribution
    type_dist: dict = {}
    for c in all_chunks:
        type_dist[c['content_type']] = type_dist.get(c['content_type'], 0) + 1
    print("\n    Content type distribution:")
    for t, n in sorted(type_dist.items()):
        bar = "█" * n
        print(f"      {t:<18}  {bar}  ({n})")

    # Show one intact example chunk
    ex_chunks = [c for c in all_chunks if c['content_type'] == 'example']
    if ex_chunks:
        ec = ex_chunks[0]
        section(f"Sample EXAMPLE chunk  [{ec['id']}]  — problem+solution intact")
        for line in ec['text'].split('\n')[:8]:
            print(f"    {line}")
        if ec['text'].count('\n') > 8:
            print("    …")

    # ── Save ──────────────────────────────────────────────────
    out = PROJECT_ROOT / 'chunks' / 'all_chunks.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    ok(f"Saved → {out}  ({len(all_chunks)} chunks)")

    return all_chunks


# ══════════════════════════════════════════════════════════════
# STAGE 2 — DUAL RETRIEVAL
# ══════════════════════════════════════════════════════════════

def run_stage2(all_chunks: list):
    """
    Builds BM25Retriever, SentenceTransformerRetriever, HybridRetriever.

    Runs 5 comparison queries showing:
      - Which retriever finds the correct section at rank 1
      - Where BM25 beats semantic and vice-versa
      - How hybrid RRF combines both

    Returns:
        HybridRetriever instance  (used in Stages 3 and 4)
    """
    stage_header(2, "DUAL RETRIEVAL  (BM25 + Sentence Transformer + Hybrid RRF)")

    step("Importing retrieval classes …")
    from stage2_retrieval import (
        BM25Retriever,
        SentenceTransformerRetriever,
        HybridRetriever,
    )

    # ── Build retrievers ──────────────────────────────────────
    section("2A  Building BM25 Index")
    bm25_ret = BM25Retriever(all_chunks)

    section("2B  Building Sentence Transformer (TF-IDF semantic vectors)")
    sem_ret = SentenceTransformerRetriever(all_chunks)
    vocab   = sem_ret.corpus_matrix.shape[1]
    ok(f"Vocab size: {vocab} dimensions  |  "
       f"Matrix: {len(all_chunks)} × {vocab}")

    section("2C  Building Hybrid Retriever (BM25 + Semantic via RRF k=60)")
    retriever = HybridRetriever(all_chunks)

    # ── Comparison tests ──────────────────────────────────────
    section("Retriever Comparison  (5 test queries)")

    test_cases = [
        # (query, expected_keyword_in_section, description)
        ("What is Newton's second law of motion?",
         "SECOND LAW",
         "Direct keyword match — BM25 should win"),

        ("How fast does velocity change when force is applied?",
         "RATE OF CHANGE",
         "Paraphrased — Semantic should help"),

        ("Why does a ship float but a stone sinks?",
         "BUOYANCY",
         "Conceptual — no exact term 'float' in section title"),

        ("Calculate kinetic energy of a 15 kg object at 4 m/s",
         "ENERGY",
         "Calculation with numbers — BM25 numbers + Semantic concept"),

        ("What is the minimum distance for an echo to be heard?",
         "ECHO",
         "Specific fact — should hit Ch12 sound content"),
    ]

    hdr = f"  {'Query':<42} {'BM25':>8} {'Semantic':>10} {'Hybrid':>8}"
    print(hdr)
    print("  " + "─" * (W - 2))

    bm25_wins = sem_wins = hyb_wins = 0
    for q, kw, desc in test_cases:
        b_top = bm25_ret.retrieve(q, 1)[0]['section']
        s_top = sem_ret.retrieve(q,  1)[0]['section']
        h_top = retriever.retrieve(q, 1)[0]['section']

        b_ok = "✓" if kw in b_top.upper() else "✗"
        s_ok = "✓" if kw in s_top.upper() else "✗"
        h_ok = "✓" if kw in h_top.upper() else "✗"

        if b_ok == "✓": bm25_wins += 1
        if s_ok == "✓": sem_wins  += 1
        if h_ok == "✓": hyb_wins  += 1

        print(f"  {q[:40]:<42} {b_ok:>8} {s_ok:>10} {h_ok:>8}")
        print(f"  {'↳ ' + desc:<42}")

    print("  " + "─" * (W - 2))
    print(f"  {'Correct rank-1':<42} {bm25_wins:>8} {sem_wins:>10} {hyb_wins:>8} / {len(test_cases)}")

    # ── Detailed comparison on one query ─────────────────────
    section("Detailed comparison: 'What is Newton's second law?'")
    retriever.compare_retrievers("What is Newton's second law?", k=3)

    ok("Hybrid retriever ready")
    return retriever


# ══════════════════════════════════════════════════════════════
# STAGE 3 — GROUNDED GENERATION
# ══════════════════════════════════════════════════════════════

def run_stage3(retriever, api_key: str = ""):
    """
    Builds GroundedAnswerSystem and runs 6 demonstration answers.

    Shows:
      1. Prompt V1 vs V2 — what changed and why
      2. 3 correct grounded answers (direct + cross-chapter + calculation)
      3. 2 refusal cases (obvious OOS + adversarial OOS)
      4. 1 paraphrased question

    Returns:
        GroundedAnswerSystem instance  (used in Stage 4 and chat mode)
    """
    stage_header(3, "GROUNDED ANSWER GENERATION")

    step("Importing generation module …")
    from stage3_generation import (
        GroundedAnswerSystem,
        PROMPT_V1,
        PROMPT_V2,
        build_context_block,
    )

    # ── Show prompt comparison ────────────────────────────────
    section("Prompt V1 vs V2 — why the refusal instruction changed")

    print("\n  V1 (permissive — what most people write first):")
    for line in PROMPT_V1.strip().split('\n')[:4]:
        print(f"    {line}")
    print()
    print("  V2 (constraint — explicit refuse rule + prescribed text):")
    for line in PROMPT_V2.strip().split('\n')[:8]:
        print(f"    {line}")
    print()
    print(wrap("Key change: V1 says 'answer using ONLY context' — the LLM reads "
               "this as a preference. V2 says 'Answer ONLY IF directly relevant' "
               "— this is a conditional that forces a relevance check first. "
               "The prescribed refusal text makes is_refusal flag deterministic."))

    # ── Build the system ──────────────────────────────────────
    section("Building GroundedAnswerSystem")
    system = GroundedAnswerSystem(retriever, api_key=api_key)

    # ── Demo questions ────────────────────────────────────────
    demo_questions = [
        {
            'q'    : "What is Newton's second law of motion? Write the formula.",
            'type' : "Direct — Ch9 (Force)",
            'expect': "grounded answer"
        },
        {
            'q'    : "What is the difference between kinetic energy and potential energy?",
            'type' : "Cross-chapter — Ch11 (Work & Energy)",
            'expect': "grounded answer"
        },
        {
            'q'    : "A bullet of 20 g is fired from a 4 kg gun at 400 m/s. Find recoil velocity.",
            'type' : "Calculation — Ch9 conservation of momentum",
            'expect': "step-by-step calculation"
        },
        {
            'q'    : "How do we measure the rate at which velocity changes over time?",
            'type' : "Paraphrased — asking about 'acceleration' differently",
            'expect': "grounded answer"
        },
        {
            'q'    : "Explain the process of photosynthesis in plants.",
            'type' : "Out-of-scope — Biology, Ch1",
            'expect': "REFUSAL"
        },
        {
            'q'    : "How does electric current flow through a copper wire?",
            'type' : "Adversarial OOS — Physics but Ch13 (not in corpus)",
            'expect': "REFUSAL"
        },
    ]

    section(f"Demo Answers  ({len(demo_questions)} questions)")

    for i, demo in enumerate(demo_questions, 1):
        print(f"\n  ── Q{i}: {demo['type']} {'─' * (W - 14 - len(demo['type']))}")
        print(f"  Question : {demo['q']}")
        print(f"  Expected : {demo['expect']}")

        result = system.answer(demo['q'])

        top = result['retrieved_chunks'][0] if result['retrieved_chunks'] else {}
        print(f"  Top chunk: [{top.get('chapter','?')}]  "
              f"{top.get('section','?')}"
              f"  (score={top.get('rrf_score', top.get('bm25_score','?'))})")

        status = "✓ REFUSAL" if result['is_refusal'] else "→ ANSWERED"
        if not result['is_refusal'] and demo['expect'] == "REFUSAL":
            status = "✗ SHOULD HAVE REFUSED  ← hallucination risk"

        print(f"\n  Answer   :")
        for line in result['answer'].split('\n'):
            print(f"    {line}")
        print(f"\n  Status   : {status}")

    ok("Stage 3 complete")
    return system


# ══════════════════════════════════════════════════════════════
# STAGE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════

def run_stage4(system, retriever, skip: bool = False):
    """
    Runs the full 25-question evaluation across all 5 chapters.

    Scoring axes:
      correctness  — correct / partial / wrong / missed_refusal /
                     correct_refusal / incorrect_refusal
      grounding    — grounded / partial / ungrounded / na
      refusal      — correct_refusal / missed_refusal (OOS only)

    Also runs BM25 vs Semantic vs Hybrid retriever comparison on 5 probes.
    Saves eval/evaluation_results.csv + eval/evaluation_results.md

    Returns:
        list of result dicts
    """
    stage_header(4, "EVALUATION  (25 questions · 5 chapters)")

    step("Importing evaluation module …")
    from stage4_evaluation import (
        EVAL_SET,
        score_correctness,
        score_grounding,
        compare_retriever_performance,
        run_full_evaluation,
        print_summary,
        save_results,
    )

    if skip:
        warn("--skip-eval flag set — skipping 25-question evaluation loop")
        warn("Remove --skip-eval to see full results")
        return []

    # ── Run all 25 questions ──────────────────────────────────
    section(f"Running {len(EVAL_SET)} evaluation questions")

    print(f"\n  {'':5} {'Type':<14} {'Result':<22} {'Ch':<6} Question")
    print("  " + "─" * (W - 2))

    results = []
    t0 = time.time()
    for eq in EVAL_SET:
        r = system.answer(eq['q'])
        correctness = score_correctness(
            r['answer'], eq['key_terms'], r['is_refusal'], eq['expected'])
        grounding = score_grounding(
            r['answer'], r['retrieved_chunks'], r['is_refusal'])

        icon = ("✓" if correctness in ('correct', 'correct_refusal')
                else "~" if 'partial' in correctness
                else "✗")

        top = r['retrieved_chunks'][0] if r['retrieved_chunks'] else {}
        results.append({
            **eq,
            'answer'      : r['answer'],
            'is_refusal'  : r['is_refusal'],
            'correctness' : correctness,
            'grounding'   : grounding,
            'top_section' : top.get('section', ''),
            'top_score'   : top.get('rrf_score', top.get('bm25_score', 0)),
        })

        print(f"  {icon} {eq['id']:<4} {eq['type']:<14} "
              f"{correctness:<22} {eq['chapter']:<6} "
              f"{eq['q'][:35]}")

    elapsed = time.time() - t0
    print(f"\n  Completed {len(results)} questions in {elapsed:.1f}s")

    # ── Print summary ─────────────────────────────────────────
    total_c, total_n = print_summary(results)

    # ── Retriever comparison ──────────────────────────────────
    section("Retriever Comparison  (BM25 vs Semantic vs Hybrid)")
    compare_retriever_performance(
        system.retriever.chunks if hasattr(system, 'retriever') else [],
        EVAL_SET
    )

    # ── Save results ──────────────────────────────────────────
    section("Saving Results")
    out_dir = str(PROJECT_ROOT / 'eval')
    save_results(results, out_dir)

    # ── Failure analysis ──────────────────────────────────────
    failures = [r for r in results
                if r['correctness'] not in ('correct', 'correct_refusal')]
    if failures:
        section(f"Failure Analysis  ({len(failures)} non-correct results)")
        for f in failures[:4]:
            print(f"\n  {f['id']}  [{f['chapter']}]  {f['q'][:55]}")
            print(f"    Correctness : {f['correctness']}")
            print(f"    Top chunk   : {f['top_section'][:40]}")

            if f['correctness'] == 'missed_refusal':
                print(wrap("Root cause: GENERATION — retriever returned plausible but "
                           "wrong chunks; prompt's refusal instruction didn't fire. "
                           "Fix: add score threshold check before generation.", 6))
            elif f['correctness'] in ('partial', 'wrong'):
                print(wrap("Root cause: RETRIEVAL or CHUNKING — content either split "
                           "across chunk boundaries or section header not used as "
                           "chunk boundary. Fix: force commit on section_header.", 6))
            elif f['correctness'] == 'incorrect_refusal':
                print(wrap("Root cause: Over-conservative — answer IS in corpus "
                           "but retrieval score was low. Check if the relevant "
                           "section is buried in a larger chunk.", 6))

    print(f"\n  ═══ FINAL SCORE: {total_c}/{total_n} "
          f"({total_c/total_n*100:.0f}%) ═══")

    return results


# ══════════════════════════════════════════════════════════════
# INTERACTIVE CHAT MODE
# ══════════════════════════════════════════════════════════════

def run_chat(system) -> None:
    """
    Drop into an interactive Q&A session after the pipeline.

    Commands:
      :quit / :q       → exit chat
      :chunks          → show number of chunks loaded
      :history         → show all questions asked this session
      :debug           → toggle debug mode (shows retrieved chunks)
      :chapter N       → filter to a specific chapter (1=Ch8 … 5=Ch12)

    The main loop:
      1. Read a question from stdin
      2. Call system.answer(question)
      3. Print the answer and retrieved chunk info
      4. Loop until :quit
    """
    banner("INTERACTIVE Q&A  —  NCERT Class 9 Physics")

    chapter_map = {
        "1": "Chapter 8",
        "2": "Chapter 9",
        "3": "Chapter 10",
        "4": "Chapter 11",
        "5": "Chapter 12",
    }

    print(wrap("Ask any physics question from Class 9 NCERT "
               "(Ch 8–12). The system answers from the textbook "
               "and refuses if the topic is not covered."))
    print()
    print("  Commands:  :quit  :history  :debug  :chunks  :chapter N")
    print("  Chapters:  1=Motion  2=Force  3=Gravitation  "
          "4=Work/Energy  5=Sound")
    print()

    history   = []
    debug     = False
    ch_filter = None     # None = search all chapters

    while True:
        try:
            # Input prompt — shows current chapter filter if active
            ch_label = (f"[{chapter_map[ch_filter]}] "
                        if ch_filter and ch_filter in chapter_map
                        else "")
            raw = input(f"  {ch_label}You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────
        if raw.lower() in (':quit', ':q', 'quit', 'exit'):
            print("\n  Goodbye!\n")
            break

        if raw.lower() == ':chunks':
            n = len(system.retriever.chunks)
            print(f"\n  {n} chunks loaded across 5 chapters\n")
            continue

        if raw.lower() == ':debug':
            debug = not debug
            print(f"\n  Debug mode {'ON' if debug else 'OFF'}\n")
            continue

        if raw.lower() == ':history':
            if not history:
                print("\n  No questions asked yet.\n")
            else:
                print(f"\n  Session history ({len(history)} questions):")
                for j, hq in enumerate(history, 1):
                    print(f"    {j}. {hq}")
                print()
            continue

        if raw.lower().startswith(':chapter'):
            parts = raw.split()
            if len(parts) == 2 and parts[1] in chapter_map:
                ch_filter = parts[1]
                print(f"\n  Filtering to {chapter_map[ch_filter]}\n")
            elif len(parts) == 1:
                ch_filter = None
                print("\n  Chapter filter cleared — searching all chapters\n")
            else:
                print(f"\n  Usage: :chapter 1-5  (or :chapter to clear)\n")
            continue

        # ── Answer the question ───────────────────────────────
        history.append(raw)
        print()

        result = system.answer(raw, k=3)

        # Status line
        if result['is_refusal']:
            status_line = "  ⊘ Out of scope"
        else:
            top = result['retrieved_chunks'][0]
            ch  = top.get('chapter', '?')
            sec = top.get('section', '?')
            sc  = top.get('rrf_score', top.get('bm25_score', 0))
            status_line = f"  ↳ Source: {ch} › {sec}  (score={sc:.3f})"

        # Print answer
        print(f"  Assistant ›")
        for line in result['answer'].split('\n'):
            print(f"    {line}")
        print(status_line)

        # Debug: show all retrieved chunks
        if debug:
            print(f"\n  ── Retrieved chunks ──")
            for rc in result['retrieved_chunks']:
                mode = rc.get('retrieval_mode', 'hybrid')
                score = rc.get('rrf_score', rc.get('bm25_score', 0))
                print(f"    [{rc.get('chapter','?')}]  "
                      f"{rc.get('section','?')[:35]}  "
                      f"mode={mode}  score={score:.4f}")

        print()


# ══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def load_chunks_from_disk() -> list:
    """
    Load pre-built chunks from JSON if Stage 1 was already run.
    Used when --stage 2/3/4 is passed without running Stage 1 first.
    """
    chunk_file = PROJECT_ROOT / 'chunks' / 'all_chunks.json'
    if not chunk_file.exists():
        fail(f"chunks/all_chunks.json not found. Run Stage 1 first:\n"
             f"  python main.py --stage 1")
    with open(chunk_file) as f:
        chunks = json.load(f)
    ok(f"Loaded {len(chunks)} chunks from {chunk_file}")
    return chunks


def print_startup_info(args: argparse.Namespace) -> None:
    """Print configuration at startup."""
    banner("PariShiksha  NCERT Class 9 Physics RAG")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Stage    : {'all' if not args.stage else args.stage}")
    print(f"  API      : {'Gemini (real)' if (args.api_key or os.environ.get('GEMINI_API_KEY')) else 'mock generation'}")
    print(f"  Chunks   : target={args.chunk_size} words, overlap={args.overlap} words")
    print(f"  Chat     : {'yes' if args.chat else 'no'}")
    print(f"  Skip eval: {'yes' if args.skip_eval else 'no'}")
    print()


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Main orchestrator. Runs stages in order based on --stage flag.

    State is threaded through function returns:
      Stage 1 → returns all_chunks (list)
      Stage 2 → returns retriever  (HybridRetriever)
      Stage 3 → returns system     (GroundedAnswerSystem)
      Stage 4 → returns results    (list of eval dicts)

    If --stage N is given, earlier stages load from disk rather
    than re-running, so you can iterate on any single stage without
    re-running everything before it.
    """
    print_startup_info(args)

    # Resolve API key: CLI arg > env variable > empty string (→ mock)
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY', '')

    # ── Track what we have ────────────────────────────────────
    all_chunks = None
    retriever  = None
    system     = None

    run_all = args.stage is None   # True if no --stage given

    # ────── STAGE 1 ──────────────────────────────────────────
    if run_all or args.stage == 1:
        all_chunks = run_stage1(
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

    # ────── STAGE 2 ──────────────────────────────────────────
    if run_all or args.stage == 2:
        if all_chunks is None:
            step("Stage 1 not run — loading chunks from disk …")
            all_chunks = load_chunks_from_disk()
        retriever = run_stage2(all_chunks)

    # ────── STAGE 3 ──────────────────────────────────────────
    if run_all or args.stage == 3:
        if all_chunks is None:
            step("Loading chunks from disk …")
            all_chunks = load_chunks_from_disk()
        if retriever is None:
            step("Stage 2 not run — building retrievers now …")
            from stage2_retrieval import HybridRetriever
            retriever = HybridRetriever(all_chunks)
        system = run_stage3(retriever, api_key=api_key)

    # ────── STAGE 4 ──────────────────────────────────────────
    if run_all or args.stage == 4:
        if all_chunks is None:
            step("Loading chunks from disk …")
            all_chunks = load_chunks_from_disk()
        if retriever is None:
            step("Building retrievers …")
            from stage2_retrieval import HybridRetriever
            retriever = HybridRetriever(all_chunks)
        if system is None:
            step("Building answer system …")
            from stage3_generation import GroundedAnswerSystem
            system = GroundedAnswerSystem(retriever, api_key=api_key)
        run_stage4(system, retriever, skip=args.skip_eval)

    # ────── INTERACTIVE CHAT ──────────────────────────────────
    if args.chat:
        if system is None:
            # Build system if we didn't run Stage 3 or 4
            if all_chunks is None:
                all_chunks = load_chunks_from_disk()
            if retriever is None:
                from stage2_retrieval import HybridRetriever
                retriever = HybridRetriever(all_chunks)
            from stage3_generation import GroundedAnswerSystem
            system = GroundedAnswerSystem(retriever, api_key=api_key)
        run_chat(system)

    # ────── DONE ──────────────────────────────────────────────
    if not args.chat:
        banner("PIPELINE COMPLETE")
        print(f"  Finished : {datetime.now().strftime('%H:%M:%S')}")
        if args.stage is None or args.stage == 4:
            csv_path = PROJECT_ROOT / 'eval' / 'evaluation_results.csv'
            md_path  = PROJECT_ROOT / 'eval' / 'evaluation_results.md'
            if csv_path.exists():
                ok(f"Results → {csv_path}")
            if md_path.exists():
                ok(f"Results → {md_path}")
        print()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows consoles often default to cp1252 which cannot print box-drawing
    # characters used in banners. Reconfigure to UTF-8 if supported.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    run_pipeline(args)
