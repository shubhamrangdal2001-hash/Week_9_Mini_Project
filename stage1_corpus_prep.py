"""
stage1_corpus_prep.py
─────────────────────
Stage 1: Corpus Preparation

Steps:
  1A. PDF extraction  (real code shown; synthetic data used here)
  1B. Text cleaning   (fix PDF artifacts)
  1C. Content classification (concept / example / exercise)
  1D. Tokenizer comparison  (Whitespace vs BPE-style vs WordPiece-style)
  1E. Chunking strategy     (300-word target, 50-word overlap, no example splits)

Each decision is explained inline with the reasoning behind it.
"""

import sys, re, json
from pathlib import Path

sys.path.insert(0, '/Users/shubh/Project/Ncert_Rag/corpus')
# from ncert_corpus import ALL_CHAPTERS
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

# ══════════════════════════════════════════════════════════════
# 1A  PDF EXTRACTION  (real code – runs with actual files)
# ══════════════════════════════════════════════════════════════

def extract_real_pdf(pdf_path: str) -> str:
    """
    What you run on the actual NCERT PDF files.

    fitz.open()        → opens the PDF as a Document object
    doc[n]             → access page n (0-indexed)
    page.get_text()    → extracts ALL text from that page
                         mode="text" = plain text, left-to-right
    The loop adds a page marker before each page so we know
    where each chunk came from (useful for Teacher Mode citations).
    """
    import fitz                           # pip install pymupdf
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()            # ← the actual extraction call
        full_text += f"\n--- PAGE {page_num + 1} ---\n"
        full_text += text
    doc.close()
    return full_text

# NCERT chapter URLs (download manually — not committed to repo):
# Ch 8:  https://ncert.nic.in/textbook/pdf/iesc108.pdf
# Ch 9:  https://ncert.nic.in/textbook/pdf/iesc109.pdf
# Ch 10: https://ncert.nic.in/textbook/pdf/iesc110.pdf
# Ch 11: https://ncert.nic.in/textbook/pdf/iesc111.pdf
# Ch 12: https://ncert.nic.in/textbook/pdf/iesc112.pdf


# ══════════════════════════════════════════════════════════════
# 1B  TEXT CLEANING
# ══════════════════════════════════════════════════════════════

# Known fused-word artifacts from NCERT PDFs.
# These happen because the PDF uses column layout — when a word
# wraps to the next line, the space is sometimes lost in extraction.
FUSED_WORD_FIXES = {
    r'\bandothers\b':     'and others',
    r'\bflowsthrough\b':  'flows through',
    r'\bnonuniform\b':    'non-uniform',
    r'\buniformmotion\b': 'uniform motion',
    r'\bcalledits\b':     'called its',
    r'\baboutmotion\b':   'about motion',
    r'\bspecifyinga\b':   'specifying a',
}


def clean_text(raw: str) -> str:
    """
    Remove PDF extraction artifacts without destroying content.

    What we fix:
      1. Fused words from column layout (see FUSED_WORD_FIXES above)
      2. Isolated page headers: lines containing only "SCIENCE"
         (re.MULTILINE makes ^ match start of EACH line, not just start of string)
      3. Figure refs: [Fig. 8.1: caption] → [FIGURE 8.1: caption]
         (we keep them — they're metadata signals for the chunker)
      4. 3+ consecutive blank lines → exactly 2 blank lines
         (normalises paragraph boundaries for our paragraph splitter)
      5. Trailing spaces on each line (rstrip, not strip — preserves indent)

    What we do NOT fix:
      • Broken equations like "v2" (lost superscript) — can't recover
      • Missing diagrams — acknowledged as limitation
      • "ms-1" units — keep as-is; our tokeniser handles them
    """
    text = raw

    # Step 1: Fix fused words
    for pattern, replacement in FUSED_WORD_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Step 2: Remove isolated page headers
    # re.MULTILINE → ^ matches start of any line (not just whole string)
    # \s*SCIENCE\s* → "SCIENCE" with optional surrounding whitespace
    # \n → the newline after the header line
    text = re.sub(r'^\s*SCIENCE\s*\n', '', text, flags=re.MULTILINE)

    # Step 3: Normalise figure references
    # \[Fig\.\s*  → literal "[Fig." with optional space
    # (\d+\.\d+)  → capture group 1: figure number like "8.1"
    # :([^\]]+)   → capture group 2: everything up to the closing "]"
    # \]          → closing bracket
    text = re.sub(r'\[Fig\.\s*(\d+\.\d+):([^\]]+)\]',
                  r'[FIGURE \1:\2]', text)

    # Step 4: Collapse excessive blank lines
    # \n{3,} matches 3 or more consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 5: Remove trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


# ══════════════════════════════════════════════════════════════
# 1C  CONTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def classify_paragraph(para: str) -> str:
    """
    Classify a paragraph into one of five types.

    This matters for chunking because:
    - 'example_problem' triggers the "don't split" mode
    - 'exercise' chunks get their own content_type in metadata
    - 'section_header' updates the current_section tracker
    - 'equation' helps identify formula-heavy paragraphs

    re.match() checks from the START of the string only.
    re.search() would check anywhere — we don't want that here
    because we're looking for specific paragraph openers.
    """
    p = para.strip()

    # Numbered exercise questions: "1. An athlete..."
    # \d+\. → one or more digits followed by a literal period
    # \s+   → one or more whitespace characters
    # [A-Z] → starts with a capital letter (NCERT exercise questions always do)
    if re.match(r'^\d+\.\s+[A-Z]', p):
        return 'exercise'

    # Exercise section header: a line that is only "EXERCISES" (singular or plural)
    # EXERCISES? → E-X-E-R-C-I-S-E followed by optional S
    # \s*$       → optional trailing whitespace, then end of string
    if re.match(r'^EXERCISES?\s*$', p, re.IGNORECASE):
        return 'exercise_header'

    # Worked example start: "EXAMPLE 8.1" or "EXAMPLE 10.3"
    # (EXAMPLE|Example) → either capitalisation
    # \s+ → space
    # \d+\.\d+ → pattern like "8.1" or "10.3"
    if re.match(r'^(EXAMPLE|Example)\s+\d+\.\d+', p):
        return 'example_problem'

    # Section headers: "8.2 MEASURING THE RATE OF MOTION"
    # \d+\.\d* → number.decimal (decimal part optional for top-level sections)
    # \s+ → space
    # [A-Z] → starts with capital
    if re.match(r'^\d+\.\d*\s+[A-Z]', p):
        return 'section_header'

    # Equation lines: contain equation markers like ...(8.1) or start with variable=
    if '...(' in p or re.match(r'^[a-zA-Z]\s*=\s*', p):
        return 'equation'

    return 'concept'


# ══════════════════════════════════════════════════════════════
# 1D  TOKENIZER COMPARISON
# ══════════════════════════════════════════════════════════════

def compare_tokenizers(passages: list) -> None:
    """
    Compare three tokenisation strategies on representative NCERT passages.

    Why this matters:
    - Our BM25 retriever uses whitespace+punctuation tokenisation
    - Our Sentence Transformer (Stage 2) uses its own internal WordPiece tokeniser
    - If we ever add BM25 + reranker, the two tokenisers must be compatible
    - Scientific terms like "ms-2", "v2", "9.8" split differently across strategies

    Strategy 1: Whitespace   → passage.lower().split()
    Strategy 2: BPE-style    → re.findall(r'\w+|[^\w\s]', ...)   keeps subwords together
    Strategy 3: WordPiece    → re.findall(r'[a-zA-Z]+|[0-9]+|[^\w\s]', ...) splits digits
    """
    print("\n" + "═"*68)
    print("1D  TOKENIZER COMPARISON")
    print("═"*68)

    header = f"{'Passage':<12} {'Whitespace':>12} {'BPE-style':>12} {'WordPiece':>12}"
    print(header)
    print("─"*50)

    interesting_terms = ['ms-1', 'ms-2', 'v2', 'u2', '9.8', 'v = u + at',
                         'F = ma', 'PE', 'KE', '10-11']

    for i, passage in enumerate(passages):
        ws  = passage.lower().split()
        bpe = re.findall(r'\w+|[^\w\s]', passage.lower())
        wp  = re.findall(r'[a-zA-Z]+|[0-9]+|[^\w\s]', passage.lower())
        print(f"Passage {i+1:<4} {len(ws):>12} {len(bpe):>12} {len(wp):>12}")

    print("\nHow scientific terms split differently:")
    print(f"  {'Term':<20} {'BPE-style':<25} {'WordPiece'}")
    print("  " + "─"*60)
    for term in interesting_terms:
        bpe_t = re.findall(r'\w+|[^\w\s]', term.lower())
        wp_t  = re.findall(r'[a-zA-Z]+|[0-9]+|[^\w\s]', term.lower())
        if bpe_t != wp_t:
            print(f"  {term:<20} {str(bpe_t):<25} {str(wp_t)}")

    print("\nDecision: whitespace+punctuation for BM25 (consistent index/query time).")
    print("SentenceTransformer handles its own tokenisation internally.")
    print("The mismatch risk is highest when combining BM25 scores with neural scores.")


# ══════════════════════════════════════════════════════════════
# 1E  CHUNKING
# ══════════════════════════════════════════════════════════════

def split_into_paragraphs(text: str) -> list:
    """
    Split cleaned text on blank lines.

    re.split(r'\n\n+', text)  → split on TWO or more consecutive newlines
    The comprehension filters out very short fragments (< 10 chars)
    which are usually stray whitespace left by our cleaning step.
    """
    raw = re.split(r'\n\n+', text)
    return [p.strip() for p in raw if len(p.strip()) >= 10]


def chunk_chapter(text: str, chapter_name: str,
                  target_words: int = 300,
                  overlap_words: int = 50) -> list:
    """
    Convert a cleaned chapter text into a list of chunk dicts.

    CHUNKING STRATEGY — the key decisions:

    target_words = 300
      At 150: worked examples split. Retriever gets the problem (chunk N)
              but not the solution (chunk N+1). LLM re-derives from scratch.
      At 500: too much unrelated content per chunk. A "velocity" query returns
              a chunk covering velocity AND 3 other topics → dilutes attention.
      300 is the sweet spot: fits a complete example AND some context around it.

    overlap_words = 50
      Without overlap, a sentence split across chunk boundaries can only be
      retrieved by the chunk that has its END. The beginning of the sentence
      is in the previous chunk and gets missed.
      With 50-word overlap, the last 50 words of chunk N appear at the start
      of chunk N+1. Either chunk retrieves the complete sentence.

    Special rule: example blocks are NEVER split.
      When we see "EXAMPLE X.Y", we set in_example=True and stop checking
      word count. We keep appending until we detect the solution is complete.
      A complete solution is detected by: "therefore" keyword OR a numeric
      result line (regex for "= some number").
      This is the most important rule — splitting examples from solutions
      is the #1 retrieval failure mode for NCERT content.
    """
    paragraphs = split_into_paragraphs(text)

    chunks       = []
    current      = []      # paragraphs collected for the current chunk
    current_wc   = 0       # word count of current
    in_example   = False   # state flag: are we inside a worked example?
    curr_section = "Introduction"
    chunk_id     = 0

    def commit(content_type: str):
        """Flush current paragraphs as a finished chunk."""
        nonlocal chunk_id
        text_body = '\n\n'.join(p for p in current if p.strip())
        if not text_body.strip():
            return
        safe_name = re.sub(r'[^a-z0-9]+', '_', chapter_name.lower()).strip('_')
        chunks.append({
            'id'           : f"{safe_name}_{chunk_id:03d}",
            'text'         : text_body,
            'chapter'      : chapter_name,
            'section'      : curr_section,
            'content_type' : content_type,
            'word_count'   : len(text_body.split()),
        })
        chunk_id += 1

    for para in paragraphs:
        pw  = len(para.split())
        ct  = classify_paragraph(para)

        # Always update section tracker
        if ct == 'section_header':
            curr_section = para.strip()

        # ── Example block entry ──────────────────────────────
        if ct == 'example_problem':
            # Flush whatever was collected before this example
            if current:
                commit('concept')
            current  = [para]
            current_wc = pw
            in_example = True
            continue

        # ── Inside example block ─────────────────────────────
        if in_example:
            current.append(para)
            current_wc += pw
            # Exit condition: line has "therefore" OR has "= -?number"
            # These are the two ways NCERT example solutions end
            ends_here = (
                'therefore' in para.lower() or
                re.search(r'=\s*-?\d+\.?\d*\s*(m|N|J|W|Pa|kg|s|Hz)', para) or
                (current_wc > 50 and re.search(r'=\s*-?\d+', para))
            )
            if ends_here:
                commit('example')
                # Overlap: carry the last N words into the next chunk
                overlap_text = ' '.join(current[-1].split()[-overlap_words:])
                current    = [overlap_text] if overlap_text.strip() else []
                current_wc = len(overlap_text.split())
                in_example = False
            continue

        # ── Normal paragraph ─────────────────────────────────
        if current_wc + pw > target_words and current:
            ctype = 'exercise' if ct == 'exercise' else 'concept'
            commit(ctype)
            # Overlap: last N words of previous last paragraph
            overlap_text = ' '.join(current[-1].split()[-overlap_words:]) if current else ''
            current    = [overlap_text, para] if overlap_text.strip() else [para]
            current_wc = len(overlap_text.split()) + pw
        else:
            current.append(para)
            current_wc += pw

    # Final flush
    if current:
        commit('concept')

    return chunks


def build_full_chunk_store() -> list:
    """
    Build chunks for all 5 physics chapters, return as a flat list.
    """
    all_chunks = []
    for chapter_name, raw_text in ALL_CHAPTERS.items():
        clean  = clean_text(raw_text)
        chunks = chunk_chapter(clean, chapter_name)
        all_chunks.extend(chunks)
        print(f"  {chapter_name:<42}: {len(chunks):>3} chunks  | "
              f"words: {min(c['word_count'] for c in chunks)}-{max(c['word_count'] for c in chunks)}")
    return all_chunks


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═"*68)
    print("STAGE 1 — CORPUS PREPARATION  (5 NCERT Physics Chapters)")
    print("═"*68)

    # 1B: Clean all chapters
    print("\n1B  CLEANING")
    cleaned = {}
    for name, raw in ALL_CHAPTERS.items():
        c = clean_text(raw)
        cleaned[name] = c
        diff = len(raw) - len(c)
        print(f"  {name:<42}: -{diff:>4} chars cleaned")

    # 1C: Show content-type breakdown for one chapter
    print("\n1C  CONTENT CLASSIFICATION (Chapter 10 sample)")
    ch10_paras = split_into_paragraphs(cleaned["Chapter 10: Gravitation"])
    counts = {}
    for p in ch10_paras:
        t = classify_paragraph(p)
        counts[t] = counts.get(t, 0) + 1
    for t, n in sorted(counts.items()):
        print(f"  {t:<20}: {n}")

    # 1D: Tokenizer comparison
    sample_passages = [
        "The rate of change of velocity is acceleration. a = (v - u) / t. SI unit is ms-2.",
        "F = G × m1 × m2 / d2 where G = 6.673 × 10-11 N m2 kg-2.",
        "KE = (1/2) × m × v2. For m=15 kg and v=4 m s-1: KE = 120 J.",
        "v = f × λ. Speed of sound in air at 25°C = 346 m s-1.",
        "W = F × s = 5 × 2 = 10 J. Power P = W/t = 1000/10 = 100 W.",
    ]
    compare_tokenizers(sample_passages)

    # 1E: Chunking
    print("\n\n1E  CHUNKING  (target=300 words, overlap=50 words)")
    all_chunks = build_full_chunk_store()

    print(f"\n  Total chunks across all chapters: {len(all_chunks)}")
    wcs = [c['word_count'] for c in all_chunks]
    print(f"  Word count: min={min(wcs)}, max={max(wcs)}, avg={sum(wcs)/len(wcs):.0f}")

    type_dist = {}
    for c in all_chunks:
        type_dist[c['content_type']] = type_dist.get(c['content_type'], 0) + 1
    print("  Content type distribution:")
    for t, n in sorted(type_dist.items()):
        print(f"    {t:<18}: {n}")

    # Show one example chunk (must have problem + solution together)
    ex_chunks = [c for c in all_chunks if c['content_type'] == 'example']
    if ex_chunks:
        ec = ex_chunks[0]
        print(f"\n  Example chunk [{ec['id']}] — problem+solution intact:")
        print("  " + "─"*50)
        for line in ec['text'].split('\n')[:10]:
            print(f"  {line}")
        if ec['text'].count('\n') >= 10:
            print("  ...")

    # Save
    out_path = Path('/Users/shubh/Project/Ncert_Rag/chunks/all_chunks.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\n✓ Saved {len(all_chunks)} chunks → {out_path}")
