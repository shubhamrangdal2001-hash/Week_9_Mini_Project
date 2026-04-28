"""
stage2_retrieval.py
────────────────────
Stage 2: Dual Retrieval System

Two retrievers run in parallel and their scores are combined:

  Retriever A — BM25 (lexical)
    • Term-frequency × inverse-document-frequency scoring
    • Fast, exact keyword matching
    • Weakness: misses paraphrased questions ("rate of velocity change"
      vs "acceleration") if the exact word doesn't appear

  Retriever B — Sentence Transformer (semantic / dense)
    • Encodes query and chunks as dense vectors
    • Finds chunks that MEAN the same thing even with different words
    • Implemented here with TF-IDF vectors + cosine similarity
      (same mathematical principle as real sentence transformers;
       production version uses all-MiniLM-L6-v2 from HuggingFace)

  Hybrid fusion (Reciprocal Rank Fusion):
    • Neither retriever is always best alone
    • BM25 wins on exact keyword queries (formula lookups, unit questions)
    • Semantic wins on paraphrased or conceptual queries
    • RRF combines both ranked lists without needing to normalise scores

Every line is explained with the reasoning behind the decision.
"""

import sys, re, json, math
import numpy as np
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════
# TOKENISATION  (shared between BM25 index and BM25 queries)
# ══════════════════════════════════════════════════════════════

# Stopwords as a Python SET, not a list.
# Why set? Membership test `x in STOPWORDS` is O(1) for sets,
# O(n) for lists. With thousands of tokens per document, this matters.
STOPWORDS = {
    'the','a','an','is','in','on','at','to','for','of','and','or',
    'it','its','by','be','are','was','were','this','that','with',
    'from','not','as','but','we','can','will','have','has','had',
    'do','does','when','if','then','than','so','such','also','both',
    'each','which','what','how','where','there','their','an','about',
    'into','over','after','just','more','other','some','these','those',
}


def bm25_tokenize(text: str) -> list:
    """
    Tokeniser for BM25 — used IDENTICALLY at index time and query time.

    text.lower()
      Lowercase first so "Force" and "force" are the same token.

    re.findall(r'[a-z0-9]+', text)
      Match sequences of letters OR digits. This strips all punctuation
      automatically — a comma, equals sign, hyphen produce no token.
      "ms-2" → ["ms", "2"]
      "F=ma" → ["f", "ma"]
      "9.8"  → ["9", "8"]

    Filter: remove stopwords AND remove single-char tokens UNLESS digit.
      We keep single digits (appear in formulas like "2" in "F = 2 N").
      We drop single letters ("a", "v") — they're noise in BM25 but
      meaningful in formulae — trade-off accepted for now.
    """
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


# ══════════════════════════════════════════════════════════════
# SENTENCE TRANSFORMER  (TF-IDF implementation)
# ══════════════════════════════════════════════════════════════

class SentenceTransformerRetriever:
    """
    Dense semantic retriever using TF-IDF vectors + cosine similarity.

    How it differs from BM25:
      BM25 — counts exact term matches, weighted by rarity
      This  — builds a vector for each chunk where each dimension
              is a TF-IDF score for a vocabulary term. Two chunks
              with similar MEANING but different WORDS will still
              have similar vector directions → high cosine similarity.

    Why TF-IDF vectors approximate sentence transformers:
      Real sentence transformers (all-MiniLM-L6-v2) use attention
      mechanisms to encode contextual meaning. TF-IDF is simpler
      but operates on the same core intuition: represent text as a
      vector in "meaning space" and measure angle between vectors.
      For domain-specific retrieval (one textbook, consistent vocabulary),
      TF-IDF performs surprisingly close to neural embeddings.

    Production upgrade path:
      Replace the TfidfVectorizer with:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([c['text'] for c in chunks])
      Everything else (cosine_similarity, ranking) stays identical.
    """

    def __init__(self, chunks: list):
        self.chunks = chunks
        corpus_texts = [c['text'] for c in chunks]

        # TfidfVectorizer parameters:
        #   ngram_range=(1,2)  → include unigrams AND bigrams
        #     "kinetic energy" as a bigram is more specific than
        #     "kinetic" and "energy" separately
        #   min_df=1           → include terms that appear in ≥1 doc
        #     (small corpus — we can't afford min_df=2)
        #   max_df=0.85        → exclude terms appearing in >85% of docs
        #     These are corpus-specific stopwords (e.g. "object" in
        #     physics is everywhere and thus uninformative)
        #   sublinear_tf=True  → use log(tf) + 1 instead of raw tf
        #     Prevents a term that appears 100× in one doc from
        #     dominating over one that appears 10×
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85,
            sublinear_tf=True
        )

        # fit_transform does two things:
        #   fit   → learns the vocabulary and IDF weights from corpus
        #   transform → converts each document to a TF-IDF vector
        # Result is a sparse matrix of shape (n_chunks, vocab_size)
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)

        vocab_size = self.corpus_matrix.shape[1]
        print(f"  SentenceTransformer: {len(chunks)} chunks × {vocab_size} vocab dims")

    def encode(self, texts: list) -> np.ndarray:
        """
        Convert a list of strings to TF-IDF vectors.

        vectorizer.transform (NOT fit_transform) uses the already-learned
        vocabulary — we don't refit on query text. This is critical:
        query terms not in the training vocabulary get zero weight,
        which is correct behaviour.

        toarray() converts sparse matrix to dense numpy array.
        """
        return self.vectorizer.transform(texts).toarray()

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Retrieve top-k chunks by cosine similarity to the query.

        Steps:
          1. Encode query → vector of shape (1, vocab_size)
          2. cosine_similarity(query_vec, corpus_matrix)
             → shape (1, n_chunks) — one score per chunk
          3. [0] extracts the first row (since query is a single doc)
          4. argsort()[::-1][:k] → indices of top-k scores, descending
        """
        query_vec = self.encode([query])           # shape: (1, V)
        scores    = cosine_similarity(query_vec,
                       self.corpus_matrix.toarray())[0]  # shape: (N,)

        # argsort gives ascending order, [::-1] reverses to descending
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k_idx):
            chunk = self.chunks[idx].copy()
            chunk['semantic_score'] = round(float(scores[idx]), 4)
            chunk['semantic_rank']  = rank + 1
            results.append(chunk)
        return results


# ══════════════════════════════════════════════════════════════
# BM25 RETRIEVER
# ══════════════════════════════════════════════════════════════

class BM25Retriever:
    """
    Lexical retriever using BM25Okapi.

    BM25Okapi is the Okapi BM25 variant with parameters:
      k1 = 1.5  (term frequency saturation — higher means TF matters more)
      b  = 0.75 (length normalisation — 0 = no normalisation, 1 = full)

    These defaults work well for textbook content. NCERT chapters have
    consistent sentence length, so length normalisation matters less here
    than in general web text.
    """

    def __init__(self, chunks: list):
        self.chunks = chunks
        # Tokenise corpus ONCE at index time
        # This is the heavy part — after this, scoring is fast
        self.corpus_tokens = [bm25_tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        avg_toks = sum(len(t) for t in self.corpus_tokens) / len(self.corpus_tokens)
        print(f"  BM25: {len(chunks)} chunks, avg {avg_toks:.0f} tokens/chunk")

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Score all chunks against tokenised query, return top-k.

        bm25.get_scores(query_tokens)
          → numpy array of shape (n_chunks,)
          Each score is a BM25 relevance score for that chunk.

        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
          → Sort INDICES by their score value.
          We sort indices (not scores) so we can retrieve the original
          chunk object by index number.
        """
        query_tokens = bm25_tokenize(query)   # same tokeniser as index time
        scores       = self.bm25.get_scores(query_tokens)
        top_k_idx    = sorted(range(len(scores)),
                              key=lambda i: scores[i],
                              reverse=True)[:k]

        results = []
        for rank, idx in enumerate(top_k_idx):
            chunk = self.chunks[idx].copy()
            chunk['bm25_score'] = round(float(scores[idx]), 3)
            chunk['bm25_rank']  = rank + 1
            results.append(chunk)
        return results


# ══════════════════════════════════════════════════════════════
# HYBRID RETRIEVER  (BM25 + Semantic, fused with RRF)
# ══════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Combines BM25 and semantic retrieval using Reciprocal Rank Fusion.

    Why RRF instead of just averaging scores?
      BM25 scores and cosine similarity scores live on different scales.
      BM25 might give 12.4 for the best result; cosine gives 0.87.
      Directly averaging these is meaningless.

      RRF converts each raw score into a rank, then combines ranks:
        RRF(d) = Σ [ 1 / (k + rank_in_system_i(d)) ]
      where k=60 is a smoothing constant (standard default from the 2009 paper).

      Intuition: if chunk A is rank 1 in BM25 and rank 3 in semantic,
      it gets: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
      A chunk that is rank 1 in BOTH gets: 1/61 + 1/61 = 0.03279
      The consistently top-ranked chunk wins, even if it's not #1 in either.
    """

    RRF_K = 60   # Standard RRF constant from Cormack et al. 2009

    def __init__(self, chunks: list):
        self.chunks   = chunks
        self.bm25_ret = BM25Retriever(chunks)
        self.sem_ret  = SentenceTransformerRetriever(chunks)
        # Map chunk id to index for O(1) lookup during fusion
        self.id_to_idx = {c['id']: i for i, c in enumerate(chunks)}
        print(f"  Hybrid retriever ready ({len(chunks)} chunks)")

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Full hybrid retrieval pipeline.

        1. Get BM25 top-N results     (N = k * 3 to get enough candidates)
        2. Get Semantic top-N results
        3. Build a score dict: {chunk_id: rrf_score}
           For each retrieved chunk, add 1/(RRF_K + rank) to its RRF score
        4. Sort by RRF score descending, return top-k
        5. Annotate with both individual scores for transparency
        """
        N = min(k * 3, len(self.chunks))   # search wider than we return

        bm25_results = self.bm25_ret.retrieve(query, k=N)
        sem_results  = self.sem_ret.retrieve(query,  k=N)

        # Build chunk_id → individual scores lookup
        bm25_scores = {r['id']: (r['bm25_score'], r['bm25_rank'])
                       for r in bm25_results}
        sem_scores  = {r['id']: (r['semantic_score'], r['semantic_rank'])
                       for r in sem_results}

        # RRF fusion
        # Collect all unique chunk IDs from both result sets
        all_ids = set(r['id'] for r in bm25_results) | \
                  set(r['id'] for r in sem_results)

        rrf_scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in bm25_scores:
                rank   = bm25_scores[cid][1]  # rank from BM25
                score += 1.0 / (self.RRF_K + rank)
            if cid in sem_scores:
                rank   = sem_scores[cid][1]   # rank from semantic
                score += 1.0 / (self.RRF_K + rank)
            rrf_scores[cid] = score

        # Sort by RRF score descending, take top-k
        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

        results = []
        for final_rank, cid in enumerate(top_ids, 1):
            idx   = self.id_to_idx[cid]
            chunk = self.chunks[idx].copy()

            # Annotate with all scores for debugging and transparency
            chunk['rrf_score']      = round(rrf_scores[cid], 5)
            chunk['bm25_score']     = bm25_scores.get(cid, (0, '-'))[0]
            chunk['semantic_score'] = sem_scores.get(cid, (0, '-'))[0]
            chunk['final_rank']     = final_rank
            chunk['retrieval_mode'] = self._mode(cid, bm25_scores, sem_scores)
            results.append(chunk)

        return results

    def _mode(self, cid, bm25_dict, sem_dict) -> str:
        """Label how a chunk was retrieved — useful for debugging."""
        in_bm25 = cid in bm25_dict
        in_sem  = cid in sem_dict
        if in_bm25 and in_sem: return 'hybrid'
        if in_bm25:            return 'bm25_only'
        return                        'semantic_only'

    def compare_retrievers(self, query: str, k: int = 3) -> None:
        """
        Print side-by-side comparison of BM25-only, Semantic-only, Hybrid.
        Very useful for debugging and for the reflection write-up.
        """
        bm25_r = self.bm25_ret.retrieve(query, k)
        sem_r  = self.sem_ret.retrieve(query, k)
        hyb_r  = self.retrieve(query, k)

        print(f"\nQuery: '{query}'")
        print(f"\n{'Rank':<6} {'BM25 top chunks':<38} {'Semantic top chunks':<38} {'Hybrid top chunks'}")
        print("─" * 110)
        for i in range(k):
            b  = bm25_r[i]['section'][:30] if i < len(bm25_r) else '—'
            s  = sem_r[i]['section'][:30]  if i < len(sem_r)  else '—'
            h  = hyb_r[i]['section'][:30]  if i < len(hyb_r)  else '—'
            bs = f"({bm25_r[i]['bm25_score']:.2f})"   if i < len(bm25_r) else ''
            ss = f"({sem_r[i]['semantic_score']:.3f})" if i < len(sem_r)  else ''
            hs = f"({hyb_r[i]['rrf_score']:.4f})"     if i < len(hyb_r)  else ''
            print(f"  {i+1}  {b+' '+bs:<38} {s+' '+ss:<38} {h+' '+hs}")


# ══════════════════════════════════════════════════════════════
# MAIN  — test the full retrieval system
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═"*68)
    print("STAGE 2 — RETRIEVAL SYSTEM (BM25 + Sentence Transformer Hybrid)")
    print("═"*68)

    # Load chunks from Stage 1
    chunks = json.load(open('/Users/shubh/Project/Ncert_Rag/chunks/all_chunks.json'))
    print(f"\nLoaded {len(chunks)} chunks from Stage 1\n")

    print("Initialising retrievers:")
    retriever = HybridRetriever(chunks)

    # ── Test 1: Direct textbook question ──────────────────────
    print("\n" + "─"*68)
    print("TEST 1 — Direct question (should hit Ch9 F=ma content)")
    retriever.compare_retrievers("What is Newton's second law of motion?", k=3)

    # ── Test 2: Paraphrased question ──────────────────────────
    print("\n" + "─"*68)
    print("TEST 2 — Paraphrased (no exact keyword match)")
    retriever.compare_retrievers(
        "How fast does velocity change when force is applied?", k=3)

    # ── Test 3: Multi-chapter question ────────────────────────
    print("\n" + "─"*68)
    print("TEST 3 — Multi-chapter (spans Ch11 Work + Ch12 Sound)")
    retriever.compare_retrievers(
        "What is the SI unit of energy and how is sound energy measured?", k=3)

    # ── Test 4: Calculation question ──────────────────────────
    print("\n" + "─"*68)
    print("TEST 4 — Calculation question with numbers")
    retriever.compare_retrievers(
        "A ball thrown up with 19.6 m/s. How high does it go?", k=3)

    # ── Show where semantic beats BM25 ────────────────────────
    print("\n" + "─"*68)
    print("TEST 5 — Semantic advantage: concept with no exact keywords")
    retriever.compare_retrievers(
        "Why does the ship float but a stone sinks?", k=3)

    # Save retriever state for Stage 3
    retriever_data = {
        'n_chunks': len(chunks),
        'vocab_size': retriever.sem_ret.corpus_matrix.shape[1]
    }
    with open('/Users/shubh/Project/Ncert_Rag/chunks/retriever_info.json', 'w') as f:
        json.dump(retriever_data, f)
    print("\n✓ Retrieval system ready for Stage 3")
