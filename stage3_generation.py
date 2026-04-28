"""
stage3_generation.py
─────────────────────
Stage 3: Grounded Answer Generation

Uses the HybridRetriever from Stage 2 and an LLM to generate
answers grounded strictly in retrieved NCERT content.

Key design decisions explained line by line:
  • Prompt V1 vs V2 — why "refuse if not present" beats "answer from context"
  • temperature=0   — why evaluation must be deterministic
  • Context formatting — why chunk metadata is included
  • is_refusal flag  — how to detect refusals programmatically
"""

import sys, re, json, os
sys.path.insert(0, '/Users/shubh/Project/Ncert_Rag')
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

from stage2_retrieval import HybridRetriever, BM25Retriever, SentenceTransformerRetriever


# ══════════════════════════════════════════════════════════════
# GROUNDING PROMPTS
# ══════════════════════════════════════════════════════════════

# ── VERSION 1  (what most people write first) ─────────────────
# Problem: "ONLY from context" is a preference, not a constraint.
# The LLM reads it as "prefer context over other knowledge."
# For out-of-scope queries, it sees plausible-looking but irrelevant
# chunks and generates a confident (but wrong) answer.
PROMPT_V1 = """\
You are a study assistant for NCERT Class 9 Science.
Answer the student's question using ONLY the provided context.

Context:
{context}

Question: {question}
Answer:"""


# ── VERSION 2  (final — explicit refusal instruction) ─────────
# Changes from V1:
#   1. "ONLY if directly relevant" — forces a relevance check BEFORE answering
#   2. Explicit refusal text — prescribes the exact string we check for in code
#   3. "Do not use any knowledge outside" — stricter than "only from context"
#   4. Step-by-step instruction for calculations — NCERT students need workings shown
#
# Why prescribe the exact refusal text?
#   So our is_refusal flag is deterministic:
#     'not in the provided chapters' in answer.lower()
#   If we left it to the LLM, it varies: "I can't find...", "Not covered...", etc.
#   We'd miss some refusals and wrongly count them as answers.
PROMPT_V2 = """\
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

Retrieved Context:
{context}

Student Question: {question}

Answer (from context only):\
"""


def build_context_block(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into a context block for the LLM.

    Each chunk is labelled with:
      - Source number (1, 2, 3 …)
      - Chapter name
      - Section name
      - Content type (concept / example / exercise)
      - Retrieval score (so LLM knows confidence)

    The '---' separator between chunks helps the LLM mentally
    distinguish "these are three separate text passages."
    Without it, the model often treats the whole context as one
    continuous piece of text.
    """
    parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        header = (f"[Source {i}: {c['chapter']} | {c['section']} "
                  f"| type={c['content_type']} | score={c.get('rrf_score', c.get('bm25_score','?'))}]")
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
# GEMINI API CALL
# ══════════════════════════════════════════════════════════════

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Call Gemini 1.5 Flash with temperature=0.

    temperature=0:
      Deterministic output — same prompt always gives same answer.
      Without this, your evaluation score is non-reproducible:
        Run 1: 14/20 correct
        Run 2: 12/20 correct  (same system, nothing changed)
      The difference is just random sampling at temperature>0.
      Always set temperature=0 during evaluation.
      Set temperature=0.3–0.7 for production (friendlier tone).

    max_output_tokens=600:
      Enough for any NCERT answer including multi-step calculation.
      Keeping it bounded prevents runaway generation and controls API cost.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={
            'temperature': 0,          # deterministic for evaluation
            'max_output_tokens': 600,  # enough for full worked example
        }
    )
    response = model.generate_content(prompt)
    return response.text   # .text extracts the generated string from the response object


# ══════════════════════════════════════════════════════════════
# MOCK GENERATION  (used when no API key is available)
# ══════════════════════════════════════════════════════════════

def mock_generate(question: str, context: str) -> str:
    """
    Simulates what a well-prompted LLM returns for demonstration.
    Not production code — replace with call_gemini() in real use.

    The logic mirrors the V2 prompt rules:
      1. Check if question is clearly out-of-scope (refusal keywords)
      2. Check if context actually contains relevant content
      3. If yes: construct a specific answer based on key terms
      4. If no:  return the prescribed refusal text
    """
    q = question.lower()
    ctx = context.lower()

    # Out-of-scope topics (not in Ch 8–12)
    oos = ['photosynthesis','cell division','dna','atom structure','periodic table',
           'chemical reaction','human body','ecosystem','quantum','relativity',
           'climate change','electricity','magnetism','optics']
    if any(t in q for t in oos):
        return ("This information is not in the provided chapters. "
                "Please refer to the relevant chapter.")

    # ── Direct answers constructed from retrieved context ────
    responses = {
        ("newton", "second"): (
            "Newton's Second Law of Motion states: The rate of change of momentum "
            "of an object is proportional to the applied unbalanced force in the "
            "direction of the force. Mathematically:\n  F = ma\nwhere F = force (N), "
            "m = mass (kg), a = acceleration (m s⁻²).\n1 Newton = 1 kg × 1 m s⁻²."),
        ("newton", "first"): (
            "Newton's First Law (Law of Inertia): An object remains in a state of rest "
            "or uniform motion in a straight line unless compelled to change that state "
            "by an applied force."),
        ("newton", "third"): (
            "Newton's Third Law: For every action, there is an equal and opposite "
            "reaction; they act on two different objects.\n"
            "Example: A gun recoils when a bullet is fired. The force on the bullet "
            "(action) equals the recoil force on the gun (reaction), but they act "
            "on different objects."),
        ("gravitational", "constant"): (
            "The Universal Gravitational Constant G = 6.673 × 10⁻¹¹ N m² kg⁻².\n"
            "Newton's Law of Gravitation: F = G × m₁ × m₂ / d²"),
        ("kinetic energy",): (
            "Kinetic Energy KE = ½ × m × v²\n"
            "where m = mass (kg), v = velocity (m s⁻¹).\n"
            "Unit: Joule (J)."),
        ("potential energy",): (
            "Gravitational Potential Energy PE = m × g × h\n"
            "where m = mass, g = 9.8 m s⁻², h = height above ground.\n"
            "Unit: Joule (J)."),
        ("conservation of energy",): (
            "Law of Conservation of Energy: Energy can neither be created nor destroyed. "
            "It can only be converted from one form to another. Total energy of an "
            "isolated system remains constant.\nDuring free fall: PE decreases, KE increases, "
            "total (PE + KE) remains constant."),
        ("acceleration due to gravity", "g"): (
            "Acceleration due to gravity g = 9.8 m s⁻² (≈ 10 m s⁻²).\n"
            "g = GM/R² where M = mass of Earth, R = radius of Earth.\n"
            "On Moon: g_Moon ≈ 1.63 m s⁻² (about 1/6 of Earth's g)."),
        ("weight", "moon"): (
            "Weight on Moon = (1/6) × Weight on Earth, because the Moon's "
            "gravitational pull is weaker.\ng_Moon ≈ 1.63 m s⁻²\n"
            "W = mg, so for m=10 kg: W_Earth=98 N, W_Moon≈16.3 N."),
        ("speed of sound",): (
            "Speed of sound depends on the medium:\n"
            "  Air (25°C): 346 m s⁻¹\n  Water: 1500 m s⁻¹\n  Steel: 5100 m s⁻¹\n"
            "Sound travels faster in denser media and at higher temperatures."),
        ("echo",): (
            "Echo: reflected sound heard after the original sound.\n"
            "Minimum distance for echo: 17 m (sound must travel ≥ 34 m in ≥ 0.1 s).\n"
            "Formula: d = v × t / 2\nwhere d = distance to reflector, v = speed of sound."),
        ("power",): (
            "Power = Rate of doing work = W / t\nSI unit: Watt (W). 1 W = 1 J s⁻¹.\n"
            "1 kWh = 3.6 × 10⁶ J (commercial unit of energy)."),
        ("pressure",): (
            "Pressure = Force / Area = F/A\nSI unit: Pascal (Pa). 1 Pa = 1 N m⁻².\n"
            "Pressure in fluid: P = hρg where h = depth, ρ = density, g = gravity."),
        ("buoyancy","float","archimedes"): (
            "Archimedes' Principle: Buoyant force = weight of fluid displaced.\n"
            "Object floats if its density < fluid density.\n"
            "Object sinks if its density > fluid density."),
        ("equations of motion", "three"): (
            "Three equations of uniformly accelerated motion:\n"
            "1. v = u + at\n2. s = ut + ½at²\n3. v² = u² + 2as\n"
            "u=initial velocity, v=final velocity, a=acceleration, t=time, s=displacement."),
        ("average speed",): (
            "Average speed = Total distance / Total time\n"
            "Example: 16 m in 4 s + 16 m in 2 s → total=32 m, time=6 s → speed=5.33 m s⁻¹."),
        ("inertia",): (
            "Inertia: tendency of an object to resist change in its state of motion.\n"
            "Determined by mass — more mass = more inertia.\n"
            "Example: stone has more inertia than a rubber ball of same size."),
        ("recoil",): (
            "By conservation of momentum: m₁u₁ + m₂u₂ = m₁v₁ + m₂v₂\n"
            "For gun (m=4 kg) + bullet (m=0.02 kg, v=400 m s⁻¹), both initially at rest:\n"
            "0 = 0.02×400 + 4×v₂ → v₂ = -2 m s⁻¹\n"
            "Gun recoils at 2 m s⁻¹ opposite to bullet direction."),
        ("sonar",): (
            "SONAR: Sound Navigation And Ranging.\n"
            "Uses ultrasound to find depth of sea / detect underwater objects.\n"
            "Formula: d = v × t / 2\nExample: echo in 4 s at 1500 m s⁻¹ → d = 3000 m."),
        ("ultrasound",): (
            "Ultrasound: sound above 20 000 Hz (20 kHz).\n"
            "Uses: medical imaging, SONAR, cleaning components, detecting metal cracks.\n"
            "Bats and dolphins use echolocation with ultrasound."),
        ("free fall",): (
            "Free fall: motion under gravity alone (no air resistance).\n"
            "Acceleration = g = 9.8 m s⁻² (downward).\n"
            "Equations: v = gt, s = ½gt² (for object dropped from rest, u=0)."),
    }

    for keywords, answer in responses.items():
        if all(kw in q for kw in keywords):
            # Verify answer content is at least partially in context
            if any(kw in ctx for kw in keywords[:2]):
                return answer

    # Generic fallback using context
    if len(ctx) > 100:
        relevant_sentence = ""
        for line in context.split('\n'):
            line_lower = line.lower()
            query_words = [w for w in q.split() if len(w) > 4]
            if sum(1 for w in query_words if w in line_lower) >= 2:
                relevant_sentence = line.strip()
                break
        if relevant_sentence:
            return f"Based on the retrieved NCERT content:\n{relevant_sentence}\n(See retrieved context for full details.)"

    return ("This information is not in the provided chapters. "
            "Please refer to the relevant chapter.")


# ══════════════════════════════════════════════════════════════
# ANSWER SYSTEM
# ══════════════════════════════════════════════════════════════

class GroundedAnswerSystem:
    """
    Combines HybridRetriever + LLM into one answer() function.

    Public interface:
      result = system.answer("What is Newton's second law?")
      print(result['answer'])          # the generated answer
      print(result['is_refusal'])      # True if system refused
      print(result['retrieved_chunks'])# list of chunk dicts
    """

    # The exact refusal phrase we prescribe in PROMPT_V2.
    # Checking for this string lets us programmatically detect refusals.
    REFUSAL_PHRASE = "not in the provided chapters"

    def __init__(self, retriever: HybridRetriever,
                 api_key: str = None,
                 prompt_version: str = 'v2'):
        self.retriever      = retriever
        self.api_key        = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.use_real_api   = bool(self.api_key)
        self.prompt_template = PROMPT_V2 if prompt_version == 'v2' else PROMPT_V1

        mode = "Gemini API" if self.use_real_api else "mock generation"
        print(f"  GroundedAnswerSystem ready | prompt={prompt_version} | mode={mode}")

    def answer(self, question: str, k: int = 3) -> dict:
        """
        Full RAG pipeline:
          Retrieve → Build context → Format prompt → Generate → Return.

        k=3: retrieve top-3 chunks. More chunks = more context but also
             more noise. 3 is a good balance for single-concept NCERT questions.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, k=k)

        # Step 2: Format context block
        context = build_context_block(chunks)

        # Step 3: Fill in the prompt template
        # .format() replaces {context} and {question} placeholders
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # Step 4: Generate answer
        if self.use_real_api:
            answer_text = call_gemini(prompt, self.api_key)
        else:
            answer_text = mock_generate(question, context)

        # Step 5: Detect refusal
        # .lower() for case-insensitive check
        is_refusal = self.REFUSAL_PHRASE in answer_text.lower()

        return {
            'question'        : question,
            'answer'          : answer_text,
            'retrieved_chunks': chunks,
            'is_refusal'      : is_refusal,
            'top_chunk'       : chunks[0] if chunks else {},
            'prompt_used'     : self.prompt_template[:80] + '...',
        }

    def demo(self, question: str) -> None:
        """Pretty-print a single answer for demonstration."""
        r = self.answer(question)
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print(f"\nTop chunk: [{r['top_chunk'].get('chapter','?')}] "
              f"{r['top_chunk'].get('section','?')} "
              f"(score={r['top_chunk'].get('rrf_score', r['top_chunk'].get('bm25_score','?'))})")
        print(f"\nA: {r['answer']}")
        print(f"\n{'✓ REFUSAL' if r['is_refusal'] else '→ ANSWERED'} | "
              f"retrieved {len(r['retrieved_chunks'])} chunks")


# ══════════════════════════════════════════════════════════════
# MAIN  — demonstrate Stage 3
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═"*68)
    print("STAGE 3 — GROUNDED ANSWER GENERATION")
    print("═"*68)

    chunks = json.load(open('/Users/shubh/Project/Ncert_Rag/chunks/all_chunks.json'))

    print("\nInitialising system...")
    retriever = HybridRetriever(chunks)
    api_key   = os.environ.get('GEMINI_API_KEY', '')
    system    = GroundedAnswerSystem(retriever, api_key=api_key)

    print("\n── Prompt V1 vs V2 comparison ──────────────────────────")
    print("V1 prompt (first 120 chars):")
    print("  " + PROMPT_V1[:120].replace('\n', '\n  '))
    print("\nV2 prompt (first 240 chars):")
    print("  " + PROMPT_V2[:240].replace('\n', '\n  '))
    print("\nKey difference: V2 has explicit REFUSE instruction + prescribed refusal text.")

    print("\n\n── Demo answers ────────────────────────────────────────")

    # Direct textbook question — should answer well
    system.demo("What is Newton's second law of motion?")

    # Cross-chapter question — tests multi-chapter retrieval
    system.demo("What is the difference between kinetic and potential energy?")

    # Calculation question
    system.demo("A bullet of 20 g is fired from a 4 kg gun at 400 m/s. Find the recoil velocity.")

    # Out-of-scope: obvious — should refuse
    system.demo("Explain the process of photosynthesis in plants.")

    # Out-of-scope: adversarial — same science domain, not in our chapters
    system.demo("Explain how electric current flows through a conductor.")

    print("\n✓ Stage 3 complete")
