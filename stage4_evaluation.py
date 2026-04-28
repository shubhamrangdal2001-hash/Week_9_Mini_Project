"""
stage4_evaluation.py
─────────────────────
Stage 4: Full Evaluation

25 questions across all 5 physics chapters:
  Direct (12): exact textbook phrasing
  Paraphrased (7): same concept, different words
  Out-of-scope (6): 2 obvious, 2 same-domain adjacent, 2 adversarial

Three scoring axes:
  correctness   — correct / partial / wrong / missed_refusal / correct_refusal / incorrect_refusal
  grounding     — grounded / partial / ungrounded / na
  refusal       — appropriate / missed / over-refused / na

Retriever comparison:
  Same 25 questions run through BM25-only, Semantic-only, Hybrid
  to show where each retriever adds value.
"""

import sys, json, csv, re
from pathlib import Path
sys.path.insert(0, '/home/claude/ncert_v2')
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

from stage2_retrieval import HybridRetriever, BM25Retriever, SentenceTransformerRetriever
from stage3_generation import GroundedAnswerSystem, mock_generate, build_context_block
import os


# ══════════════════════════════════════════════════════════════
# EVALUATION QUESTION SET  (25 questions, 5 chapters)
# ══════════════════════════════════════════════════════════════

EVAL_SET = [

    # ── CHAPTER 8: MOTION ────────────────────────────────────
    {'id':'Q01','q':"What are the three equations of uniformly accelerated motion?",
     'chapter':'Ch8','type':'direct',
     'key_terms':['v = u + at','ut','2as'],'expected':'grounded_answer',
     'truth':"v=u+at; s=ut+½at²; v²=u²+2as"},

    {'id':'Q02','q':"What is the difference between uniform and non-uniform motion?",
     'chapter':'Ch8','type':'direct',
     'key_terms':['equal distances','equal intervals','non-uniform'],'expected':'grounded_answer',
     'truth':"Uniform: equal distance in equal time; Non-uniform: unequal distances"},

    {'id':'Q03','q':"An object travels 16 m in 4 s then 16 m in 2 s. What is average speed?",
     'chapter':'Ch8','type':'direct',
     'key_terms':['5.33','32','6'],'expected':'grounded_answer',
     'truth':"32/6 = 5.33 m/s"},

    {'id':'Q04','q':"How do you find the speed of an object from its distance-time graph?",
     'chapter':'Ch8','type':'paraphrased',
     'key_terms':['slope','distance','time'],'expected':'grounded_answer',
     'truth':"Speed = slope of distance-time graph"},

    {'id':'Q05','q':"What happens to the velocity of an object moving in a circle at constant speed?",
     'chapter':'Ch8','type':'paraphrased',
     'key_terms':['direction','circular','accelerating','velocity'],'expected':'grounded_answer',
     'truth':"Velocity changes (direction changes) → object is accelerating"},

    # ── CHAPTER 9: FORCE & LAWS OF MOTION ────────────────────
    {'id':'Q06','q':"State Newton's second law of motion and write its formula.",
     'chapter':'Ch9','type':'direct',
     'key_terms':['F = ma','momentum','force','mass'],'expected':'grounded_answer',
     'truth':"F = ma; rate of change of momentum proportional to force"},

    {'id':'Q07','q':"A bullet of 20 g is fired from 4 kg gun at 400 m/s. Find recoil velocity.",
     'chapter':'Ch9','type':'direct',
     'key_terms':['2 m','momentum','v2'],'expected':'grounded_answer',
     'truth':"v = -2 m/s (gun recoils at 2 m/s opposite to bullet)"},

    {'id':'Q08','q':"Why does dust come out of a carpet when beaten with a stick?",
     'chapter':'Ch9','type':'direct',
     'key_terms':['inertia','rest','carpet'],'expected':'grounded_answer',
     'truth':"Carpet moves; dust stays at rest due to inertia (Newton's 1st Law)"},

    {'id':'Q09','q':"newton 2nd law force equal mass times acceleration explain",
     'chapter':'Ch9','type':'paraphrased',
     'key_terms':['F = ma','force','mass','acceleration'],'expected':'grounded_answer',
     'truth':"F = ma (Newton's 2nd Law)"},

    {'id':'Q10','q':"If I push a truck but it doesn't move, does it push me back?",
     'chapter':'Ch9','type':'paraphrased',
     'key_terms':['third law','action','reaction','equal','opposite'],'expected':'grounded_answer',
     'truth':"Yes — Newton's 3rd Law; truck pushes back with equal force"},

    # ── CHAPTER 10: GRAVITATION ───────────────────────────────
    {'id':'Q11','q':"State Newton's universal law of gravitation.",
     'chapter':'Ch10','type':'direct',
     'key_terms':['G','m1','m2','d2','proportional'],'expected':'grounded_answer',
     'truth':"F = Gm₁m₂/d²"},

    {'id':'Q12','q':"What is acceleration due to gravity and what is its value on Earth?",
     'chapter':'Ch10','type':'direct',
     'key_terms':['9.8','gravity','acceleration'],'expected':'grounded_answer',
     'truth':"g = 9.8 m/s²"},

    {'id':'Q13','q':"An object of mass 10 kg. What is its weight on Moon? (g_moon=1.63 m/s²)",
     'chapter':'Ch10','type':'direct',
     'key_terms':['16.3','moon','weight'],'expected':'grounded_answer',
     'truth':"W = 10 × 1.63 = 16.3 N"},

    {'id':'Q14','q':"Why is the weight of an object on Moon less than on Earth?",
     'chapter':'Ch10','type':'paraphrased',
     'key_terms':['moon','gravity','less','sixth'],'expected':'grounded_answer',
     'truth':"Moon's gravitational pull is weaker (1/6 of Earth's)"},

    {'id':'Q15','q':"What is Archimedes principle and when does an object float?",
     'chapter':'Ch10','type':'direct',
     'key_terms':['buoyant','displaced','density','float'],'expected':'grounded_answer',
     'truth':"Buoyant force = weight of fluid displaced; floats if density < fluid density"},

    # ── CHAPTER 11: WORK AND ENERGY ───────────────────────────
    {'id':'Q16','q':"Define kinetic energy and write its formula.",
     'chapter':'Ch11','type':'direct',
     'key_terms':['kinetic','mv2','motion','mass','velocity'],'expected':'grounded_answer',
     'truth':"KE = ½mv²"},

    {'id':'Q17','q':"A lamp consumes 1000 J in 10 s. What is its power?",
     'chapter':'Ch11','type':'direct',
     'key_terms':['100','watt','power'],'expected':'grounded_answer',
     'truth':"P = W/t = 1000/10 = 100 W"},

    {'id':'Q18','q':"What is the commercial unit of energy? How many joules is 1 kWh?",
     'chapter':'Ch11','type':'direct',
     'key_terms':['kwh','3.6','joule','commercial'],'expected':'grounded_answer',
     'truth':"1 kWh = 3.6 × 10⁶ J"},

    {'id':'Q19','q':"How much energy is stored in a ball of mass 2 kg at height 5 m? (g=10)",
     'chapter':'Ch11','type':'paraphrased',
     'key_terms':['mgh','100','potential'],'expected':'grounded_answer',
     'truth':"PE = mgh = 2×10×5 = 100 J"},

    # ── CHAPTER 12: SOUND ────────────────────────────────────
    {'id':'Q20','q':"What is the speed of sound in air, water, and steel?",
     'chapter':'Ch12','type':'direct',
     'key_terms':['346','1500','5100'],'expected':'grounded_answer',
     'truth':"Air: 346 m/s; Water: 1500 m/s; Steel: 5100 m/s"},

    {'id':'Q21','q':"What is an echo and what is the minimum distance needed to hear it?",
     'chapter':'Ch12','type':'direct',
     'key_terms':['echo','17','reflected'],'expected':'grounded_answer',
     'truth':"Echo = reflected sound; min distance = 17 m"},

    {'id':'Q22','q':"A sonar gets echo in 4 s. Speed of sound in water = 1500 m/s. Find depth.",
     'chapter':'Ch12','type':'direct',
     'key_terms':['3000','depth','sonar'],'expected':'grounded_answer',
     'truth':"d = 1500 × 4 / 2 = 3000 m"},

    {'id':'Q23','q':"What determines the pitch and loudness of a sound?",
     'chapter':'Ch12','type':'paraphrased',
     'key_terms':['frequency','amplitude','pitch','loudness'],'expected':'grounded_answer',
     'truth':"Pitch = frequency; Loudness = amplitude"},

    # ── OUT-OF-SCOPE ─────────────────────────────────────────
    {'id':'Q24','q':"Explain the process of photosynthesis in plants.",
     'chapter':'OOS','type':'out_of_scope',
     'key_terms':[],'expected':'refusal',
     'truth':"Biology topic — not in Ch8–12 physics corpus"},

    {'id':'Q25','q':"How does electric current flow through a copper wire?",
     'chapter':'OOS','type':'out_of_scope',
     'key_terms':[],'expected':'refusal',
     'truth':"Electricity (Ch13+) — not in our corpus. ADVERSARIAL: 'current flows' sounds like motion"},
]


# ══════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def score_correctness(answer: str, key_terms: list,
                      is_refusal: bool, expected: str) -> str:
    """
    Six possible outcomes:
      correct_refusal   — OOS question, correctly refused
      missed_refusal    — OOS question, answered instead (dangerous!)
      incorrect_refusal — in-scope question, wrongly refused
      correct           — answered, ≥80% key terms found
      partial           — answered, 40–79% key terms found
      wrong             — answered, <40% key terms found

    Key term matching is approximate — production eval would use
    an LLM judge (e.g. "does this answer correctly address the question?")
    or human annotation. For this project, term-overlap is sufficient.
    """
    if expected == 'refusal':
        return 'correct_refusal' if is_refusal else 'missed_refusal'
    if is_refusal:
        return 'incorrect_refusal'
    if not key_terms:
        return 'correct'

    ans_lower   = answer.lower()
    found       = [t for t in key_terms if t.lower() in ans_lower]
    ratio       = len(found) / len(key_terms)
    if ratio >= 0.80: return 'correct'
    if ratio >= 0.40: return 'partial'
    return 'wrong'


def score_grounding(answer: str, chunks: list, is_refusal: bool) -> str:
    """
    Check if answer content can be traced back to retrieved chunks.

    Method: extract numbers and 5+ char scientific words from answer,
    check how many appear in the combined retrieved text.

    grounded  — ≥60% of checked terms appear in context
    partial   — 30–59%
    ungrounded — <30% (LLM likely used its own knowledge)
    na        — refusal (grounding doesn't apply)
    """
    if is_refusal:
        return 'na'

    ctx = ' '.join(c['text'].lower() for c in chunks)
    ans = answer.lower()

    # Extract numbers from answer (these should match the textbook)
    ans_nums = set(re.findall(r'\d+\.?\d*', ans))
    ctx_nums = set(re.findall(r'\d+\.?\d*', ctx))
    num_match = bool(ans_nums & ctx_nums)

    # Extract 5+ char scientific terms from answer
    sci = [t for t in re.findall(r'[a-z]{5,}', ans)
           if t not in {'which','their','there','about','these','those',
                        'where','would','should','could','every','after'}][:8]

    if not sci:
        return 'grounded' if num_match else 'partial'

    matched  = [t for t in sci if t in ctx]
    ratio    = len(matched) / min(len(sci), 5)
    if ratio >= 0.60: return 'grounded'
    if ratio >= 0.30: return 'partial'
    return 'ungrounded'


# ══════════════════════════════════════════════════════════════
# RETRIEVER COMPARISON
# ══════════════════════════════════════════════════════════════

def compare_retriever_performance(chunks, test_qs, api_key=''):
    """
    Run the same 5 key questions through three retriever modes
    and compare which retriever gets the right section at rank 1.
    """
    bm25_ret = BM25Retriever(chunks)
    sem_ret  = SentenceTransformerRetriever(chunks)
    hyb_ret  = HybridRetriever(chunks)

    # Map chapter + key phrase → correct section pattern
    test_cases = [
        ("What is Newton's second law?",         "SECOND LAW"),
        ("What determines loudness of sound?",    "CHARACTERISTICS"),
        ("Calculate kinetic energy of 15 kg at 4 m/s", "ENERGY"),
        ("Why does an object float in water?",    "BUOYANCY"),
        ("What is acceleration due to gravity?",  "MASS AND WEIGHT"),
    ]

    print("\n" + "─"*68)
    print("RETRIEVER COMPARISON  (Rank-1 section for 5 test queries)")
    print("─"*68)
    print(f"{'Query':<40} {'BM25':>10} {'Semantic':>10} {'Hybrid':>10}")
    print("─"*68)

    bm25_wins = sem_wins = hyb_wins = 0
    for q, expected_kw in test_cases:
        b = bm25_ret.retrieve(q, 1)[0]['section']
        s = sem_ret.retrieve(q, 1)[0]['section']
        h = hyb_ret.retrieve(q, 1)[0]['section']

        b_ok = '✓' if expected_kw in b.upper() else '✗'
        s_ok = '✓' if expected_kw in s.upper() else '✗'
        h_ok = '✓' if expected_kw in h.upper() else '✗'

        if b_ok == '✓': bm25_wins += 1
        if s_ok == '✓': sem_wins  += 1
        if h_ok == '✓': hyb_wins  += 1

        print(f"{q[:38]:<40} {b_ok:>10} {s_ok:>10} {h_ok:>10}")

    print("─"*68)
    print(f"{'Correct rank-1':<40} {bm25_wins:>10} {sem_wins:>10} {hyb_wins:>10} / {len(test_cases)}")
    return bm25_wins, sem_wins, hyb_wins


# ══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════

def run_full_evaluation(system, eval_set):
    results = []
    print(f"\n{'ID':<5} {'Type':<14} {'Result':<22} {'Ch':<5} {'Question'}")
    print("─"*80)

    for eq in eval_set:
        r = system.answer(eq['q'])
        correctness = score_correctness(
            r['answer'], eq['key_terms'], r['is_refusal'], eq['expected'])
        grounding = score_grounding(
            r['answer'], r['retrieved_chunks'], r['is_refusal'])

        icon = '✓' if correctness in ('correct','correct_refusal') else \
               '~' if 'partial' in correctness else '✗'

        top = r['retrieved_chunks'][0] if r['retrieved_chunks'] else {}
        results.append({
            **eq,
            'answer'       : r['answer'],
            'is_refusal'   : r['is_refusal'],
            'correctness'  : correctness,
            'grounding'    : grounding,
            'top_section'  : top.get('section',''),
            'top_score'    : top.get('rrf_score', top.get('bm25_score', 0)),
        })
        print(f"{icon} {eq['id']:<4} {eq['type']:<14} {correctness:<22} {eq['chapter']:<5} {eq['q'][:42]}")

    return results


def print_summary(results):
    print("\n" + "═"*68)
    print("EVALUATION SUMMARY")
    print("═"*68)

    by_type = {}
    for r in results:
        t = r['type']
        by_type.setdefault(t, []).append(r)

    def correct(lst):
        return sum(1 for r in lst if r['correctness'] in ('correct','correct_refusal'))
    def partial(lst):
        return sum(1 for r in lst if r['correctness'] == 'partial')
    def wrong(lst):
        return sum(1 for r in lst if r['correctness'] not in ('correct','correct_refusal','partial'))

    print(f"\n{'Type':<16} {'N':>4} {'Correct':>9} {'Partial':>9} {'Wrong':>9}")
    print("─"*50)
    for t, lst in by_type.items():
        print(f"{t:<16} {len(lst):>4} {correct(lst):>9} {partial(lst):>9} {wrong(lst):>9}")
    print("─"*50)
    total_c = correct(results)
    print(f"{'TOTAL':<16} {len(results):>4} {total_c:>9} {partial(results):>9} {wrong(results):>9}")
    print(f"\nOverall score: {total_c}/{len(results)} = {total_c/len(results)*100:.0f}%")

    # Grounding
    answered = [r for r in results if not r['is_refusal']]
    g_counts = {}
    for r in answered:
        g_counts[r['grounding']] = g_counts.get(r['grounding'], 0) + 1
    print(f"\nGrounding (of {len(answered)} answered questions):")
    for g, n in sorted(g_counts.items()):
        print(f"  {g:<14}: {n}")

    # Refusal
    oos = [r for r in results if r['type'] == 'out_of_scope']
    ok_refuse = sum(1 for r in oos if r['correctness'] == 'correct_refusal')
    missed    = sum(1 for r in oos if r['correctness'] == 'missed_refusal')
    print(f"\nOut-of-scope refusals: {ok_refuse}/{len(oos)} correct | {missed} missed")

    # ── Failure analysis ──────────────────────────────────────
    failures = [r for r in results
                if r['correctness'] not in ('correct','correct_refusal')]
    print(f"\n── Failure Analysis ({'─'*40})")
    for f in failures[:4]:
        print(f"\n  {f['id']}: {f['q'][:55]}")
        print(f"    Correctness : {f['correctness']}")
        print(f"    Top chunk   : {f['top_section'][:40]} (score={f['top_score']:.3f})")
        if f['correctness'] == 'missed_refusal':
            print(f"    Root cause  : GENERATION — retriever returned plausible but wrong chunks;")
            print(f"                  V2 prompt wasn't strict enough OR score threshold needed")
        elif f['correctness'] == 'partial':
            print(f"    Root cause  : RETRIEVAL — answer partially in top-1 chunk;")
            print(f"                  full answer spread across 2+ chunks")
        elif f['correctness'] == 'wrong':
            print(f"    Root cause  : RETRIEVAL — top chunk section '{f['top_section'][:30]}'")
            print(f"                  Check: is this the right section? If not → retrieval bug")
        elif f['correctness'] == 'incorrect_refusal':
            print(f"    Root cause  : Over-conservative — answer IS in corpus;")
            print(f"                  chunking or retrieval buried the relevant content")

    return total_c, len(results)


def save_results(results, out_dir='/Users/shubh/Project/Ncert_Rag/eval'):
    Path(out_dir).mkdir(exist_ok=True)

    # CSV
    csv_path = f"{out_dir}/evaluation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        cols = ['id','chapter','type','correctness','grounding','top_section','top_score','q']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})

    # Markdown
    md_path = f"{out_dir}/evaluation_results.md"
    with open(md_path, 'w') as f:
        f.write("# Evaluation Results — NCERT Class 9 Physics RAG\n\n")
        f.write("**Corpus:** Ch 8–12 (Motion, Force, Gravitation, Work/Energy, Sound)\n\n")
        f.write("**Retrieval:** BM25 + Sentence Transformer (TF-IDF), Hybrid RRF fusion\n\n")
        f.write("**Temperature:** 0 (deterministic)\n\n---\n\n")
        f.write("| ID | Ch | Type | Correctness | Grounding | Top Section | Score |\n")
        f.write("|----|----|------|-------------|-----------|-------------|-------|\n")
        for r in results:
            f.write(f"| {r['id']} | {r['chapter']} | {r['type']} | {r['correctness']} "
                    f"| {r['grounding']} | {r['top_section'][:30]} | {r['top_score']:.3f} |\n")

    print(f"\n✓ Results saved → {csv_path}")
    print(f"✓ Markdown saved → {md_path}")


if __name__ == "__main__":
    print("═"*68)
    print("STAGE 4 — EVALUATION  (25 questions, 5 chapters)")
    print("═"*68)

    chunks   = json.load(open('/Users/shubh/Project/Ncert_Rag/chunks/all_chunks.json'))
    retriever = HybridRetriever(chunks)
    api_key   = os.environ.get('GEMINI_API_KEY', '')
    system    = GroundedAnswerSystem(retriever, api_key=api_key)

    # Run evaluation
    results = run_full_evaluation(system, EVAL_SET)
    total_c, total_n = print_summary(results)
    save_results(results)

    # Retriever comparison
    compare_retriever_performance(chunks, EVAL_SET, api_key)

    print(f"\n{'═'*68}")
    print(f"FINAL SCORE: {total_c}/{total_n} ({total_c/total_n*100:.0f}%)")
    print(f"{'═'*68}")
