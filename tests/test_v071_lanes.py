"""
KOS v0.7.1 Lane Test -- Validates hard gates, grounding, and preference checks
across all three lanes (math, factual, comparison).

Score bands: EXCELLENT >= 0.85, STRONG >= 0.70, USABLE >= 0.55, WEAK < 0.55
"""
import json
import urllib.request

BASE = "http://localhost:8080"

def query(prompt):
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/query", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

def band(score):
    if score >= 0.85: return "EXCELLENT"
    if score >= 0.70: return "STRONG"
    if score >= 0.55: return "USABLE"
    return "WEAK"

# ── MATH LANE (10 queries) ──────────────────────────────────────
MATH = [
    "integrate x^2",
    "derivative of sin(x)",
    "sqrt(144)",
    "345000000 * 0.0825",
    "derivative sin(x)*x^3",
    "solve x^2 - 5x + 6 = 0",
    "simplify (x^2-4)/(x-2)",
    "factorial of 7",
    "log base 2 of 64",
    "15% of 380",
]

# ── FACTUAL LANE (10 queries) ───────────────────────────────────
FACTUAL = [
    "What is Toronto?",
    "When was Toronto founded?",
    "Population of Toronto",
    "What is perovskite?",
    "Tell me about apixaban",
    "What is Montreal?",
    "Explain backpropagation",
    "What is Newton's second law?",
    "What produces insulin?",
    "What is the capital of Canada?",
]

# ── COMPARISON LANE (5 queries) ─────────────────────────────────
COMPARISON = [
    "Compare Toronto and Montreal",
    "What is the difference between backpropagation and gradient descent?",
    "Apixaban vs Warfarin",
    "Compare perovskite and silicon solar cells",
    "How does Python compare to JavaScript?",
]

def run_lane(name, queries):
    scores = []
    results = []
    for q in queries:
        try:
            r = query(q)
            s = r.get("relevance_score", 0)
            trust = r.get("trust_label", "?")
            verify = r.get("stages", {}).get("verify", {})
            hard_fail = verify.get("hard_fail", False)
            failure_tags = verify.get("failure_tags", [])
            grounding = verify.get("grounding_score", -1)

            scores.append(s)
            results.append(r)
            tag_str = f" TAGS={failure_tags}" if failure_tags else ""
            ground_str = f" ground={grounding:.3f}" if grounding >= 0 else ""
            hf_str = " HARD_FAIL" if hard_fail else ""
            print(f"  {s:.3f} [{band(s):9s}] trust={trust:14s}"
                  f"{ground_str}{hf_str}{tag_str} | {q}")
        except Exception as e:
            print(f"  ERROR | {q} -- {e}")
            scores.append(0)

    avg = sum(scores) / len(scores) if scores else 0
    mn = min(scores) if scores else 0
    mx = max(scores) if scores else 0
    passing = sum(1 for s in scores if s >= 0.55)
    print(f"  --- {name}: avg={avg:.3f} [{band(avg)}] "
          f"min={mn:.3f} max={mx:.3f} pass={passing}/{len(scores)}")
    return scores, results

print("=" * 72)
print("KOS v0.7.1 LANE TEST")
print("=" * 72)

print("\nMATH LANE:")
math_scores, math_results = run_lane("MATH", MATH)

print("\nFACTUAL LANE:")
fact_scores, fact_results = run_lane("FACTUAL", FACTUAL)

print("\nCOMPARISON LANE:")
comp_scores, comp_results = run_lane("COMPARISON", COMPARISON)

# ── SUMMARY ─────────────────────────────────────────────────────
all_scores = math_scores + fact_scores + comp_scores
overall = sum(all_scores) / len(all_scores) if all_scores else 0
total_pass = sum(1 for s in all_scores if s >= 0.55)

# Count trust labels
trust_counts = {}
for r in math_results + fact_results + comp_results:
    t = r.get("trust_label", "unknown")
    trust_counts[t] = trust_counts.get(t, 0) + 1

# Count hard fails
hard_fails = sum(
    1 for r in math_results + fact_results + comp_results
    if r.get("stages", {}).get("verify", {}).get("hard_fail", False))

# Count grounding issues
grounding_flags = sum(
    1 for r in math_results + fact_results + comp_results
    if "V_LOW_GROUNDING" in r.get("stages", {}).get("verify", {}).get("failure_tags", []))

# Count preference flags
pref_flags = sum(
    1 for r in math_results + fact_results + comp_results
    if "V_UNSUPPORTED_PREFERENCE" in r.get("stages", {}).get("verify", {}).get("failure_tags", []))

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  MATH      : avg={sum(math_scores)/len(math_scores):.3f} "
      f"pass={sum(1 for s in math_scores if s >= 0.55)}/{len(math_scores)}")
print(f"  FACTUAL   : avg={sum(fact_scores)/len(fact_scores):.3f} "
      f"pass={sum(1 for s in fact_scores if s >= 0.55)}/{len(fact_scores)}")
print(f"  COMPARISON: avg={sum(comp_scores)/len(comp_scores):.3f} "
      f"pass={sum(1 for s in comp_scores if s >= 0.55)}/{len(comp_scores)}")
print(f"  OVERALL   : avg={overall:.3f} "
      f"pass={total_pass}/{len(all_scores)}")
print(f"\n  Trust: {trust_counts}")
print(f"  Hard fails: {hard_fails}")
print(f"  Low grounding: {grounding_flags}")
print(f"  Unsupported preference: {pref_flags}")
print("=" * 72)
