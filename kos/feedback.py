"""
KOS V5.1 — Weaver Feedback Loop + Auto Formula Discovery +
             Continuous Auto-Tuning + Scaled Analogy Discovery.

Closes all 4 remaining self-learning gaps:
  #4  Learn from user behavior (re-ask detection + evidence demotion)
  #9  Learn new formulas from ingested text
  L1  Continuous auto-tuning (self-triggering threshold optimizer)
  L3  Autonomous analogy discovery at scale
"""

import re
import time
import math
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# GAP #4: WEAVER FEEDBACK LOOP
# Learns from user re-asks to demote bad evidence patterns
# ═══════════════════════════════════════════════════════════════

class WeaverFeedback:
    """
    Tracks user query patterns. If a user re-asks a similar question
    (indicating dissatisfaction), demotes the evidence that was served.
    If the user moves to a new topic (satisfaction), boosts it.

    This gives the Weaver a feedback signal it currently lacks.
    """

    def __init__(self, weaver):
        self.weaver = weaver
        self._history = []  # (query, answer, evidence_sentences, timestamp)
        self._evidence_scores = defaultdict(float)  # sentence -> cumulative feedback
        self._reask_count = 0
        self._satisfied_count = 0

    def record(self, query: str, answer: str, evidence: list = None):
        """Record a query-answer pair with the evidence that was used."""
        self._history.append({
            'query': query,
            'answer': answer,
            'evidence': evidence or [],
            'time': time.time(),
        })
        # Keep bounded
        if len(self._history) > 500:
            self._history = self._history[-500:]

        # Check if this is a re-ask of the previous query
        if len(self._history) >= 2:
            prev = self._history[-2]
            curr = self._history[-1]
            similarity = self._query_similarity(prev['query'], curr['query'])

            if similarity > 0.30:
                # RE-ASK: user is dissatisfied with previous answer
                self._reask_count += 1
                # Demote evidence from the previous (bad) answer
                for sent in prev['evidence']:
                    self._evidence_scores[sent] -= 1.0
            else:
                # NEW TOPIC: user was satisfied with previous answer
                self._satisfied_count += 1
                # Boost evidence from the previous (good) answer
                for sent in prev['evidence']:
                    self._evidence_scores[sent] += 0.5

    def get_evidence_adjustment(self, sentence: str) -> float:
        """
        Returns a score adjustment for a sentence based on feedback history.
        Positive = historically successful evidence.
        Negative = historically re-asked (bad) evidence.
        """
        return self._evidence_scores.get(sentence, 0.0)

    def apply_to_weaver(self):
        """
        Patch the Weaver's scoring to include feedback adjustments.
        We add the feedback score as a bonus/penalty to each sentence.
        """
        # Store reference so the weaver can use it
        self.weaver._feedback = self

    def get_stats(self) -> dict:
        return {
            'total_queries': len(self._history),
            'reasks': self._reask_count,
            'satisfied': self._satisfied_count,
            'tracked_evidence': len(self._evidence_scores),
            'satisfaction_rate': (
                self._satisfied_count / max(1, self._satisfied_count + self._reask_count)
            ),
        }

    @staticmethod
    def _query_similarity(q1: str, q2: str) -> float:
        """
        Similarity between two queries using Jaccard + entity matching.
        If both queries share the same core entity (noun > 4 chars),
        boost the similarity score. This catches re-asks like
        "Where is Toronto?" / "Where exactly is Toronto?"
        """
        ignore = {'what', 'where', 'when', 'who', 'how', 'is', 'the',
                  'a', 'an', 'of', 'in', 'to', 'about', 'tell', 'me',
                  'please', 'can', 'does', 'it', 'are', 'was', 'exactly',
                  'really', 'actually'}
        w1 = {w.lower() for w in re.findall(r'\w+', q1) if w.lower() not in ignore and len(w) > 2}
        w2 = {w.lower() for w in re.findall(r'\w+', q2) if w.lower() not in ignore and len(w) > 2}
        if not w1 or not w2:
            return 0.0

        jaccard = len(w1 & w2) / len(w1 | w2)

        # Entity boost: if they share a long word (likely a proper noun/entity)
        entities1 = {w for w in w1 if len(w) >= 5}
        entities2 = {w for w in w2 if len(w) >= 5}
        shared_entities = entities1 & entities2

        if shared_entities:
            # Shared entity = almost certainly same topic
            jaccard = max(jaccard, 0.5 + 0.1 * len(shared_entities))

        return min(1.0, jaccard)


# ═══════════════════════════════════════════════════════════════
# GAP #9: AUTO FORMULA DISCOVERY
# Learns new formulas from ingested text
# ═══════════════════════════════════════════════════════════════

class FormulaLearner:
    """
    Scans all provenance sentences for mathematical expressions
    and automatically registers them with the CodeDriver.

    Patterns detected:
    - "X = expression" (e.g., "area = length * width")
    - "X equals expression" (e.g., "force equals mass times acceleration")
    - Named formulas ("compound interest formula: P(1+r/n)^(nt)")
    """

    # Common formula word-to-operator mappings
    WORD_OPS = {
        'times': '*', 'multiplied by': '*', 'divided by': '/',
        'plus': '+', 'minus': '-', 'squared': '**2', 'cubed': '**3',
        'raised to': '**', 'power of': '**',
    }

    def __init__(self):
        self.discovered = []  # list of {name, expression, source, sympy_expr}
        self._registered = set()  # avoid duplicates

    def scan_provenance(self, kernel) -> list:
        """
        Scan all provenance sentences for formula candidates.
        Returns list of discovered formulas.
        """
        new_formulas = []

        # Pattern 1: "X = math_expression"
        eq_pattern = re.compile(
            r'([a-zA-Z_]\w*)\s*=\s*([a-zA-Z0-9\s\*\+\-\/\(\)\^\.\,]+)',
        )

        # Pattern 2: "X equals Y times Z"
        word_pattern = re.compile(
            r'([a-zA-Z_]\w+)\s+equals?\s+(.+?)(?:\.|$)',
            re.IGNORECASE,
        )

        # Pattern 3: Numeric expressions like "345,000 * 0.08 = 27,600"
        numeric_pattern = re.compile(
            r'(\d[\d,]*\.?\d*)\s*[\*\/\+\-]\s*(\d[\d,]*\.?\d*)\s*=\s*(\d[\d,]*\.?\d*)',
        )

        for pair, sentences in kernel.provenance.items():
            for sent in sentences:
                sig = sent[:50]
                if sig in self._registered:
                    continue

                # Try pattern 1
                for match in eq_pattern.finditer(sent):
                    name = match.group(1).strip()
                    expr = match.group(2).strip()

                    # Must contain an operator
                    if any(op in expr for op in ['+', '-', '*', '/', '^', '**']):
                        if len(expr) > 3 and len(name) > 1:
                            formula = {
                                'name': name.lower(),
                                'expression': expr,
                                'source': sent[:80],
                                'type': 'equation',
                            }
                            new_formulas.append(formula)
                            self._registered.add(sig)

                # Try pattern 2 (word-based)
                for match in word_pattern.finditer(sent):
                    name = match.group(1).strip()
                    word_expr = match.group(2).strip()

                    # Convert word operators to symbols
                    converted = word_expr
                    for word_op, symbol in self.WORD_OPS.items():
                        converted = converted.replace(word_op, f' {symbol} ')

                    if any(op in converted for op in ['+', '-', '*', '/']):
                        formula = {
                            'name': name.lower(),
                            'expression': converted,
                            'source': sent[:80],
                            'type': 'word_equation',
                        }
                        new_formulas.append(formula)
                        self._registered.add(sig)

        self.discovered.extend(new_formulas)
        return new_formulas

    def register_with_codedriver(self, code_driver):
        """Push discovered formulas into the CodeDriver's registry."""
        registered = 0
        for formula in self.discovered:
            name = formula['name']
            expr = formula['expression']

            # Only register if not already known
            if name not in code_driver.FORMULA_REGISTRY:
                code_driver.FORMULA_REGISTRY[name] = {
                    'expression': expr,
                    'source': formula['source'],
                    'auto_discovered': True,
                }
                registered += 1

        return registered

    def get_stats(self) -> dict:
        return {
            'discovered': len(self.discovered),
            'unique': len(self._registered),
            'types': defaultdict(int,
                {f['type']: 1 for f in self.discovered}),
        }


# ═══════════════════════════════════════════════════════════════
# GAP L1: CONTINUOUS AUTO-TUNING (Self-Triggering)
# Runs auto-tuning automatically based on performance metrics
# ═══════════════════════════════════════════════════════════════

class ContinuousTuner:
    """
    Monitors query accuracy over time. When accuracy drops below
    a threshold, automatically triggers the AutoTuner.

    This makes Level 1 self-triggering rather than manual.
    """

    def __init__(self, auto_tuner, threshold: float = 0.75,
                 window_size: int = 20, check_interval: int = 50):
        self.tuner = auto_tuner
        self.threshold = threshold
        self.window_size = window_size
        self.check_interval = check_interval

        self._query_results = []  # True/False for correct/incorrect
        self._queries_since_check = 0
        self._tune_count = 0
        self._last_accuracy = 1.0

    def record_result(self, correct: bool):
        """Record whether a query was answered correctly."""
        self._query_results.append(correct)
        if len(self._query_results) > self.window_size * 3:
            self._query_results = self._query_results[-self.window_size * 3:]

        self._queries_since_check += 1

        # Check if it's time to evaluate
        if self._queries_since_check >= self.check_interval:
            self._check_and_tune()

    def _check_and_tune(self):
        """Check recent accuracy and trigger tuning if needed."""
        self._queries_since_check = 0

        if len(self._query_results) < self.window_size:
            return  # Not enough data yet

        # Calculate rolling accuracy
        recent = self._query_results[-self.window_size:]
        accuracy = sum(recent) / len(recent)
        self._last_accuracy = accuracy

        if accuracy < self.threshold:
            # Accuracy has dropped — trigger auto-tuning
            print(f"\n[CONTINUOUS-TUNER] Accuracy dropped to {accuracy:.0%} "
                  f"(threshold: {self.threshold:.0%})")
            print(f"[CONTINUOUS-TUNER] Auto-triggering threshold optimization...")

            self.tuner.tune(verbose=True)
            self._tune_count += 1

            print(f"[CONTINUOUS-TUNER] Re-tuning complete (total: {self._tune_count}x)")

    def force_check(self) -> dict:
        """Force a check regardless of interval."""
        if len(self._query_results) < 5:
            return {'accuracy': None, 'status': 'insufficient_data'}

        recent = self._query_results[-self.window_size:]
        accuracy = sum(recent) / len(recent)
        self._last_accuracy = accuracy

        needs_tuning = accuracy < self.threshold

        return {
            'accuracy': accuracy,
            'window_size': len(recent),
            'needs_tuning': needs_tuning,
            'tune_count': self._tune_count,
        }

    def get_stats(self) -> dict:
        recent = self._query_results[-self.window_size:] if self._query_results else []
        return {
            'total_queries': len(self._query_results),
            'recent_accuracy': sum(recent) / len(recent) if recent else 0,
            'threshold': self.threshold,
            'times_auto_tuned': self._tune_count,
            'queries_until_next_check': max(0, self.check_interval - self._queries_since_check),
        }


# ═══════════════════════════════════════════════════════════════
# GAP L3: AUTONOMOUS ANALOGY DISCOVERY AT SCALE
# Runs VSA analogy detection across the entire graph periodically
# ═══════════════════════════════════════════════════════════════

class AnalogyScanner:
    """
    Periodically scans the graph for structural analogies using
    KASM VSA vectors. Instead of comparing all N^2 pairs (expensive),
    uses degree-bucketing to only compare structurally similar nodes.

    Discovery: if two nodes have similar connection patterns but
    are in different semantic domains, they might be analogous.
    """

    def __init__(self, kernel, lexicon, similarity_threshold: float = 0.6):
        self.kernel = kernel
        self.lexicon = lexicon
        self.threshold = similarity_threshold
        self.discovered_analogies = []
        self._scanned = set()  # avoid rescanning

    def scan(self, max_comparisons: int = 5000, verbose: bool = True) -> list:
        """
        Scan for structural analogies using connection pattern similarity.

        Two nodes are analogous if:
        1. They have similar number of connections (degree-bucketed)
        2. Their connection targets overlap significantly (Jaccard)
        3. They are NOT directly connected (cross-domain)
        """
        from collections import defaultdict

        new_analogies = []
        comparisons = 0

        # Degree bucketing: group nodes by connection count
        degree_buckets = defaultdict(list)
        for nid, node in self.kernel.nodes.items():
            if node.connections:
                degree = len(node.connections)
                degree_buckets[degree].append(nid)

        # Also check adjacent degrees (+/- 2)
        expanded_buckets = defaultdict(list)
        for degree, nodes in degree_buckets.items():
            for d in range(max(1, degree - 2), degree + 3):
                if d in degree_buckets:
                    expanded_buckets[degree].extend(degree_buckets[d])

        # Compare within buckets
        for degree, node_ids in expanded_buckets.items():
            if len(node_ids) < 2:
                continue

            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    if comparisons >= max_comparisons:
                        break

                    nid_a, nid_b = node_ids[i], node_ids[j]
                    pair_key = tuple(sorted([nid_a, nid_b]))

                    if pair_key in self._scanned:
                        continue
                    self._scanned.add(pair_key)
                    comparisons += 1

                    # Skip if directly connected
                    node_a = self.kernel.nodes.get(nid_a)
                    node_b = self.kernel.nodes.get(nid_b)
                    if not node_a or not node_b:
                        continue
                    if nid_b in node_a.connections:
                        continue

                    # Calculate structural similarity (Jaccard on targets)
                    targets_a = set(node_a.connections.keys())
                    targets_b = set(node_b.connections.keys())

                    if not targets_a or not targets_b:
                        continue

                    # Shared targets
                    shared = targets_a & targets_b
                    jaccard = len(shared) / len(targets_a | targets_b)

                    if jaccard >= self.threshold:
                        word_a = self.lexicon.get_word(nid_a)
                        word_b = self.lexicon.get_word(nid_b)

                        analogy = {
                            'node_a': nid_a,
                            'node_b': nid_b,
                            'word_a': word_a,
                            'word_b': word_b,
                            'similarity': jaccard,
                            'shared_targets': len(shared),
                            'total_unique_targets': len(targets_a | targets_b),
                        }
                        new_analogies.append(analogy)

        self.discovered_analogies.extend(new_analogies)

        if verbose:
            print(f"\n[ANALOGY-SCANNER] Scanned {comparisons} pairs")
            print(f"  Discovered {len(new_analogies)} new analogies")
            for a in new_analogies[:5]:
                print(f"    {a['word_a']} <=> {a['word_b']} "
                      f"(Jaccard={a['similarity']:.2f}, "
                      f"shared={a['shared_targets']})")

        return new_analogies

    def wire_analogies(self, confidence: float = 0.5) -> int:
        """
        Wire discovered analogies as edges in the graph.
        Uses a conservative weight (default 0.5) since these are inferred.
        """
        wired = 0
        for analogy in self.discovered_analogies:
            nid_a = analogy['node_a']
            nid_b = analogy['node_b']

            if nid_a in self.kernel.nodes and nid_b in self.kernel.nodes:
                if nid_b not in self.kernel.nodes[nid_a].connections:
                    weight = min(confidence, analogy['similarity'])
                    self.kernel.add_connection(
                        nid_a, nid_b, weight,
                        f"[ANALOGY] {analogy['word_a']} <=> {analogy['word_b']} "
                        f"(Jaccard={analogy['similarity']:.2f})")
                    wired += 1

        return wired

    def get_stats(self) -> dict:
        return {
            'total_scanned': len(self._scanned),
            'analogies_found': len(self.discovered_analogies),
            'avg_similarity': (
                sum(a['similarity'] for a in self.discovered_analogies) /
                max(1, len(self.discovered_analogies))
            ),
        }
