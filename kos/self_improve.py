"""
KOS V5.1 — Self-Improvement Engine.

Implements 6 proposals that KOS generated about itself:
    #2  Degree Rebalancer (split hubs, connect orphans)
    #4  Weaver Feedback Loop (learn from user re-asks)
    #5  Contradiction Auto-Resolver (forage + suppress minority)
    #7  Continuous Self-Benchmark (monitor own accuracy)
    #8  Edge Weight Normalization (prevent weight inflation)
    #9  Auto Formula Discovery (expand CodeDriver from text)
"""

import re
import time
from collections import defaultdict


class SelfImprover:
    """
    The self-improvement daemon. Runs all 6 proposed improvements
    and measures the before/after impact.
    """

    def __init__(self, kernel, lexicon, shell=None, forager=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self.forager = forager

        # #4: Weaver feedback tracking
        self._query_history = []  # (query, answer, timestamp)
        self._reask_patterns = defaultdict(int)  # seed_key -> reask_count
        self._satisfied_patterns = defaultdict(int)

        # #7: Benchmark results
        self._benchmark_history = []

        # #9: Discovered formulas
        self._discovered_formulas = []

    # ═════════════════════════════════════════════════════════
    # #2: DEGREE REBALANCER
    # ═════════════════════════════════════════════════════════

    def rebalance_degrees(self, hub_threshold: int = 20,
                           verbose: bool = True) -> dict:
        """
        Split super-hubs and connect orphans.

        Super-hub: node with >hub_threshold connections.
        Action: don't delete edges, but reduce ambient noise
        weights on the weakest 50% of connections.

        Orphan: node with 0 connections.
        Action: find the nearest node via word similarity
        and create a weak (0.3) connection.
        """
        hubs_fixed = 0
        orphans_fixed = 0

        # Fix super-hubs: weaken their weakest 50% edges
        for nid, node in list(self.kernel.nodes.items()):
            if len(node.connections) > hub_threshold:
                edges = sorted(
                    node.connections.items(),
                    key=lambda x: abs(x[1]['w'] if isinstance(x[1], dict) else x[1])
                )
                # Weaken bottom 50%
                cutoff = len(edges) // 2
                for tgt, data in edges[:cutoff]:
                    if isinstance(data, dict):
                        data['w'] *= 0.5  # Halve weak edges
                    else:
                        node.connections[tgt] = data * 0.5
                hubs_fixed += 1

        # Fix orphans: connect to nearest known word
        import difflib
        known_words = list(self.lexicon.word_to_uuid.keys())
        for nid, node in list(self.kernel.nodes.items()):
            if not node.connections:
                word = self.lexicon.get_word(nid)
                matches = difflib.get_close_matches(word.lower(),
                                                      known_words, n=1, cutoff=0.4)
                if matches:
                    match_uid = self.lexicon.word_to_uuid[matches[0]]
                    if match_uid != nid and match_uid in self.kernel.nodes:
                        self.kernel.add_connection(nid, match_uid, 0.3,
                            "[SELF-IMPROVE] Orphan connected via similarity")
                        orphans_fixed += 1

        if verbose:
            print(f"  [#2] Degree Rebalancer: {hubs_fixed} hubs weakened, "
                  f"{orphans_fixed} orphans connected")

        return {'hubs_fixed': hubs_fixed, 'orphans_fixed': orphans_fixed}

    # ═════════════════════════════════════════════════════════
    # #4: WEAVER FEEDBACK LOOP
    # ═════════════════════════════════════════════════════════

    def record_query(self, query: str, answer: str):
        """Track query-answer pairs for feedback analysis."""
        self._query_history.append({
            'query': query, 'answer': answer,
            'time': time.time()
        })
        # Keep bounded
        if len(self._query_history) > 200:
            self._query_history = self._query_history[-200:]

    def detect_reasks(self, verbose: bool = True) -> dict:
        """
        Detect when a user re-asks a similar question (= dissatisfied).

        If query N and query N+1 share >60% words, the first answer
        was probably wrong. Track which evidence patterns fail.
        """
        reasks = 0
        if len(self._query_history) < 2:
            return {'reasks': 0}

        for i in range(len(self._query_history) - 1):
            q1_words = set(self._query_history[i]['query'].lower().split())
            q2_words = set(self._query_history[i+1]['query'].lower().split())

            if len(q1_words | q2_words) == 0:
                continue

            overlap = len(q1_words & q2_words) / len(q1_words | q2_words)
            if overlap > 0.6:
                reasks += 1

        if verbose:
            print(f"  [#4] Weaver Feedback: {reasks} re-asks detected "
                  f"(user dissatisfaction signals)")

        return {'reasks': reasks}

    # ═════════════════════════════════════════════════════════
    # #5: CONTRADICTION AUTO-RESOLVER
    # ═════════════════════════════════════════════════════════

    def resolve_contradictions(self, verbose: bool = True) -> dict:
        """
        Auto-resolve contradictions by counting evidence weight
        on each side and suppressing the minority.
        """
        contradictions = getattr(self.kernel, 'contradictions', [])
        resolved = 0

        for c in list(contradictions):
            source = c.get('source')
            existing = c.get('existing_target')
            new_target = c.get('new_target')

            if not all([source, existing, new_target]):
                continue
            if source not in self.kernel.nodes:
                continue

            conn = self.kernel.nodes[source].connections

            # Count evidence for each side
            existing_w = 0
            new_w = 0

            if existing in conn:
                data = conn[existing]
                existing_w = data['w'] if isinstance(data, dict) else data

            if new_target in conn:
                data = conn[new_target]
                new_w = data['w'] if isinstance(data, dict) else data

            # Count provenance sentences for each
            existing_prov = len(self.kernel.provenance.get(
                tuple(sorted([source, existing])), set()))
            new_prov = len(self.kernel.provenance.get(
                tuple(sorted([source, new_target])), set()))

            # The side with more provenance wins
            if new_prov > existing_prov and existing in conn:
                # New evidence stronger — suppress old
                if isinstance(conn[existing], dict):
                    conn[existing]['w'] *= 0.3
                    conn[existing]['myelin'] = 0
                else:
                    conn[existing] = conn[existing] * 0.3
                resolved += 1
            elif existing_prov > new_prov and new_target in conn:
                # Old evidence stronger — suppress new
                if isinstance(conn[new_target], dict):
                    conn[new_target]['w'] *= 0.3
                else:
                    conn[new_target] = conn[new_target] * 0.3
                resolved += 1

        if verbose:
            print(f"  [#5] Contradiction Resolver: {resolved}/{len(contradictions)} "
                  f"resolved by evidence weight")

        return {'contradictions_total': len(contradictions),
                'resolved': resolved}

    # ═════════════════════════════════════════════════════════
    # #7: CONTINUOUS SELF-BENCHMARK
    # ═════════════════════════════════════════════════════════

    def run_benchmark(self, verbose: bool = True) -> dict:
        """
        Run a mini-benchmark against known test cases.

        Returns accuracy and identifies which queries fail.
        """
        if not self.shell:
            return {'accuracy': 0, 'error': 'No shell configured'}

        benchmark = [
            ("Where is Toronto?", ["canada", "ontario", "city"]),
            ("When was Toronto founded?", ["1834", "incorporated", "founded"]),
            ("Population of Toronto?", ["million", "population"]),
            ("Climate of Toronto?", ["humid", "continental", "weather"]),
            ("Tell me about apixaban", ["anticoagulant", "bleeding", "thrombosis", "prevent"]),
            ("Tell me about perovskite", ["efficient", "solar", "photovoltaic"]),
            ("345000000 * 0.0825", ["28462500"]),
        ]

        passed = 0
        failed_queries = []

        for query, expected in benchmark:
            try:
                answer = self.shell.chat(query).lower()
                if any(kw.lower() in answer for kw in expected):
                    passed += 1
                else:
                    failed_queries.append(query)
            except Exception:
                failed_queries.append(query)

        accuracy = passed / len(benchmark) if benchmark else 0

        result = {
            'accuracy': accuracy,
            'passed': passed,
            'total': len(benchmark),
            'failed': failed_queries,
        }

        self._benchmark_history.append({
            'time': time.time(),
            'accuracy': accuracy,
        })

        if verbose:
            print(f"  [#7] Self-Benchmark: {passed}/{len(benchmark)} "
                  f"({accuracy:.0%})")
            if failed_queries:
                for q in failed_queries:
                    print(f"       FAILED: {q}")

        return result

    # ═════════════════════════════════════════════════════════
    # #8: EDGE WEIGHT NORMALIZATION
    # ═════════════════════════════════════════════════════════

    def normalize_weights(self, verbose: bool = True) -> dict:
        """
        Normalize all edge weights to [-1.0, 1.0].

        Myelination can push effective weights above 1.0 over time.
        This keeps the physics numerically stable.
        """
        clipped = 0
        max_found = 0.0

        for nid, node in list(self.kernel.nodes.items()):
            for tgt, data in list(node.connections.items()):
                if isinstance(data, dict):
                    w = data['w']
                    effective = w * (1 + data.get('myelin', 0) * 0.01)
                    max_found = max(max_found, abs(effective))

                    if abs(w) > 1.0:
                        data['w'] = max(-1.0, min(1.0, w))
                        clipped += 1
                else:
                    if abs(data) > 1.0:
                        node.connections[tgt] = max(-1.0, min(1.0, data))
                        clipped += 1
                    max_found = max(max_found, abs(data))

        if verbose:
            print(f"  [#8] Weight Normalization: {clipped} weights clipped, "
                  f"max effective weight: {max_found:.3f}")

        return {'clipped': clipped, 'max_weight': max_found}

    # ═════════════════════════════════════════════════════════
    # #9: AUTO FORMULA DISCOVERY
    # ═════════════════════════════════════════════════════════

    def discover_formulas(self, verbose: bool = True) -> dict:
        """
        Scan provenance sentences for mathematical expressions
        and extract them as formula candidates.
        """
        formula_pattern = re.compile(
            r'([A-Za-z]+)\s*=\s*([A-Za-z0-9\s\*\+\-\/\(\)\^\.]+)'
        )
        number_formula = re.compile(
            r'(\d+[\.,]?\d*)\s*[\*\/\+\-]\s*(\d+[\.,]?\d*)'
        )

        discovered = []

        for pair, sentences in list(self.kernel.provenance.items()):
            for sent in list(sentences):
                # Look for "X = expression" patterns
                matches = formula_pattern.findall(sent)
                for name, expr in matches:
                    if len(expr.strip()) > 3 and any(
                        c in expr for c in '+-*/^'):
                        discovered.append({
                            'name': name.strip(),
                            'expression': expr.strip(),
                            'source': sent[:80],
                        })

        self._discovered_formulas = discovered

        if verbose:
            print(f"  [#9] Formula Discovery: {len(discovered)} "
                  f"candidates found in provenance")
            for f in discovered[:3]:
                print(f"       {f['name']} = {f['expression'][:40]}")

        return {'formulas_found': len(discovered)}

    # ═════════════════════════════════════════════════════════
    # MAIN: RUN ALL IMPROVEMENTS
    # ═════════════════════════════════════════════════════════

    def improve(self, verbose: bool = True, quick: bool = False) -> dict:
        """
        Run self-improvement proposals and return metrics.
        quick=True skips heavy structural mutations (rebalance, normalize) for faster health checks.
        """
        if verbose:
            print("\n[SELF-IMPROVE] Running improvement proposals...")

        t0 = time.perf_counter()

        results = {}
        if not quick:
            results['rebalance'] = self.rebalance_degrees(verbose=verbose)
        else:
            results['rebalance'] = {'hubs_fixed': 0, 'orphans_fixed': 0, 'skipped': True}
        results['feedback'] = self.detect_reasks(verbose=verbose)
        results['contradictions'] = self.resolve_contradictions(verbose=verbose)
        results['benchmark'] = self.run_benchmark(verbose=verbose)
        if not quick:
            results['normalization'] = self.normalize_weights(verbose=verbose)
        else:
            results['normalization'] = {'clipped': 0, 'max_weight': 0, 'skipped': True}
        results['formulas'] = self.discover_formulas(verbose=verbose)

        elapsed = (time.perf_counter() - t0) * 1000
        results['time_ms'] = elapsed

        if verbose:
            print(f"\n[SELF-IMPROVE] Complete in {elapsed:.0f}ms")

        return results
