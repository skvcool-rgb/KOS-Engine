"""
KOS Auto-Generated Agent: computerscienceAgent
Domain: computer science
Generated: 2026-03-25T14:57:16.426033

Specialized agent for computer science domain with 48 knowledge nodes. Core concepts: shop.n.01, memory.n.01, KASM_a61fa7f2, cerebral_cortex.n.01, comparison.n.03. Can answer questions about deep question answering, quantitative analysis, concept retrieval.

Capabilities: deep question answering, quantitative analysis, concept retrieval, relationship mapping

SAFETY: This agent can only READ the kernel graph.
It cannot modify files, make network calls, or execute arbitrary code.
"""


class computerscienceAgent:
    """
    Specialized agent for computer science domain.

    Core knowledge nodes: 20
    Capabilities: deep question answering, quantitative analysis, concept retrieval, relationship mapping
    """

    DOMAIN = "computer science"
    CORE_CONCEPTS = ['shop.n.01', 'memory.n.01', 'KASM_a61fa7f2', 'cerebral_cortex.n.01', 'comparison.n.03', 'KASM_c14cdb19', 'entree.n.02', 'database.n.01', 'structure.n.01', 'question.n.01', 'detector.n.01', 'row.n.01', 'key.n.01', 'document.n.01', 'KASM_ca7224dc', 'standardization.n.02', 'monitoring.n.01', 'redundancy.n.01', 'cassandra.n.01', 'KASM_1cc9a79b']
    ENTRY_QUERIES = ['What is shop.n.01?', 'How does shop.n.01 work?', 'What is memory.n.01?', 'How does memory.n.01 work?', 'What is KASM_a61fa7f2?', 'How does KASM_a61fa7f2 work?']
    CAPABILITIES = ['deep question answering', 'quantitative analysis', 'concept retrieval', 'relationship mapping']

    def __init__(self, kernel, lexicon, shell=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self._query_count = 0
        self._hits = 0

    def can_handle(self, query: str) -> float:
        """
        Score how well this agent can handle a query.
        Returns 0.0-1.0 confidence score.
        """
        query_lower = query.lower()
        score = 0.0

        # Check concept overlap
        for concept in self.CORE_CONCEPTS:
            if concept.lower() in query_lower:
                score += 0.3
                break

        # Check domain keywords
        domain_words = self.DOMAIN.lower().split()
        for word in domain_words:
            if word in query_lower:
                score += 0.2

        # Check if query mentions any connected concepts
        words = query_lower.split()
        for word in words:
            uid = self.lexicon.word_to_uuid.get(word)
            if uid and uid in self.kernel.nodes:
                node = self.kernel.nodes[uid]
                for tgt in node.connections:
                    if tgt in self.CORE_CONCEPTS[:10]:
                        score += 0.15
                        break

        return min(score, 1.0)

    def query(self, question: str) -> dict:
        """
        Answer a question using domain-specific knowledge.

        Returns: {
            "answer": str,
            "evidence": list,
            "confidence": float,
            "domain": str,
        }
        """
        self._query_count += 1

        # Use shell if available for full retrieval pipeline
        if self.shell:
            try:
                result = self.shell.query(question)
                if result and result.get("answer"):
                    self._hits += 1
                    return {
                        "answer": result["answer"],
                        "evidence": result.get("evidence", []),
                        "confidence": result.get("confidence", 0.5),
                        "domain": self.DOMAIN,
                        "agent": "computerscienceAgent",
                    }
            except Exception:
                pass

        # Fallback: direct graph traversal
        evidence = []
        words = question.lower().split()
        for word in words:
            uid = self.lexicon.word_to_uuid.get(word)
            if uid and uid in self.kernel.nodes:
                node = self.kernel.nodes[uid]
                for tgt, data in node.connections.items():
                    w = data.get("w", 0) if isinstance(data, dict) else data
                    if abs(w) > 0.3:
                        evidence.append({
                            "from": uid,
                            "to": tgt,
                            "weight": round(w, 3),
                        })

        # Sort by weight
        evidence.sort(key=lambda e: -abs(e.get("weight", 0)))
        evidence = evidence[:10]

        if evidence:
            self._hits += 1
            answer = "Based on %d connections: %s relates to %s" % (
                len(evidence),
                evidence[0]["from"],
                ", ".join(e["to"] for e in evidence[:3]))
        else:
            answer = "No strong evidence found in %s domain" % self.DOMAIN

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": min(len(evidence) * 0.1, 0.9),
            "domain": self.DOMAIN,
            "agent": "computerscienceAgent",
        }

    def get_stats(self) -> dict:
        """Return agent performance stats."""
        return {
            "name": "computerscienceAgent",
            "domain": self.DOMAIN,
            "queries": self._query_count,
            "hits": self._hits,
            "hit_rate": self._hits / max(self._query_count, 1),
            "core_concepts": len(self.CORE_CONCEPTS),
            "capabilities": self.CAPABILITIES,
        }

    def explain_domain(self) -> dict:
        """Return a summary of what this agent knows."""
        concept_details = []
        for concept in self.CORE_CONCEPTS[:10]:
            if concept in self.kernel.nodes:
                node = self.kernel.nodes[concept]
                connections = len(node.connections)
                top_neighbors = sorted(
                    node.connections.items(),
                    key=lambda x: -abs(x[1].get("w", 0)
                                       if isinstance(x[1], dict) else x[1])
                )[:5]
                concept_details.append({
                    "concept": concept,
                    "connections": connections,
                    "top_neighbors": [t[0] for t in top_neighbors],
                })

        return {
            "agent": "computerscienceAgent",
            "domain": self.DOMAIN,
            "total_concepts": len(self.CORE_CONCEPTS),
            "capabilities": self.CAPABILITIES,
            "concept_map": concept_details,
        }
