"""
KOS V6.1 — User Modeling System.

Tracks per-user profiles to adapt KOS responses to individual needs:
    - Expertise detection (beginner / intermediate / expert)
    - Preferred detail level (brief / normal / detailed)
    - Domain interests (from query history)
    - Satisfaction scoring (from feedback)
    - Response adaptation (simplify for beginners, cite for experts)

Profiles persist across sessions via .cache/user_profiles.json.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Technical term sets for expertise detection
# ---------------------------------------------------------------------------

EXPERT_TERMS: Set[str] = {
    # Scientific
    "enzyme", "catalyst", "photovoltaic", "perovskite", "quantum",
    "entanglement", "eigenvalue", "eigenstate", "hamiltonian",
    "stochastic", "heuristic", "bayesian", "markov", "gaussian",
    "spectroscopy", "chromatography", "thermodynamics", "entropy",
    "isotope", "isomer", "polymer", "monomer", "stoichiometry",
    # Technical / CS
    "algorithm", "backpropagation", "gradient", "convolution",
    "recurrent", "transformer", "attention", "embedding", "latent",
    "hyperparameter", "regularization", "overfitting", "sigmoid",
    "kernel", "manifold", "topology", "orthogonal", "tensor",
    "mutex", "semaphore", "deadlock", "heap", "amortized",
    # Medical
    "pharmacokinetics", "pharmacodynamics", "contraindication",
    "pathogenesis", "etiology", "prognosis", "bioavailability",
    "receptor", "agonist", "antagonist", "metabolite", "cytokine",
    "apoptosis", "mitosis", "meiosis", "transcription", "ribosome",
    # Legal
    "jurisprudence", "precedent", "statute", "tort", "liability",
    "adjudication", "arbitration", "subpoena", "deposition",
}

INTERMEDIATE_TERMS: Set[str] = {
    "molecule", "atom", "cell", "protein", "gene", "genome",
    "neural", "network", "database", "api", "function", "variable",
    "hypothesis", "experiment", "correlation", "regression",
    "probability", "statistics", "derivative", "integral",
    "diagnosis", "symptom", "treatment", "dosage", "side effect",
    "contract", "regulation", "amendment", "jurisdiction",
}


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------

class UserProfile:
    """Per-user profile tracking preferences and behaviour."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.query_history: List[Dict] = []
        self.expertise_level: str = "intermediate"  # beginner/intermediate/expert
        self.preferred_detail: str = "normal"        # brief/normal/detailed
        self.domain_interests: Counter = Counter()
        self.satisfaction_score: float = 0.5         # 0.0 - 1.0
        self.total_queries: int = 0
        self.positive_feedback: int = 0
        self.negative_feedback: int = 0
        self.created_at: float = time.time()
        self.last_active: float = time.time()

    def to_dict(self) -> Dict:
        """Serialise profile to a JSON-safe dict."""
        return {
            'user_id': self.user_id,
            'expertise_level': self.expertise_level,
            'preferred_detail': self.preferred_detail,
            'domain_interests': dict(self.domain_interests.most_common(20)),
            'satisfaction_score': round(self.satisfaction_score, 3),
            'total_queries': self.total_queries,
            'positive_feedback': self.positive_feedback,
            'negative_feedback': self.negative_feedback,
            'created_at': self.created_at,
            'last_active': self.last_active,
            'query_history': self.query_history[-50:],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        """Deserialise profile from a dict."""
        profile = cls(data.get('user_id', 'unknown'))
        profile.expertise_level = data.get('expertise_level', 'intermediate')
        profile.preferred_detail = data.get('preferred_detail', 'normal')
        profile.domain_interests = Counter(data.get('domain_interests', {}))
        profile.satisfaction_score = data.get('satisfaction_score', 0.5)
        profile.total_queries = data.get('total_queries', 0)
        profile.positive_feedback = data.get('positive_feedback', 0)
        profile.negative_feedback = data.get('negative_feedback', 0)
        profile.created_at = data.get('created_at', time.time())
        profile.last_active = data.get('last_active', time.time())
        profile.query_history = data.get('query_history', [])
        return profile


# ---------------------------------------------------------------------------
# User Model
# ---------------------------------------------------------------------------

class UserModel:
    """Tracks and adapts to individual user behaviour.

    Maintains per-user profiles with query history, expertise level,
    preferred detail level, domain interests, and satisfaction scores.
    Profiles persist to .cache/user_profiles.json.

    Example::

        model = UserModel()
        model.update_from_interaction("user1", "What is pharmacokinetics?",
                                      "Pharmacokinetics is...", "helpful")
        profile = model.get_profile("user1")
        adapted = model.adapt_response("Full answer here.", profile)
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        """Initialise the user model.

        Args:
            cache_dir: Directory for persistent profile storage.
        """
        self._profiles: Dict[str, UserProfile] = {}
        self._cache_dir = cache_dir
        self._cache_file = os.path.join(cache_dir, "user_profiles.json")
        self._load_profiles()

    # ----- Profile Management ---------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile.

        Args:
            user_id: Unique user identifier.

        Returns:
            The UserProfile for this user.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id)
        return self._profiles[user_id]

    def get_all_profiles(self) -> Dict[str, UserProfile]:
        """Return all tracked user profiles."""
        return dict(self._profiles)

    # ----- Expertise Detection --------------------------------------------

    def detect_expertise(self, queries: List[str]) -> str:
        """Classify user expertise from their query language.

        Scans queries for technical vocabulary:
        - 3+ expert terms -> 'expert'
        - 2+ intermediate terms -> 'intermediate'
        - Otherwise -> 'beginner'

        Args:
            queries: List of raw query strings from the user.

        Returns:
            One of 'beginner', 'intermediate', 'expert'.
        """
        if not queries:
            return "intermediate"

        all_words: Set[str] = set()
        for q in queries:
            words = set(re.findall(r'\w+', q.lower()))
            all_words.update(words)

        expert_count = len(all_words & EXPERT_TERMS)
        intermediate_count = len(all_words & INTERMEDIATE_TERMS)

        if expert_count >= 3:
            return "expert"
        elif expert_count >= 1 or intermediate_count >= 2:
            return "intermediate"
        else:
            return "beginner"

    # ----- Response Adaptation --------------------------------------------

    def adapt_response(self, answer: str,
                       user_profile: UserProfile) -> str:
        """Adapt an answer based on user profile.

        Adaptation rules:
        - beginner: prepend a simplified explanation
        - expert: append citation/evidence note
        - brief: return first sentence only
        - detailed: return full answer with evidence chain marker

        Args:
            answer: The raw answer from the Weaver.
            user_profile: The user's profile.

        Returns:
            Adapted answer string.
        """
        if not answer or not answer.strip():
            return answer

        adapted = answer

        # Detail level adaptation
        if user_profile.preferred_detail == "brief":
            # Return first sentence only
            sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
            if sentences:
                adapted = sentences[0]

        elif user_profile.preferred_detail == "detailed":
            # Add evidence chain marker
            if not adapted.endswith('.'):
                adapted += '.'
            adapted += ("\n[Evidence chain available. "
                        "Ask 'show evidence' for full provenance.]")

        # Expertise level adaptation
        if user_profile.expertise_level == "beginner":
            # Prepend simplified framing
            adapted = ("In simple terms: " + adapted)

        elif user_profile.expertise_level == "expert":
            # Append technical note
            if not adapted.endswith('.'):
                adapted += '.'
            adapted += (" [Source: KOS graph evidence with "
                        "spreading activation scoring.]")

        return adapted

    # ----- Interaction Learning -------------------------------------------

    def update_from_interaction(self, user_id: str, query: str,
                                answer: str,
                                was_helpful: bool = True,
                                feedback: Optional[str] = None) -> UserProfile:
        """Learn user preferences from an interaction.

        Records the query, updates domain interests, re-evaluates
        expertise level, and adjusts satisfaction from feedback.

        Args:
            user_id: Unique user identifier.
            query: The query that was asked.
            answer: The answer that was returned.
            was_helpful: Whether the user found the answer helpful.
                         True boosts satisfaction, False reduces it.
            feedback: Optional feedback string. Positive keywords
                      ('helpful', 'good', 'thanks', 'correct', 'yes')
                      boost satisfaction. Negative keywords ('wrong',
                      'bad', 'incorrect', 'no', 'unhelpful') reduce it.

        Returns:
            The updated UserProfile.
        """
        profile = self.get_profile(user_id)
        profile.last_active = time.time()
        profile.total_queries += 1

        # Record query
        profile.query_history.append({
            'query': query[:200],
            'answer_preview': answer[:100] if answer else '',
            'timestamp': time.time(),
            'feedback': feedback,
        })

        # Trim history to last 200 entries
        if len(profile.query_history) > 200:
            profile.query_history = profile.query_history[-200:]

        # Update domain interests from query words
        query_words = set(re.findall(r'\w+', query.lower()))
        domain_keywords = {
            'medical': {'disease', 'drug', 'treatment', 'symptom',
                        'patient', 'clinical', 'therapy', 'diagnosis',
                        'health', 'medical', 'doctor', 'hospital'},
            'scientific': {'experiment', 'hypothesis', 'molecule',
                           'quantum', 'physics', 'chemistry', 'biology',
                           'science', 'research', 'theory', 'equation'},
            'technical': {'algorithm', 'code', 'software', 'api',
                          'database', 'server', 'programming', 'debug',
                          'function', 'system', 'network', 'protocol'},
            'legal': {'law', 'legal', 'court', 'statute', 'contract',
                      'jurisdiction', 'attorney', 'plaintiff', 'defendant'},
            'geographic': {'city', 'country', 'river', 'mountain',
                           'continent', 'ocean', 'region', 'capital',
                           'population', 'border', 'island'},
            'historical': {'history', 'war', 'century', 'dynasty',
                           'empire', 'revolution', 'ancient', 'medieval',
                           'colonial', 'independence'},
        }
        for domain, keywords in domain_keywords.items():
            overlap = query_words & keywords
            if overlap:
                profile.domain_interests[domain] += len(overlap)

        # Re-evaluate expertise based on recent queries
        recent_queries = [
            q['query'] for q in profile.query_history[-20:]
        ]
        profile.expertise_level = self.detect_expertise(recent_queries)

        # Process was_helpful boolean
        if was_helpful:
            profile.positive_feedback += 1
            profile.satisfaction_score = min(1.0,
                profile.satisfaction_score + 0.05)
        else:
            profile.negative_feedback += 1
            profile.satisfaction_score = max(0.0,
                profile.satisfaction_score - 0.08)

        # Process optional feedback string (overrides was_helpful if present)
        if feedback:
            feedback_lower = feedback.lower()
            positive = {'helpful', 'good', 'thanks', 'correct', 'yes',
                        'great', 'perfect', 'excellent', 'right', 'accurate'}
            negative = {'wrong', 'bad', 'incorrect', 'no', 'unhelpful',
                        'poor', 'inaccurate', 'false', 'terrible'}

            feedback_words = set(feedback_lower.split())
            if feedback_words & positive:
                profile.positive_feedback += 1
                profile.satisfaction_score = min(1.0,
                    profile.satisfaction_score + 0.05)
            elif feedback_words & negative:
                profile.negative_feedback += 1
                profile.satisfaction_score = max(0.0,
                    profile.satisfaction_score - 0.08)

            # Detect detail preference from feedback
            if any(w in feedback_lower for w in ['brief', 'shorter', 'concise', 'tldr']):
                profile.preferred_detail = "brief"
            elif any(w in feedback_lower for w in ['detail', 'more', 'elaborate', 'explain']):
                profile.preferred_detail = "detailed"

        # Persist after update
        self._save_profiles()

        return profile

    # ----- Persistence ----------------------------------------------------

    def save(self) -> None:
        """Public save — persist all profiles to the cache file."""
        self._save_profiles()

    def load(self) -> None:
        """Public load — reload profiles from the cache file."""
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load profiles from the cache file."""
        if not os.path.exists(self._cache_file):
            return
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for uid, profile_data in data.items():
                self._profiles[uid] = UserProfile.from_dict(profile_data)
        except (json.JSONDecodeError, IOError, KeyError):
            pass  # Silently skip corrupt cache

    def _save_profiles(self) -> None:
        """Save profiles to the cache file."""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            data = {
                uid: profile.to_dict()
                for uid, profile in self._profiles.items()
            }
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError):
            pass  # Silently skip write failures

    # ----- Utilities ------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics across all users."""
        total_users = len(self._profiles)
        if total_users == 0:
            return {
                'total_users': 0,
                'avg_satisfaction': 0.0,
                'expertise_distribution': {},
            }

        expertise_counts = Counter(
            p.expertise_level for p in self._profiles.values()
        )
        avg_satisfaction = sum(
            p.satisfaction_score for p in self._profiles.values()
        ) / total_users

        return {
            'total_users': total_users,
            'avg_satisfaction': round(avg_satisfaction, 3),
            'expertise_distribution': dict(expertise_counts),
            'total_queries': sum(
                p.total_queries for p in self._profiles.values()),
            'top_domains': Counter(
                {domain: count
                 for p in self._profiles.values()
                 for domain, count in p.domain_interests.items()}
            ).most_common(5),
        }

    def __repr__(self) -> str:
        return f"UserModel(users={len(self._profiles)})"
