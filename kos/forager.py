"""
KOS V5.0 — Autonomous Web Forager.

Layer 5 Action System: When System Entropy is too high (the OS
doesn't know enough to answer safely), the Forager autonomously
reaches out to the internet, acquires structured knowledge, and
wires it into the graph.

The Forager does NOT use an LLM to summarize pages. It uses the
existing TextDriver SVO extraction pipeline to convert raw text
into graph edges with provenance. This preserves the zero-hallucination
guarantee — every fact in the graph traces back to a source sentence.

Usage:
    forager = WebForager(kernel, lexicon)
    facts_added = forager.forage("https://en.wikipedia.org/wiki/Toronto")
    facts_added = forager.forage_query("Toronto climate temperature")
"""

import re
import requests
from bs4 import BeautifulSoup


class WebForager:
    """
    Autonomous knowledge acquisition agent.

    Given a URL or search query, the Forager:
    1. Fetches the page content
    2. Extracts clean text (strips HTML, navigation, ads)
    3. Passes the text to the TextDriver for SVO extraction
    4. The TextDriver wires edges + provenance into the kernel

    The Forager is the OS's "hands" — it acts on the world to
    reduce its own uncertainty.
    """

    def __init__(self, kernel, lexicon, text_driver=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.driver = text_driver

        # Lazy-init TextDriver if not provided
        if self.driver is None:
            try:
                from .drivers.text import TextDriver
                self.driver = TextDriver(kernel, lexicon)
            except ImportError:
                raise RuntimeError(
                    "TextDriver required for WebForager. "
                    "Install nltk and jellyfish."
                )

        self.headers = {
            "User-Agent": "KOS-Engine/5.0 (Knowledge Forager; "
                          "+https://github.com/skvcool-rgb/KOS-Engine)"
        }
        self.max_chars = 50_000  # Don't ingest more than 50K chars per page

    def _fetch_and_clean(self, url: str) -> str:
        """
        Fetch a URL and extract clean paragraph text.
        Strips navigation, scripts, sidebars, footers.
        """
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[FORAGER] Fetch failed: {e}")
            return ""

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove noise elements
        for tag in soup.find_all(['script', 'style', 'nav', 'footer',
                                   'header', 'aside', 'form', 'button',
                                   'noscript', 'iframe']):
            tag.decompose()

        # Remove Wikipedia-specific noise
        for cls in ['navbox', 'sidebar', 'mw-references-wrap',
                     'reflist', 'toc', 'infobox', 'metadata',
                     'noprint', 'mw-editsection']:
            for el in soup.find_all(class_=cls):
                el.decompose()
        for el in soup.find_all(id=['references', 'external-links',
                                     'see-also', 'notes']):
            el.decompose()

        # Extract paragraph text
        paragraphs = []
        for p in soup.find_all(['p', 'li']):
            text = p.get_text(separator=' ', strip=True)
            # Skip very short or reference-heavy lines
            if len(text) > 40 and not text.startswith('['):
                # Clean up Wikipedia citation brackets [1][2]
                text = re.sub(r'\[\d+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                paragraphs.append(text)

        full_text = '\n'.join(paragraphs)

        # Truncate to max chars
        if len(full_text) > self.max_chars:
            full_text = full_text[:self.max_chars]
            # Cut at last sentence boundary
            last_period = full_text.rfind('.')
            if last_period > self.max_chars * 0.8:
                full_text = full_text[:last_period + 1]

        return full_text

    def forage(self, url: str, verbose: bool = True) -> int:
        """
        Fetch a URL and ingest its content into the knowledge graph.

        Returns the number of nodes in the graph after ingestion
        (to measure how much was learned).
        """
        if verbose:
            print(f"[FORAGER] Fetching: {url}")

        before_nodes = len(self.kernel.nodes)

        text = self._fetch_and_clean(url)
        if not text:
            if verbose:
                print("[FORAGER] No usable text extracted.")
            return 0

        if verbose:
            print(f"[FORAGER] Extracted {len(text):,} chars. Ingesting into graph...")

        self.driver.ingest(text)

        after_nodes = len(self.kernel.nodes)
        new_nodes = after_nodes - before_nodes

        if verbose:
            print(f"[FORAGER] Ingestion complete. +{new_nodes} new concepts wired.")

        return new_nodes

    def forage_query(self, query: str, verbose: bool = True) -> int:
        """
        Search Wikipedia for a topic and ingest the result.

        This is the simplest possible search — hit Wikipedia's API
        for the best matching article and ingest it. No LLM needed.
        """
        if verbose:
            print(f"[FORAGER] Searching Wikipedia for: '{query}'")

        # Use Wikipedia's opensearch API to find the best article
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "format": "json",
        }

        try:
            resp = requests.get(search_url, params=params,
                                headers=self.headers, timeout=10)
            data = resp.json()
            if len(data) >= 4 and data[3]:
                article_url = data[3][0]
                if verbose:
                    print(f"[FORAGER] Found article: {article_url}")
                return self.forage(article_url, verbose=verbose)
            else:
                if verbose:
                    print("[FORAGER] No Wikipedia article found.")
                return 0
        except Exception as e:
            if verbose:
                print(f"[FORAGER] Search failed: {e}")
            return 0

    def forage_multiple(self, queries: list, verbose: bool = True) -> int:
        """Forage multiple topics sequentially."""
        total = 0
        for q in queries:
            total += self.forage_query(q, verbose=verbose)
        return total
