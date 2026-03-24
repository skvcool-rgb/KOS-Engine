"""
KOS V5.1 — Autonomous Web Forager (Multi-Backend).

Layer 5 Action System with pluggable search backends:
    - Wikipedia (default, always available)
    - arXiv (scientific papers)
    - Local PDF/text files
    - Any URL (direct fetch)

Fix #11: Multiple search backends with priority routing.

Usage:
    forager = WebForager(kernel, lexicon)
    forager.forage("https://en.wikipedia.org/wiki/Toronto")
    forager.forage_query("Toronto climate temperature")
    forager.forage_arxiv("perovskite solar cell efficiency")
    forager.forage_file("/path/to/document.txt")
"""

import os
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

    # ── FIX #11: Multi-Backend Search ────────────────────────

    def forage_arxiv(self, query: str, max_results: int = 3,
                     verbose: bool = True) -> int:
        """
        Search arXiv for scientific papers and ingest abstracts.

        Uses the arXiv API (no authentication required).
        Ingests paper titles + abstracts (not full text).
        """
        if verbose:
            print(f"[FORAGER-ARXIV] Searching: '{query}'")

        search_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

        try:
            resp = requests.get(search_url, params=params,
                                headers=self.headers, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')

            entries = soup.find_all('entry')
            if not entries:
                if verbose:
                    print("[FORAGER-ARXIV] No results found.")
                return 0

            total_new = 0
            before = len(self.kernel.nodes)

            for entry in entries:
                title = entry.find('title')
                summary = entry.find('summary')
                if title and summary:
                    text = f"{title.get_text(strip=True)}. {summary.get_text(strip=True)}"
                    text = re.sub(r'\s+', ' ', text).strip()
                    if verbose:
                        print(f"  Paper: {title.get_text(strip=True)[:80]}...")
                    self.driver.ingest(text)

            total_new = len(self.kernel.nodes) - before
            if verbose:
                print(f"[FORAGER-ARXIV] +{total_new} concepts from "
                      f"{len(entries)} papers")
            return total_new

        except Exception as e:
            if verbose:
                print(f"[FORAGER-ARXIV] Error: {e}")
            return 0

    def forage_file(self, filepath: str, verbose: bool = True) -> int:
        """
        Ingest a local text file into the knowledge graph.

        Supports .txt and .md files. For PDFs, pre-convert to text.
        """
        if not os.path.exists(filepath):
            if verbose:
                print(f"[FORAGER-FILE] Not found: {filepath}")
            return 0

        if verbose:
            print(f"[FORAGER-FILE] Reading: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            if verbose:
                print(f"[FORAGER-FILE] Read error: {e}")
            return 0

        if len(text) > self.max_chars:
            text = text[:self.max_chars]
            last_period = text.rfind('.')
            if last_period > self.max_chars * 0.8:
                text = text[:last_period + 1]

        before = len(self.kernel.nodes)
        self.driver.ingest(text)
        new_nodes = len(self.kernel.nodes) - before

        if verbose:
            print(f"[FORAGER-FILE] +{new_nodes} concepts from {filepath}")

        return new_nodes

    def forage_smart(self, query: str, verbose: bool = True) -> int:
        """
        Smart foraging: tries Wikipedia first, then arXiv if
        the query looks scientific.

        Heuristic: if query contains scientific terms, try arXiv.
        """
        # Try Wikipedia first
        result = self.forage_query(query, verbose=verbose)
        if result > 50:  # Got substantial content
            return result

        # If Wikipedia was thin, try arXiv for scientific queries
        science_words = {"cell", "solar", "quantum", "neural", "protein",
                         "molecule", "genome", "algorithm", "theorem",
                         "perovskite", "photovoltaic", "semiconductor",
                         "enzyme", "catalyst", "polymer", "nanoscale"}
        query_words = set(query.lower().split())
        if query_words & science_words:
            if verbose:
                print("[FORAGER] Wikipedia thin. Trying arXiv...")
            arxiv_result = self.forage_arxiv(query, verbose=verbose)
            return result + arxiv_result

        return result
