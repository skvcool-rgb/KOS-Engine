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

    def _wikipedia_search(self, query: str, verbose: bool = False) -> str:
        """Search Wikipedia opensearch API. Returns article URL or empty string."""
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
                return data[3][0]
        except Exception:
            pass
        return ""

    def forage_query(self, query: str, verbose: bool = True) -> int:
        """
        Search Wikipedia for a topic and ingest the result.

        Uses progressive fallback:
        1. Try full query ("tungsten boiling point properties")
        2. Try first two words ("tungsten boiling")
        3. Try first word only ("tungsten")
        4. Try each word individually

        This ensures maximum hit rate on Wikipedia's search.
        """
        if verbose:
            print(f"[FORAGER] Searching Wikipedia for: '{query}'")

        words = query.strip().split()

        # Progressive search strategies
        search_attempts = [
            query,                                    # Full query
            ' '.join(words[:2]) if len(words) > 2 else None,  # First 2 words
            words[0] if len(words) > 1 else None,     # First word only
        ]
        # Add individual words as last resort (skip stopwords)
        stopwords = {'what', 'is', 'the', 'of', 'how', 'does', 'do', 'are',
                     'was', 'were', 'can', 'will', 'a', 'an', 'and', 'or',
                     'in', 'on', 'at', 'to', 'for', 'with', 'about', 'tell',
                     'me', 'point', 'properties'}
        for w in words:
            if w.lower() not in stopwords and len(w) > 3:
                search_attempts.append(w)

        # Remove None and duplicates while preserving order
        seen = set()
        clean_attempts = []
        for s in search_attempts:
            if s and s not in seen:
                seen.add(s)
                clean_attempts.append(s)

        for attempt in clean_attempts:
            if verbose:
                print(f"[FORAGER] Trying: '{attempt}'")

            url = self._wikipedia_search(attempt, verbose=verbose)
            if url:
                if verbose:
                    print(f"[FORAGER] Found article: {url}")
                return self.forage(url, verbose=verbose)

        if verbose:
            print(f"[FORAGER] No Wikipedia article found after {len(clean_attempts)} attempts.")
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

    # ── PubMed Search Backend ─────────────────────────────────

    def forage_pubmed(self, query: str, max_results: int = 3,
                      verbose: bool = True) -> int:
        """
        Search PubMed for biomedical literature and ingest abstracts.

        Uses NCBI E-utilities API (no authentication required for
        low-volume requests). Ingests paper titles + abstracts.

        Args:
            query: Search terms for PubMed.
            max_results: Maximum number of papers to fetch (default 3).
            verbose: Print progress messages.

        Returns:
            Number of new concepts added to the graph.
        """
        if verbose:
            print(f"[FORAGER-PUBMED] Searching: '{query}'")

        # Step 1: Search for PubMed IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }

        try:
            resp = requests.get(search_url, params=search_params,
                                headers=self.headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                if verbose:
                    print("[FORAGER-PUBMED] No results found.")
                return 0

            if verbose:
                print(f"[FORAGER-PUBMED] Found {len(id_list)} papers. "
                      "Fetching abstracts...")

            # Step 2: Fetch abstracts for all IDs
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "text",
            }

            fetch_resp = requests.get(fetch_url, params=fetch_params,
                                      headers=self.headers, timeout=20)
            fetch_resp.raise_for_status()
            abstract_text = fetch_resp.text

            if not abstract_text or len(abstract_text.strip()) < 50:
                if verbose:
                    print("[FORAGER-PUBMED] No usable abstracts retrieved.")
                return 0

            # Truncate to max chars
            if len(abstract_text) > self.max_chars:
                abstract_text = abstract_text[:self.max_chars]
                last_period = abstract_text.rfind('.')
                if last_period > self.max_chars * 0.8:
                    abstract_text = abstract_text[:last_period + 1]

            before = len(self.kernel.nodes)
            self.driver.ingest(abstract_text)
            total_new = len(self.kernel.nodes) - before

            if verbose:
                print(f"[FORAGER-PUBMED] +{total_new} concepts from "
                      f"{len(id_list)} papers")
            return total_new

        except Exception as e:
            if verbose:
                print(f"[FORAGER-PUBMED] Error: {e}")
            return 0

    # ── Google Search Backend ─────────────────────────────────

    def forage_google(self, query: str, max_results: int = 3,
                       verbose: bool = True) -> int:
        """
        Search Google and ingest top results.

        Uses Google's public search to find relevant pages,
        then fetches and ingests each one.

        No API key needed — uses the public search URL.
        Respects rate limits (1 search per call).
        """
        if verbose:
            print(f"[FORAGER-GOOGLE] Searching: '{query}'")

        try:
            # Use Google search via requests
            search_url = "https://www.google.com/search"
            params = {"q": query, "num": max_results}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }

            resp = requests.get(search_url, params=params,
                                headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Extract URLs from search results
            urls = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Google wraps URLs in /url?q=...
                if '/url?q=' in href:
                    real_url = href.split('/url?q=')[1].split('&')[0]
                    if real_url.startswith('http') and 'google.com' not in real_url:
                        urls.append(real_url)
                elif href.startswith('http') and 'google.com' not in href:
                    urls.append(href)

            urls = list(dict.fromkeys(urls))[:max_results]  # Dedupe, limit

            if not urls:
                if verbose:
                    print("[FORAGER-GOOGLE] No URLs found in search results.")
                return 0

            if verbose:
                print(f"[FORAGER-GOOGLE] Found {len(urls)} URLs. Ingesting...")

            total_new = 0
            for url in urls:
                try:
                    new = self.forage(url, verbose=verbose)
                    total_new += new
                    if total_new > 200:  # Cap per search
                        break
                except Exception as e:
                    if verbose:
                        print(f"[FORAGER-GOOGLE] Failed on {url[:50]}: {e}")

            if verbose:
                print(f"[FORAGER-GOOGLE] Total: +{total_new} concepts from {len(urls)} pages")

            return total_new

        except Exception as e:
            if verbose:
                print(f"[FORAGER-GOOGLE] Search failed: {e}")
            return 0

    # ── Smart Foraging with Domain Routing ─────────────────────

    def forage_smart(self, query: str, domain: str = None,
                     verbose: bool = True) -> int:
        """
        Smart foraging with domain-aware source routing and fallback.

        Routes queries to the most appropriate source based on domain:
        - domain="medical" -> PubMed first, then Wikipedia
        - domain="science" -> arXiv first, then Wikipedia
        - domain=None      -> Wikipedia first (default)

        If the primary source returns thin results (<50 concepts),
        automatically tries the next source in the chain.

        Args:
            query: Search query string.
            domain: Optional domain hint ("medical", "science", or None).
            verbose: Print progress messages.

        Returns:
            Total number of new concepts added to the graph.
        """
        # Define source chains per domain
        if domain == "medical":
            chain = [
                ("PubMed", lambda: self.forage_pubmed(query, verbose=verbose)),
                ("Wikipedia", lambda: self.forage_query(query, verbose=verbose)),
                ("Google", lambda: self.forage_google(query, verbose=verbose)),
            ]
        elif domain == "science":
            chain = [
                ("arXiv", lambda: self.forage_arxiv(query, verbose=verbose)),
                ("Wikipedia", lambda: self.forage_query(query, verbose=verbose)),
                ("Google", lambda: self.forage_google(query, verbose=verbose)),
            ]
        else:
            # Default: Wikipedia first, then try specialised sources
            chain = [
                ("Wikipedia", lambda: self.forage_query(query, verbose=verbose)),
            ]
            # Auto-detect scientific queries for fallback
            science_words = {
                "cell", "solar", "quantum", "neural", "protein",
                "molecule", "genome", "algorithm", "theorem",
                "perovskite", "photovoltaic", "semiconductor",
                "enzyme", "catalyst", "polymer", "nanoscale",
                "hypothesis", "experiment", "equation", "reactor",
            }
            medical_words = {
                "disease", "drug", "treatment", "symptom", "patient",
                "clinical", "therapy", "diagnosis", "receptor",
                "pharmaceutical", "dosage", "contraindication",
                "pathology", "surgery", "infection", "antibody",
                "vaccine", "prognosis", "syndrome", "tumor",
            }
            query_words = set(query.lower().split())
            if query_words & medical_words:
                chain.append(
                    ("PubMed", lambda: self.forage_pubmed(query, verbose=verbose)))
            elif query_words & science_words:
                chain.append(
                    ("arXiv", lambda: self.forage_arxiv(query, verbose=verbose)))

            # Google is ALWAYS the last resort for any topic
            chain.append(
                ("Google", lambda: self.forage_google(query, verbose=verbose)))

        total = 0
        for source_name, fetcher in chain:
            if verbose:
                print(f"[FORAGER-SMART] Trying {source_name}...")
            try:
                result = fetcher()
                total += result
                if result > 50:
                    # Got substantial content — no need for fallback
                    if verbose:
                        print(f"[FORAGER-SMART] {source_name} returned "
                              f"{result} concepts. Done.")
                    return total
                elif result > 0 and verbose:
                    print(f"[FORAGER-SMART] {source_name} returned "
                          f"{result} concepts (thin). Trying next source...")
            except Exception as e:
                if verbose:
                    print(f"[FORAGER-SMART] {source_name} failed: {e}. "
                          "Trying next source...")

        if verbose:
            print(f"[FORAGER-SMART] Total: {total} concepts from all sources.")
        return total
