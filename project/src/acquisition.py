"""
src/acquisition.py

arXiv paper acquisition: reading the paper list, HTTP fetching with
robots.txt-compliant rate limiting, and local disk caching.

Allowed endpoints (per arxiv.org/robots.txt and professor clarification):
    /abs/{id}   — abstract / metadata page
    /html/{id}  — LaTeXML HTML rendering (primary source)
    /pdf/{id}   — PDF (fallback when HTML is unavailable)

NEVER fetched here: /src, /e-print (disallowed by robots.txt).
NEVER called at runtime: any LLM or external generation API.

Crawl-delay enforced: 15 seconds after every network request (robots.txt
specifies Crawl-delay: 15 for the wildcard user-agent).  Cache hits are
free — they make zero network calls and incur no sleep penalty.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .audit import AuditTrail

logger = logging.getLogger(__name__)

# ── module-level constants ─────────────────────────────────────────────────────

ARXIV_BASE = "https://arxiv.org"

# Mandated by arxiv robots.txt Crawl-delay: 15 (wildcard user-agent)
CRAWL_DELAY: int = 15

# Descriptive User-Agent as required by polite-crawler convention
USER_AGENT: str = (
    "OTH-NLP-EquationsKG/1.0 "
    "(Academic project, OTH Amberg-Weiden; "
    "quantum physics equations extraction; "
    "compliant with arxiv.org/robots.txt)"
)

# HTTP status code returned for cache hits (not a real HTTP code)
_CACHE_HIT_STATUS: int = -1


# ── paper list ─────────────────────────────────────────────────────────────────

def read_paper_list(path: str) -> List[str]:
    """Read arXiv IDs from the assigned paper list file.

    Strips the ``arXiv:`` prefix, preserves file order, and deduplicates
    to the first occurrence of each ID.  Duplicate count is logged at
    WARNING level.

    Parameters
    ----------
    path : str
        Path to ``paper_list_44.txt`` (or equivalent).

    Returns
    -------
    list of str
        Bare arXiv IDs (e.g. ``'2401.13506'``) in file order, no duplicates.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> ids = read_paper_list("paper_list_44.txt")
    >>> ids[0]
    '2401.13506'
    """
    ids: List[str] = []
    seen: set = set()
    duplicate_count: int = 0

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines

            # normalise: strip optional "arXiv:" prefix (case-insensitive)
            arxiv_id = re.sub(r"(?i)^arxiv:", "", line)

            if arxiv_id in seen:
                duplicate_count += 1
                logger.warning("Duplicate arXiv ID skipped: %s", arxiv_id)
            else:
                seen.add(arxiv_id)
                ids.append(arxiv_id)

    if duplicate_count:
        logger.info(
            "read_paper_list: dropped %d duplicate ID(s), %d unique IDs retained.",
            duplicate_count,
            len(ids),
        )
    else:
        logger.info("read_paper_list: %d unique IDs loaded, no duplicates.", len(ids))

    return ids


# ── Fetcher ────────────────────────────────────────────────────────────────────

class Fetcher:
    """Fetches arXiv papers with local caching and polite rate limiting.

    Strategy per paper:
        1. ``/abs/{id}``  — always fetched (metadata / context text).
        2. ``/html/{id}`` — primary source; gives LaTeXML-rendered HTML with
                           MathML equations and clean paragraph text.
        3. ``/pdf/{id}``  — fallback if HTML returns non-200 or empty content.
        4. If both fail, source is recorded as ``'none'``.

    Cache layout::

        data/cache/
            {safe_id}_abs.html
            {safe_id}_html.html
            {safe_id}_pdf.pdf

    where ``safe_id`` replaces ``/`` with ``_`` (for versioned IDs like
    ``2401.13506v2``).

    Second run makes zero network requests for any previously fetched paper.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Root directory for cached files.  Created if absent.
        Default: ``'data/cache'``.

    Attributes
    ----------
    cache_dir : Path
        Resolved cache directory.
    session : requests.Session
        Shared HTTP session with User-Agent pre-set.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # ── internal helpers ───────────────────────────────────────────────────────

    def _cache_path(self, arxiv_id: str, endpoint: str) -> Path:
        """Return the local path for a cached response.

        Parameters
        ----------
        arxiv_id : str
            Bare arXiv ID (e.g. ``'2401.13506'``).
        endpoint : str
            One of ``'abs'``, ``'html'``, ``'pdf'``.

        Returns
        -------
        Path
            e.g. ``data/cache/2401.13506_html.html``
        """
        # replace "/" to handle versioned IDs (e.g. 2401.13506v2 → 2401.13506v2)
        safe_id = arxiv_id.replace("/", "_")
        ext = "pdf" if endpoint == "pdf" else "html"
        return self.cache_dir / f"{safe_id}_{endpoint}.{ext}"

    def _get(
        self,
        url: str,
        cache_path: Path,
        audit: AuditTrail,
        method: str,
    ) -> Tuple[Optional[bytes], int]:
        """Core fetch helper: cache-first, then network with mandatory sleep.

        Cache hit path:
            Reads file bytes and returns immediately — no network call, no sleep.

        Network path:
            Performs HTTP GET, sleeps CRAWL_DELAY seconds regardless of success
            or failure (polite crawler behaviour), then caches on HTTP 200.
            Non-200 responses are NOT cached so they can be retried on re-runs.

        Parameters
        ----------
        url : str
            Full URL to fetch.
        cache_path : Path
            Where to store (and later load) the response.
        audit : AuditTrail
            Receives one log entry describing this fetch.
        method : str
            Method label for the audit entry (e.g. ``'fetch_html'``).

        Returns
        -------
        content : bytes or None
            Response body, or ``None`` on failure.
        status : int
            HTTP status code, ``_CACHE_HIT_STATUS`` (-1) for cache hits,
            or ``0`` on network exception.
        """
        # ── cache hit: return immediately, no network, no sleep ────────────────
        if cache_path.exists() and cache_path.stat().st_size > 0:
            content = cache_path.read_bytes()
            audit.log(
                method,
                f"cache hit, {len(content)} bytes, file={cache_path.name}",
            )
            logger.debug("Cache hit: %s", cache_path.name)
            return content, _CACHE_HIT_STATUS

        # ── network request ────────────────────────────────────────────────────
        try:
            response = self.session.get(url, timeout=30)
            status = response.status_code
        except requests.RequestException as exc:
            # sleep even on exception — we made a request, stay polite
            time.sleep(CRAWL_DELAY)
            audit.log(method, f"request exception: {exc!r}")
            logger.error("Request failed for %s: %s", url, exc)
            return None, 0

        # mandatory 15-second sleep after every successful network call
        time.sleep(CRAWL_DELAY)

        audit.log(method, f"HTTP {status}, url={url}")

        if status == 200:
            # persist to cache only on success
            cache_path.write_bytes(response.content)
            logger.info("Fetched and cached: %s (HTTP %d)", cache_path.name, status)
            return response.content, status

        # non-200: return without caching so a re-run can retry
        logger.warning("HTTP %d for %s", status, url)
        return None, status

    # ── public fetch methods ───────────────────────────────────────────────────

    def fetch_abs(self, arxiv_id: str, audit: AuditTrail) -> Optional[bytes]:
        """Fetch the ``/abs`` abstract page for a paper.

        The abstract page provides title, author, and abstract text that
        can supplement meaning and symbol extraction.

        Parameters
        ----------
        arxiv_id : str
            Bare arXiv ID.
        audit : AuditTrail
            Audit log for this paper/equation.

        Returns
        -------
        bytes or None
            HTML of the abstract page, or ``None`` on failure.
        """
        url = f"{ARXIV_BASE}/abs/{arxiv_id}"
        cache_path = self._cache_path(arxiv_id, "abs")
        content, _ = self._get(url, cache_path, audit, "fetch_abs")
        return content

    def fetch_html(
        self, arxiv_id: str, audit: AuditTrail
    ) -> Tuple[Optional[bytes], int]:
        """Fetch the ``/html`` LaTeXML rendering of a paper.

        This is the primary source: equations appear as MathML inside
        ``<math>`` / ``ltx_equation`` elements, and surrounding paragraph
        text is clean unicode suitable for NLP pattern matching.

        Parameters
        ----------
        arxiv_id : str
            Bare arXiv ID.
        audit : AuditTrail
            Audit log for this paper/equation.

        Returns
        -------
        content : bytes or None
        status : int
            ``-1`` for cache hit, ``0`` for network error, HTTP code otherwise.
        """
        url = f"{ARXIV_BASE}/html/{arxiv_id}"
        cache_path = self._cache_path(arxiv_id, "html")
        return self._get(url, cache_path, audit, "fetch_html")

    def fetch_pdf(
        self, arxiv_id: str, audit: AuditTrail
    ) -> Tuple[Optional[bytes], int]:
        """Fetch the ``/pdf`` version of a paper.

        Used as a fallback when the HTML rendering is unavailable.
        PDF yields lower-fidelity equation text (rendered Unicode rather
        than MathML-derived LaTeX) but still allows equation numbering
        detection and surrounding-text extraction.

        Parameters
        ----------
        arxiv_id : str
            Bare arXiv ID.
        audit : AuditTrail
            Audit log for this paper/equation.

        Returns
        -------
        content : bytes or None
        status : int
            ``-1`` for cache hit, ``0`` for network error, HTTP code otherwise.
        """
        url = f"{ARXIV_BASE}/pdf/{arxiv_id}"
        cache_path = self._cache_path(arxiv_id, "pdf")
        return self._get(url, cache_path, audit, "fetch_pdf")

    def fetch_paper(self, arxiv_id: str, audit: AuditTrail) -> Dict:
        """Fetch the best available version of a paper.

        Acquisition order:
            1. ``/abs``  — always attempted (metadata).
            2. ``/html`` — primary; used if HTTP 200 (or cached).
            3. ``/pdf``  — fallback if HTML is unavailable (non-200 / empty).
            4. If both fail, ``source`` is ``'none'`` and ``content`` is ``None``.

        Every paper, regardless of outcome, gets a result dict so the
        pipeline can write an empty-equation JSON entry rather than silently
        skipping it (spec requirement).

        Parameters
        ----------
        arxiv_id : str
            Bare arXiv ID.
        audit : AuditTrail
            Audit log; receives entries for every fetch attempt.

        Returns
        -------
        dict with keys:

            ``'arxiv_id'`` : str
            ``'abs_content'`` : bytes or None
            ``'content'`` : bytes or None  — primary body (HTML or PDF bytes)
            ``'source'`` : str             — ``'html'``, ``'pdf'``, or ``'none'``
            ``'html_status'`` : int or None
            ``'pdf_status'`` : int or None
        """
        result: Dict = {
            "arxiv_id": arxiv_id,
            "abs_content": None,
            "content": None,
            "source": "none",
            "html_status": None,
            "pdf_status": None,
        }

        # step 1: always fetch /abs for metadata context
        result["abs_content"] = self.fetch_abs(arxiv_id, audit)

        # step 2: try /html (preferred — clean MathML + paragraph text)
        html_content, html_status = self.fetch_html(arxiv_id, audit)
        result["html_status"] = html_status

        # a cache hit (_CACHE_HIT_STATUS) is as good as HTTP 200
        html_ok = html_content is not None and html_status in (200, _CACHE_HIT_STATUS)

        if html_ok:
            result["content"] = html_content
            result["source"] = "html"
            audit.log(
                "fetch_paper",
                f"{arxiv_id}: resolved via html (status={html_status})",
            )
            logger.info("Paper %s: source=html", arxiv_id)
            return result

        # step 3: HTML unavailable — fall through to /pdf
        audit.log(
            "fetch_paper",
            f"{arxiv_id}: html unavailable (status={html_status}), trying pdf",
        )
        logger.info("Paper %s: html failed (%s), falling back to pdf", arxiv_id, html_status)

        pdf_content, pdf_status = self.fetch_pdf(arxiv_id, audit)
        result["pdf_status"] = pdf_status

        pdf_ok = pdf_content is not None and pdf_status in (200, _CACHE_HIT_STATUS)

        if pdf_ok:
            result["content"] = pdf_content
            result["source"] = "pdf"
            audit.log(
                "fetch_paper",
                f"{arxiv_id}: resolved via pdf (status={pdf_status})",
            )
            logger.info("Paper %s: source=pdf", arxiv_id)
        else:
            # both endpoints failed — paper will have an empty equation dict
            audit.log(
                "fetch_paper",
                (
                    f"{arxiv_id}: source=none; "
                    f"html_status={html_status}, pdf_status={pdf_status}"
                ),
            )
            logger.error(
                "Paper %s: both html and pdf failed (html=%s, pdf=%s)",
                arxiv_id,
                html_status,
                pdf_status,
            )

        return result
