"""Fetch arXiv HTML pages with cache-first strategy and robots.txt compliance."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests


ARXIV_BASE = "https://arxiv.org"
USER_AGENT = "arxiv-rag/0.1 (student project; respects arxiv robots.txt)"
CRAWL_DELAY = 15.0


@dataclass
class RawPage:
    """Raw fetched page with metadata."""

    arxiv_id: str
    status_code: int
    html: str
    url: str
    from_cache: bool


class ArxivFetcher:
    """Cache-first HTTP client for arXiv HTML endpoints.

    Respects robots.txt by using a 15-second crawl delay and only
    hitting the allowed ``/html`` endpoint.

    Parameters
    ----------
    cache_dir : Path
        Directory for caching downloaded HTML files.
    sleep_seconds : float
        Seconds to wait between network requests.
    """

    def __init__(self, cache_dir: Path, sleep_seconds: float = CRAWL_DELAY) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, arxiv_id: str) -> RawPage:
        """Fetch an arXiv HTML page or read from local cache.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier (e.g. ``2401.13506``).

        Returns
        -------
        RawPage
            The fetched page with status, HTML content, and cache info.
        """
        cache_path = self.cache_dir / f"{arxiv_id.replace('/', '_')}.html"
        url = f"{ARXIV_BASE}/html/{arxiv_id}"

        if cache_path.exists() and cache_path.stat().st_size > 0:
            html = cache_path.read_text(encoding="utf-8", errors="replace")
            return RawPage(arxiv_id, 200, html, url, True)

        response = self.session.get(url, timeout=40)
        time.sleep(self.sleep_seconds)

        if response.status_code == 200 and response.text.strip():
            cache_path.write_text(response.text, encoding="utf-8")

        return RawPage(arxiv_id, response.status_code, response.text, url, False)
