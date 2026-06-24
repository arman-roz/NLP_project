"""arXiv HTML fetching and equation extraction."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from html.entities import codepoint2name
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from .common import AuditTrail, short

ARXIV_BASE = "https://arxiv.org"
USER_AGENT = "OTH-NLP-EquationKG/0.2 (student project; respects arxiv robots.txt)"
CRAWL_DELAY_SECONDS = 15.0

_EQ_NUMBER_RE = re.compile(r"^\s*\(\s*([A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)\s*\)\s*$")

# pylatexenc renders inline LaTeX in the prose context to readable Unicode
# (e.g. ``I_{\rm OFF}`` -> ``I_OFF``, ``\eta_{\text{D}}`` -> ``η_D``). This is
# used only for the surrounding text the NLP methods read; the ``equation`` field
# keeps the verbatim LaTeX. Done with a library rather than ad-hoc regex so the
# full LaTeX macro set (``\rm``, ``\cal``, ``\hat``, Greek letters, ...) is
# handled correctly instead of leaking command names like "rm" into the text.
try:
    from pylatexenc.latex2text import LatexNodes2Text

    _LATEX2TEXT = LatexNodes2Text(math_mode="text", strict_latex_spaces="based-on-source")
except Exception:  # pragma: no cover - pylatexenc is a hard dependency in practice
    _LATEX2TEXT = None


@lru_cache(maxsize=50000)
def _latex_inline_to_text(latex: str) -> str:
    """Convert a small inline LaTeX fragment to readable Unicode text."""

    if _LATEX2TEXT is None or not latex:
        return latex
    try:
        return _LATEX2TEXT.latex_to_text(latex).strip()
    except Exception:
        return latex


@dataclass
class HtmlPage:
    """Fetched arXiv HTML page."""

    arxiv_id: str
    status_code: int
    html: str
    url: str
    from_cache: bool


@dataclass
class EquationBlock:
    """One numbered equation and the surrounding text window."""

    number: str
    latex: str
    before: str
    after: str
    mathml_symbols: List[str] = field(default_factory=list)
    audit: AuditTrail = field(default_factory=AuditTrail)


class ArxivHtmlClient:
    """Cache-first client for the robots.txt-allowed ``/html`` endpoint."""

    def __init__(self, cache_dir: Path, sleep_seconds: float = CRAWL_DELAY_SECONDS) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, arxiv_id: str, audit: AuditTrail) -> HtmlPage:
        """Fetch an arXiv HTML page or read it from cache."""

        url = f"{ARXIV_BASE}/html/{arxiv_id}"
        cache_path = self.cache_dir / f"{arxiv_id.replace('/', '_')}.html"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            html = cache_path.read_text(encoding="utf-8", errors="replace")
            audit.add("fetch_html", f"cache hit {cache_path.name}, chars={len(html)}")
            return HtmlPage(arxiv_id, 200, html, url, True)

        response = self.session.get(url, timeout=40)
        time.sleep(self.sleep_seconds)
        audit.add("fetch_html", f"HTTP {response.status_code} {url}")
        if response.status_code == 200 and response.text.strip():
            cache_path.write_text(response.text, encoding="utf-8")
            audit.add("cache_html", f"saved {cache_path.name}, chars={len(response.text)}")
        return HtmlPage(arxiv_id, response.status_code, response.text, url, False)


class ArxivHtmlPaper:
    """BeautifulSoup wrapper around arXiv's LaTeXML HTML."""

    def __init__(self, html: str) -> None:
        self.soup = BeautifulSoup(html, "lxml")

    def title(self) -> str:
        """Return the paper title if arXiv HTML exposes it."""

        node = self.soup.find("h1", class_="ltx_title") or self.soup.find("title")
        return _clean_text(node.get_text(" ", strip=True)) if node else ""

    def paper_text(self) -> str:
        """Return article text with equation tables removed and inline math kept."""

        soup = BeautifulSoup(str(self.soup), "lxml")
        for table in soup.find_all("table", class_="ltx_eqn_table"):
            table.decompose()
        for math in soup.find_all("math"):
            math.replace_with(" " + _latex_inline_to_text(_math_text(math)) + " ")
        return _clean_text(soup.get_text(" ", strip=True))

    def equations(self, max_equations: int, audit: AuditTrail) -> List[EquationBlock]:
        """Extract numbered equations in document order."""

        tables = self.soup.find_all("table", class_="ltx_eqn_table")
        audit.add("find_equation_tables", f"{len(tables)} equation table candidates")
        out: List[EquationBlock] = []
        seen: set[str] = set()

        for table in tables:
            if len(out) >= max_equations:
                break
            hits = self._row_equations(table) or [self._table_equation(table)]
            for hit in hits:
                if hit is None or hit.number in seen:
                    continue
                hit.before, hit.after = self._local_context(table)
                hit.audit.add("extract_equation", f"({hit.number}) {short(hit.latex, 120)}")
                out.append(hit)
                seen.add(hit.number)
                if len(out) >= max_equations:
                    break

        audit.add("extract_equations", f"kept {len(out)} numbered equations")
        return out

    def _table_equation(self, table: Tag) -> Optional[EquationBlock]:
        tag = table.find("span", class_=["ltx_tag_equation", "ltx_tag_equationgroup"])
        number = _equation_number(tag)
        math_nodes = table.find_all("math")
        latex = _latex_from_math(math_nodes)
        if number is None or not latex:
            return None
        return EquationBlock(number, latex, "", "", _mi_symbols(math_nodes))

    def _row_equations(self, table: Tag) -> List[EquationBlock]:
        out: List[EquationBlock] = []
        rows = table.find_all("tr")
        for row in rows:
            number = _equation_number(row.find("span", class_="ltx_tag_equation"))
            if number is None:
                continue
            math_nodes = row.find_all("math")
            if not math_nodes:
                previous = _previous_math(rows, row)
                math_nodes = [previous] if previous is not None else []
            latex = _latex_from_math(math_nodes)
            if latex:
                out.append(EquationBlock(number, latex, "", "", _mi_symbols(math_nodes)))
        return out

    def _local_context(self, table: Tag) -> tuple[str, str]:
        para = _main_paragraph(table)
        before = _text_inside(para, table, before=True) if para else ""
        after = _text_inside(para, table, before=False) if para else ""
        before = _dedupe_sentences(_clean_text(" ".join([_nearby_paragraph(table, True), before])))
        after = _dedupe_sentences(_clean_text(" ".join([after, _nearby_paragraph(table, False)])))
        return before, after


def _equation_number(tag: Optional[Tag]) -> Optional[str]:
    if tag is None:
        return None
    match = _EQ_NUMBER_RE.match(tag.get_text(" ", strip=True))
    return match.group(1) if match else None


def _latex_from_math(math_nodes: List[Tag]) -> str:
    parts: List[str] = []
    for math in math_nodes:
        if math is not None:
            text = _math_text(math)
            if text:
                parts.append(_clean_latex(text))
    return " \\\\\n".join(parts)


def _math_text(math: Tag) -> str:
    annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
    if annotation and annotation.get_text(strip=True):
        return annotation.get_text(" ", strip=True)
    alt = math.get("alttext", "").strip()
    if alt:
        return alt
    return math.get_text(" ", strip=True)


def _clean_latex(text: str) -> str:
    text = text.replace("%\n", "")
    text = re.sub(r"^\\displaystyle\s*", "", text.strip())
    return " ".join(text.split())


def _previous_math(rows: List[Tag], row: Tag) -> Optional[Tag]:
    index = rows.index(row)
    for previous in reversed(rows[:index]):
        math = previous.find("math")
        if math is not None:
            return math
    return None


def _main_paragraph(node: Tag) -> Optional[Tag]:
    return node.find_parent(lambda item: item.name == "div" and "ltx_para" in item.get("class", [])) or node.find_parent("p")


def _text_inside(container: Tag, marker: Tag, before: bool) -> str:
    parts: List[str] = []
    seen_marker = False
    for child in container.children:
        if child is marker or _contains(child, marker):
            seen_marker = True
            continue
        if (before and not seen_marker) or ((not before) and seen_marker):
            parts.append(_text_keep_inline_math(child))
    return _clean_text(" ".join(parts))


def _contains(parent, child: Tag) -> bool:
    return parent is child or (isinstance(parent, Tag) and parent.find(lambda item: item is child) is not None)


def _nearby_paragraph(node: Tag, previous: bool) -> str:
    iterator = node.find_all_previous if previous else node.find_all_next
    for candidate in iterator(lambda item: item.name in {"p", "div"}):
        if candidate.find(lambda item: item is node):
            continue
        if candidate.find("table", class_="ltx_eqn_table"):
            continue
        if candidate.name == "p" or "ltx_para" in candidate.get("class", []):
            text = _text_keep_inline_math(candidate)
            if len(text.split()) >= 6:
                return text
    return ""


def _text_keep_inline_math(node) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    soup = BeautifulSoup(str(node), "lxml")
    for table in soup.find_all("table"):
        table.decompose()
    for math in soup.find_all("math"):
        math.replace_with(" " + _latex_inline_to_text(_math_text(math)) + " ")
    for tag in soup.find_all("span"):
        classes = set(tag.get("class", []))
        if any(name.startswith("ltx_tag") for name in classes):
            tag.decompose()
    return _clean_text(soup.get_text(" ", strip=True))


def _mi_symbols(math_nodes: List[Tag]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for math in math_nodes:
        if math is None:
            continue
        for node in math.find_all(["mi", "msub", "msubsup"]):
            symbol = _structured_identifier(node)
            if not symbol:
                continue
            if symbol and symbol not in seen:
                seen.add(symbol)
                out.append(symbol)
    return out


def _structured_identifier(node: Tag) -> str:
    if node.name in {"msub", "msubsup"}:
        return _subscripted_identifier(node)
    if node.name != "mi":
        return ""
    if _has_identifier_parent(node):
        return ""
    return _atomic_identifier(node, allow_text=False)


def _subscripted_identifier(node: Tag) -> str:
    """Identifier for an ``msub``/``msubsup`` node, keeping a *semantic* subscript.

    A subscript that names a quantity (``\\eta_{\\rm D}`` -> ``eta_D``,
    ``P_{\\rm th}`` -> ``P_th``, ``I_{\\rm OFF}`` -> ``I_OFF``) is kept so that
    physically distinct symbols sharing a base letter (``eta_D`` vs ``eta_path``)
    stay separate in the symbols dictionary. A plain index subscript (a digit or
    a single lower-case letter, e.g. ``x_1`` or ``\\rho_t``) carries no naming
    information and is dropped, leaving just the base identifier.
    """

    children = _element_children(node)
    if len(children) < 2:
        return ""
    base = _node_identifier(children[0], allow_text=False)
    if not base:
        return ""
    sub = _semantic_subscript(children[1])
    return f"{base}_{sub}" if sub else base


def _semantic_subscript(node: Tag) -> str:
    """Return a subscript's letters if they name a quantity, else ``""``.

    Kept when the subscript is alphabetic and either multi-letter (``th``,
    ``eff``, ``path``, ``OFF``) or a single capital used as a label (``D``,
    ``A``); dropped for digits and single lower-case indices (``1``, ``t``).
    """

    text = unicodedata.normalize("NFKC", _clean_text(node.get_text("", strip=True)))
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) >= 2 or (len(letters) == 1 and letters.isupper()):
        return letters
    return ""


def _node_identifier(node: Tag, allow_text: bool) -> str:
    if node.name == "mi":
        return _atomic_identifier(node, allow_text=allow_text)
    if node.name in {"mrow", "mstyle", "mtext"}:
        parts = [
            _node_identifier(child, allow_text=allow_text)
            for child in _element_children(node)
        ]
        return " ".join(part for part in parts if part).strip()
    return ""


def _atomic_identifier(mi: Tag, allow_text: bool) -> str:
    raw = mi.get_text("", strip=True)
    if not raw:
        return ""
    if _is_function_identifier(raw):
        return ""
    if _is_text_identifier(mi, raw) and not allow_text:
        return ""
    return _canonical_identifier(raw)


def _has_identifier_parent(mi: Tag) -> bool:
    parent = mi.parent
    return isinstance(parent, Tag) and parent.name in {"msub", "msubsup", "mmultiscripts"}


def _is_text_identifier(mi: Tag, raw: str) -> bool:
    variant = str(mi.get("mathvariant", "")).lower()
    if variant in {"normal", "upright"}:
        return True
    classes = " ".join(mi.get("class", [])).lower()
    if "text" in classes or "mathrm" in classes:
        return True
    return len(raw) > 1 and raw.isascii() and raw.isalpha()


def _is_function_identifier(raw: str) -> bool:
    return raw.casefold() in {"log", "ln", "sin", "cos", "tan", "exp", "lim", "max", "min"}


def _canonical_identifier(raw: str) -> str:
    text = unicodedata.normalize("NFKC", _clean_text(raw))
    if not text or text.isdigit():
        return ""
    if len(text) == 1:
        return _unicode_letter_name(text) or text
    return text


def _unicode_letter_name(char: str) -> str:
    entity_name = codepoint2name.get(ord(char), "")
    if entity_name and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", entity_name):
        return entity_name
    name = unicodedata.name(char, "")
    match = re.fullmatch(r"GREEK (SMALL|CAPITAL) LETTER ([A-Z]+)(?: .*)?", name)
    if match:
        normalized = match.group(2).lower()
        return normalized.capitalize() if match.group(1) == "CAPITAL" else normalized
    match = re.fullmatch(r"GREEK ([A-Z]+) SYMBOL", name)
    return match.group(1).lower() if match else ""


def _element_children(tag: Tag) -> List[Tag]:
    return [child for child in tag.children if isinstance(child, Tag)]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _dedupe_sentences(text: str) -> str:
    pieces = re.split(r"(?<=[.!?:])\s+", text)
    seen: set[str] = set()
    out: List[str] = []
    for piece in pieces:
        clean = piece.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return " ".join(out)
