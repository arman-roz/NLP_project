"""Parse arXiv HTML into structured equation records and text chunks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from html.entities import codepoint2name
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag


_EQ_NUM_RE = re.compile(r"^\s*\(\s*([A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)\s*\)\s*$")


@dataclass
class EquationRecord:
    """One extracted equation with all surrounding context."""

    paper_id: str
    eq_num: str
    latex: str
    mathml_symbols: List[str] = field(default_factory=list)
    before: str = ""
    after: str = ""
    section: str = ""
    context: str = ""


@dataclass
class TextChunk:
    """A paragraph-level text chunk for indexing."""

    paper_id: str
    chunk_id: str
    text: str
    section: str = ""
    chunk_type: str = "paragraph"


class ArxivParser:
    """Extract equations, symbols, and text from arXiv HTML.

    Works on LaTeXML-generated HTML from the ``/html`` endpoint.
    Extracts numbered equations with their MathML symbols and
    surrounding prose context.
    """

    def __init__(self, html: str, paper_id: str) -> None:
        self.html = html
        self.paper_id = paper_id
        self.soup = BeautifulSoup(html, "lxml")

    def title(self) -> str:
        """Return paper title if found."""
        node = self.soup.find("h1", class_="ltx_title") or self.soup.find("title")
        return self._clean(node.get_text(" ", strip=True)) if node else ""

    def full_text(self) -> str:
        """Return full article text with equation tables removed."""
        soup = BeautifulSoup(str(self.soup), "lxml")
        for table in soup.find_all("table", class_="ltx_eqn_table"):
            table.decompose()
        for math in soup.find_all("math"):
            math.replace_with(" " + self._math_text(math) + " ")
        return self._clean(soup.get_text(" ", strip=True))

    def section_map(self) -> Dict[str, str]:
        """Map section titles to their text content."""
        sections: Dict[str, str] = {}
        for sec in self.soup.find_all(["div", "section"]):
            cls = " ".join(sec.get("class", []))
            if "ltx_section" in cls:
                title_node = sec.find(["h2", "h3", "h4"])
                if title_node:
                    title = self._clean(title_node.get_text(" ", strip=True))
                    paragraphs = [p.get_text(" ", strip=True) for p in sec.find_all("p")]
                    sections[title] = " ".join(paragraphs)
        return sections

    def equations(self, max_eq: int = 7) -> List[EquationRecord]:
        """Extract numbered equations in document order.

        Parameters
        ----------
        max_eq : int
            Maximum number of equations to extract per paper.

        Returns
        -------
        list[EquationRecord]
            Extracted equations with symbols and context.
        """
        tables = self.soup.find_all("table", class_="ltx_eqn_table")
        records: List[EquationRecord] = []
        seen: set = set()

        for table in tables:
            if len(records) >= max_eq:
                break
            hits = self._row_equations(table) or [self._table_equation(table)]
            for hit in hits:
                if hit is None or hit.eq_num in seen:
                    continue
                hit.before, hit.after = self._local_context(table)
                hit.section = self._find_section(table)
                hit.context = f"{hit.before} {hit.after}"
                records.append(hit)
                seen.add(hit.eq_num)
                if len(records) >= max_eq:
                    break

        return records

    def text_chunks(self) -> List[TextChunk]:
        """Extract paragraph-level text chunks for indexing.

        Returns
        -------
        list[TextChunk]
            Paragraph chunks with section info.
        """
        chunks: List[TextChunk] = []
        idx = 0
        for section, text in self.section_map().items():
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            for para in paragraphs:
                chunks.append(TextChunk(
                    paper_id=self.paper_id,
                    chunk_id=f"p{idx}",
                    text=para,
                    section=section,
                    chunk_type="paragraph",
                ))
                idx += 1
        return chunks

    def _table_equation(self, table: Tag) -> Optional[EquationRecord]:
        tag = table.find("span", class_=["ltx_tag_equation", "ltx_tag_equationgroup"])
        num = self._eq_number(tag)
        math_nodes = table.find_all("math")
        latex = self._latex(math_nodes)
        if num is None or not latex:
            return None
        return EquationRecord(
            paper_id=self.paper_id,
            eq_num=num,
            latex=latex,
            mathml_symbols=self._mi_symbols(math_nodes),
        )

    def _row_equations(self, table: Tag) -> List[EquationRecord]:
        out: List[EquationRecord] = []
        rows = table.find_all("tr")
        for row in rows:
            num = self._eq_number(row.find("span", class_="ltx_tag_equation"))
            if num is None:
                continue
            math_nodes = row.find_all("math")
            if not math_nodes:
                prev = self._prev_math(rows, row)
                math_nodes = [prev] if prev is not None else []
            latex = self._latex(math_nodes)
            if latex:
                out.append(EquationRecord(
                    paper_id=self.paper_id,
                    eq_num=num,
                    latex=latex,
                    mathml_symbols=self._mi_symbols(math_nodes),
                ))
        return out

    def _local_context(self, table: Tag) -> Tuple[str, str]:
        para = table.find_parent(
            lambda item: item.name == "div" and "ltx_para" in item.get("class", [])
        ) or table.find_parent("p")
        if para is None:
            before = self._nearby_text(table, True)
            after = self._nearby_text(table, False)
            return before, after
        before = self._text_before(table, para)
        after = self._text_after(table, para)
        before = self._dedupe(self._clean(self._nearby_text(table, True) + " " + before))
        after = self._dedupe(self._clean(after + " " + self._nearby_text(table, False)))
        return before, after

    def _find_section(self, table: Tag) -> str:
        for parent in table.parents:
            cls = " ".join(parent.get("class", []))
            if "ltx_section" in cls:
                title_node = parent.find(["h2", "h3", "h4"])
                if title_node:
                    return self._clean(title_node.get_text(" ", strip=True))
        return ""

    def _text_before(self, table: Tag, para: Tag) -> str:
        parts: List[str] = []
        for child in para.children:
            if child is table:
                break
            parts.append(self._text_node(child))
        return self._clean(" ".join(parts))

    def _text_after(self, table: Tag, para: Tag) -> str:
        parts: List[str] = []
        seen = False
        for child in para.children:
            if child is table:
                seen = True
                continue
            if seen:
                parts.append(self._text_node(child))
        return self._clean(" ".join(parts))

    def _nearby_text(self, table: Tag, before: bool) -> str:
        iterator = table.find_all_previous if before else table.find_all_next
        for candidate in iterator(lambda item: item.name in {"p", "div"}):
            if candidate.find(lambda item: item is table):
                continue
            if candidate.find("table", class_="ltx_eqn_table"):
                continue
            text = self._text_node(candidate)
            if len(text.split()) >= 6:
                return text
        return ""

    def _text_node(self, node) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        soup = BeautifulSoup(str(node), "lxml")
        for table in soup.find_all("table"):
            table.decompose()
        for math in soup.find_all("math"):
            math.replace_with(" " + self._math_text(math) + " ")
        for tag in soup.find_all("span"):
            if tag is None or not isinstance(tag, Tag):
                continue
            try:
                classes = set(tag.get("class", []))
            except AttributeError:
                continue
            if any(name.startswith("ltx_tag") for name in classes):
                tag.decompose()
        return self._clean(soup.get_text(" ", strip=True))

    def _mi_symbols(self, math_nodes: List[Tag]) -> List[str]:
        out: List[str] = []
        seen: set = set()
        for math in math_nodes:
            if math is None:
                continue
            for node in math.find_all(["mi", "msub", "msubsup"]):
                sym = self._symbol_id(node)
                if sym and sym not in seen:
                    seen.add(sym)
                    out.append(sym)
        return out

    def _symbol_id(self, node: Tag) -> str:
        if node.name in {"msub", "msubsup"}:
            children = self._elem_children(node)
            if len(children) < 2:
                return ""
            base = self._node_id(children[0])
            return base if base else ""
        if node.name != "mi":
            return ""
        if self._has_id_parent(node):
            return ""
        return self._atomic_id(node)

    def _node_id(self, node: Tag) -> str:
        if node.name == "mi":
            return self._atomic_id(node)
        if node.name in {"mrow", "mstyle", "mtext"}:
            parts = [self._node_id(ch) for ch in self._elem_children(node)]
            return " ".join(p for p in parts if p).strip()
        return ""

    def _atomic_id(self, mi: Tag) -> str:
        raw = mi.get_text("", strip=True)
        if not raw:
            return ""
        if raw.lower() in {"log", "ln", "sin", "cos", "tan", "exp", "lim", "max", "min"}:
            return ""
        variant = str(mi.get("mathvariant", "")).lower()
        if variant in {"normal", "upright"}:
            return ""
        classes = " ".join(mi.get("class", [])).lower()
        if "text" in classes or "mathrm" in classes:
            return ""
        return self._canonical(raw)

    def _has_id_parent(self, mi: Tag) -> bool:
        parent = mi.parent
        return isinstance(parent, Tag) and parent.name in {"msub", "msubsup"}

    def _latex(self, math_nodes: List[Tag]) -> str:
        parts: List[str] = []
        for math in math_nodes:
            if math is not None:
                text = self._math_text(math)
                if text:
                    text = re.sub(r"^\\displaystyle\s*", "", text.strip())
                    parts.append(" ".join(text.split()))
        return " \\\\\n".join(parts)

    def _math_text(self, math: Tag) -> str:
        ann = math.find("annotation", attrs={"encoding": "application/x-tex"})
        if ann and ann.get_text(strip=True):
            return ann.get_text(" ", strip=True)
        alt = math.get("alttext", "").strip()
        if alt:
            return alt
        return math.get_text(" ", strip=True)

    def _eq_number(self, tag: Optional[Tag]) -> Optional[str]:
        if tag is None:
            return None
        match = _EQ_NUM_RE.match(tag.get_text(" ", strip=True))
        return match.group(1) if match else None

    @staticmethod
    def _prev_math(rows: List[Tag], row: Tag) -> Optional[Tag]:
        index = rows.index(row)
        for prev in reversed(rows[:index]):
            math = prev.find("math")
            if math is not None:
                return math
        return None

    @staticmethod
    def _elem_children(tag: Tag) -> List[Tag]:
        return [ch for ch in tag.children if isinstance(ch, Tag)]

    @staticmethod
    def _canonical(raw: str) -> str:
        text = unicodedata.normalize("NFKC", raw.strip())
        if not text or text.isdigit():
            return ""
        if len(text) == 1:
            return ArxivParser._unicode_name(text) or text
        return text

    @staticmethod
    def _unicode_name(char: str) -> str:
        entity = codepoint2name.get(ord(char), "")
        if entity and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", entity):
            return entity
        name = unicodedata.name(char, "")
        m = re.fullmatch(r"GREEK (SMALL|CAPITAL) LETTER ([A-Z]+)(?: .*)?", name)
        if m:
            normalized = m.group(2).lower()
            return normalized.capitalize() if m.group(1) == "CAPITAL" else normalized
        m2 = re.fullmatch(r"GREEK ([A-Z]+) SYMBOL", name)
        return m2.group(1).lower() if m2 else ""

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    @staticmethod
    def _dedupe(text: str) -> str:
        pieces = re.split(r"(?<=[.!?:])\s+", text)
        seen: set = set()
        out: List[str] = []
        for piece in pieces:
            clean = piece.strip()
            if clean and clean not in seen:
                seen.add(clean)
                out.append(clean)
        return " ".join(out)
