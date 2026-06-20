"""
src/symbols.py

Symbol definition extraction for equations in the same-paper BM25 context.

Two-phase pipeline (Stage 3 — per equation):
  Phase 1 — Symbol identification
    Primary: parse MathML <mi> elements from the equation's <math> block in
    the paper HTML.  <mi> is the MathML "identifier" element; it excludes
    operators (<mo>), numbers (<mn>), and structural elements.  This gives a
    precise, noise-free symbol list without regex heuristics.
    Fallback: LaTeX regex tokeniser (used when HTML is unavailable, e.g. PDF
    papers or unit tests).

  Phase 2 — Definition retrieval and confidence gating
    For each symbol, run a BM25 query via symbol_chunk_view (structure-aware)
    or the general retriever, then apply five definitional regex patterns.
    Only definitions that pass the confidence gate (high or medium) are emitted.
    Low-confidence matches (appositive-only) and rejected definitions are logged
    as symbol_reject audit entries.

Confidence mapping:
    HIGH   — where_is  ("where X is/denotes/represents …")
               let_be   ("let X be …")
               sym_is   ("X is/denotes/represents …")
    MEDIUM — the_noun_sym  ("the total Hamiltonian H")
    LOW    — sym_the       ("H, the Hamiltonian …")  → rejected
    LOW    — dep_appos     (spaCy appositive only)    → rejected

Rejection conditions (applied after finding any match):
    - Single-char definition
    - Definition equals the symbol key itself
    - Definition's last word is a dangling function word (the/a/of/in/…)
    - All words are stop words
    - Contains zero-width / artifact characters
    - Definition word count outside [1, 15]

No text generation.  Every emitted definition is a verbatim phrase extracted
from the paper's own text.

Audit trail per accepted symbol:
    symbol -> "sym='H' def='system Hamiltonian' chunk=S2.p3 char=88
               confidence=high pattern=where_is"
Audit trail per rejected symbol:
    symbol_reject -> "sym='k' def='k' reason='single_char_def'"
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── spaCy optional import ─────────────────────────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.info("spaCy not available — symbol extraction will use regex-only mode")

# ── BeautifulSoup optional import (for MathML parsing) ───────────────────────
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.info("beautifulsoup4 not available — MathML <mi> extraction disabled")

# ── LaTeX structural / operator commands to skip ──────────────────────────────
_SKIP_COMMANDS: Set[str] = {
    # structural
    "begin", "end", "frac", "dfrac", "tfrac", "cfrac",
    "sqrt", "root", "over", "atop",
    "left", "right", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "big", "Big",
    # decorators
    "hat", "bar", "tilde", "vec", "dot", "ddot", "dddot", "breve", "acute",
    "grave", "check", "widehat", "widetilde", "overline", "underline",
    "overbrace", "underbrace", "overleftarrow", "overrightarrow",
    "overleftrightarrow",
    # text/font commands
    "text", "rm", "mathrm", "mathbf", "mathit", "mathcal", "mathbb",
    "mathfrak", "mathsf", "mathtt", "boldsymbol", "bm", "emph", "bf",
    "it", "tt", "sf",
    # spacing/layout
    "quad", "qquad", "hspace", "vspace", "hfill", "vfill",
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
    "limits", "nolimits", "mspace", "mkern",
    # operators and relations
    "cdot", "cdots", "ldots", "vdots", "ddots", "times", "div", "pm", "mp",
    "otimes", "oplus", "ominus", "oslash", "odot", "wedge", "vee",
    "leq", "geq", "neq", "approx", "equiv", "sim", "simeq", "propto",
    "ll", "gg", "subset", "supset", "subseteq", "supseteq",
    "in", "notin", "ni", "cup", "cap", "setminus",
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
    "leftrightarrow", "Leftrightarrow", "to", "gets",
    "langle", "rangle", "vert", "Vert",
    "dagger", "ddagger", "circ", "bullet", "star",
    "partial", "nabla", "infty", "emptyset", "forall", "exists",
    "int", "oint", "iint", "iiint", "sum", "prod", "coprod",
    # standard math functions
    "exp", "log", "ln", "lg", "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "arccot", "sinh", "cosh", "tanh",
    "max", "min", "sup", "inf", "lim", "limsup", "liminf",
    "det", "tr", "Tr", "ker", "dim", "deg", "gcd", "lcm", "mod",
    "arg", "Re", "Im",
    # labels/references
    "label", "ref", "eqref", "nonumber", "notag", "tag",
    # misc
    "not", "ne", "le", "ge",
    "leavevmode", "nobreak", "strut", "hbox",
}

# Summation indices — included in symbol extraction but omitted from output if
# no definition is found (see _extract_symbols_from_latex and extract()).
_SKIP_SINGLE: Set[str] = {"d", "n", "i", "j", "k", "l", "m"}

# ── Greek letter maps ─────────────────────────────────────────────────────────
# LaTeX command name → list of text/Unicode surface forms
_GREEK_TEXT: Dict[str, List[str]] = {
    "alpha": ["α", "alpha"],       "beta":  ["β", "beta"],
    "gamma": ["γ", "gamma"],       "delta": ["δ", "delta"],
    "epsilon": ["ε", "epsilon"],   "zeta":  ["ζ", "zeta"],
    "eta":   ["η", "eta"],         "theta": ["θ", "theta"],
    "iota":  ["ι", "iota"],        "kappa": ["κ", "kappa"],
    "lambda":["λ", "lambda"],      "mu":    ["μ", "mu"],
    "nu":    ["ν", "nu"],          "xi":    ["ξ", "xi"],
    "pi":    ["π", "pi"],          "rho":   ["ρ", "rho"],
    "sigma": ["σ", "sigma"],       "tau":   ["τ", "tau"],
    "upsilon":["υ","upsilon"],     "phi":   ["φ", "ϕ", "phi"],
    "chi":   ["χ", "chi"],         "psi":   ["ψ", "psi"],
    "omega": ["ω", "omega"],
    "Gamma": ["Γ", "Gamma"],       "Delta": ["Δ", "Delta"],
    "Theta": ["Θ", "Theta"],       "Lambda":["Λ", "Lambda"],
    "Xi":    ["Ξ", "Xi"],          "Pi":    ["Π", "Pi"],
    "Sigma": ["Σ", "Sigma"],       "Upsilon":["Υ","Upsilon"],
    "Phi":   ["Φ", "Phi"],         "Psi":   ["Ψ", "Psi"],
    "Omega": ["Ω", "Omega"],
    "hbar":  ["ℏ", "ħ", "hbar"],
}

# Unicode character → Greek symbol name (built from _GREEK_TEXT)
_UNICODE_TO_SYMBOL: Dict[str, str] = {}
for _name, _forms in _GREEK_TEXT.items():
    for _f in _forms:
        if len(_f) == 1 and ord(_f) > 127:
            _UNICODE_TO_SYMBOL[_f] = _name

# Additional single Unicode chars
_UNICODE_TO_SYMBOL.update({
    'ℏ': 'hbar', 'ħ': 'hbar',
})

# ── Retrieval constant ────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5

# ── Inline math stripper (for spaCy input) ────────────────────────────────────
_INLINE_MATH_RE = re.compile(
    r'\$[^$]+\$'
    r'|\\\([^)]+\\\)'
    r'|\\[a-zA-Z]+\{[^}]*\}'
)

# ── Definitional regex patterns ────────────────────────────────────────────────
# (name, template, confidence)
# {SYM} is replaced at match time with the symbol's regex fragment.
_DEF_PATTERNS: List[Tuple[str, str, str]] = [
    (
        "where_is",
        r'\bwhere\s+{SYM}\s+(?:is|are|denotes?|represents?|gives?|'
        r'stands?\s+for|corresponds?\s+to)\s+(?:the\s+)?(.+?)(?:[,;]|\s+and\s+|\.$|$)',
        "high",
    ),
    (
        "let_be",
        r'\blet\s+{SYM}\s+be\s+(?:the\s+)?(.+?)(?:[,;.]|$)',
        "high",
    ),
    (
        "sym_is",
        r'\b{SYM}\s+(?:is|are|denotes?|represents?|corresponds?\s+to)\s+'
        r'(?:the\s+)?(.+?)(?:[,;.]|$)',
        "high",
    ),
    (
        "the_noun_sym",
        r'\bthe\s+((?:\w+\s+){0,4}\w+)\s+{SYM}(?:\b|$)',
        "medium",
    ),
    (
        "sym_the",
        r'\b{SYM}\s*,\s+the\s+([\w\s\-]+?)(?:[,;.]|$)',
        "low",
    ),
]

# ── Rejection helpers ─────────────────────────────────────────────────────────
_DANGLING_WORDS: frozenset = frozenset({
    'the', 'a', 'an', 'of', 'in', 'for', 'to', 'is', 'are', 'and', 'or',
    'with', 'by', 'from', 'at', 'on', 'its', 'their',
    # Additional dangling prepositions/conjunctions seen in corpus noise
    'where', 'into', 'that', 'as', 'which', 'each', 'given', 'such',
    'than', 'but', 'only', 'between', 'within',
})
_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "it", "we", "as", "if", "not", "no", "also",
})
_ZERO_WIDTH: str = '​‌‍﻿'


def _build_sym_pattern(sym_key: str) -> str:
    """Build regex fragment matching sym_key in plain prose text."""
    if sym_key in _GREEK_TEXT:
        forms = [re.escape(f) for f in _GREEK_TEXT[sym_key]]
    else:
        forms = [re.escape(sym_key)]
    dollar_forms = [
        r'\$\\?' + re.escape(sym_key) + r'\$',
        r'\$' + re.escape(sym_key) + r'\$',
    ]
    return r'(?:' + '|'.join(forms + dollar_forms) + r')'


def _reject_reason(defn: str, sym_key: str, all_keys: Set[str]) -> Optional[str]:
    """Return a rejection reason string, or None if the definition is acceptable."""
    stripped = defn.strip()
    words    = stripped.split()

    # empty or single non-alphabetic char
    if not stripped or len(stripped) <= 1:
        return "single_char_def"

    # definition equals the symbol key
    if stripped.lower() == sym_key.lower():
        return "def_equals_symbol"

    # definition equals another symbol in the equation
    if stripped in all_keys or stripped.lower() in {k.lower() for k in all_keys} - {sym_key.lower()}:
        return "def_equals_other_symbol"

    # starts with a function word (the, a, an, of, …)
    if words and words[0].lower() in _DANGLING_WORDS:
        return "starts_with_function_word"

    # ends in a dangling function word
    if words and words[-1].lower() in _DANGLING_WORDS:
        return "dangling_function_word"

    # all stop words
    if all(w.lower() in _STOP_WORDS for w in words):
        return "stop_words_only"

    # fewer than 2 meaningful (non-stop, length > 1) content words
    content_words = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
    if len(content_words) < 2:
        return "too_few_content_words"

    # SYM placeholder leaked into definition
    if 'SYM' in stripped:
        return "contains_SYM_placeholder"

    # raw LaTeX fragment leaked into definition
    if any(c in stripped for c in ('\\', '{', '}')):
        return "contains_latex_fragment"

    # zero-width / artifact characters
    if any(ch in stripped for ch in _ZERO_WIDTH):
        return "artifact_chars"

    return None


class SymbolExtractor:
    """Extract symbol definitions from equation LaTeX/MathML and surrounding text.

    Parameters
    ----------
    mode : str
        ``"regex"`` — regex patterns only.
        ``"regex+dep"`` — regex first; spaCy dependency parsing as fallback for
        symbols that passed the confidence gate via dep-parse.
        Defaults to ``"regex+dep"``.

    Examples
    --------
    >>> se = SymbolExtractor(mode="regex+dep")
    >>> syms = se.extract("3", r"H \\psi = E \\psi", ctx_b, ctx_a, audit,
    ...                   retriever=bm25, html_content=html_bytes,
    ...                   parsed_paper=paper)
    >>> syms
    {'H': 'system Hamiltonian', 'psi': 'wave function', 'E': 'energy eigenvalue'}
    """

    def __init__(self, mode: str = "regex+dep") -> None:
        if mode not in ("regex", "regex+dep"):
            raise ValueError(f"mode must be 'regex' or 'regex+dep', got {mode!r}")
        self.mode = mode
        self._nlp = None

    def _load_nlp(self) -> bool:
        """Lazy-load spaCy model.  Returns True if available."""
        if self._nlp is not None:
            return True
        if not _SPACY_AVAILABLE:
            return False
        try:
            self._nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found")
            return False

    # ── public interface ───────────────────────────────────────────────────────

    def extract(
        self,
        eq_num: str,
        latex: str,
        context_before: str,
        context_after: str,
        audit: Any,
        retriever: Any = None,
        html_content: Optional[bytes] = None,
        parsed_paper: Any = None,
    ) -> Dict[str, str]:
        """Extract symbol definitions from equation LaTeX/MathML and context.

        Phase 1: obtain symbol keys from MathML <mi> elements (when HTML is
        available) or from the LaTeX string (fallback).

        Phase 2: for each symbol, retrieve definition candidates via BM25 or
        symbol_chunk_view, then run definitional regex patterns.  Apply the
        confidence gate and rejection conditions; emit only passing definitions.

        Parameters
        ----------
        eq_num : str
            Equation number (for MathML lookup in the paper HTML).
        latex : str
            LaTeX source string.
        context_before : str
            Text immediately before the equation.
        context_after : str
            Text immediately after the equation.
        audit : AuditTrail
        retriever : BM25Retriever or None, optional
        html_content : bytes or None, optional
            Full paper HTML bytes — enables MathML <mi> extraction.
        parsed_paper : ParsedPaper or None, optional
            Structured paper model — enables symbol_chunk_view retrieval.

        Returns
        -------
        dict
            ``{symbol_key: definition_phrase}`` for accepted symbols only.
        """
        # ── phase 1: identify symbols ──────────────────────────────────────────
        mi_symbols: Optional[List[str]] = None
        if html_content is not None and _BS4_AVAILABLE:
            mi_symbols = self._extract_mi_symbols_from_html(html_content, eq_num)
            if mi_symbols:
                audit.log(
                    "symbol_source",
                    f"eq=({eq_num}): MathML <mi> symbols={mi_symbols}",
                )

        if mi_symbols is not None:
            # from MathML: keep _SKIP_SINGLE members; they will be rejected
            # in phase 2 unless a definition is found.
            all_sym_keys: List[str] = mi_symbols
            use_skip_gate = True   # gate out _SKIP_SINGLE if no def found
        else:
            # LaTeX fallback
            all_sym_keys = self._extract_symbols_from_latex(latex)
            use_skip_gate = False  # already pre-filtered in LaTeX extractor

        if not all_sym_keys:
            audit.log("symbol_source", f"eq=({eq_num}): no symbols identified")
            return {}

        all_keys_set: Set[str] = set(all_sym_keys)
        base_context = (context_before + " " + context_after).strip()
        result: Dict[str, str] = {}

        # ── phase 2: per-symbol retrieval + confidence gate ───────────────────
        for sym in all_sym_keys:
            # Build retrieval context: symbol_chunk_view > BM25 > base context
            context = self._build_context(
                sym, base_context, retriever, parsed_paper
            )

            defn, pattern_name, confidence = self._find_definition_with_confidence(
                sym, context
            )

            if defn is None:
                # No definition found.
                # For MathML-sourced _SKIP_SINGLE members: omit silently.
                if use_skip_gate and sym in _SKIP_SINGLE:
                    continue
                # Otherwise: no entry in output (silent skip, no reject log needed).
                continue

            # Apply confidence gate — reject LOW confidence
            if confidence == "low":
                audit.log(
                    "symbol_reject",
                    f"sym={sym!r} def={defn!r} reason=low_confidence "
                    f"pattern={pattern_name}",
                )
                continue

            defn = defn.strip()

            # Apply rejection conditions
            reason = _reject_reason(defn, sym, all_keys_set)
            if reason:
                audit.log(
                    "symbol_reject",
                    f"sym={sym!r} def={defn!r} reason={reason} "
                    f"pattern={pattern_name}",
                )
                continue

            # Accepted — determine best chunk reference for audit
            chunk_ref = self._chunk_ref_for(sym, retriever, parsed_paper)

            audit.log(
                "symbol",
                f"sym={sym!r} def={defn[:60]!r} "
                f"chunk={chunk_ref} "
                f"confidence={confidence} "
                f"pattern={pattern_name}",
            )
            result[sym] = defn

        return result

    # ── symbol identification ──────────────────────────────────────────────────

    def _extract_mi_symbols_from_html(
        self, html_content: bytes, eq_num: str
    ) -> Optional[List[str]]:
        """Extract symbol keys from <mi> elements in the equation's <math> block.

        Searches the LaTeXML-generated HTML for the equation table whose tag
        reads ``({eq_num})``, then extracts all <mi> identifiers from the
        corresponding <math> element.

        Returns None if the equation cannot be located in the HTML.
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(html_content, "html.parser")
            except Exception:
                return None

        # Find the <span class="ltx_tag_equation"> matching this equation number.
        target_text = f"({eq_num})"
        for tag_span in soup.find_all("span", class_="ltx_tag_equation"):
            if tag_span.get_text(strip=True) == target_text:
                table = tag_span.find_parent("table", class_="ltx_eqn_table")
                if table:
                    math_elem = table.find("math")
                    if math_elem:
                        return self._collect_mi_from_math(math_elem)
        return None

    def _collect_mi_from_math(self, math_elem: Any) -> List[str]:
        """Walk a <math> element and collect identifier keys from <mi> children.

        Handles:
          - bare <mi>: single identifier, e.g. <mi>H</mi> → "H"
          - <msub><mi>base</mi><X>sub</X></msub>: subscripted form → "base_sub"
            (only when both base and subscript are simple letters/digits)
          - Greek Unicode in <mi> text → mapped to symbol name (e.g. ψ → "psi")

        Skips: pure numbers, multi-char identifiers that match _SKIP_COMMANDS,
        invisible operators, whitespace-only.
        """
        seen: set = set()
        symbols: List[str] = []

        for mi in math_elem.find_all("mi"):
            # Only process <mi> that are NOT inside <mo>, <mn> etc.
            key = self._mi_to_key(mi.get_text(strip=True))
            if key is None:
                continue

            # Check for subscripted parent (<msub>, first child is the base)
            parent = mi.parent
            if parent and parent.name == "msub":
                children = [c for c in parent.children if hasattr(c, 'name') and c.name]
                if len(children) >= 2 and children[0] is mi:
                    sub_text = children[1].get_text(strip=True)
                    if sub_text and (sub_text.isalnum() or sub_text in _UNICODE_TO_SYMBOL):
                        sub_key = _UNICODE_TO_SYMBOL.get(sub_text, sub_text)
                        subscripted_key = f"{key}_{sub_key}"
                        if subscripted_key not in seen:
                            seen.add(subscripted_key)
                            symbols.append(subscripted_key)
                        # Also add the base symbol alone
                        if key not in seen:
                            seen.add(key)
                            symbols.append(key)
                        continue

            if key not in seen:
                seen.add(key)
                symbols.append(key)

        return symbols

    def _mi_to_key(self, text: str) -> Optional[str]:
        """Convert <mi> text content to a symbol key, or None to skip."""
        if not text:
            return None
        # Pure whitespace or invisible chars
        if not text.strip():
            return None
        # Map Greek Unicode to symbol name
        if text in _UNICODE_TO_SYMBOL:
            return _UNICODE_TO_SYMBOL[text]
        # Strip backslash (shouldn't appear in <mi> but just in case)
        if text.startswith('\\'):
            text = text[1:]
        # Skip known operators/structural commands
        if text in _SKIP_COMMANDS:
            return None
        # Skip multi-char non-Greek identifiers that look like function names
        if len(text) > 1 and text.lower() in {c.lower() for c in _SKIP_COMMANDS}:
            return None
        # Skip pure numbers
        if text.isdigit():
            return None
        # Single alphabetic char or known multi-char physics symbol
        if text.isalpha() or re.match(r'^[A-Za-z][A-Za-z0-9]*$', text):
            return text
        return None

    def _extract_symbols_from_latex(self, latex: str) -> List[str]:
        """Extract symbol keys from a LaTeX string (fallback when HTML unavailable).

        Behaviour mirrors the old _tokenize_latex: Greek commands and uppercase/
        lowercase letters, excluding structural commands, operators, and index
        variables.
        """
        seen: set = set()
        symbols: List[str] = []
        tokens = re.findall(r'\\[a-zA-Z]+|[A-Za-z]', latex)
        for tok in tokens:
            if tok.startswith('\\'):
                name = tok[1:]
                if name in _SKIP_COMMANDS:
                    continue
                if name in _GREEK_TEXT:
                    key = name
                elif name.lower() in {"ket", "bra", "braket"}:
                    continue
                else:
                    continue
            else:
                if tok in _SKIP_SINGLE:
                    continue   # pre-filtered in LaTeX mode
                if not tok.isalpha():
                    continue
                key = tok
            if key not in seen:
                seen.add(key)
                symbols.append(key)
        return symbols

    # ── definition retrieval ───────────────────────────────────────────────────

    def _build_context(
        self,
        sym: str,
        base_context: str,
        retriever: Any,
        parsed_paper: Any,
    ) -> str:
        """Build the text to search for sym's definition."""
        parts: List[str] = []

        # symbol_chunk_view (structure-aware, prefers exact symbol occurrences)
        if parsed_paper is not None:
            try:
                text_forms: List[str] = list(_GREEK_TEXT.get(sym, [sym]))
                if sym not in text_forms:
                    text_forms.insert(0, sym)
                view_chunks = parsed_paper.symbol_chunk_view(parsed_paper, text_forms)
                if view_chunks:
                    parts.append(" ".join(c.text for c in view_chunks[:5]))
            except Exception:
                pass

        # BM25 retriever
        if retriever is not None:
            text_forms = list(_GREEK_TEXT.get(sym, [sym]))
            if sym not in text_forms:
                text_forms.insert(0, sym)
            query = " ".join(text_forms[:3])
            parts.append(retriever.retrieve_text(query, top_k=RETRIEVAL_TOP_K))

        parts.append(base_context)
        return " ".join(p for p in parts if p).strip()

    def _find_definition_with_confidence(
        self, sym: str, context: str
    ) -> Tuple[Optional[str], str, str]:
        """Search context for sym's definition; return (defn, pattern_name, confidence).

        Returns (None, "", "") when no pattern matches.
        """
        sym_frag = _build_sym_pattern(sym)

        for pattern_name, template, confidence in _DEF_PATTERNS:
            pattern = re.compile(
                template.replace("{SYM}", sym_frag),
                re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(context)
            if m:
                captured = m.group(1).strip()
                words = captured.split()
                if 1 <= len(words) <= 15:
                    return captured, pattern_name, confidence

        # spaCy dependency fallback (only for medium confidence if mode allows)
        if self.mode == "regex+dep" and self._load_nlp():
            defn = self._find_by_dep(sym, context)
            if defn:
                return defn, "dep_nsubj", "medium"

        return None, "", ""

    def _find_by_dep(self, sym: str, context: str) -> Optional[str]:
        """Use spaCy dependency parsing as last-resort fallback.

        Only nsubj+copula patterns are used (dep=nsubj with head lemma "be").
        Appositive patterns (dep=appos) are skipped here because they are LOW
        confidence and would be rejected by the confidence gate anyway.
        """
        clean = _INLINE_MATH_RE.sub(" SYM ", context)
        for sent_text in re.split(r'(?<=[.!?])\s+', clean):
            if "SYM" not in sent_text:
                if sym not in sent_text and sym.lower() not in sent_text.lower():
                    continue
            try:
                doc = self._nlp(sent_text)
            except Exception:
                continue
            for token in doc:
                if token.text.lower() not in (sym.lower(), "sym"):
                    continue
                if token.dep_ == "nsubj" and token.head.lemma_ == "be":
                    attrs = [
                        c for c in token.head.children
                        if c.dep_ in ("attr", "xcomp", "acomp") and c.i != token.i
                    ]
                    if attrs:
                        phrase = " ".join(
                            t.text for t in attrs[0].subtree if not t.is_punct
                        )
                        if 1 <= len(phrase.split()) <= 15:
                            return phrase
        return None

    # ── audit helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_ref_for(
        sym: str,
        retriever: Any,
        parsed_paper: Any,
    ) -> str:
        """Return a short chunk reference string for the audit trail."""
        if parsed_paper is not None:
            try:
                text_forms = list(_GREEK_TEXT.get(sym, [sym]))
                if sym not in text_forms:
                    text_forms.insert(0, sym)
                view_chunks = parsed_paper.symbol_chunk_view(parsed_paper, text_forms)
                if view_chunks:
                    return view_chunks[0].chunk_id
            except Exception:
                pass
        if retriever is not None:
            hits = retriever.search(sym, top_k=1)
            if hits:
                return hits[0]["chunk_id"]
        return "context_window"

    # ── utility (backward compat) ─────────────────────────────────────────────

    @staticmethod
    def clean_math_for_nlp(text: str) -> str:
        """Replace inline math with a placeholder so spaCy sees clean English."""
        return _INLINE_MATH_RE.sub("MATHSYM", text)
