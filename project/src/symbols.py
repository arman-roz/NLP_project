"""
src/symbols.py

Symbol definition extraction from equation LaTeX and surrounding text.

Two modes (set at construction time):
  "regex"      — regex definitional patterns only.
  "regex+dep"  — regex first; spaCy dependency parsing as fallback.

The mode parameter exists so you can run both, count how many definitions
each finds, and report the comparison.  regex is always tried first so the
results of the two modes differ only in the *additional* definitions that
dependency parsing recovers.

No text generation.  Every description returned is a phrase or clause
copied verbatim from the paper's surrounding text.

Symbol output key:  LaTeX command without leading backslash (e.g. "alpha",
"phi"), or the bare letter for single-letter variables (e.g. "H", "E").
Standard operators (+, -, =, \\nabla, etc.) are never emitted as symbols.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── spaCy optional import ─────────────────────────────────────────────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.info("spaCy not available — symbol extraction will use regex-only mode")

# ── LaTeX structural / operator commands to skip ──────────────────────────────
# These are never meaningful symbols — they are either structural formatting,
# bracket sizing, common operators, or standard functions.
_SKIP_COMMANDS: Set[str] = {
    # structural
    "begin", "end", "frac", "dfrac", "tfrac", "cfrac",
    "sqrt", "root", "over", "atop",
    "left", "right", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "big", "Big",
    # decorators (base letter is the symbol, not the decorator)
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
    "langle", "rangle", "vert", "Vert", "|",
    "dagger", "ddagger", "circ", "bullet", "star",
    "partial", "nabla", "infty", "emptyset", "forall", "exists",
    "int", "oint", "iint", "iiint", "sum", "prod", "coprod",
    # standard math functions
    "exp", "log", "ln", "lg", "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "arccot", "sinh", "cosh", "tanh",
    "max", "min", "sup", "inf", "lim", "limsup", "liminf",
    "det", "tr", "Tr", "ker", "dim", "deg", "gcd", "lcm", "mod",
    "arg", "Re", "Im",
    # labels/references (appear in annotation tags)
    "label", "ref", "eqref", "nonumber", "notag", "tag",
    # misc
    "not", "ne", "le", "ge", "to",
    "leavevmode", "nobreak", "strut", "hbox",
}

# Single-letter tokens that are almost always indices/differentials, not symbols
# worth defining.  Small set — erring on the side of inclusion.
_SKIP_SINGLE: Set[str] = {"d", "n", "i", "j", "k", "l", "m"}

# ── Greek letter reverse map: LaTeX command name → common text forms ──────────
# Used to search for symbol mentions in plain text.
# key = command name WITHOUT backslash  (e.g. "alpha")
# value = list of text forms that might appear in prose
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
    # common physics symbols
    "hbar":  ["ℏ", "ħ", "hbar"],
}

# ── Inline math stripper (for spaCy input) ────────────────────────────────────
# Replaces $...$ and \(...\) with a placeholder so spaCy doesn't parse LaTeX.
_INLINE_MATH_RE = re.compile(
    r'\$[^$]+\$'           # $...$
    r'|\\\([^)]+\\\)'      # \(...\)
    r'|\\[a-zA-Z]+\{[^}]*\}'  # \cmd{...}
)

# ── Definitional regex patterns ────────────────────────────────────────────────
# Each pattern has a group(1) that captures the definition text.
# {SYM} is a placeholder replaced at match time with the actual symbol pattern.
# Patterns are tried in order; first match wins.
_DEF_PATTERN_TEMPLATES: List[Tuple[str, str]] = [
    # "where X is/denotes/represents ..."
    (
        "where_is",
        r'\bwhere\s+{SYM}\s+(?:is|are|denotes?|represents?|gives?|'
        r'stands?\s+for|corresponds?\s+to)\s+(?:the\s+)?(.+?)(?:[,;]|\s+and\s+|\.$|$)',
    ),
    # "let X be ..."
    (
        "let_be",
        r'\blet\s+{SYM}\s+be\s+(?:the\s+)?(.+?)(?:[,;.]|$)',
    ),
    # "X is/denotes/represents the ..."  (standalone)
    (
        "sym_is",
        r'\b{SYM}\s+(?:is|are|denotes?|represents?|corresponds?\s+to)\s+'
        r'(?:the\s+)?(.+?)(?:[,;.]|$)',
    ),
    # "the ... X"  appositive: "the total Hamiltonian H"
    (
        "the_noun_sym",
        r'\bthe\s+((?:\w+\s+){0,4}\w+)\s+{SYM}(?:\b|$)',
    ),
    # "X, the ..."  appositive reversed: "H, the Hamiltonian"
    (
        "sym_the",
        r'\b{SYM}\s*,\s+the\s+([\w\s\-]+?)(?:[,;.]|$)',
    ),
]


def _build_sym_pattern(sym_key: str) -> str:
    """Build a regex fragment that matches sym_key in plain text.

    For a LaTeX command like "alpha", also matches "α" and "alpha".
    For a single letter like "H", matches the bare letter with word boundary.

    Parameters
    ----------
    sym_key : str
        Symbol key as stored in the output dict (no backslash).

    Returns
    -------
    str
        Regex fragment (not compiled).
    """
    forms: List[str] = []

    if sym_key in _GREEK_TEXT:
        forms = [re.escape(f) for f in _GREEK_TEXT[sym_key]]
    else:
        # single-letter or multi-letter variable
        forms = [re.escape(sym_key)]

    # also look for the dollar-delimited form: $H$, $\alpha$
    dollar_forms = [
        r'\$\\?' + re.escape(sym_key) + r'\$',
        r'\$' + re.escape(sym_key) + r'\$',
    ]
    all_forms = forms + dollar_forms
    return r'(?:' + '|'.join(all_forms) + r')'


class SymbolExtractor:
    """Extract symbol definitions from equation LaTeX and surrounding text.

    Parameters
    ----------
    mode : str
        ``"regex"`` — regex patterns only.
        ``"regex+dep"`` — regex first, spaCy dependency parsing as fallback.
        Defaults to ``"regex+dep"``.

    Examples
    --------
    >>> se = SymbolExtractor(mode="regex+dep")
    >>> syms = se.extract(r"E = mc^2", ctx_before, ctx_after, audit)
    >>> syms
    {'E': 'total energy', 'm': 'rest mass', 'c': 'speed of light'}
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
        latex: str,
        context_before: str,
        context_after: str,
        audit,
    ) -> Dict[str, str]:
        """Extract symbol definitions from LaTeX and surrounding text.

        Parameters
        ----------
        latex : str
            The equation LaTeX string.
        context_before : str
            Text immediately before the equation.
        context_after : str
            Text immediately after the equation.
        audit : AuditTrail
            Receives one entry per symbol found.

        Returns
        -------
        dict
            ``{symbol_key: description_phrase}`` where ``symbol_key`` is the
            LaTeX command name without backslash, or the bare letter.
            Standard operators and structural commands are excluded.
        """
        symbols = self._tokenize_latex(latex)
        context = (context_before + " " + context_after).strip()
        result: Dict[str, str] = {}

        for sym in symbols:
            defn = self._find_definition(sym, context, audit)
            if defn:
                result[sym] = defn.strip()
                audit.log(
                    "symbol",
                    f"sym={sym!r}: definition={defn[:80]!r}",
                )

        return result

    # ── LaTeX tokeniser ────────────────────────────────────────────────────────

    def _tokenize_latex(self, latex: str) -> List[str]:
        """Extract meaningful symbol keys from a LaTeX string.

        Finds:
          - ``\\command`` tokens whose name is a known Greek letter or
            physics symbol (hbar, etc.).
          - Bare uppercase/lowercase letters that are not index variables.

        Skips:
          - Structural commands (``\\frac``, ``\\begin``, decorators, …).
          - Common operators (``\\sum``, ``\\int``, ``\\nabla``, …).
          - Single-letter index variables (d, i, j, k, l, m, n).
          - Digits and punctuation.

        Parameters
        ----------
        latex : str
            Raw LaTeX string.

        Returns
        -------
        list of str
            Ordered, deduplicated symbol keys (no backslash).
        """
        seen: set = set()
        symbols: List[str] = []

        # tokenise: \command OR single alphanumeric char
        tokens = re.findall(r'\\[a-zA-Z]+|[A-Za-z]', latex)

        for tok in tokens:
            if tok.startswith('\\'):
                name = tok[1:]  # strip backslash
                if name in _SKIP_COMMANDS:
                    continue
                # keep Greek letters and known physics symbols
                if name in _GREEK_TEXT:
                    key = name
                elif name.lower() in {"ket", "bra", "braket"}:
                    continue  # Dirac notation wrappers, not symbols
                else:
                    continue  # unknown command — skip
            else:
                # bare letter
                if tok in _SKIP_SINGLE:
                    continue
                if not tok.isalpha():
                    continue
                key = tok

            if key not in seen:
                seen.add(key)
                symbols.append(key)

        return symbols

    # ── definition search ──────────────────────────────────────────────────────

    def _find_definition(
        self, sym: str, context: str, audit
    ) -> Optional[str]:
        """Search for sym's definition in context text.

        Tries regex patterns first.  If mode is "regex+dep" and regex fails,
        falls back to spaCy dependency parsing.

        Parameters
        ----------
        sym : str
            Symbol key (no backslash).
        context : str
            Plain text to search.
        audit : AuditTrail
            Receives a note if dep parsing was used.

        Returns
        -------
        str or None
        """
        # ── regex path ────────────────────────────────────────────────────────
        defn = self._find_by_regex(sym, context)
        if defn:
            return defn

        # ── dep path (only if mode allows and spaCy is available) ─────────────
        if self.mode == "regex+dep" and self._load_nlp():
            defn = self._find_by_dep(sym, context, audit)
            if defn:
                return defn

        return None

    def _find_by_regex(self, sym: str, context: str) -> Optional[str]:
        """Apply definitional regex patterns to find sym's description.

        Returns the first captured group from the first matching pattern,
        or None if no pattern matches.
        """
        sym_frag = _build_sym_pattern(sym)

        for pattern_name, template in _DEF_PATTERN_TEMPLATES:
            pattern = re.compile(
                template.replace("{SYM}", sym_frag),
                re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(context)
            if m:
                captured = m.group(1).strip()
                # sanity: skip captures that are too short or too long
                words = captured.split()
                if 1 <= len(words) <= 15:
                    return captured
        return None

    def _find_by_dep(self, sym: str, context: str, audit) -> Optional[str]:
        """Use spaCy dependency parsing to find sym's definition.

        Strips inline math from the sentence first, then walks the
        dependency tree looking for:
          - nsubj of a copula (X is ...)
          - appositive (X, the ...) or (the ... X)

        Parameters
        ----------
        sym : str
            Symbol key.
        context : str
            Raw context text (may contain inline math).
        audit : AuditTrail

        Returns
        -------
        str or None
        """
        # strip inline math so spaCy parses clean English
        clean = _INLINE_MATH_RE.sub(" SYM ", context)

        # process sentence by sentence to keep the dep tree manageable
        for sent_text in re.split(r'(?<=[.!?])\s+', clean):
            if "SYM" not in sent_text:
                # if the symbol was replaced, sent must contain SYM placeholder
                # for single-letter vars that were NOT in inline math, search raw
                if sym not in sent_text and sym.lower() not in sent_text.lower():
                    continue

            doc = self._nlp(sent_text)

            for token in doc:
                tok_text = token.text.lower()
                if tok_text != sym.lower() and tok_text != "sym":
                    continue

                # pattern A: X is <attr>  →  token is nsubj, head lemma "be"
                if token.dep_ == "nsubj" and token.head.lemma_ == "be":
                    attrs = [
                        child for child in token.head.children
                        if child.dep_ in ("attr", "xcomp", "acomp")
                           and child.i != token.i
                    ]
                    if attrs:
                        phrase = " ".join(
                            t.text for t in attrs[0].subtree
                            if not t.is_punct
                        )
                        if 1 <= len(phrase.split()) <= 15:
                            audit.log(
                                "symbol_dep",
                                f"sym={sym!r}: dep=nsubj+attr, phrase={phrase!r}",
                            )
                            return phrase

                # pattern B: appositive  X , the <noun phrase>
                if token.dep_ == "appos" or any(
                    c.dep_ == "appos" and c.i == token.i
                    for c in token.head.children
                ):
                    phrase = " ".join(
                        t.text for t in token.subtree
                        if not t.is_punct and t.i != token.i
                    )
                    if 1 <= len(phrase.split()) <= 15:
                        audit.log(
                            "symbol_dep",
                            f"sym={sym!r}: dep=appos, phrase={phrase!r}",
                        )
                        return phrase

        return None

    # ── utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def clean_math_for_nlp(text: str) -> str:
        """Replace inline math with a placeholder so spaCy sees clean English.

        Parameters
        ----------
        text : str
            Raw sentence possibly containing ``$...$`` or ``\\(...\\)``.

        Returns
        -------
        str
            Text with inline math replaced by ``MATHSYM``.
        """
        return _INLINE_MATH_RE.sub("MATHSYM", text)
