"""
src/audit.py

Per-equation audit trail.

The project spec defines audit-trail as a dict whose keys are method names.
The spec's own example shows the same key appearing twice (e.g. two calls to
find_symbol), which is invalid JSON.  We resolve this by collecting entries as
an ordered list internally and adding a numeric suffix (_1, _2, …) to any
duplicate key at serialisation time.  This keeps the output a valid JSON dict
while preserving every log entry.
"""

from collections import OrderedDict
from typing import List, Tuple


class AuditTrail:
    """Collects per-equation extraction log entries.

    Each entry is a (method_name, detail_string) pair recorded by the
    method that produced a piece of data.  When serialised, duplicate
    method names receive a numeric suffix so no key is silently lost.

    Parameters
    ----------
    None

    Examples
    --------
    >>> trail = AuditTrail()
    >>> trail.log("fetch_html", "HTTP 200, cached, 84321 bytes")
    >>> trail.log("extract_eq", "found eq (1): E = mc^2")
    >>> trail.log("find_symbol", "found symbol: E")
    >>> trail.log("find_symbol", "found symbol: m")
    >>> trail.to_dict()
    {'fetch_html': 'HTTP 200, cached, 84321 bytes',
     'extract_eq': 'found eq (1): E = mc^2',
     'find_symbol': 'found symbol: E',
     'find_symbol_1': 'found symbol: m'}
    """

    def __init__(self) -> None:
        # ordered list so audit entries appear in extraction order
        self._entries: List[Tuple[str, str]] = []

    def log(self, method: str, detail: str) -> None:
        """Append one audit entry.

        Parameters
        ----------
        method : str
            Name of the method that produced the data (e.g. ``'fetch_html'``).
        detail : str
            Short, human-readable description of what was found or decided.
            Keep it under ~150 characters so the JSON stays readable.
        """
        self._entries.append((method, str(detail)))

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible ordered dict.

        Duplicate method names are disambiguated with a numeric suffix:
        ``find_symbol``, ``find_symbol_1``, ``find_symbol_2``, …

        Returns
        -------
        dict
            {method_key: detail_string} in insertion order.
        """
        result: OrderedDict = OrderedDict()
        # track how many times each base method name has appeared
        counts: dict = {}

        for method, detail in self._entries:
            if method not in counts:
                counts[method] = 0
                result[method] = detail
            else:
                counts[method] += 1
                result[f"{method}_{counts[method]}"] = detail

        return dict(result)

    def entries(self) -> List[Tuple[str, str]]:
        """Return raw (method, detail) pairs in insertion order.

        Returns
        -------
        list of tuple
            Each tuple is (method_name, detail_string).
        """
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuditTrail({len(self._entries)} entries)"
