"""Small shared helpers for the extraction pipeline."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple


class AuditTrail:
    """Collects short method-level evidence for one equation."""

    def __init__(self) -> None:
        self._items: List[Tuple[str, str]] = []

    def add(self, method: str, detail: str) -> None:
        """Add one audit item."""

        self._items.append((method, short(detail)))

    def extend(self, other: "AuditTrail", prefix: str = "") -> None:
        """Copy items from another trail."""

        for method, detail in other.items():
            self.add(prefix + method, detail)

    def items(self) -> List[Tuple[str, str]]:
        """Return raw items."""

        return list(self._items)

    def as_dict(self) -> dict:
        """Return a JSON-friendly dictionary with stable repeated keys."""

        out: OrderedDict[str, str] = OrderedDict()
        seen: dict[str, int] = {}
        for method, detail in self._items:
            index = seen.get(method, 0)
            key = method if index == 0 else f"{method}_{index}"
            out[key] = detail
            seen[method] = index + 1
        return dict(out)


def short(text: object, limit: int = 180) -> str:
    """Compress a value into a short audit string."""

    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def read_paper_list(path: Path) -> List[str]:
    """Read arXiv IDs in exactly the assigned file order."""

    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                ids.append(re.sub(r"(?i)^arxiv:", "", line))
    return ids


def paper_key(arxiv_id: str) -> str:
    """Return the JSON key used by the assignment."""

    return f"arXiv:{arxiv_id}"

