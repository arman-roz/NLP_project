"""Validation for the assignment JSON shape."""

from __future__ import annotations

from typing import Dict, List

REQUIRED_KEYS = {"equation", "meaning", "symbols", "relations", "audit-trail"}
VALID_GRADES = {"none", "strong", "potential"}


class DatasetError(ValueError):
    """Raised when the output shape is invalid."""


def check_dataset(data: Dict) -> None:
    """Validate the final JSON object."""

    errors: List[str] = []
    if not isinstance(data, dict):
        raise DatasetError("root must be a dictionary")
    for paper_id, equations in data.items():
        if not isinstance(equations, dict):
            errors.append(f"{paper_id}: paper value must be a dictionary")
            continue
        eq_numbers = list(equations)
        for eq_num, entry in equations.items():
            errors.extend(_check_equation(paper_id, eq_num, entry, eq_numbers))
    if errors:
        raise DatasetError("\n".join(errors[:30]))


def _check_equation(paper_id: str, eq_num: str, entry: object, eq_numbers: List[str]) -> List[str]:
    prefix = f"{paper_id}/{eq_num}"
    if not isinstance(entry, dict):
        return [f"{prefix}: entry must be a dictionary"]
    errors: List[str] = []
    if set(entry) != REQUIRED_KEYS:
        errors.append(f"{prefix}: keys must be {sorted(REQUIRED_KEYS)}, got {sorted(entry)}")
    if not isinstance(entry.get("equation"), str):
        errors.append(f"{prefix}: equation must be a string")
    if not isinstance(entry.get("meaning"), str):
        errors.append(f"{prefix}: meaning must be a string")
    if not isinstance(entry.get("symbols"), dict):
        errors.append(f"{prefix}: symbols must be a dictionary")
    if not isinstance(entry.get("audit-trail"), dict):
        errors.append(f"{prefix}: audit-trail must be a dictionary")
    errors.extend(_check_relations(prefix, eq_num, entry.get("relations"), eq_numbers))
    return errors


def _check_relations(prefix: str, eq_num: str, relations: object, eq_numbers: List[str]) -> List[str]:
    if not isinstance(relations, dict):
        return [f"{prefix}: relations must be a dictionary"]
    errors: List[str] = []
    expected = {number for number in eq_numbers if number != eq_num}
    if set(relations) != expected:
        errors.append(f"{prefix}: relation keys must be {sorted(expected)}, got {sorted(relations)}")
    for other, relation in relations.items():
        if not isinstance(relation, dict):
            errors.append(f"{prefix}/relations/{other}: relation must be a dictionary")
            continue
        if relation.get("grade") not in VALID_GRADES:
            errors.append(f"{prefix}/relations/{other}: invalid grade {relation.get('grade')!r}")
        if not isinstance(relation.get("description", ""), str):
            errors.append(f"{prefix}/relations/{other}: description must be a string")
    return errors

