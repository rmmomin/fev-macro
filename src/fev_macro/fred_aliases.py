from __future__ import annotations

from typing import Iterable

# Shared manual aliases used across FRED/ALFRED ingestion scripts.
MANUAL_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "CLAIMSx": ("ICSA",),
    "S&P 500": ("SP500",),
    "S&P div yield": ("SPDIVY",),
    "S&P PE ratio": ("SP500PE", "SPEARN"),
}


def candidate_series_ids(variable_name: str) -> list[str]:
    """Return prioritized candidate FRED series IDs for a template variable."""
    raw = str(variable_name).strip()
    if not raw:
        return []

    candidates: list[str] = []
    manual = MANUAL_SERIES_ALIASES.get(raw)
    if manual:
        candidates.extend(list(manual))

    candidates.append(raw)

    no_spaces = raw.replace(" ", "")
    if no_spaces != raw:
        candidates.append(no_spaces)

    if raw.endswith("^") and len(raw) > 1:
        candidates.append(raw[:-1])

    if raw.endswith("x") and len(raw) > 1:
        stem = raw[:-1]
        candidates.append(stem)
        candidates.append(stem.replace(" ", ""))

    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        c = str(cand).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out
