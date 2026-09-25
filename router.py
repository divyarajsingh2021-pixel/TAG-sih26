"""
backend/router.py
-----------------
Classifies user queries into intent categories so the UI can
show the right visualization alongside the LLM answer.
"""

from __future__ import annotations


INTENT_MAP = {
    "temperature": ["temperature", "temp", "sst", "warm", "cold", "heat", "thermal"],
    "salinity":    ["salinity", "salt", "psu", "fresh", "brackish"],
    "location":    ["where", "location", "map", "position", "coordinate", "lat", "lon", "float"],
    "depth":       ["depth", "deep", "profile", "layer", "500m", "1000m", "mixed layer"],
    "comparison":  ["compare", "vs", "versus", "difference", "between", "arabian", "bay of bengal"],
    "trend":       ["trend", "over time", "monthly", "seasonal", "change", "increase", "decrease"],
    "ts_diagram":  ["t-s", "ts diagram", "water mass", "thermohaline"],
}


def classify_intent(query: str) -> str:
    """Return the primary intent of the query."""
    q = query.lower()
    scores = {intent: 0 for intent in INTENT_MAP}
    for intent, keywords in INTENT_MAP.items():
        for kw in keywords:
            if kw in q:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def extract_region(query: str) -> str:
    """Extract mentioned ocean region from query."""
    q = query.lower()
    if "arabian" in q:
        return "Arabian Sea"
    if "bay of bengal" in q or "bengal" in q:
        return "Bay of Bengal"
    return "All"


def extract_month(query: str) -> int:
    """Extract mentioned month number (0 = all months)."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    q = query.lower()
    for name, num in months.items():
        if name in q:
            return num
    return 0
