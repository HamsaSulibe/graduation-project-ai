"""
arabic.py – Single-source Arabic text normalisation for the entire project.

Consolidates the duplicated normalisers that previously lived in:
  - intent_fields._norm()
  - context_utils._norm_simple()
  - plant_data_store._norm()
  - app.py  normalize_ar_text() / normalize_ar_name()

Public API
----------
DIACRITICS_RE          compiled regex for Arabic diacritics
normalize(text)        full normalisation (diacritics, hamza, ya, waw)
normalize_name(name)   normalize + strip ال/لل/ل prefixes  (for fuzzy plant-name matching)
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────
# Shared compiled regex
# ─────────────────────────────────────────────────────────────────
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

# Prefix patterns for normalize_name  (longest first)
_PREFIX_RE = re.compile(
    r"^(?:بال|وال|فال|كال|لل|ال|ب|ل|ك|و|ف)\s*",
    re.UNICODE,
)

# ─────────────────────────────────────────────────────────────────
# Arabic word-prefix constants (for query-token normalisation)
# ─────────────────────────────────────────────────────────────────
# Ordered longest-first so that "بال" is stripped before bare "ب"
_AR_WORD_PREFIXES: tuple[str, ...] = (
    "بال", "وال", "فال", "كال",   # preposition + definite article
    "لل",                          # lam + lam  (للنعنع)
    "ال",                          # definite article
    "ب", "ل", "ك", "و", "ف",      # single prepositions
)


# ─────────────────────────────────────────────────────────────────
# Core normaliser  (replaces _norm / _norm_simple / normalize_ar_text)
# ─────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """
    Normalise an Arabic string:
      - Remove tatweel (kashida)
      - Strip diacritical marks (tashkeel)
      - Unify hamza forms → ا
      - Unify ى → ي, ؤ → و, ئ → ي
    """
    if not text:
        return ""
    s = str(text)
    s = s.replace(_TATWEEL, "")
    s = DIACRITICS_RE.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    s = s.replace("ؤ", "و").replace("ئ", "ي")
    return s.strip()


# ─────────────────────────────────────────────────────────────────
# Extended normaliser for query / plant-name matching
# (replaces app.py normalize_ar_text + normalize_ar_name)
# ─────────────────────────────────────────────────────────────────
def normalize_query(text: str) -> str:
    """
    Normalise for search / keyword matching:
      - All of ``normalize()``
      - Replace punctuation (؟ ، , .) with spaces
      - Collapse whitespace
    """
    if not text:
        return ""
    s = normalize(text)
    s = s.replace("؟", " ").replace("،", " ").replace(",", " ").replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(name: str) -> str:
    """
    Normalise a plant name for fuzzy matching:
      - ``normalize_query()``
      - Strip leading ال / لل / بال / ل / ب … prefixes
    """
    s = normalize_query(name)
    for _ in range(3):
        prev = s
        s = _PREFIX_RE.sub("", s).strip()
        if s == prev:
            break
    return s


# ─────────────────────────────────────────────────────────────────
# Token-level prefix stripping  (for extract_target_plant)
# ─────────────────────────────────────────────────────────────────

def strip_ar_word_prefix(word: str) -> list[str]:
    """
    Return all prefix-stripped variants of a single (normalised)
    Arabic word token.

    Example
    -------
    >>> strip_ar_word_prefix("للنبات")
    ["للنبات", "نبات"]
    >>> strip_ar_word_prefix("بالنبات")
    ["بالنبات", "نبات"]
    >>> strip_ar_word_prefix("النبات")
    ["النبات", "نبات"]
    """
    variants: set[str] = {word}
    for pfx in _AR_WORD_PREFIXES:
        if word.startswith(pfx) and len(word) > len(pfx) + 1:
            stripped = word[len(pfx):]
            if len(stripped) >= 2:
                variants.add(stripped)
    return list(variants)
