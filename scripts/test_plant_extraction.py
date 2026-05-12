"""
test_plant_extraction.py – Verify plant name extraction and intent detection.

Run from the project root:
    .venv\Scripts\python.exe scripts/test_plant_extraction.py

Tests every example question from the requirements and prints:
  - extracted plant name
  - detected intents
  - whether it would succeed (plant found + response mode)
"""
from __future__ import annotations

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.arabic import normalize, normalize_name, strip_ar_word_prefix
from src.inference.intent_fields import classify_intents

# ── Inline extract_target_plant test (no faiss needed) ───────────
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZY = True
except ImportError:
    _HAS_FUZZY = False


def _extract_plant(question: str, all_plant_names: list[str]):
    """Replicate extract_target_plant logic without importing faiss."""
    q_norm = normalize(question).lower()
    _openers = ("نبتة ", "نبات ", "عشبة ")

    # 1. Direct substring match
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm:
            continue
        if name_norm in q_norm:
            return original_name
        for pfx in _openers:
            if name_norm.startswith(pfx):
                stripped = name_norm[len(pfx):].strip()
                if stripped and len(stripped) > 2 and stripped in q_norm:
                    return original_name

    # 2. Token-level direct match with prefix stripping
    q_words = [w for w in q_norm.split() if len(w) >= 2]
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm or len(name_norm) < 2:
            continue
        for word in q_words:
            for variant in strip_ar_word_prefix(word):
                if variant == name_norm:
                    return original_name

    # 3. Fuzzy match with prefix-stripped token variants
    if _HAS_FUZZY:
        seen_variants: set[str] = set()
        q_tokens_expanded: list[str] = []
        for raw_tok in q_words:
            for variant in strip_ar_word_prefix(raw_tok):
                if variant not in seen_variants and len(variant) >= 2:
                    seen_variants.add(variant)
                    q_tokens_expanded.append(variant)

        name_norms = {normalize(n).lower(): n for n in all_plant_names if n}
        for token in q_tokens_expanded:
            match = process.extractOne(token, list(name_norms.keys()), scorer=fuzz.WRatio)
            if match and match[1] >= 80:
                return name_norms[match[0]]

    return None


# ── Token prefix stripping unit tests ────────────────────────────
print("=" * 60)
print("1. Token prefix stripping")
print("=" * 60)
strip_tests = [
    ("للنعنع",   "نعنع"),
    ("بالنعنع",  "نعنع"),
    ("النعنع",   "نعنع"),
    ("والنعنع",  "نعنع"),
    ("فالنعنع",  "نعنع"),
    ("نعنع",     "نعنع"),   # no prefix
    ("نعناع",    "نعناع"),  # no prefix (official name)
]
for word, expected in strip_tests:
    word_n = normalize(word).lower()
    variants = strip_ar_word_prefix(word_n)
    ok = expected in variants
    status = "✓" if ok else "✗ FAIL"
    print(f"  strip_ar_word_prefix({word!r:12s}) → {variants} [{status}]")

# ── Intent detection tests ────────────────────────────────────────
print()
print("=" * 60)
print("2. Intent detection")
print("=" * 60)
intent_tests = [
    ("كم مرة أسقي النعنع؟",                "watering"),
    ("ما هو الضوء المناسب للنعنع؟",        "light"),
    ("درجة الحرارة المناسبة للنعنع؟",       "temperature"),
    ("شو تربة النعنع؟",                    "soil"),
    ("تسميد النعنع؟",                      "fertilizing"),
    ("مهام العناية بالنعنع؟",              "tasks"),
    ("هل النعنع مناسب لبيتي؟",             "suitability"),
    ("احكيلي عن النعنع",                   "plant_overview"),
    ("معلومات عن النعنع",                  "general_summary"),
    ("كيف ازرع النعنع؟",                   "planting"),
    ("متى أحصد النعنع؟",                   "harvest_storage"),
    ("أمراض النعنع",                       "pests_diseases"),
    ("رطوبة النعنع",                       "humidity"),
]
for q, expected_intent in intent_tests:
    intents = classify_intents(q)
    ok = expected_intent in intents
    status = "✓" if ok else f"✗ FAIL (got {intents})"
    print(f"  {q[:42]:42s} → {intents[0]:20s} [{status}]")

# ── Plant extraction tests ────────────────────────────────────────
print()
print("=" * 60)
print("3. Plant name extraction  (DB has: 'نعناع')")
print("=" * 60)
DB_NAMES = ["نعناع", "ريحان", "بقدونس", "زعتر", "طماطم"]

extraction_tests = [
    # (question, expected_plant)
    ("نعناع",                              "نعناع"),
    ("النعناع",                            "نعناع"),
    ("نعنع",                               "نعناع"),
    ("للنعنع",                             "نعناع"),
    ("بالنعنع",                            "نعناع"),
    ("عن النعنع",                          "نعناع"),
    ("اسقي النعنع",                        "نعناع"),
    ("كم مرة أسقي النعنع؟",               "نعناع"),
    ("كم مرة أسقي نعنع؟",                 "نعناع"),
    ("كم مرة أسقي للنعنع؟",               "نعناع"),
    ("ما هو الضوء المناسب للنعنع؟",       "نعناع"),
    ("درجة الحرارة المناسبة للنعنع؟",      "نعناع"),
    ("شو تربة النعنع؟",                   "نعناع"),
    ("شو مهام العناية بالنعنع؟",          "نعناع"),
    ("هل النعنع مناسب لبيتي؟",            "نعناع"),
    ("احكيلي عن النعنع",                   "نعناع"),
    ("احكيلي عن نعنع",                    "نعناع"),
    # also test other plants
    ("كيف أسقي الريحان؟",                 "ريحان"),
    ("تربة البقدونس",                      "بقدونس"),
]

passed = 0
failed = 0
for q, expected in extraction_tests:
    result = _extract_plant(q, DB_NAMES)
    ok = result == expected
    status = "✓" if ok else f"✗ FAIL (got {result!r})"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {q[:48]:48s} → {str(result):10s} [{status}]")

print()
print(f"Results: {passed}/{passed+failed} passed", "✓" if failed == 0 else f"— {failed} FAILED")

# ── normalize_name tests ─────────────────────────────────────────
print()
print("=" * 60)
print("4. normalize_name prefix stripping")
print("=" * 60)
name_tests = [
    ("نعناع",   "نعناع"),
    ("النعناع", "نعناع"),
    ("للنعناع", "نعناع"),
    ("بالنعناع","نعناع"),
    ("نعنع",    "نعنع"),
]
for name, expected in name_tests:
    result = normalize_name(name)
    ok = result == expected
    status = "✓" if ok else f"✗ FAIL (got {result!r})"
    print(f"  normalize_name({name!r:15s}) → {result!r:12s} [{status}]")
