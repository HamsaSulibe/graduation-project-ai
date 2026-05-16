"""Plant-name extraction helpers."""
from __future__ import annotations

import re
from typing import Optional

from src.utils.arabic import normalize, strip_ar_word_prefix


_NON_PLANT_TOKENS: frozenset[str] = frozenset({
    # Generic plant-domain nouns
    "نبات", "نبتة", "نباتات", "عشبة", "اعشاب", "أعشاب",
    "زراعة", "ازرع", "أزرع", "يزرع", "تزرع", "زرع",
    "ري", "تربة", "سماد", "حصاد", "ضوء", "شمس",
    "اصيص", "موسم", "فلسطين",
    "مباشر", "مباشرة", "مباشره", "غير", "تحب", "يحب",
    "يحتاج", "تحتاج", "بحب", "بتحب", "بدها", "بده",
    # Care/task schedule words
    "مهام", "مهمة", "مهمه",
    "جدول",
    "عناية", "عنايه",
    "تذكير", "تذكيرات",
    "روتين",
    # Watering/fertilizing action words
    "سقي", "اسقي", "أسقي",
    "تسميد",
    # Generic question / info words
    "معلومات", "معلومه", "معلومة",
    "سؤال", "اسئلة", "أسئلة",
    # Time / frequency words
    "يوم", "أيام", "ايام", "أسبوع", "اسبوع", "شهر",
    "مرة", "مره", "كم",
    # ── Benefit / use / info intent words (never plant names) ────
    "فوائد", "فوايد", "فائدة", "فايدة",
    "استخدامات", "استخدام",
    "ملاحظة", "ملاحظات",
    "نصائح", "نصايح", "نصيحة",
    "وصف", "تعريف",
    # ── Descriptive adjectives ────────────────────────────────────
    "مناسب", "مناسبة", "مناسبون",
    "إضافية", "اضافية", "إضافي", "اضافي",
    "أفضل", "افضل",
    "أكثر", "اكثر",
    "أقل", "اقل",
    # ── Plural-of-شهر (months) + other time words ────────────────
    "أشهر", "اشهر",
    "وقت", "أوقات", "اوقات",
    "فصل", "فصول",
    "فترة", "فترات",
    # ── Quantity / function words ─────────────────────────────────
    "كل",
    "نوع", "أنواع", "انواع",
    # ── Possessive / colloquial verbs ─────────────────────────────
    "عندك", "عندي", "عنده", "عندها", "عندنا",
    "خمنلي",
    # English plant-domain and question words
    "plant", "plants", "herb", "herbs", "care", "watering", "water",
    "light", "sunlight", "sun", "soil", "fertilizer", "fertilizing",
    "harvest", "uses", "use", "tasks", "task", "schedule", "routine",
    "temperature", "temp", "cold", "heat", "frost", "humidity", "moisture",
    "spacing", "space", "distance", "germination", "germinate",
    "how", "often", "what", "when", "where", "which", "compare",
    "comparison", "difference", "between", "easy", "easier", "best",
    "better", "grow", "growing", "planting", "steps",
    "do", "does", "did", "is", "are", "am", "was", "were", "be",
    "should", "would", "could", "can", "need", "needs", "needed",
    "have", "has", "had", "for", "from", "to", "in", "on", "of",
    "the", "a", "an", "it", "its", "this", "that", "my", "your",
})


def extract_plant_name(card: str) -> str:
    """Return the Arabic plant name from the first line of a card."""
    first_line = card.split("\n")[0]
    if ":" in first_line:
        after_colon = first_line.split(":", 1)[1].strip()
        return after_colon.split("|")[0].strip()
    return first_line.strip()

def extract_target_plant(
    question: str,
    all_plant_names: list[str],
) -> Optional[str]:
    """
    Try to find the plant name mentioned in the user's question.

    Strategy
    --------
    1. Direct substring match (normalised) – handles "نعناع" ⊂ "للنعناع".
    2. Token-level direct match with Arabic prefix stripping –
       handles "النبات" / "للنبات" / "بالنبات" → "نبات" == name_norm.
    3. Fuzzy match (rapidfuzz) on prefix-stripped token variants –
       handles informal spellings like "نعنع" vs "نعناع".

    Returns the *original* (un-normalised) plant name, or ``None``.
    """
    q_norm = normalize(question).lower()

    # Pre-strip common question-opening phrases so only the subject remains
    _openers = ("نبتة ", "نبات ", "عشبة ")

    # 1. Direct substring match
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm:
            continue
        # exact containment (e.g. "نعناع" inside "للنعناع")
        if name_norm in q_norm:
            return original_name
        # also try without common nominal prefixes on the name itself
        for pfx in _openers:
            if name_norm.startswith(pfx):
                stripped = name_norm[len(pfx):].strip()
                if stripped and len(stripped) > 2 and stripped in q_norm:
                    return original_name

    # 2. Token-level direct match with prefix stripping
    #    e.g. "نبات" (stripped from "النبات") == name_norm "نبات"
    q_words = [w for w in q_norm.split() if len(w) >= 2]
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm or len(name_norm) < 2:
            continue
        for word in q_words:
            for variant in strip_ar_word_prefix(word):
                if variant == name_norm:
                    return original_name

    # 3. Fuzzy match (optional – graceful if rapidfuzz absent)
    #    Use prefix-stripped token variants to bridge spelling gaps
    #    e.g. "نعنع" (stripped from "للنعنع") fuzzy-matches "نعناع"
    try:
        from rapidfuzz import process, fuzz

        # Build expanded token set: original tokens + prefix-stripped variants
        # Use min length 3 for fuzzy to avoid short words (e.g. 'ما') causing false positives
        # Skip tokens that are domain-intent words (e.g. "مهام", "جدول", "عناية").
        q_raw_tokens = [t for t in q_norm.split() if len(t) >= 3]
        seen_variants: set[str] = set()
        q_tokens_expanded: list[str] = []
        for raw_tok in q_raw_tokens:
            for variant in strip_ar_word_prefix(raw_tok):
                if variant not in seen_variants and len(variant) >= 2:
                    # Skip variants that are known non-plant domain words
                    if variant in _NON_PLANT_TOKENS:
                        continue
                    seen_variants.add(variant)
                    q_tokens_expanded.append(variant)

        name_norms = {normalize(n).lower(): n for n in all_plant_names if n}

        for token in q_tokens_expanded:
            match = process.extractOne(
                token,
                list(name_norms.keys()),
                scorer=fuzz.WRatio,
            )
            if match and match[1] >= 80:
                return name_norms[match[0]]
    except ImportError:
        pass

    return None

def extract_all_plants_from_question(
    question: str,
    all_plant_names: list[str],
) -> list[str]:
    """
    Return every plant name that appears in *question*.

    Used for comparison queries where we need two or more plants.
    Returns original (un-normalised) names, deduplicated, in order
    of first appearance.
    """
    q_norm = normalize(question).lower()
    found: list[str] = []
    found_norms: set[str] = set()

    # Pass 1: word-boundary-aware substring match.
    # A plant name must NOT be preceded or followed by an Arabic letter to
    # prevent "\u0631\u064a\u062d\u0627\u0646" from matching inside "\u0627\u0644\u0631\u064a\u062d\u0627\u0646".
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm or len(name_norm) < 2:
            continue
        if name_norm in found_norms:
            continue
        # \u0621-\u064a covers standard Arabic letters (hamza through ya)
        _wb_pat = r'(?<![\u0621-\u064a])' + re.escape(name_norm) + r'(?![\u0621-\u064a])'
        if re.search(_wb_pat, q_norm):
            found.append(original_name)
            found_norms.add(name_norm)

    # Pass 2: token-level match with Arabic prefix stripping
    q_words = [w for w in q_norm.split() if len(w) >= 2]
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm or name_norm in found_norms:
            continue
        for word in q_words:
            for variant in strip_ar_word_prefix(word):
                if variant == name_norm:
                    found.append(original_name)
                    found_norms.add(name_norm)
                    break

    # Pass 3: fuzzy match – only for tokens that look like plant name candidates.
    # We skip common question/domain words to avoid false positives where a word
    # like "اسهل" or "المبتدئين" fuzzy-matches to a real plant name.
    _COMPARISON_SKIP_TOKENS: frozenset[str] = _NON_PLANT_TOKENS | frozenset({
        "اسهل", "أسهل", "اصعب", "أصعب", "افضل", "أفضل",
        "اسرع", "أسرع", "اكثر", "أكثر", "اقل", "أقل",
        "مين", "مينو", "ايهما", "أيهما", "ايهم",
        "مبتدئين", "المبتدئين", "مبتدئ",
        "قارن", "بين", "ولا", "وين", "مقارنة",
        "يتحمل", "يحتاج", "يحتج", "احتاج",
        "حرارة", "برودة", "حصاد",
        "من", "في", "على", "الى", "إلى", "عن",
        "هو", "هي", "هم", "انا", "انت",
        # Short ambiguous tokens that cause false positives via WRatio partial_ratio
        # e.g. "اي" matches inside "شاي" → false positive for "الشاي الأخضر"
        "اي", "أي", "ايه", "أيه",
        "و", "او", "أو",
    })
    try:
        from rapidfuzz import process, fuzz as _fuzz
        seen_variants: set[str] = set()
        q_tokens_expanded: list[str] = []
        for raw_tok in q_words:
            # If the raw token itself is a skip word, skip ALL its stripped variants
            if raw_tok in _COMPARISON_SKIP_TOKENS:
                continue
            for variant in strip_ar_word_prefix(raw_tok):
                # Require ≥4 chars for fuzzy: prevents short tokens (e.g. "اي")
                # from matching multi-word plant names via partial_ratio.
                if variant not in seen_variants and len(variant) >= 4:
                    # Skip non-plant domain words and question/comparison words
                    if variant in _COMPARISON_SKIP_TOKENS:
                        continue
                    seen_variants.add(variant)
                    q_tokens_expanded.append(variant)

        name_norms_map = {
            normalize(n).lower(): n
            for n in all_plant_names
            if n and normalize(n).lower() not in found_norms
        }
        if name_norms_map and q_tokens_expanded:
            for token in q_tokens_expanded:
                match = process.extractOne(
                    token, list(name_norms_map.keys()), scorer=_fuzz.WRatio
                )
                # Use higher threshold (85) than single-plant extraction to
                # avoid false positives from question words in comparison queries.
                if match and match[1] >= 85:
                    matched_norm = match[0]
                    if matched_norm not in found_norms:
                        found.append(name_norms_map[matched_norm])
                        found_norms.add(matched_norm)
    except ImportError:
        pass

    return found
