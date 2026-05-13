"""
plant_resolution.py – Plant-name lookup helpers for the Garssa Smart
Plant Assistant.

All functions are pure; no FastAPI / app dependencies.
"""
import re
from typing import Optional

from src.utils.arabic import normalize, normalize_query, strip_ar_word_prefix

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# Intents where a specific plant name is required to answer (Rule 9/10/17)
_PLANT_SPECIFIC_INTENTS: frozenset[str] = frozenset({
    "watering", "light", "temperature", "soil", "fertilizing",
    "season", "harvest_storage", "pests_diseases", "germination",
    "planting", "care_summary", "humidity", "suitability", "tasks",
    "spacing", "uses", "description", "plant_identity",
})

# Common Arabic question-opening phrases stripped before token extraction
_QUESTION_OPENERS: list[str] = [
    "اخبرني عن",
    "احكيلي عن",
    "حدثني عن",
    "اعطيني معلومات عن",
    "أريد معلومات عن",
    "بدي معلومات عن",
    "بدي أعرف عن",
    "معلومات عن",
    "دليل زراعة",
    "كيف أزرع",
    "كيف ازرع",
    "كيف اعتني ب",
    "كيف أعتني ب",
    "ما هو",
    "ما هي",
    "شو هو",
    "شو هي",
    "عن",
    "tell me about",
    "information about",
    "info about",
    "how do i grow",
    "how to grow",
    "how do i plant",
    "how to plant",
    "how do i care for",
    "how to care for",
    "what is",
    "about",
]

# Generic domain tokens that are never plant-name candidates
_DOMAIN_SKIP_TOKENS: frozenset[str] = frozenset({
    # Generic plant-domain nouns (never a specific plant name)
    "نبات", "نبتة", "نباتات", "عشبة", "اعشاب", "أعشاب",
    "زراعة", "ازرع", "أزرع", "يزرع", "تزرع", "زرع",
    "ري", "تربة", "سماد", "حصاد", "ضوء", "شمس",
    "اصيص", "موسم", "فلسطين",
    "مباشر", "مباشرة", "مباشره", "غير", "تحب", "يحب",
    "يحتاج", "تحتاج", "بحب", "بتحب", "بدها", "بده",
    # Care/task schedule words
    "مهام", "مهمة", "مهمه",
    "جدول",
    "عناية", "عنايه", "اعتني", "أعتني",
    "تذكير", "تذكيرات",
    "روتين",
    # Watering/fertilizing actions
    "سقي", "اسقي", "أسقي", "اسقيه", "اسقيها", "أسقيه", "أسقيها",
    "احصده", "احصدها", "أحصده", "أحصدها",
    "تسميد",
    # Generic question / info words
    "معلومات", "معلومه", "معلومة",
    "سؤال", "اسئلة", "أسئلة", "كيف",
    # Time / frequency words
    "متى", "متي", "أيمتى", "ايمتى", "يوم", "أيام", "ايام", "أسبوع", "اسبوع", "شهر",
    "مرة", "مره", "كم",
    # Benefit / use / info intent words (never plant names)
    "فوائد", "فوايد", "فائدة", "فايدة",
    "استخدامات", "استخدام", "استخداماته", "استخداماتها", "استخدامها",
    "تربته", "تربتها",
    "ملاحظة", "ملاحظات",
    "نصائح", "نصايح", "نصيحة",
    "وصف", "تعريف",
    # Descriptive adjectives
    "مناسب", "مناسبة", "مناسبون",
    "إضافية", "اضافية", "إضافي", "اضافي",
    "أفضل", "افضل",
    "أكثر", "اكثر",
    "أقل", "اقل",
    # Plural-of-شهر (months) + other time words
    "أشهر", "اشهر",
    "وقت", "أوقات", "اوقات",
    "فصل", "فصول",
    "فترة", "فترات",
    # Quantity / function words
    "كل",
    "نوع", "أنواع", "انواع",
    # Possessive / colloquial verbs
    "عندك", "عندي", "عنده", "عندها", "عندنا",
    "فيه", "فيها", "اله", "إله", "له", "لها",
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


# ──────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────

def _needs_plant_name(intents: list[str]) -> bool:
    """True when the detected intents require a specific plant name."""
    if not intents:
        return False
    return any(intent in _PLANT_SPECIFIC_INTENTS for intent in intents)


def _detect_unlisted_plant_mention(
    question: str,
    known_plant_names: list[str],
) -> Optional[str]:
    """
    Try to identify a specific entity (plant name) in the question that is NOT
    in our plant database.  Returns the candidate token if found, else None.
    """
    q = normalize_query(question).lower().strip()
    q = re.sub(r"[^0-9a-zA-Z\u0600-\u06FF]+", " ", q).strip()

    # Strip common question-opening phrases so only the subject remains
    for opener in _QUESTION_OPENERS:
        op_n = normalize(opener).lower()
        if q.startswith(op_n):
            q = q[len(op_n):].strip()
            break

    tokens = [t for t in q.split() if len(t) >= 3]
    if not tokens:
        return None

    # Take the first token that is not a generic domain keyword.
    # Apply prefix stripping before checking so that e.g. "العناية"
    # (stripped → "عناية") and "المهام" (stripped → "مهام") are both skipped.
    candidate = None
    for tok in tokens:
        tok_variants = strip_ar_word_prefix(tok)
        if not any(v in _DOMAIN_SKIP_TOKENS for v in tok_variants):
            candidate = tok
            break

    if not candidate:
        return None

    # If the candidate closely matches a known plant (score >= 85) it should have
    # been caught by extract_target_plant already → not a true plant_not_found.
    try:
        from rapidfuzz import process as _rfp, fuzz as _rff
        name_norms = [normalize(n).lower() for n in known_plant_names if n]
        if name_norms:
            best = _rfp.extractOne(candidate, name_norms, scorer=_rff.WRatio)
            if best and best[1] >= 85:
                return None
    except ImportError:
        pass

    return candidate


def _find_additional_unlisted_plant(
    question: str,
    known_plant_names: list[str],
    found_plant: str,
) -> Optional[str]:
    """
    After *found_plant* was successfully matched, look for a second plant-like
    token in the question that is NOT in the database.

    Returns the candidate token string (raw, not normalised) or None.

    Used to detect questions about two plants where one plant is found
    in the store but the second plant-like token is not in the database.
    """
    if not question or not known_plant_names or not found_plant:
        return None

    # Remove the matched plant name from the question text (with common prefixes)
    q_work = normalize_query(question).lower()
    q_work = re.sub(r"[^0-9a-zA-Z\u0600-\u06FF]+", " ", q_work)
    for prefix in ("", "ال", "ب", "و", "ف", "ك", "ل"):
        q_work = q_work.replace(prefix + normalize(found_plant).lower(), " ")
    q_work = q_work.strip()
    if not q_work:
        return None

    return _detect_unlisted_plant_mention(q_work, known_plant_names)


def build_safe_domain_fallback(suggested_name: Optional[str] = None) -> str:
    """Clear, polite no-hallucination fallback for domain questions."""
    msg = (
        "عذرًا، المعلومات المتاحة في قاعدة البيانات الحالية غير كافية للإجابة بدقة على هذا السؤال. "
        "إذا أحببت، يمكنني مساعدتك بسؤال نباتي آخر مدعوم بشكل أوضح."
    )
    if suggested_name:
        msg += f"\n\nهل تقصد: {suggested_name}؟"
    return msg


def suggest_plant_name(query: str) -> Optional[dict]:
    """
    Return the best fuzzy-matched plant name from the data store for *query*.

    Builds the normalised-name map dynamically from the live store so this
    function has no dependency on app.py globals.  Safe to call from both the
    /search endpoint and assistant_service.

    Returns ``{"suggested": <name>, "score": <int>}`` or ``None``.
    """
    from src.inference.plant_data_store import get_all_plant_names, is_store_loaded
    from src.utils.arabic import normalize_name

    if not is_store_loaded():
        return None

    all_names = get_all_plant_names()
    if not all_names:
        return None

    # Build normalised → original map (first occurrence wins)
    norm_map: dict[str, str] = {}
    for n in all_names:
        nn = normalize_name(n)
        if nn and nn not in norm_map:
            norm_map[nn] = n

    q_norm = normalize_name(query)
    norm_names = list(norm_map.keys())

    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(q_norm, norm_names, scorer=fuzz.WRatio)
    except ImportError:
        return None

    if not match:
        return None

    best_norm, score, _ = match
    if score >= 80:
        suggested = norm_map.get(best_norm, best_norm)
        return {"suggested": suggested, "score": int(score)}

    return None

