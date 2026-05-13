"""
context_utils.py – Retrieval helpers, direct-answer building, and context utilities.
يأخذ سؤال المستخدم → يحوله embedding → يبحث في FAISS → يرجّع أفضل كروت نباتات.
Public API
----------
# ── Retrieval ───────────────────
retrieve_relevant_context(…)           → list[dict]
get_best_retrieval_score(retrieved)    → float
is_retrieval_strong(retrieved)         → bool

# ── Plant name extraction ───────
extract_plant_name(card)               → str
extract_target_plant(question, …)      → str | None

# ── Excel-based answer builders ─
build_direct_answer(plant, intents)    → str
build_answer_draft(plant, intents)     → str
has_sufficient_data(plant, intents)    → bool

# ── Context helpers ─────────────
collect_user_and_climate_context(…)    → tuple[str|None, str|None]
suggest_alternative_plants_if_needed(…)→ list[str]
format_fallback_answer(retrieved, …)   → str
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.columns import get_column
from src.config.constants import is_junk
from src.config.settings import CONFIDENCE_THRESHOLD, SAFE_NO_ANSWER
from src.inference.intent_fields import (
    get_fields_for_intents,
    resolve_fields_for_plant,
    _INTENT_INDEX,
)
from src.utils.arabic import normalize, strip_ar_word_prefix

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 0.  Retrieval-strength helpers
# ─────────────────────────────────────────────────────────────────
def get_best_retrieval_score(retrieved: list[dict]) -> float:
    """Return the highest ``score`` from retrieved results (0.0 if empty)."""
    if not retrieved:
        return 0.0
    try:
        return max(r.get("score", 0.0) for r in retrieved)
    except (ValueError, TypeError):
        return 0.0


def is_retrieval_strong(
    retrieved: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """True when at least one result has score >= *threshold*."""
    if not retrieved:
        return False
    return get_best_retrieval_score(retrieved) >= threshold


# ─────────────────────────────────────────────────────────────────
# 1.  FAISS retrieval
# ─────────────────────────────────────────────────────────────────
def retrieve_relevant_context(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    cards: list[str],
    top_k: int = 3,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Encode *query* and return top-k plant cards above *threshold*."""
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        if score < threshold:
            continue
        results.append({"card": cards[idx], "score": round(float(score), 4)})
    return results


# ─────────────────────────────────────────────────────────────────
# 2.  Plant name extraction
# ─────────────────────────────────────────────────────────────────

# Base-form tokens that must NEVER be treated as plant name candidates.
# Checked (after prefix stripping) inside extract_target_plant fuzzy step
# and in _detect_unlisted_plant_mention in app.py.
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


# ─────────────────────────────────────────────────────────────────
# 3.  Excel-based answer builders
# ─────────────────────────────────────────────────────────────────

# _g is now imported from src.config.columns.get_column
# _is_junk is now imported from src.config.constants.is_junk
# Both use the shared COLUMN_ALIASES and EMPTY_MARKERS.
_g = get_column
_is_junk = is_junk

_EN_CANONICAL_OVERRIDES: dict[str, str] = {
    "arabic_name_primary": "english_name_primary",
    "short_summary": "short_summary_en",
    "watering_rule_text": "watering_info_en",
    "light_level": "light_info_en",
    "soil_texture_preference": "soil_info_en",
    "fertilizer_notes_ar": "fertilizer_notes_en",
    "harvest_method": "harvest_info_en",
    "care_steps_json": "planting_steps_en",
    "planting_steps_ar": "planting_steps_en",
    "seed_care_ar": "seed_care_en",
    "care_info_ar": "care_info_en",
    "beginner_tips": "care_info_en",
    "uses_info_ar": "uses_info_en",
    "humidity_notes_ar": "humidity_notes_en",
    "adjustment_tip_ar": "adjustment_tip_en",
    "planting_months_pal": "planting_note_en",
    "season_notes_pal": "planting_note_en",
}

_FIELD_LABELS_EN: dict[str, str] = {
    "arabic_name_primary": "Arabic name",
    "english_name_primary": "English name",
    "scientific_name": "Scientific name",
    "short_summary": "Description",
    "short_summary_en": "Description",
    "category": "Category",
    "difficulty_level": "Difficulty",
    "watering_need": "Watering need",
    "watering_rule_text": "Watering",
    "watering_interval_days_min": "Watering interval (days)",
    "light_level": "Light",
    "temperature_optimal_min_c": "Minimum temperature",
    "temperature_optimal_max_c": "Maximum temperature",
    "temperature_sensitivity": "Temperature sensitivity",
    "soil_texture_preference": "Soil",
    "soil_moisture_min": "Minimum soil moisture",
    "soil_moisture_max": "Maximum soil moisture",
    "fertilizer_type": "Fertilizer type",
    "fertilizer_stage": "Fertilizer stage",
    "fertilizer_frequency_days": "Fertilizing frequency (days)",
    "fertilizer_notes_ar": "Fertilizing notes",
    "harvest_method": "Harvest",
    "harvest_after_days_min": "Minimum days to harvest",
    "harvest_after_days_max": "Maximum days to harvest",
    "care_steps_json": "Planting steps",
    "propagation_method_primary": "Propagation method",
    "germination_days_min": "Germination days",
    "seed_to_seedling_days": "Seed to seedling days",
    "plant_spacing_cm_min": "Minimum spacing (cm)",
    "plant_spacing_cm_max": "Maximum spacing (cm)",
    "seed_care_ar": "Seed care",
    "establishment_days": "Establishment days",
    "care_info_ar": "Care information",
    "beginner_tips": "Care tips",
    "uses_info_ar": "Uses",
    "humidity_preference": "Humidity preference",
    "air_humidity_min": "Minimum air humidity (%)",
    "air_humidity_max": "Maximum air humidity (%)",
    "humidity_notes_ar": "Humidity notes",
    "adjustment_tip_ar": "Adjustment tip",
    "planting_months_pal": "Planting months and notes",
    "season_notes_pal": "Season",
    "task_type": "Task",
    "interval_days": "Every (days)",
}

_TASK_TYPE_EN: dict[str, str] = {
    "WATERING": "Watering",
    "FERTILIZING": "Fertilizing",
    "PRUNING": "Pruning",
    "HARVEST": "Harvest",
    "CARE": "Care",
    "watering": "Watering",
    "fertilizing": "Fertilizing",
    "pruning": "Pruning",
    "harvest": "Harvest",
    "care": "Care",
}


def _is_english(language: str) -> bool:
    return (language or "").lower().startswith("en")


def _g_lang(d: dict, col: str, language: str = "ar") -> Optional[str]:
    """Resolve a column in the selected answer language."""
    if _is_english(language):
        override = _EN_CANONICAL_OVERRIDES.get(col)
        if override:
            value = _g(d, override)
            if value:
                return _clean_english_text(value)
        # Neutral numeric/code fields can be reused; Arabic prose fields cannot.
        if col.endswith("_ar") or col in {
            "short_summary", "watering_rule_text", "light_level",
            "soil_texture_preference", "harvest_method", "care_steps_json",
            "planting_steps_ar", "beginner_tips", "uses_info_ar",
            "planting_months_pal", "season_notes_pal",
        }:
            return None
        return _clean_english_text(_g(d, col))
    return _g(d, col)


def _label_for_field(col: str, fallback_label: str, language: str) -> str:
    if _is_english(language):
        return _FIELD_LABELS_EN.get(col, fallback_label)
    return fallback_label


# ── Text cleaning for natural Arabic output ───────────────────────

_PAREN_EN_RE = re.compile(r'\([A-Za-z][A-Za-z\s./%°,;:\'"\x2d\u201c\u201d\u2018\u2019]+\)')

# Matches sequences of English words (including hyphens/slashes between words).
# Used by _strip_inline_english to remove English text embedded in Arabic strings.
_EN_CHUNK_RE = re.compile(r'[A-Za-z][A-Za-z\-\'\/]*(?:\s+[A-Za-z][A-Za-z\-\'\/]*)+|[A-Za-z]{3,}')
_EN_CHAR_RE  = re.compile(r'[A-Za-z]')
_AR_CHAR_RE  = re.compile(r'[\u0600-\u06FF]')
_EN_OR_DIGIT_RE = re.compile(r'[A-Za-z0-9]')


def _clean_text(s: str) -> str:
    """Strip technical artefacts: |, *, English parens, junk substrings, excess whitespace."""
    if not s:
        return s
    s = s.replace(" | ", "، ").replace("|", "، ")
    s = s.replace(" + ", " و")
    s = s.replace("• ", "").replace("* ", "")
    s = s.replace("\\n", " ").replace("\n", " ")
    s = _PAREN_EN_RE.sub("", s)
    # Remove junk placeholders embedded in longer text
    s = re.sub(r'مش موجود|غير متوفر|غير متوفرة|غير معروف|غير مذكور', '', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = s.replace("..", ".").replace("،.", ".").replace(".،", "،")
    return s


def _clean_english_text(value: str) -> Optional[str]:
    """Return an English-only display value, or None if no usable English remains."""
    if value is None:
        return None
    text = _clean_text(str(value))
    if not text:
        return None

    text = re.sub(r'(?<=\d)\s*سم(?![\u0600-\u06FF])', ' cm', text)
    text = re.sub(r'(?<=\d)\s*سنتيمتر(?![\u0600-\u06FF])', ' cm', text)
    text = re.sub(r'(?<=\d)\s*(?:إنش|انش|بوصة)(?![\u0600-\u06FF])', ' inch', text)
    text = re.sub(r'(?<![\u0600-\u06FF])نعم(?![\u0600-\u06FF])', 'yes', text)
    text = re.sub(r'(?<![\u0600-\u06FF])لا(?![\u0600-\u06FF])', 'no', text)
    text = text.replace("°م", "C").replace("م°", "C")

    text = re.sub(r'\bsource data\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u0600-\u06FF]+', ' ', text)
    text = re.sub(r'\s+([,.;:!?%)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace(" ,", ",").replace("..", ".").strip(" -،,.;:")

    if not text or _AR_CHAR_RE.search(text):
        return None
    if not _EN_OR_DIGIT_RE.search(text):
        return None
    return text


def _strip_inline_english(text: str) -> str:
    """Remove inline English word sequences from Arabic text.

    Removes: multi-word English phrases and standalone English words ≥3 chars.
    Preserves: degree symbols (°C/°F), % numbers, and short abbreviations like pH.
    """
    if not text:
        return text
    result = _EN_CHUNK_RE.sub('', text)
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r'\s+([،,.:؛])', r'\1', result)
    # Remove orphaned leading/trailing punctuation and stray digits at boundaries
    result = re.sub(r'^[\s.،,]+', '', result)
    result = re.sub(r'[\s.،,]+$', '', result)
    # Clean up double periods from stripping
    result = re.sub(r'\.\.+', '.', result)
    result = re.sub(r'\s{2,}', ' ', result)
    return result.strip()


def _has_too_much_english(text: str, threshold: int = 10) -> bool:
    """Return True if *text* contains more than *threshold* Latin characters."""
    return bool(text) and len(_EN_CHAR_RE.findall(text)) > threshold


_MAX_DIRECT_SENTENCES = 4


def _postprocess_direct(text: str) -> str:
    """Final cleanup for direct answers: paragraph form, no technical artefacts."""
    if not text:
        return text
    text = _clean_text(text)
    # Remove stray bullet markers
    text = re.sub(r'(?:^|\n)\s*[-•]\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Per-intent sentence builders ──────────────────────────────────

def _sentences_watering(p: str, d: dict) -> list[str]:
    rule     = _g(d, "watering_rule_text")         # → wateringInfoAr
    need     = _g(d, "watering_need")              # → wateringNeed
    interval = _g(d, "watering_interval_days_min") # → wateringIntervalDays (unified)

    if not rule and not need and not interval:
        return []

    # Header: prefer interval for precision
    if interval:
        header = f"حسب بيانات غرسة، {p} يحتاج ري كل {interval} أيام."
    elif need:
        header = f"حسب بيانات غرسة، {p} يحتاج ري {_clean_text(need)}."
    else:
        header = f"حسب بيانات غرسة، {p} يحتاج للري بانتظام."

    lines = [header]
    if rule:
        # Strip English inline text, then take first 2 Arabic sentences only
        clean_rule = _strip_inline_english(_clean_text(rule))
        if clean_rule and clean_rule != header:
            sents = [s.strip() for s in re.split(r'(?<=[.،؟!])\s+', clean_rule) if s.strip()]
            short_rule = ' '.join(sents[:2])
            if short_rule:
                lines.append(f"معلومة الري: {short_rule}")
    return ["\n".join(lines)]


def _sentences_light(p: str, d: dict) -> list[str]:
    # lightInfoAr (Care_Details) via light_level alias — full text block
    info = _g(d, "light_level")  # → lightInfoAr
    if info:
        clean_info = _strip_inline_english(_clean_text(info))
        if clean_info:
            return [f"الضوء المناسب لـ{p}:\n{clean_info}"]
    return []


def _sentences_temperature(p: str, d: dict) -> list[str]:
    out: list[str] = []
    t_min = _g(d, "temperature_optimal_min_c")  # → minTemp
    t_max = _g(d, "temperature_optimal_max_c")  # → maxTemp
    sens  = _g(d, "temperature_sensitivity")    # → temperatureSensitivity

    if t_min and t_max:
        out.append(f"تتراوح درجة الحرارة المثالية لـ{p} بين {t_min}°م و{t_max}°م.")
    elif t_min:
        out.append(f"تبدأ الحرارة المثالية لـ{p} من {t_min}°م.")
    elif t_max:
        out.append(f"لا تتجاوز الحرارة المثالية لـ{p} {t_max}°م.")
    if sens:
        out.append(f"حساسيته لدرجة الحرارة: {_clean_text(sens)}.")
    return out


def _sentences_soil(p: str, d: dict) -> list[str]:
    # soilInfoAr (Care_Details) via soil_texture_preference alias — full text block
    info      = _g(d, "soil_texture_preference")  # → soilInfoAr
    moist_min = _g(d, "soil_moisture_min")        # → soilMoistureMin
    moist_max = _g(d, "soil_moisture_max")        # → soilMoistureMax

    if not info and not moist_min:
        return []

    lines = [f"التربة المناسبة لـ{p}:"]
    if info:
        clean_info = _strip_inline_english(_clean_text(info))
        if clean_info:
            lines.append(clean_info)
    if moist_min and moist_max:
        lines.append(f"رطوبة التربة المناسبة: {moist_min}%–{moist_max}%.")
    elif moist_min:
        lines.append(f"لا تقل رطوبة التربة عن {moist_min}%.")
    return ["\n".join(lines)] if len(lines) > 1 else []


def _sentences_fertilizing(p: str, d: dict) -> list[str]:
    out: list[str] = []
    ftype = _g(d, "fertilizer_type")           # → fertilizerType
    stage = _g(d, "fertilizer_stage")           # → fertilizerStage
    freq  = _g(d, "fertilizer_frequency_days")  # → fertilizerFrequencyDays
    notes = _g(d, "fertilizer_notes_ar")        # → fertilizerNotesAr

    if ftype and freq:
        out.append(f"يُفضَّل استخدام {_clean_text(ftype)} مرة كل {freq} يومًا.")
    elif ftype:
        out.append(f"يُفضَّل استخدام {_clean_text(ftype)} لتسميد {p}.")
    elif freq:
        out.append(f"يُسمَّد {p} كل {freq} يومًا.")
    if stage:
        out.append(f"مرحلة التسميد المناسبة: {_clean_text(stage)}.")
    if notes:
        out.append(_clean_text(notes))
    return out


def _sentences_season(p: str, d: dict) -> list[str]:
    """
    Return Month_Plants data ONLY as a structured bullet list.

    Intent: season
    Source: month_plants list (monthName, season, plantingNoteAr).
    Fallback: planting_months_pal scalar field.

    Rules:
    - NEVER mix in plantingSteps, soil, spacing, or other fields.
    - Each month on its own bullet line.
    - If no month_plants and no fallback scalar → return [].
    """
    out: list[str] = []
    month_plants = d.get("month_plants")
    if month_plants:
        lines: list[str] = [f"حسب بيانات غرسة، أشهر زراعة {p} هي:"]
        for item in month_plants:
            month_name = item.get("monthName", "")
            season_val = item.get("season", "")
            note       = item.get("plantingNoteAr", "")
            if not month_name:
                continue
            entry = f"- {month_name}"
            if season_val:
                entry += f" ({season_val})"
            if note:
                clean_note = _clean_text(str(note))
                if clean_note:
                    entry += f": {clean_note}"
            lines.append(entry)
        if len(lines) > 1:
            out.append("\n".join(lines))
        return out

    # Fallback: plantingNoteAr scalar via planting_months_pal alias
    plant_m = _g(d, "planting_months_pal")  # → plantingNoteAr
    if plant_m:
        out.append(f"حسب بيانات غرسة، يُزرع {p} في: {_clean_text(plant_m)}.")
    return out


def _sentences_harvest_storage(p: str, d: dict) -> list[str]:
    out: list[str] = []
    # harvestInfoAr (Care_Details) via harvest_method alias — full text block
    method = _g(d, "harvest_method")          # → harvestInfoAr
    h_min  = _g(d, "harvest_after_days_min")  # → daysToHarvestMin
    h_max  = _g(d, "harvest_after_days_max")  # → daysToHarvestMax
    # NOTE: drying/storage fields no longer exist in the new Excel.

    if h_min and h_max and h_min != h_max:
        out.append(f"يُحصد {p} بعد {h_min}–{h_max} يومًا من الزراعة.")
    elif h_min:
        out.append(f"يُحصد {p} بعد حوالي {h_min} يومًا من الزراعة.")
    if method:
        out.append(_clean_text(method))
    return out


def _sentences_pests_diseases(p: str, d: dict) -> list[str]:
    out: list[str] = []
    # common_pests / common_diseases removed (plants_pests_diseases sheet gone).
    # Fallback to careInfoAr which may contain pest/disease guidance.
    care = _g(d, "care_info_ar")  # → careInfoAr
    if care:
        out.append(_clean_text(care))
    return out


def _sentences_beginner(p: str, d: dict) -> list[str]:
    out: list[str] = []
    diff = _g(d, "difficulty_level")  # → difficultyLevel
    tips = _g(d, "beginner_tips")     # → careInfoAr (proxy)
    # time_commitment no longer in new Excel — removed.

    if diff:
        out.append(f"زراعة {p} تُعتبر {diff}.")
    if tips:
        out.append(_clean_text(tips))
    return out


def _sentences_container(p: str, d: dict) -> list[str]:
    out: list[str] = []
    # container_possible / pot dimensions not in new Excel.
    # Use plantingStepsAr as the most relevant guidance, with soilInfoAr as fallback.
    steps     = _g(d, "planting_steps_ar")      # → plantingStepsAr
    soil_info = _g(d, "soil_texture_preference") # → soilInfoAr

    if steps:
        out.append(_clean_text(steps))
    elif soil_info:
        out.append(_clean_text(soil_info))
    return out


def _sentences_planting(p: str, d: dict) -> list[str]:
    out: list[str] = []
    # plantingStepsAr (Care_Details) via care_steps_json alias — primary source
    steps    = _g(d, "care_steps_json")              # → plantingStepsAr
    method   = _g(d, "propagation_method_primary")   # → plantingMethod
    germ     = _g(d, "germination_days_min")         # → germinationDays (unified)
    spacing  = _g(d, "plant_spacing_cm")              # → plantSpacingCmMin
    seed     = _g(d, "seed_care_ar")                  # → seedCareInstructionsAr
    est_days = _g(d, "establishment_days")            # → establishmentDays
    s2s_days = _g(d, "seed_to_seedling_days")         # → seedToSeedlingDays
    # transplanting_ok no longer in new Excel — removed.

    if steps:
        out.append(_clean_text(steps))
        return out   # steps cover everything; skip individual fields

    # Fallback: build from individual fields
    if method:
        out.append(f"يُكثَّر {p} عن طريق {_clean_text(method)}.")
    if germ:
        out.append(f"تنبت البذور في غضون {germ} يومًا.")
    if spacing:
        out.append(f"يُوصى بترك مسافة {_clean_text(spacing)} سم بين النباتات.")
    if seed:
        out.append(_clean_text(seed))
    if s2s_days:
        out.append(f"يصل النبات من البذرة إلى الشتلة في غضون {s2s_days} يوم.")
    if est_days:
        out.append(f"تستغرق مرحلة التأسيس نحو {est_days} يوم.")
    return out


def _sentences_general_summary(p: str, d: dict) -> list[str]:
    out: list[str] = []
    # shortDescriptionAr (Plants) via short_summary alias
    summary  = _g(d, "short_summary")   # → shortDescriptionAr
    category = _g(d, "category")        # → category (same name)
    uses     = _g(d, "uses_info_ar")    # → usesInfoAr (Care_Details)
    diff     = _g(d, "difficulty_level") # → difficultyLevel
    # growth_habit / life_cycle / fragrance_level / edible_parts removed from Excel.

    if summary:
        out.append(_clean_text(summary))
        extras: list[str] = []
        s_low = summary.lower()
        if category and category.lower() not in s_low:
            extras.append(f"يُصنَّف ضمن {_clean_text(category)}")
        if diff and diff.lower() not in s_low:
            extras.append(f"مستوى صعوبة زراعته {diff}")
        if extras:
            out.append("، ".join(extras) + ".")
    else:
        parts: list[str] = [p]
        if category:
            parts.append(f"يُصنَّف ضمن {_clean_text(category)}")
        if diff:
            parts.append(f"مستوى صعوبته {diff}")
        out.append("، ".join(parts) + ".")

    if uses:
        out.append(_clean_text(uses))
    return out


def _sentences_uses(p: str, d: dict) -> list[str]:
    uses = _g(d, "uses_info_ar")
    if not uses:
        return []
    return [_clean_text(uses)]




def _sentences_humidity(p: str, d: dict) -> list[str]:
    out: list[str] = []
    pref    = _g(d, "humidity_preference")  # → humidityPreference
    air_min = _g(d, "air_humidity_min")     # → airHumidityMin
    air_max = _g(d, "air_humidity_max")     # → airHumidityMax
    notes   = _g(d, "humidity_notes_ar")    # → humidityNotesAr

    if pref:
        out.append(f"يُفضِّل {p} رطوبة {_clean_text(pref)}.")
    if air_min and air_max:
        out.append(f"تتراوح رطوبة الهواء المثالية بين {air_min}% و{air_max}%.")
    elif air_min:
        out.append(f"يحتاج {p} إلى رطوبة هواء لا تقل عن {air_min}%.")
    elif air_max:
        out.append(f"يُفضَّل ألا تتجاوز رطوبة الهواء {air_max}%.")
    if notes:
        out.append(_clean_text(notes))
    return out


def _sentences_suitability(p: str, d: dict) -> list[str]:
    out: list[str] = []
    suitability_tips = d.get("suitability_tips", [])
    if suitability_tips:
        tips = [
            _clean_text(str(item.get("adjustmentTipAr", "")))
            for item in suitability_tips
            if item.get("adjustmentTipAr")
        ]
        if tips:
            out.append(f"نصائح الملاءمة لـ{p}: " + " ".join(tips) + ".")
    return out


# Arabic display names for English task type codes from the Tasks sheet
_TASK_TYPE_AR: dict[str, str] = {
    "WATERING":    "الري",
    "FERTILIZING": "التسميد",
    "PRUNING":     "التقليم",
    "HARVEST":     "الحصاد",
    "CARE":        "العناية",
    "watering":    "الري",
    "fertilizing": "التسميد",
    "pruning":     "التقليم",
    "harvest":     "الحصاد",
    "care":        "العناية",
}


def _sentences_tasks(p: str, d: dict) -> list[str]:
    """Build a formatted bullet-list of care tasks from the Tasks sheet data."""
    out: list[str] = []
    tasks = d.get("tasks", [])
    if not tasks:
        return out

    task_lines: list[str] = []
    for item in tasks:
        t_type     = str(item.get("taskType", "")).strip()
        t_interval = item.get("intervalDays", "")
        if not t_type:
            continue
        # Map English codes → Arabic; keep unknown values as-is
        t_ar = _TASK_TYPE_AR.get(t_type, _TASK_TYPE_AR.get(t_type.upper(), t_type))
        if t_interval:
            task_lines.append(f"- {t_ar}: كل {t_interval} أيام")
        else:
            task_lines.append(f"- {t_ar}")

    if task_lines:
        header = f"مهام العناية لـ {p} حسب بيانات غرسة:"
        out.append(header + "\n" + "\n".join(task_lines))
    return out


def _sentences_care_summary(p: str, d: dict) -> list[str]:
    """Build a structured care template directly from Excel data (no LLM)."""
    lines = [f"عناية {p} حسب بيانات غرسة:"]

    # Watering
    interval     = _g(d, "watering_interval_days_min")
    watering_info = _g(d, "watering_rule_text")
    if interval:
        lines.append(f"- الري: كل {interval} أيام")
    elif watering_info:
        lines.append(f"- الري: {_clean_text(watering_info)[:100]}")

    # Light
    light = _g(d, "light_level")
    if light:
        lines.append(f"- الضوء: {_clean_text(light)[:100]}")

    # Soil
    soil = _g(d, "soil_texture_preference")
    if soil:
        lines.append(f"- التربة: {_clean_text(soil)[:100]}")

    # Care notes (careInfoAr preferred, fertilizer notes as fallback)
    care_notes = _g(d, "care_info_ar") or _g(d, "fertilizer_notes_ar")
    if care_notes:
        lines.append(f"- ملاحظات العناية: {_clean_text(care_notes)[:120]}")

    if len(lines) == 1:  # only the header — no data available
        return []
    return ["\n".join(lines)]


def _sentences_plant_names(p: str, d: dict) -> list[str]:
    out: list[str] = []
    name_ar  = _g(d, "arabic_name_primary")   # → nameAr  (Plants)
    name_en  = _g(d, "english_name_primary")  # → nameEn  (Plants)
    name_sci = _g(d, "scientific_name")       # → nameScientific (Plants)
    category = _g(d, "category")              # → category (Plants)

    parts = []
    if name_ar:
        parts.append(f"الاسم العربي: {name_ar}")
    if name_en:
        parts.append(f"الاسم الإنجليزي: {name_en}")
    if name_sci:
        parts.append(f"الاسم العلمي: {name_sci}")
    if parts:
        out.append("، ".join(parts) + ".")
    if category:
        out.append(f"تُصنَّف {p} ضمن {_clean_text(category)}.")
    return out

def _sentences_germination(p: str, d: dict) -> list[str]:
    """Strict single-field answer: germinationDays only."""
    germ = _g(d, "germination_days_min")
    if not germ:
        return []
    return [f"حسب بيانات غرسة، {p} يحتاج حوالي {germ} أيام للإنبات."]


def _sentences_spacing(p: str, d: dict) -> list[str]:
    """Strict dual-field answer: plantSpacingCmMin / plantSpacingCmMax only."""
    sp_min = _g(d, "plant_spacing_cm_min")
    sp_max = _g(d, "plant_spacing_cm_max")
    sp_min_s = str(sp_min).strip() if sp_min is not None else ""
    sp_max_s = str(sp_max).strip() if sp_max is not None else ""
    if sp_min_s and sp_max_s and sp_min_s != sp_max_s:
        return [f"حسب بيانات غرسة، المسافة المناسبة بين نباتات {p} هي {sp_min_s}–{sp_max_s} سم."]
    elif sp_min_s:
        return [f"حسب بيانات غرسة، المسافة المناسبة بين نباتات {p} هي {sp_min_s} سم."]
    elif sp_max_s:
        return [f"حسب بيانات غرسة، المسافة المناسبة بين نباتات {p} هي {sp_max_s} سم."]
    return []


# Dispatch map: intent_name → sentence-builder function
_NATURALIZERS: dict = {
    "watering":        _sentences_watering,
    "light":           _sentences_light,
    "temperature":     _sentences_temperature,
    "soil":            _sentences_soil,
    "fertilizing":     _sentences_fertilizing,
    "season":          _sentences_season,
    "harvest_storage": _sentences_harvest_storage,
    "pests_diseases":  _sentences_pests_diseases,
    "beginner":        _sentences_beginner,
    "container":       _sentences_container,
    "planting":        _sentences_planting,
    "general_summary": _sentences_general_summary,
    "humidity":        _sentences_humidity,
    "suitability":     _sentences_suitability,
    "tasks":           _sentences_tasks,
    "care_summary":    _sentences_care_summary,
    "plant_names":     _sentences_plant_names,
    "uses":            _sentences_uses,
    "germination":     _sentences_germination,
    "spacing":         _sentences_spacing,
}


def has_sufficient_data(
    plant_data: dict,
    intents: list[str],
    min_fields: int = 1,
    language: str = "ar",
) -> bool:
    """
    True when the plant profile has at least *min_fields* non-empty
    values among the columns required by the given intents.

    Special case: ``tasks`` intent checks the Tasks list directly
    (``plant_data["tasks"]``) because task data is stored as a nested
    list, not as flat columns that resolve_column can reach.
    """
    if plant_data is None:
        return False

    # season intent: data lives in plant_data["month_plants"] list, not columns
    if "season" in intents:
        if bool(plant_data.get("month_plants")):
            if not _is_english(language):
                return True
            return any(item.get("plantingNoteEn") or item.get("monthName") for item in plant_data.get("month_plants", []))
        # Fallback: check planting_months_pal scalar field
        other_intents = [i for i in intents if i != "season"]
        if not other_intents:
            # Only season intent → check scalar fallback
            from src.config.columns import resolve_column
            scalar = _g_lang(plant_data, "planting_months_pal", language)
            return bool(scalar)
        fields = get_fields_for_intents(other_intents)
        resolved = [
            _g_lang(plant_data, col, language)
            for col in fields
        ]
        return len([v for v in resolved if v]) >= min_fields

    # tasks intent: data lives in plant_data["tasks"] list, not columns
    if "tasks" in intents:
        if bool(plant_data.get("tasks")):
            return True
        # If other intents are also present (e.g. care_summary + tasks),
        # fall through so the other intent can still produce an answer.
        other_intents = [i for i in intents if i != "tasks"]
        if not other_intents:
            return False  # tasks is the only intent → no data
        fields = get_fields_for_intents(other_intents)
        resolved = [
            _g_lang(plant_data, col, language)
            for col in fields
        ]
        return len([v for v in resolved if v]) >= min_fields

    fields = get_fields_for_intents(intents)
    if _is_english(language):
        resolved = [_g_lang(plant_data, col, language) for col in fields]
        return len([v for v in resolved if v]) >= min_fields
    resolved = resolve_fields_for_plant(plant_data, fields)
    return len(resolved) >= min_fields


# ─────────────────────────────────────────────────────────────────
# 3b.  Comparison helpers  (comparison intent)
# ─────────────────────────────────────────────────────────────────

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


# Criterion → (field_canonical, arabic_label, higher_is_better, description_ar)
_COMPARISON_CRITERIA_MAP: dict[str, tuple[str, str, bool, str]] = {
    "difficulty":  ("difficulty_level",           "مستوى الصعوبة",  False, "الأسهل للمبتدئين"),
    "watering":    ("watering_interval_days_min",  "فترة الري (أيام)", False, "يحتاج ري أكثر تكرارًا"),
    "heat":        ("temperature_optimal_max_c",   "أقصى حرارة (°C)", True,  "يتحمل حرارة أعلى"),
    "cold":        ("temperature_optimal_min_c",   "أدنى حرارة (°C)", False, "يتحمل برودة أشد"),
    "harvest":     ("harvest_after_days_min",       "أيام الحصاد",   False, "أسرع حصادًا"),
}
_COMPARISON_CRITERIA_MAP_EN: dict[str, tuple[str, str, bool, str]] = {
    "difficulty": ("difficulty_level", "difficulty", False, "is easier for beginners"),
    "watering": ("watering_interval_days_min", "watering interval", False, "needs more frequent watering"),
    "heat": ("temperature_optimal_max_c", "maximum temperature", True, "tolerates higher heat"),
    "cold": ("temperature_optimal_min_c", "minimum temperature", False, "tolerates colder conditions"),
    "harvest": ("harvest_after_days_min", "days to harvest", False, "is faster to harvest"),
}

# Difficulty level: numeric rank for comparison (lower = easier)
_DIFFICULTY_RANK: dict[str, int] = {
    "EASY": 1, "MEDIUM": 2, "HARD": 3,
    "easy": 1, "medium": 2, "hard": 3,
    "سهل": 1,  "متوسط": 2,  "صعب": 3,
    "1": 1, "2": 2, "3": 3,
}
_DIFFICULTY_AR: dict[str, str] = {
    "EASY": "سهل", "MEDIUM": "متوسط", "HARD": "صعب",
    "easy": "سهل", "medium": "متوسط", "hard": "صعب",
}


def _detect_comparison_criterion(question: str) -> str:
    """
    Return one of: "difficulty" | "watering" | "heat" | "cold" | "harvest" | "unknown".
    Checked from most-specific to least-specific.
    """
    q = normalize(question).lower()

    # harvest
    if any(p in q for p in (
        "اسرع حصاد", "حصاد اسرع", "اقرب حصاد", "حصاد اقرب", "يحصد اسرع",
        "اسرع للحصاد", "أسرع حصادا", "اقل ايام حصاد",
        "faster harvest", "fastest harvest", "harvest faster", "quickest harvest",
        "least days to harvest", "fewest days to harvest",
    )):
        return "harvest"

    # cold tolerance
    if any(p in q for p in (
        "يتحمل برد", "يتحمل الصقيع", "يتحمل البرد", "برودة", "الصقيع",
        "اكثر تحملا للبرد", "أكثر تحملا للبرد", "يتحمل برودة",
        "cold tolerant", "tolerates cold", "tolerate cold", "frost tolerant",
        "handles cold",
    )):
        return "cold"

    # heat tolerance
    if any(p in q for p in (
        "يتحمل حرارة", "يتحمل الحر", "حرارة اكثر", "اكثر تحملا للحر",
        "اكثر تحملا للحرارة", "أكثر تحملا للحرارة", "يتحمل حر",
        "heat tolerant", "tolerates heat", "tolerate heat", "handles heat",
        "higher temperature",
    )):
        return "heat"

    # watering – expanded to catch "اكثر حاجة للماء", "أكثر ري", "الأقل ريًا", etc.
    if any(p in q for p in (
        "يحتاج ري", "ري اكثر", "سقي اكثر", "اسقي اكثر",
        "يحتاج ماء", "يحتاج مياه", "حاجة ري",
        "من ناحية الري", "ناحية الري", "معيار الري",
        # new patterns
        "حاجة للماء", "حاجه للماء", "احتياج للماء", "احتاج ماء",
        "اكثر ري", "اكثر ريا", "اقل ري", "اقل ريا",
        "ري اقل", "اكثر سقيا", "اقل سقيا",
        "اكثر ماء", "اقل ماء", "اكثر مياه", "اقل مياه",
        "اكثر حاجه", "اكثر حاجة",
        "الاقل ريا", "الاكثر ريا",
        "يحتاج للماء", "يحتاج للري",
        "needs more water", "need more water", "more watering", "less watering",
        "water more", "water less", "watering frequency", "how often",
    )):
        return "watering"

    # difficulty
    if any(p in q for p in (
        "اسهل", "اصعب", "مبتدئين", "مستوى الصعوبة", "الاسهل",
        "اسهل للمبتدئين", "الاصعب", "ادق",
        "easier", "easy", "harder", "difficult", "beginner", "beginners",
        "difficulty",
    )):
        return "difficulty"

    return "unknown"


# Public alias so app.py can import and use it before calling build_comparison_answer
detect_comparison_criterion = _detect_comparison_criterion


def build_comparison_answer(
    plant_names: list[str],
    plant_data_map: dict[str, dict],
    question: str = "",
    language: str = "ar",
) -> tuple[str, str, str, list[str], list[str]]:
    """
    Build a data-only comparison answer between two or more plants.

    Returns
    -------
    answer          : str   – full Arabic answer to show the user
    conclusion      : str   – one-line conclusion (for logging)
    criterion       : str   – detected criterion code
    used_fields     : list  – field names that had data
    missing_fields  : list  – field names where data was absent
    """
    from src.config.columns import resolve_column

    criterion = _detect_comparison_criterion(question)

    # Unknown criterion → ask the user to be more specific
    if criterion == "unknown":
        if _is_english(language):
            answer = (
                "Please choose a comparison criterion:\n"
                "- easier for beginners\n"
                "- watering frequency\n"
                "- faster harvest\n"
                "- better heat tolerance\n"
                "- better cold tolerance"
            )
        else:
            answer = (
                "حدد معيار المقارنة لأقدر أساعدك:\n"
                "- الأسهل للمبتدئين\n"
                "- الأقل ريًا (أو الأكثر احتياجًا للماء)\n"
                "- الأسرع حصادًا\n"
                "- الأكثر تحملًا للحرارة\n"
                "- الأكثر تحملًا للبرودة"
            )
        return answer, "unknown_criterion", "unknown", [], []

    criteria_map = _COMPARISON_CRITERIA_MAP_EN if _is_english(language) else _COMPARISON_CRITERIA_MAP
    field_col, field_label, higher_is_better, description = criteria_map[criterion]

    rows: list[str] = []          # "- [plant_name]: [value]" lines
    values: dict[str, object] = {}  # raw numeric/string values per plant
    used_fields: list[str] = []
    missing_fields: list[str] = []

    for plant in plant_names:
        pdata = plant_data_map.get(plant, {})
        raw = resolve_column(pdata, field_col)

        if criterion == "difficulty":
            display = str(raw).strip() if raw else None
            if not _is_english(language):
                display = _DIFFICULTY_AR.get(str(raw).strip(), str(raw).strip()) if raw else None
            missing_text = "data not available" if _is_english(language) else "البيانات غير متوفرة"
            rows.append(f"- {plant}: {display}" if display else f"- {plant}: {missing_text}")
            values[plant] = _DIFFICULTY_RANK.get(str(raw).strip().upper(), None) if raw else None
        else:
            if raw is not None:
                try:
                    values[plant] = float(raw)
                    if criterion == "watering":
                        rows.append(
                            f"- {plant}: every {int(values[plant])} days"
                            if _is_english(language)
                            else f"- {plant}: كل {int(values[plant])} أيام"
                        )
                    elif criterion in ("heat", "cold"):
                        rows.append(f"- {plant}: {values[plant]:.0f}°C")
                    else:  # harvest
                        rows.append(
                            f"- {plant}: {int(values[plant])} days"
                            if _is_english(language)
                            else f"- {plant}: {int(values[plant])} يوم"
                        )
                except (ValueError, TypeError):
                    rows.append(f"- {plant}: {'data not available' if _is_english(language) else 'البيانات غير متوفرة'}")
                    values[plant] = None
            else:
                rows.append(f"- {plant}: {'data not available' if _is_english(language) else 'البيانات غير متوفرة'}")
                values[plant] = None

        if values.get(plant) is not None:
            used_fields.append(field_col)
        else:
            missing_fields.append(f"{plant}:{field_col}")

    # Build conclusion
    valid_vals = {p: v for p, v in values.items() if v is not None}
    conclusion = ""
    if len(valid_vals) >= 2:
        if higher_is_better:
            winner = max(valid_vals, key=lambda p: valid_vals[p])  # type: ignore[arg-type]
        else:
            winner = min(valid_vals, key=lambda p: valid_vals[p])  # type: ignore[arg-type]

        if criterion == "watering":
            conclusion = (
                f"Result: {winner} needs more frequent watering according to Gharsa data. "
                f"(Fewer days means more frequent watering.)"
                if _is_english(language)
                else (
                    f"النتيجة: {winner} يحتاج ري أكثر تكرارًا "
                    f"حسب بيانات غرسة. "
                    f"(الأقل أيامًا يعني ري أكثر تكرارًا)"
                )
            )
        else:
            conclusion = (
                f"Result: {winner} {description} according to Gharsa data."
                if _is_english(language)
                else f"النتيجة: {winner} {description} حسب بيانات غرسة."
            )
    elif len(valid_vals) == 0:
        conclusion = (
            f"No {field_label} data is available for the mentioned plants in Gharsa."
            if _is_english(language)
            else f"لا تتوفر بيانات {field_label} لأي من النباتات المذكورة في غرسة."
        )
    else:
        only_plant = next(iter(valid_vals))
        conclusion = (
            f"{field_label.capitalize()} data is not available for all mentioned plants. Only {only_plant} has data."
            if _is_english(language)
            else f"لا تتوفر بيانات {field_label} لكل النباتات المذكورة. فقط {only_plant} لديه بيانات."
        )

    # Assemble full answer
    lines = [("According to Gharsa data:" if _is_english(language) else "حسب بيانات غرسة:")] + rows + ["", conclusion]
    answer = "\n".join(lines)

    return answer, conclusion, criterion, list(dict.fromkeys(used_fields)), missing_fields


# Template intents: answer is structured (newlines + bullet points preserved).
# These are NOT collapsed into a flowing paragraph.
_TEMPLATE_INTENTS: frozenset[str] = frozenset({
    "watering", "light", "soil", "care_summary", "tasks", "season",
})


def _build_list_context_for_language(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
    language: str,
) -> list[str]:
    """Build nested-list data (tasks/months/suitability) in the answer language."""
    lines: list[str] = []

    if "tasks" in intents and plant_data.get("tasks"):
        task_lines = []
        for item in plant_data.get("tasks", []):
            t_type = str(item.get("taskType", "")).strip()
            t_interval = item.get("intervalDays", "")
            if not t_type:
                continue
            label = _TASK_TYPE_EN.get(t_type, _TASK_TYPE_EN.get(t_type.upper(), t_type))
            task_lines.append(
                f"- {label}: every {t_interval} days" if t_interval else f"- {label}"
            )
        if task_lines:
            lines.append(f"Care tasks for {plant_name}:\n" + "\n".join(task_lines))

    if "season" in intents and plant_data.get("month_plants"):
        month_lines = []
        for item in plant_data.get("month_plants", []):
            month_name = _clean_english_text(str(item.get("monthName", "")).strip())
            season_val = _clean_english_text(str(item.get("season", "")).strip())
            note = _clean_english_text(str(item.get("plantingNoteEn", "")).strip())
            if not month_name and not note:
                continue
            entry = f"- {month_name}" if month_name else "-"
            if season_val:
                entry += f" ({season_val})"
            if note:
                entry += f": {note}" if month_name else f" {note}"
            month_lines.append(entry)
        if month_lines:
            lines.append(f"Planting months for {plant_name}:\n" + "\n".join(month_lines))

    if "suitability" in intents and plant_data.get("suitability_tips"):
        tips = [
            clean_tip
            for item in plant_data.get("suitability_tips", [])
            for clean_tip in [_clean_english_text(str(item.get("adjustmentTipEn", "")).strip())]
            if item.get("adjustmentTipEn")
            if clean_tip
        ]
        if tips:
            lines.append(f"Adjustment tips for {plant_name}: " + " ".join(tips))

    return lines


def _build_structured_answer_en(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
) -> str:
    fields = get_fields_for_intents(intents)
    parts: list[str] = []
    seen_values: set[str] = set()

    for col, fallback_label in fields.items():
        value = _g_lang(plant_data, col, "en")
        if not value:
            continue
        label = _label_for_field(col, fallback_label, "en")
        clean_value = _clean_english_text(str(value))
        if clean_value and clean_value not in seen_values:
            parts.append(f"- {label}: {clean_value}")
            seen_values.add(clean_value)

    for extra in _build_list_context_for_language(plant_name, plant_data, intents, "en"):
        if extra and extra not in seen_values:
            parts.append(extra)
            seen_values.add(extra)

    if not parts:
        return ""

    answer = f"According to Gharsa data, {plant_name}:\n" + "\n".join(parts)
    if _AR_CHAR_RE.search(answer):
        logger.warning("[build_structured_answer_en] Arabic text leaked into English answer")
        return ""
    return answer


def build_direct_answer(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
    language: str = "ar",
) -> str:
    """
    Build a natural Arabic answer directly from Excel data (no LLM).

    Template intents (watering, light, soil, care_summary, tasks) produce
    structured output with newlines/bullets preserved.  All other intents
    produce a flowing prose paragraph.
    """
    if _is_english(language):
        return _build_structured_answer_en(plant_name, plant_data, intents)

    template_parts: list[str] = []
    prose_sentences: list[str] = []

    for intent_name in intents:
        builder = _NATURALIZERS.get(intent_name, _sentences_general_summary)
        sentences = builder(plant_name, plant_data)
        clean = [s for s in sentences if s and not _is_junk(s)]

        if intent_name in _TEMPLATE_INTENTS:
            template_parts.extend(clean)
        else:
            prose_sentences.extend(clean)

    if not template_parts and not prose_sentences:
        return ""

    parts: list[str] = []

    if template_parts:
        # Preserve structure: newlines and bullet markers stay intact
        combined = "\n\n".join(template_parts)
        combined = combined.replace(" | ", "، ").replace("|", "، ")
        combined = combined.replace("\\n", "\n")
        combined = re.sub(r'[ \t]+\n', '\n', combined)
        combined = re.sub(r'\n{3,}', '\n\n', combined)
        parts.append(combined.strip())

    if prose_sentences:
        trimmed = prose_sentences[:_MAX_DIRECT_SENTENCES]
        capped = []
        for s in trimmed:
            if len(s) > 200:
                cut = s[:200].rsplit(' ', 1)[0]
                capped.append(cut.rstrip('،. ') + '.')
            else:
                capped.append(s)
        raw = " ".join(capped)
        parts.append(_postprocess_direct(raw))

    final = "\n\n".join(parts)

    # ── Post-check: strip remaining inline English ────────────────
    # Some fields may still contain English text (e.g. soilInfoAr).
    # Strip it once; if heavy English remains, return "" so the caller
    # falls through to responseMode = "missing_data".
    if _has_too_much_english(final):
        final = _strip_inline_english(final)
    if _has_too_much_english(final, threshold=8):
        logger.warning(
            "[build_direct_answer] answer still has heavy English after strip – "
            "returning empty to trigger missing_data"
        )
        return ""

    return final


def build_answer_draft(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
    language: str = "ar",
) -> str:
    """
    Build a data-dense **draft** for the LLM to rewrite into
    natural Arabic.  Used for multi-field intents (care_summary,
    plant_overview, growing_guide, beginner_overview).

    Values are cleaned (no |, no English parens, no junk) so the
    LLM receives only factual, rewrite-ready content.
    """
    if _is_english(language):
        return _build_structured_answer_en(plant_name, plant_data, intents)

    fields = get_fields_for_intents(intents)
    resolved = resolve_fields_for_plant(plant_data, fields)

    lines: list[str] = [f"معلومات عن {plant_name}:"]

    for label, value in resolved.items():
        clean_val = _clean_text(value)
        if clean_val:
            lines.append(f"{label}: {clean_val}")

    # ── List-based intents ─────────────────────────────────────────
    # suitability_tips / tasks / month_plants are stored as nested
    # lists inside the profile — resolve_column cannot reach them.
    # Call their dedicated sentence-builders so the LLM draft includes
    # this data when relevant intents are present.
    _LIST_BASED_INTENTS = {"suitability", "tasks", "season"}
    seen_text: set[str] = set(resolved.values())
    for intent_name in intents:
        if intent_name in _LIST_BASED_INTENTS:
            builder = _NATURALIZERS.get(intent_name)
            if builder:
                for s in builder(plant_name, plant_data):
                    if s and not _is_junk(s) and s not in seen_text:
                        lines.append(s)
                        seen_text.add(s)

    # ── Multi-field intents that reference season data ─────────────
    # plant_overview and growing_guide reference planting_months_pal
    # (stored as month_plants list). Add season sentences if not already
    # included via the list-based block above.
    _SEASON_ENRICHED_INTENTS = {"plant_overview", "growing_guide", "care_summary"}
    if any(i in _SEASON_ENRICHED_INTENTS for i in intents) and "season" not in intents:
        season_builder = _NATURALIZERS.get("season")
        if season_builder:
            for s in season_builder(plant_name, plant_data):
                if s and not _is_junk(s) and s not in seen_text:
                    lines.append(s)
                    seen_text.add(s)

    if len(lines) <= 1:  # only the header line — no data
        return ""

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 4.  Collect & merge user + climate context
# ─────────────────────────────────────────────────────────────────
def collect_user_and_climate_context(
    climate_data: Optional[dict] = None,
    user_conditions: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Convert raw dicts into free-text Arabic strings for the prompt."""
    climate_text: Optional[str] = None
    user_text: Optional[str] = None

    if climate_data:
        parts: list[str] = []
        if "city" in climate_data:
            parts.append(f"المدينة: {climate_data['city']}")
        if "temperature" in climate_data:
            parts.append(f"الحرارة الحالية: {climate_data['temperature']}°C")
        if "humidity" in climate_data:
            parts.append(f"الرطوبة: {climate_data['humidity']}%")
        if "season" in climate_data:
            parts.append(f"الموسم: {climate_data['season']}")
        if "month" in climate_data:
            parts.append(f"الشهر: {climate_data['month']}")
        for k, v in climate_data.items():
            if k not in {"city", "temperature", "humidity", "season", "month"}:
                parts.append(f"{k}: {v}")
        if parts:
            climate_text = " | ".join(parts)

    if user_conditions:
        parts = []
        if "location_type" in user_conditions:
            parts.append(f"مكان الزراعة: {user_conditions['location_type']}")
        if "pot_size_cm" in user_conditions:
            parts.append(f"حجم الأصيص: {user_conditions['pot_size_cm']} سم")
        if "sunlight_hours" in user_conditions:
            parts.append(
                f"ساعات الشمس المتاحة: {user_conditions['sunlight_hours']}"
            )
        for k, v in user_conditions.items():
            if k not in {"location_type", "pot_size_cm", "sunlight_hours"}:
                parts.append(f"{k}: {v}")
        if parts:
            user_text = " | ".join(parts)

    return climate_text, user_text


# ─────────────────────────────────────────────────────────────────
# 5.  Alternative plant suggestions
# ─────────────────────────────────────────────────────────────────
_TEMP_RE = re.compile(r"الحرارة المثالية:\s*([\d.]+)-([\d.]+)")


def _extract_temp_range(card: str) -> Optional[tuple[float, float]]:
    m = _TEMP_RE.search(card)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def suggest_alternative_plants_if_needed(
    target_cards: list[dict],
    all_cards: list[str],
    user_temperature: Optional[float] = None,
) -> list[str]:
    """If user temp is outside the plant's ideal range, suggest alternatives."""
    if user_temperature is None or not target_cards:
        return []

    top_card = target_cards[0]["card"]
    top_range = _extract_temp_range(top_card)
    if top_range is None:
        return []

    t_min, t_max = top_range
    if t_min <= user_temperature <= t_max:
        return []

    alternatives: list[str] = []
    seen: set[str] = set()
    for card in all_cards:
        r = _extract_temp_range(card)
        if r and r[0] <= user_temperature <= r[1]:
            name = extract_plant_name(card)
            if name and name not in seen:
                seen.add(name)
                alternatives.append(name)
        if len(alternatives) >= 5:
            break
    return alternatives


# ─────────────────────────────────────────────────────────────────
# 6.  Fallback formatter
# ─────────────────────────────────────────────────────────────────
def format_fallback_answer(
    retrieved: list[dict],
    fallback_prefix: str,
) -> str:
    """Build a readable Arabic answer from raw retrieved cards."""
    if not retrieved:
        return SAFE_NO_ANSWER

    plant_sections: list[str] = []
    for item in retrieved:
        card = item["card"]
        lines = [ln.strip() for ln in card.splitlines() if ln.strip()]
        if not lines:
            continue

        first = lines[0]
        if ":" in first:
            name_part = first.split(":", 1)[1].strip()
            plant_name = name_part.split("|")[0].strip()
        else:
            plant_name = first.strip()

        info_lines: list[str] = []
        for ln in lines[1:]:
            if ":" in ln:
                label, _, value = ln.partition(":")
                label = label.strip()
                value = value.strip()
                if label and value and len(value) > 2:
                    info_lines.append(f"• {label}: {value}")
            if len(info_lines) >= 5:
                break

        if plant_name and info_lines:
            plant_sections.append(f"🌱 {plant_name}:\n" + "\n".join(info_lines))
        elif plant_name:
            plant_sections.append(f"🌱 {plant_name}")

    if not plant_sections:
        return SAFE_NO_ANSWER

    return fallback_prefix + "\n\n".join(plant_sections)
