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
from src.utils.arabic import normalize

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

    Strategy:
      1. Direct substring match (normalised) against all known names.
      2. Fuzzy match using rapidfuzz (if available).

    Returns the *original* (un-normalised) plant name, or ``None``.
    """
    q_norm = normalize(question).lower()

    # Strip common prefixes from the question for matching
    prefixes = ("ال", "نبتة ", "نبات ", "عشبة ")

    # 1. Direct substring match
    for original_name in all_plant_names:
        name_norm = normalize(original_name).lower()
        if not name_norm:
            continue
        # exact containment
        if name_norm in q_norm:
            return original_name
        # also try without ال prefix
        for pfx in prefixes:
            stripped = name_norm
            if stripped.startswith(pfx):
                stripped = stripped[len(pfx):].strip()
            if stripped and len(stripped) > 2 and stripped in q_norm:
                return original_name

    # 2. Fuzzy match (optional – graceful if rapidfuzz absent)
    try:
        from rapidfuzz import process, fuzz

        # Extract candidate tokens from question (3+ chars)
        q_tokens = [t for t in q_norm.split() if len(t) >= 3]
        name_norms = {normalize(n).lower(): n for n in all_plant_names if n}

        for token in q_tokens:
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


# ── Text cleaning for natural Arabic output ───────────────────────

_PAREN_EN_RE = re.compile(r'\([A-Za-z][A-Za-z\s./%°,;:\'"\x2d\u201c\u201d\u2018\u2019]+\)')


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
    out: list[str] = []
    need   = _g(d, "watering_need")
    rule   = _g(d, "watering_rule_text")
    d_min  = _g(d, "watering_interval_days_min")
    d_max  = _g(d, "watering_interval_days_max")
    fix_u  = _g(d, "fix_underwatering")
    fix_o  = _g(d, "fix_overwatering")

    if rule:
        out.append(_clean_text(rule))
    elif need:
        out.append(f"يحتاج {p} إلى ري {need}.")

    if d_min and d_max and d_min != d_max:
        out.append(f"يُوصى بريّه كل {d_min}–{d_max} يومًا.")
    elif d_min:
        out.append(f"يُوصى بريّه كل {d_min} يومًا على الأقل.")

    if fix_u:
        brief = _clean_text(fix_u).split('.')[0]
        if brief:
            out.append(f"عند نقص الري، {brief}.")
    if fix_o:
        brief = _clean_text(fix_o).split('.')[0]
        if brief:
            out.append(f"عند الإفراط بالري، {brief}.")
    return out


def _sentences_light(p: str, d: dict) -> list[str]:
    out: list[str] = []
    level  = _g(d, "light_level")
    hours  = _g(d, "min_sun_hours")
    window = _g(d, "indoor_window_direction")

    if level:
        out.append(f"يحتاج {p} إلى {_clean_text(level)}.")
    if hours:
        out.append(f"يفضّل ما لا يقل عن {hours} ساعات من ضوء الشمس يوميًا.")
    if window:
        w = _clean_text(window)
        # Avoid "نافذة … نحو نافذة" repetition
        if w.startswith("نافذة"):
            w = w[len("نافذة"):].strip()
        out.append(f"للزراعة الداخلية، يُنصح بوضعه قرب نافذة {w}.")
    return out


def _sentences_temperature(p: str, d: dict) -> list[str]:
    out: list[str] = []
    t_min  = _g(d, "temperature_optimal_min_c")
    t_max  = _g(d, "temperature_optimal_max_c")
    heat   = _g(d, "heat_tolerance")
    frost  = _g(d, "frost_tolerance")

    if t_min and t_max:
        out.append(f"تتراوح درجة الحرارة المثالية لـ{p} بين {t_min}°م و{t_max}°م.")
    elif t_min:
        out.append(f"تبدأ الحرارة المثالية لـ{p} من {t_min}°م.")
    elif t_max:
        out.append(f"لا تتجاوز الحرارة المثالية لـ{p} {t_max}°م.")

    if heat:
        out.append(f"تحمّله للحرارة المرتفعة {heat}.")
    if frost:
        out.append(f"تحمّله للصقيع {frost}.")
    return out


def _sentences_soil(p: str, d: dict) -> list[str]:
    out: list[str] = []
    texture    = _g(d, "soil_texture_preference")
    drainage   = _g(d, "drainage_need")
    ph_min     = _g(d, "soil_ph_min")
    ph_max     = _g(d, "soil_ph_max")
    amendments = _g(d, "soil_amendments")

    parts: list[str] = []
    if texture:
        parts.append(f"يفضّل {p} التربة {_clean_text(texture)}")
    if drainage:
        parts.append(f"مع أهمية {_clean_text(drainage)}")
    if parts:
        out.append("، ".join(parts) + ".")

    if ph_min and ph_max:
        out.append(f"تتراوح الحموضة المثالية بين {ph_min} و{ph_max}.")
    elif ph_min:
        out.append(f"الحد الأدنى المناسب للحموضة {ph_min}.")

    if amendments:
        out.append(f"يُفيد إضافة {_clean_text(amendments)} لتحسين التربة.")
    return out


def _sentences_fertilizing(p: str, d: dict) -> list[str]:
    out: list[str] = []
    need      = _g(d, "fertilizer_need")
    ftype     = _g(d, "fertilizer_type")
    freq      = _g(d, "fertilizer_frequency_days")
    compost   = _g(d, "compost_recommended")

    if need:
        n = need
        # Avoid "تسميد تسميد" repetition
        if n.startswith("تسميد"):
            n = n[len("تسميد"):].strip()
        out.append(f"يحتاج {p} إلى تسميد {n}.")
    if ftype and freq:
        out.append(f"يُفضَّل استخدام {_clean_text(ftype)} مرة كل {freq} يومًا.")
    elif ftype:
        out.append(f"يُفضَّل استخدام {_clean_text(ftype)}.")
    elif freq:
        out.append(f"يُسمَّد كل {freq} يومًا.")

    if compost:
        val_low = compost.strip().lower()
        if val_low in ("نعم", "yes", "1", "true"):
            out.append("يُنصح باستخدام الكمبوست لتقوية التربة.")
    return out


def _sentences_season(p: str, d: dict) -> list[str]:
    out: list[str] = []
    plant_m  = _g(d, "planting_months_pal")
    harvest_m = _g(d, "harvest_months_pal")
    notes    = _g(d, "season_notes_pal")
    region   = _g(d, "pal_region")

    region_suffix = f" في {region}" if region else " في فلسطين"

    if plant_m:
        out.append(f"يُزرع {p}{region_suffix} خلال {_clean_text(plant_m)}.")
    if harvest_m:
        out.append(f"ويُحصد في {_clean_text(harvest_m)}.")
    if notes:
        out.append(_clean_text(notes))
    return out


def _sentences_harvest_storage(p: str, d: dict) -> list[str]:
    out: list[str] = []
    method   = _g(d, "harvest_method")
    h_min    = _g(d, "harvest_after_days_min")
    h_max    = _g(d, "harvest_after_days_max")
    drying   = _g(d, "drying_method")
    storage  = _g(d, "storage_method")
    duration = _g(d, "storage_duration_months")

    if method and h_min and h_max and h_min != h_max:
        out.append(f"يُحصد {p} بعد {h_min}–{h_max} يومًا من الزراعة عن طريق {_clean_text(method)}.")
    elif method:
        out.append(f"يُحصد {p} عن طريق {_clean_text(method)}.")

    if drying:
        out.append(f"يمكن تجفيفه عبر {_clean_text(drying)}.")
    if storage and duration:
        out.append(f"يُخزَّن {_clean_text(storage)} لمدة تصل إلى {duration} أشهر.")
    elif storage:
        out.append(f"يُخزَّن {_clean_text(storage)}.")
    return out


def _sentences_pests_diseases(p: str, d: dict) -> list[str]:
    out: list[str] = []
    pests    = _g(d, "common_pests")
    diseases = _g(d, "common_diseases")

    if pests:
        out.append(f"من أبرز الآفات التي قد تصيب {p}: {_clean_text(pests)}.")
    if diseases:
        out.append(f"ومن الأمراض الشائعة: {_clean_text(diseases)}.")
    return out


def _sentences_beginner(p: str, d: dict) -> list[str]:
    out: list[str] = []
    diff  = _g(d, "difficulty_level")
    time_ = _g(d, "time_commitment")
    tips  = _g(d, "beginner_tips")

    if diff:
        out.append(f"زراعة {p} تُعتبر {diff}.")
    if time_:
        out.append(f"يتطلب {_clean_text(time_)}.")
    if tips:
        out.append(_clean_text(tips))
    return out


def _sentences_container(p: str, d: dict) -> list[str]:
    out: list[str] = []
    possible  = _g(d, "container_possible")
    diam      = _g(d, "pot_diameter_cm_min")
    depth     = _g(d, "pot_depth_cm_min")
    drainage  = _g(d, "drainage_need")

    if possible:
        val_low = possible.strip().lower()
        if val_low in ("نعم", "yes", "1", "true"):
            out.append(f"{p} مناسبة تمامًا للزراعة في الأصيص.")
        elif val_low in ("لا", "no", "0", "false"):
            out.append(f"{p} غير مناسبة للزراعة في الأصيص.")
        else:
            out.append(f"بالنسبة لزراعة {p} في الأصيص، {_clean_text(possible)}.")

    if diam and depth:
        out.append(f"يُفضَّل أصيص بقطر لا يقل عن {diam} سم وعمق {depth} سم.")
    elif diam:
        out.append(f"يُفضَّل أصيص بقطر لا يقل عن {diam} سم.")

    if drainage:
        out.append(f"تأكد من {_clean_text(drainage)} لمنع تعفّن الجذور.")
    return out


def _sentences_planting(p: str, d: dict) -> list[str]:
    out: list[str] = []
    method    = _g(d, "propagation_method_primary")
    spacing   = _g(d, "plant_spacing_cm")
    g_min     = _g(d, "germination_days_min")
    g_max     = _g(d, "germination_days_max")
    trans_ok  = _g(d, "transplanting_ok")

    if method:
        out.append(f"يُكثَّر {p} عن طريق {_clean_text(method)}.")
    if g_min and g_max and g_min != g_max:
        out.append(f"تنبت البذور في غضون {g_min}–{g_max} يومًا.")
    elif g_min:
        out.append(f"تنبت البذور بعد حوالي {g_min} يومًا.")
    if spacing:
        out.append(f"يُوصى بترك مسافة {_clean_text(spacing)} بين النباتات.")
    if trans_ok:
        val_low = trans_ok.strip().lower()
        if val_low in ("نعم", "yes", "1", "true"):
            out.append("يمكن نقله وشتله بسهولة.")
    return out


def _sentences_general_summary(p: str, d: dict) -> list[str]:
    out: list[str] = []
    summary   = _g(d, "short_summary")
    category  = _g(d, "category")
    habit     = _g(d, "growth_habit")
    cycle     = _g(d, "life_cycle")
    fragrance = _g(d, "fragrance_level")
    edible    = _g(d, "edible_parts")

    if summary:
        out.append(_clean_text(summary))
        # Only add details NOT already mentioned in the summary
        extras: list[str] = []
        s_low = summary.lower()
        if category and category.lower() not in s_low:
            extras.append(f"يُصنَّف ضمن {_clean_text(category)}")
        if fragrance and fragrance.lower() not in s_low:
            extras.append(f"بمستوى عطر {fragrance}")
        if edible and edible.lower() not in s_low:
            extras.append(f"والجزء الصالح للأكل فيه {_clean_text(edible)}")
        if extras:
            out.append("، ".join(extras) + ".")
    else:
        # No summary → build a composite natural sentence
        base = p
        if habit:
            base += f" نبات {habit}"
        if cycle:
            base += f" {cycle}"
        parts: list[str] = [base]
        if category:
            parts.append(f"يُصنَّف ضمن {_clean_text(category)}")
        if fragrance:
            parts.append(f"بمستوى عطر {fragrance}")
        if edible:
            parts.append(f"والجزء الصالح للأكل فيه {_clean_text(edible)}")
        out.append("، ".join(parts) + ".")
    return out


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
}


def has_sufficient_data(
    plant_data: dict,
    intents: list[str],
    min_fields: int = 1,
) -> bool:
    """
    True when the plant profile has at least *min_fields* non-empty
    values among the columns required by the given intents.
    """
    fields = get_fields_for_intents(intents)
    resolved = resolve_fields_for_plant(plant_data, fields)
    return len(resolved) >= min_fields


def build_direct_answer(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
) -> str:
    """
    Build a natural Arabic answer directly from Excel data (no LLM).

    Each intent has a dedicated sentence-builder that converts raw
    column values into coherent Arabic sentences.  Junk/empty values
    are silently skipped; no field labels or technical strings leak
    into the output.
    """
    all_sentences: list[str] = []

    for intent_name in intents:
        builder = _NATURALIZERS.get(intent_name)
        if builder is None:
            builder = _sentences_general_summary

        sentences = builder(plant_name, plant_data)
        all_sentences.extend(s for s in sentences if s and not _is_junk(s))

    if not all_sentences:
        return ""

    # Limit to a concise answer and join into a flowing paragraph
    trimmed = all_sentences[:_MAX_DIRECT_SENTENCES]
    # Truncate overly long individual sentences
    capped = []
    for s in trimmed:
        if len(s) > 200:
            cut = s[:200].rsplit(' ', 1)[0]
            capped.append(cut.rstrip('،. ') + '.')
        else:
            capped.append(s)
    raw = " ".join(capped)
    return _postprocess_direct(raw)


def build_answer_draft(
    plant_name: str,
    plant_data: dict,
    intents: list[str],
) -> str:
    """
    Build a data-dense **draft** for the LLM to rewrite into
    natural Arabic.  Used for multi-field intents (care_summary,
    plant_overview, growing_guide, beginner_overview).

    Values are cleaned (no |, no English parens, no junk) so the
    LLM receives only factual, rewrite-ready content.
    """
    fields = get_fields_for_intents(intents)
    resolved = resolve_fields_for_plant(plant_data, fields)

    if not resolved:
        return ""

    lines: list[str] = [f"معلومات عن {plant_name}:"]

    for label, value in resolved.items():
        clean_val = _clean_text(value)
        if clean_val:
            lines.append(f"{label}: {clean_val}")

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
