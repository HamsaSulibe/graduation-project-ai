"""Excel-backed direct-answer builders."""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.config.columns import get_column
from src.config.constants import is_junk
from src.inference.answer_cleaning import (
    _AR_CHAR_RE,
    _clean_english_text,
    _clean_text,
    _has_too_much_english,
    _postprocess_direct,
    _strip_inline_english,
)
from src.config.assistant_messages import (
    _EN_CANONICAL_OVERRIDES,
    _FIELD_LABELS_EN,
    _TASK_TYPE_AR,
    _TASK_TYPE_EN,
)
from src.inference.intent_fields import (
    get_fields_for_intents,
    resolve_fields_for_plant,
)

logger = logging.getLogger(__name__)

# _g is now imported from src.config.columns.get_column
# _is_junk is now imported from src.config.constants.is_junk
# Both use the shared COLUMN_ALIASES and EMPTY_MARKERS.
_g = get_column
_is_junk = is_junk


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

_MAX_DIRECT_SENTENCES = 4

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
