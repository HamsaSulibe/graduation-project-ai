"""
columns.py – Single canonical column-alias registry.

Merges what was previously duplicated in:
  - intent_fields.COLUMN_ALIASES  (17 entries)
  - context_utils._COL_ALIASES   (42 entries)
  - build_training_data.py       (inline pick_value calls)

Every module that needs to resolve column names imports from here.

Updated to match plant_data_final_v4.xlsx (5 sheets).
Canonical names are preserved so intents.py and context_utils.py
compile unchanged.  New Excel column names are added as first alias
so they are found first during lookup.

Public API
----------
COLUMN_ALIASES          dict[str, tuple[str, ...]]
resolve_column(data, col)  → value | None
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.config.constants import EMPTY_MARKERS, clean_value, is_junk


# ─────────────────────────────────────────────────────────────────
# Master alias table   canonical_col → tuple of alternative names
#
# Convention:
#   - Canonical names are the internal codes used by intents.py,
#     context_utils.py, and build_training_data.py.  Do NOT rename them
#     here — change intents.py instead.
#   - The FIRST alias is always the actual Excel column name from
#     plant_data_final_v4.xlsx so it is resolved first.
#   - Old aliases follow as fallbacks for backward compatibility.
#   - Entries marked [NO EQUIVALENT] have no matching column in the
#     new Excel; get_column() will return None for them until
#     intents.py is updated in a later phase.
# ─────────────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {

    # ════════════════════════════════════════════════════════════
    # Sheet: Plants  (primary sheet, keyed by nameAr)
    # ════════════════════════════════════════════════════════════

    # ── Names / identity ─────────────────────────────────────────
    # Plants.nameAr is the join key; also the primary Arabic name.
    "arabic_name_primary":        ("nameAr",),
    # Secondary sheets use plantNameAr for joining back to Plants.
    "plant_name_ar":              ("plantNameAr",),
    "english_name_primary":       ("nameEn",),
    "scientific_name":            ("nameScientific",),

    # ── Description / summary ────────────────────────────────────
    "short_summary":              ("shortDescriptionAr",),          # Plants.shortDescriptionAr
    "short_summary_en":           ("shortDescriptionEn",),          # Plants.shortDescriptionEn [NEW]
    "category":                   (),                               # Plants.category (same name)
    "difficulty_level":           ("difficultyLevel",),             # Plants.difficultyLevel

    # ── Watering ─────────────────────────────────────────────────
    # wateringIntervalDays is now a single unified value (was min/max).
    "watering_need":              ("wateringNeed",),                # Plants.wateringNeed
    "watering_interval_days_min": ("wateringIntervalDays",),        # Plants.wateringIntervalDays (unified)
    "watering_interval_days_max": ("wateringIntervalDays",),        # Plants.wateringIntervalDays (unified)
    # watering_rule_text is now a long Arabic text block in Care_Details.
    "watering_rule_text":         ("wateringInfoAr",),              # Care_Details.wateringInfoAr
    "watering_info_en":           ("wateringInfoEn",),              # Care_Details.wateringInfoEn [NEW]
    # [NO EQUIVALENT] fix_underwatering, fix_overwatering removed from Excel.
    "fix_underwatering":          (),
    "fix_overwatering":           (),

    # ── Light ─────────────────────────────────────────────────────
    # Light data is now a long Arabic text block in Care_Details.
    "light_level":                ("lightInfoAr",),                 # Care_Details.lightInfoAr (was structured field)
    "light_info_en":              ("lightInfoEn",),                 # Care_Details.lightInfoEn [NEW]
    # [NO EQUIVALENT] sub-fields removed from Excel.
    "min_sun_hours":              (),
    "indoor_window_direction":    (),

    # ── Temperature ──────────────────────────────────────────────
    "temperature_optimal_min_c":  ("minTemp",),                     # Plants.minTemp
    "temperature_optimal_max_c":  ("maxTemp",),                     # Plants.maxTemp
    "temperature_sensitivity":    ("temperatureSensitivity",),      # Plants.temperatureSensitivity [NEW]
    # [NO EQUIVALENT] detailed tolerance fields removed from Excel.
    "heat_tolerance":             (),
    "frost_tolerance":            (),

    # ── Soil ─────────────────────────────────────────────────────
    # Soil data is now a long Arabic text block in Care_Details.
    "soil_texture_preference":    ("soilInfoAr",),                  # Care_Details.soilInfoAr (was structured field)
    "soil_info_en":               ("soilInfoEn",),                  # Care_Details.soilInfoEn [NEW]
    "drainage_need":              ("soilInfoAr",),                  # Care_Details.soilInfoAr (approximate)
    "soil_moisture_min":          ("soilMoistureMin",),             # Plants.soilMoistureMin [NEW]
    "soil_moisture_max":          ("soilMoistureMax",),             # Plants.soilMoistureMax [NEW]
    # [NO EQUIVALENT] pH and amendments removed from Excel.
    "soil_ph_min":                (),
    "soil_ph_max":                (),
    "soil_amendments":            (),

    # ── Humidity ─────────────────────────────────────────────────
    "humidity_preference":        ("humidityPreference",),          # Plants.humidityPreference [NEW]
    "air_humidity_min":           ("airHumidityMin",),              # Plants.airHumidityMin [NEW]
    "air_humidity_max":           ("airHumidityMax",),              # Plants.airHumidityMax [NEW]
    "humidity_notes_ar":          ("humidityNotesAr",),             # Plants.humidityNotesAr [NEW]
    "humidity_notes_en":          ("humidityNotesEn",),             # Plants.humidityNotesEn [NEW]

    # ── Fertilizer ───────────────────────────────────────────────
    "fertilizer_type":            ("fertilizerType",),              # Plants.fertilizerType
    "fertilizer_frequency_days":  ("fertilizerFrequencyDays",),     # Plants.fertilizerFrequencyDays
    "fertilizer_stage":           ("fertilizerStage",),             # Plants.fertilizerStage [NEW]
    "fertilizer_notes_ar":        ("fertilizerNotesAr",),           # Plants.fertilizerNotesAr [NEW]
    "fertilizer_notes_en":        ("fertilizerNotesEn",),           # Plants.fertilizerNotesEn [NEW]
    # [NO EQUIVALENT] fertilizer_need, compost_recommended removed from Excel.
    "fertilizer_need":            (),
    "compost_recommended":        (),

    # ── Harvest ──────────────────────────────────────────────────
    "harvest_after_days_min":     ("daysToHarvestMin",),            # Plants.daysToHarvestMin
    "harvest_after_days_max":     ("daysToHarvestMax",),            # Plants.daysToHarvestMax
    # harvest_method is now a long Arabic text block in Care_Details.
    "harvest_method":             ("harvestInfoAr",),               # Care_Details.harvestInfoAr (was structured field)
    "harvest_info_en":            ("harvestInfoEn",),               # Care_Details.harvestInfoEn [NEW]
    # [NO EQUIVALENT] storage / drying fields removed from Excel.
    "drying_method":              (),
    "storage_method":             (),
    "storage_duration_months":    (),

    # ── Planting / propagation ───────────────────────────────────
    "propagation_method_primary": ("plantingMethod",),              # Plants.plantingMethod
    "plant_spacing_cm":           ("plantSpacingCmMin",),           # Plants.plantSpacingCmMin (approximate)
    "plant_spacing_cm_min":       ("plantSpacingCmMin",),           # Plants.plantSpacingCmMin [NEW]
    "plant_spacing_cm_max":       ("plantSpacingCmMax",),           # Plants.plantSpacingCmMax [NEW]
    # germinationDays is now a single unified value (was min/max).
    "germination_days_min":       ("germinationDays",),             # Plants.germinationDays (unified)
    "germination_days_max":       ("germinationDays",),             # Plants.germinationDays (unified)
    "establishment_days":         ("establishmentDays",),           # Plants.establishmentDays [NEW]
    "seed_to_seedling_days":      ("seedToSeedlingDays",),          # Plants.seedToSeedlingDays [NEW]
    "seed_care_ar":               ("seedCareInstructionsAr",),      # Plants.seedCareInstructionsAr [NEW]
    "seed_care_en":               ("seedCareInstructionsEn",),      # Plants.seedCareInstructionsEn [NEW]
    # [NO EQUIVALENT] transplanting_ok removed from Excel.
    "transplanting_ok":           (),

    # ── Planting steps (Care_Details) ────────────────────────────
    # care_steps_json was a JSON-encoded field; now a plain Arabic text column.
    "care_steps_json":            ("plantingStepsAr",),             # Care_Details.plantingStepsAr (was JSON)
    "planting_steps_ar":          ("plantingStepsAr",),             # Care_Details.plantingStepsAr [NEW canonical]
    "planting_steps_en":          ("plantingStepsEn",),             # Care_Details.plantingStepsEn [NEW]

    # ── Care info (Care_Details) ─────────────────────────────────
    "care_info_ar":               ("careInfoAr",),                  # Care_Details.careInfoAr [NEW]
    "care_info_en":               ("careInfoEn",),                  # Care_Details.careInfoEn [NEW]
    # beginner_tips has no direct column; use careInfoAr as closest proxy.
    "beginner_tips":              ("careInfoAr",),                  # Care_Details.careInfoAr (approximate)

    # ── Uses (Care_Details) ──────────────────────────────────────
    "uses_info_ar":               ("usesInfoAr",),                  # Care_Details.usesInfoAr [NEW]
    "uses_info_en":               ("usesInfoEn",),                  # Care_Details.usesInfoEn [NEW]

    # ════════════════════════════════════════════════════════════
    # Sheet: Suitability  (keyed by plantNameAr)
    # ════════════════════════════════════════════════════════════
    "adjustment_tip_ar":          ("adjustmentTipAr",),             # Suitability.adjustmentTipAr [NEW]
    "adjustment_tip_en":          ("adjustmentTipEn",),             # Suitability.adjustmentTipEn [NEW]

    # ════════════════════════════════════════════════════════════
    # Sheet: Tasks  (keyed by plantNameAr)
    # ════════════════════════════════════════════════════════════
    "task_type":                  ("taskType",),                    # Tasks.taskType [NEW]
    "interval_days":              ("intervalDays",),                # Tasks.intervalDays [NEW]

    # ════════════════════════════════════════════════════════════
    # Sheet: Month_Plants  (keyed by plantNameAr)
    # ════════════════════════════════════════════════════════════
    # Replaces the old planting_months_pal text field with structured rows.
    "month_number":               ("monthNumber",),                 # Month_Plants.monthNumber [NEW]
    "planting_note_ar":           ("plantingNoteAr",),              # Month_Plants.plantingNoteAr [NEW]
    "planting_note_en":           ("plantingNoteEn",),              # Month_Plants.plantingNoteEn [NEW]
    "month_name":                 ("monthName",),                   # Month_Plants.monthName [NEW]
    "season":                     ("season",),                      # Month_Plants.season [NEW]
    # Old calendar canonical names kept so intents.py does not break.
    # planting_months_pal → use Month_Plants sheet logic instead (no single column).
    "planting_months_pal":        ("plantingNoteAr",),              # approximate via Month_Plants
    "season_notes_pal":           ("plantingNoteAr",),              # Month_Plants.plantingNoteAr (approximate)
    # [NO EQUIVALENT] harvest calendar and region removed from Excel.
    "harvest_months_pal":         (),
    "pal_region":                 (),

    # ════════════════════════════════════════════════════════════
    # [NO EQUIVALENT] — sheet plants_pests_diseases removed entirely
    # ════════════════════════════════════════════════════════════
    "common_pests":               (),
    "common_diseases":            (),

    # ════════════════════════════════════════════════════════════
    # [NO EQUIVALENT] — miscellaneous fields removed from Excel
    # ════════════════════════════════════════════════════════════
    "time_commitment":            (),
    "container_possible":         (),
    "pot_diameter_cm_min":        (),
    "pot_depth_cm_min":           (),
    "growth_habit":               (),
    "life_cycle":                 (),
    "fragrance_level":            (),
    "edible_parts":               (),
}


# ─────────────────────────────────────────────────────────────────
# Resolution helpers
# ─────────────────────────────────────────────────────────────────
def _clean_display_value(v) -> Optional[str]:
    """
    Return a display-ready string, or None if the value is empty.
    Handles JSON-encoded lists / dicts (e.g. care_steps_json).
    """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in EMPTY_MARKERS:
        return None

    # JSON-encoded lists / dicts
    if s.startswith("[") or s.startswith("{"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return "\n".join(
                    f"  {i}. {item}" if isinstance(item, str)
                    else f"  {i}. {json.dumps(item, ensure_ascii=False)}"
                    for i, item in enumerate(parsed, 1)
                )
            if isinstance(parsed, dict):
                return " | ".join(f"{k}: {v}" for k, v in parsed.items())
        except (json.JSONDecodeError, TypeError):
            pass
    return s


def resolve_column(data: dict, col: str) -> Optional[str]:
    """
    Read *col* from *data* (with alias fallback).
    Returns cleaned display-ready string, or None if absent / empty.
    """
    if data is None:
        return None
    val = _clean_display_value(data.get(col))
    if val is not None:
        return val
    for alias in COLUMN_ALIASES.get(col, ()):
        val = _clean_display_value(data.get(alias))
        if val is not None:
            return val
    return None


def get_column(data: dict, col: str) -> Optional[str]:
    """
    Read *col* from *data* (with alias fallback) and apply
    junk filtering + value cleaning.  Returns None for junk values.

    This replaces the former ``context_utils._g()`` helper.
    """
    if data is None:
        return None

    def _try(key: str) -> Optional[str]:
        raw = data.get(key)
        if raw is None:
            return None
        try:
            if pd.isna(raw):
                return None
        except (TypeError, ValueError):
            pass
        s = clean_value(str(raw))
        return None if is_junk(s) else s

    v = _try(col)
    if v is not None:
        return v
    for alias in COLUMN_ALIASES.get(col, ()):
        v = _try(alias)
        if v is not None:
            return v
    return None


def pick_value(row: dict, *keys: str) -> str:
    """
    Return the first non-empty value from *keys* (with alias
    expansion for each key).  Used by offline scripts (card building,
    training data).
    """
    for k in keys:
        raw = row.get(k)
        if raw is not None and not pd.isna(raw):
            s = str(raw).strip()
            if s:
                return s
        # Try aliases
        for alias in COLUMN_ALIASES.get(k, ()):
            raw = row.get(alias)
            if raw is not None and not pd.isna(raw):
                s = str(raw).strip()
                if s:
                    return s
    return ""
