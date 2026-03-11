"""
columns.py – Single canonical column-alias registry.

Merges what was previously duplicated in:
  - intent_fields.COLUMN_ALIASES  (17 entries)
  - context_utils._COL_ALIASES   (42 entries)
  - build_training_data.py       (inline pick_value calls)

Every module that needs to resolve column names imports from here.

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
# ─────────────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    # ── Names / identity ──────────────────────────────────────────
    "arabic_name_primary":       ("arabic_name_alt", "name_ar", "plant_name_ar", "arabic_name"),
    "english_name_primary":      ("name_en", "plant_name_en", "english_name"),
    "scientific_name":           ("scientific",),

    # ── Watering ──────────────────────────────────────────────────
    "watering_need":             ("water_need", "watering"),
    "watering_rule_text":        ("watering_rule", "watering_instructions"),
    "watering_interval_days_min":("interval_min",),
    "watering_interval_days_max":("interval_max",),
    "fix_underwatering":         (),
    "fix_overwatering":          (),

    # ── Light ─────────────────────────────────────────────────────
    "light_level":               ("sunlight", "light"),
    "min_sun_hours":             ("sun_hours_min",),
    "indoor_window_direction":   ("window_direction",),

    # ── Temperature ───────────────────────────────────────────────
    "temperature_optimal_min_c": ("temperature_optimal_min", "temp_opt_min", "temp_min"),
    "temperature_optimal_max_c": ("temperature_optimal_max", "temp_opt_max", "temp_max"),
    "heat_tolerance":            (),
    "frost_tolerance":           (),

    # ── Soil ──────────────────────────────────────────────────────
    "soil_texture_preference":   ("soil", "soil_type"),
    "soil_ph_min":               (),
    "soil_ph_max":               (),
    "drainage_need":             ("drainage",),
    "soil_amendments":           (),

    # ── Fertiliser ────────────────────────────────────────────────
    "fertilizer_need":           (),
    "fertilizer_type":           (),
    "fertilizer_frequency_days": (),
    "compost_recommended":       (),

    # ── Season / Palestine calendar ───────────────────────────────
    "planting_months_pal":       (),
    "harvest_months_pal":        (),
    "season_notes_pal":          (),
    "pal_region":                (),

    # ── Harvest / storage ─────────────────────────────────────────
    "harvest_method":            (),
    "harvest_after_days_min":    (),
    "harvest_after_days_max":    (),
    "drying_method":             (),
    "storage_method":            (),
    "storage_duration_months":   (),

    # ── Pests & diseases ──────────────────────────────────────────
    "common_pests":              ("pests",),
    "common_diseases":           ("diseases",),

    # ── Beginner / difficulty ─────────────────────────────────────
    "difficulty_level":          (),
    "time_commitment":           (),
    "beginner_tips":             ("tips_beginner",),

    # ── Container ─────────────────────────────────────────────────
    "container_possible":        (),
    "pot_diameter_cm_min":       (),
    "pot_depth_cm_min":          (),

    # ── Planting / propagation ────────────────────────────────────
    "propagation_method_primary":(),
    "plant_spacing_cm":          (),
    "germination_days_min":      (),
    "germination_days_max":      (),
    "transplanting_ok":          (),
    "care_steps_json":           ("care_steps",),

    # ── Summary / general ─────────────────────────────────────────
    "short_summary":             ("description_ar", "description"),
    "category":                  (),
    "growth_habit":              (),
    "life_cycle":                (),
    "fragrance_level":           (),
    "edible_parts":              (),
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
