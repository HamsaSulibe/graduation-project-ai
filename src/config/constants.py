"""
constants.py – Shared constants used across the Garssa project.

Consolidates values that were previously duplicated across:
  - intent_fields._EMPTY_MARKERS
  - context_utils._EMPTY_MARKERS_DIR / _JUNK_RE
  - plant_data_store._NAME_COLS
  - app.py  name_col_candidates / SAFE_NO_ANSWER
  - build_training_data.py  pick_value junk checks
Updated to match plant_data_final_v4.xlsx structure (5 sheets)."""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────
# 1.  Empty / junk value detection
# ─────────────────────────────────────────────────────────────────

# Exact-match tokens that are considered "no data"
EMPTY_MARKERS: frozenset[str] = frozenset({
    "مش موجود", "غير متوفر", "غير متوفرة",
    "لا يوجد", "غير معروف", "غير مذكور",
    "nan", "none", "n/a", "-", "—", "لا", "", " ",
})

# Substring patterns that mark a value as placeholder / junk
JUNK_RE = re.compile(
    r"غير محدد|غير متوفر|غير متوفرة|لا يوجد|غير معروف|غير مذكور"
    r"|غير محدده|غير معروفه|غير ثابت|بمصدر علمي",
    re.IGNORECASE | re.UNICODE,
)


def is_junk(value: str | None) -> bool:
    """True when *value* is empty, a placeholder, or internal fallback text."""
    if not value:
        return True
    vs = value.strip()
    if len(vs) == 0:
        return True
    if vs.lower() in EMPTY_MARKERS:
        return True
    if JUNK_RE.search(vs):
        return True
    return False


# ─────────────────────────────────────────────────────────────────
# 2.  Plant-name column candidates (tried in order)
#
#  Primary sheet (Plants) uses "nameAr" as the main key.
#  Secondary sheets use "plantNameAr" for joining.
#  Fallbacks "nameEn" / "nameScientific" are used when Arabic is absent.
#
#  REMOVED (old structure, no longer in Excel):
#    "arabic_name_primary", "arabic_name_alt", "name_ar",
#    "arabic_name", "اسم_النبتة"
# ─────────────────────────────────────────────────────────────────
NAME_COLUMNS: tuple[str, ...] = (
    "nameAr",           # Plants sheet — primary key
    "plantNameAr",      # Care_Details / Suitability / Tasks / Month_Plants
    "nameEn",           # fallback: English name
    "nameScientific",   # fallback: scientific name
)

# ─────────────────────────────────────────────────────────────────
# 3.  Excel sheet names & join keys
#
#  New structure (plant_data_final_v4.xlsx) — 5 sheets:
#    Plants        → master plant data, keyed by nameAr
#    Care_Details  → long-form Arabic/English care text, keyed by plantNameAr
#    Suitability   → adjustment tips, keyed by plantNameAr
#    Tasks         → recurring task schedule, keyed by plantNameAr
#    Month_Plants  → monthly planting calendar, keyed by plantNameAr
#
#  REMOVED (old sheet names, no longer exist):
#    "plants_core", "plants_care", "plants_calendar_pal",
#    "plants_pests_diseases"
#
#  Join logic:
#    Plants.nameAr  ←→  <other_sheet>.plantNameAr
#    REMOVED old keys: "plant_id", "scientific_name", "arabic_name_primary"
# ─────────────────────────────────────────────────────────────────
EXCEL_SHEETS: list[str] = [
    "Plants",
    "Care_Details",
    "Suitability",
    "Tasks",
    "Month_Plants",
]

# Key used in the primary Plants sheet
PRIMARY_NAME_KEY: str = "nameAr"

# Key used in all secondary sheets to join back to Plants
SECONDARY_JOIN_KEY: str = "plantNameAr"

# EXCEL_JOIN_KEYS: kept for backward compatibility with plant_data_store.py
# Will be updated in plant_data_store.py in the next phase.
EXCEL_JOIN_KEYS: tuple[str, ...] = (
    "nameAr",       # Plants sheet primary key
    "plantNameAr",  # secondary sheets join key
    # REMOVED: "plant_id", "scientific_name", "arabic_name_primary"
)

# ─────────────────────────────────────────────────────────────────
# 4.  Value-cleaning helper
# ─────────────────────────────────────────────────────────────────
def clean_value(v: str) -> str:
    """
    Clean a raw Excel value for natural-text output:
      - Replace escaped / literal newlines with a space
      - Collapse multiple spaces
      - Strip surrounding whitespace
    """
    v = v.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    v = re.sub(r"\s{2,}", " ", v).strip()
    return v
