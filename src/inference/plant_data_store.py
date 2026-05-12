"""
plant_data_store.py – Structured plant data loaded from multi-sheet Excel.

At startup the Excel is read ONCE.  Data from all sheets is merged
into a unified plant profile keyed by normalised Arabic name, giving
O(1) field lookups at inference time.

Sheets consumed (plant_data_final_v4.xlsx)
------------------------------------------
- Plants        → master plant data, keyed by nameAr
- Care_Details  → long-form Arabic/English care text, keyed by plantNameAr
- Suitability   → adjustment tips per plant, keyed by plantNameAr
- Tasks         → recurring task schedule, keyed by plantNameAr
- Month_Plants  → monthly planting calendar, keyed by plantNameAr

Merge strategy
--------------
- Plants        → flat merge into profile (the base)
- Care_Details  → flat merge (one row per plant, adds text fields)
- Suitability   → profile["suitability_tips"]  : list[str]
- Tasks         → profile["tasks"]             : list[dict]
- Month_Plants  → profile["month_plants"]      : list[dict]

Public API
----------
load_plant_store(xlsx_path)   → int          number of plants loaded
get_plant_data(name)          → dict | None  unified profile for a plant
is_store_loaded()             → bool
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.constants import (
    EXCEL_SHEETS,
    NAME_COLUMNS,
    PRIMARY_NAME_KEY,
    SECONDARY_JOIN_KEY,
)
from src.utils.arabic import normalize, normalize_name

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Module‑level singleton store
# ─────────────────────────────────────────────────────────────────
_store: dict[str, dict] = {}        # norm_name → unified profile dict
_name_map: dict[str, str] = {}      # norm_name → original Arabic name
_lookup_names: set[str] = set()     # Arabic/English/scientific names users may type

_LOOKUP_SPLIT_RE = re.compile(r"\s*(?:/|\||,|;|\bor\b)\s*", re.IGNORECASE)
_LOOKUP_PUNCT_RE = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")

# Sheets that are one-to-many (stored as lists inside the profile)
_LIST_SHEETS: dict[str, str] = {
    "Suitability":  "suitability_tips",
    "Tasks":        "tasks",
    "Month_Plants": "month_plants",
}

# Columns to keep from each list sheet (others are dropped)
_LIST_SHEET_COLS: dict[str, tuple[str, ...]] = {
    "Suitability":  ("adjustmentTipAr", "adjustmentTipEn"),
    "Tasks":        ("taskType", "intervalDays"),
    "Month_Plants": ("monthNumber", "plantingNoteAr", "plantingNoteEn",
                     "monthName", "season"),
}


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _find_header_row(xlsx_path: Path, sheet_name: str,
                     probe_cols: tuple[str, ...]) -> int:
    """
    Scan the first 10 rows of *sheet_name* to find the row index
    that actually contains the column headers.

    Returns the 0-based row index of the header, or 0 as a safe
    default if no clear header row is found.
    """
    try:
        df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name,
                               header=None, nrows=10)
        for i, row in df_raw.iterrows():
            # Normalize cell values: strip whitespace and Excel merged-cell
            # artifacts like "plantNameAr+A3:K5" → "plantNameAr"
            row_vals = {
                re.sub(r'\+[A-Z]+\d+:[A-Z]+\d+$', '', str(v)).strip()
                for v in row.values if pd.notna(v)
            }
            matches = sum(1 for col in probe_cols if col in row_vals)
            if matches >= 2:
                return int(i)
    except Exception:
        pass
    return 0


def _read_sheet(xlsx_path: Path, sheet_name: str,
                probe_cols: tuple[str, ...]) -> Optional[pd.DataFrame]:
    """
    Read *sheet_name* from *xlsx_path*.

    Detects the real header row automatically so that any descriptive
    rows above the header are skipped.  Returns None on any error or
    when the sheet does not exist.
    """
    try:
        header_row = _find_header_row(xlsx_path, sheet_name, probe_cols)
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row)
        # Strip leading/trailing spaces and fix Excel merged-cell artifacts
        # e.g. "plantNameAr+A3:K5" → "plantNameAr"
        df.columns = [
            re.sub(r'\+[A-Z]+\d+:[A-Z]+\d+$', '', str(c)).strip()
            for c in df.columns
        ]
        # Drop completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)
        logger.info("  Sheet '%s': header at row %d, %d data rows, %d cols",
                    sheet_name, header_row, len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.debug("  Sheet '%s' not found or unreadable: %s", sheet_name, exc)
        return None


def _merge_row_dicts(base: dict, extra: dict) -> dict:
    """Merge extra into base without overwriting existing non-NaN values."""
    for k, v in extra.items():
        if k not in base or base[k] is None:
            base[k] = v
    return base


def _join_key(name: str) -> str:
    """Normalized plant-name key for cross-sheet joins."""
    return normalize_name(str(name)).lower().strip()


def _lookup_key(name: str) -> str:
    """Normalize Arabic/English/scientific names for direct lookup."""
    if not name:
        return ""
    key = normalize_name(str(name)).lower()
    key = _LOOKUP_PUNCT_RE.sub(" ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _lookup_variants(name: str) -> set[str]:
    """Return useful lookup variants for a stored name or alias."""
    raw = str(name or "").strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return set()

    variants: set[str] = {raw}

    # Names such as "Syrian thyme / Wild thyme" should resolve by either side.
    for part in _LOOKUP_SPLIT_RE.split(raw):
        part = part.strip(" ()[]{}")
        if part:
            variants.add(part)

    # Also keep text without parenthetical qualifiers.
    without_parens = re.sub(r"\([^)]*\)", " ", raw)
    without_parens = re.sub(r"\s+", " ", without_parens).strip()
    if without_parens:
        variants.add(without_parens)

    keys = {_lookup_key(v) for v in variants}
    return {k for k in keys if k}


def _register_lookup_name(profile: dict, original_name: str, lookup_name: str) -> int:
    """Register one display name/alias and return how many new keys were added."""
    added = 0
    for key in _lookup_variants(lookup_name):
        if key not in _store:
            _store[key] = profile
            _name_map[key] = original_name
            added += 1
        _lookup_names.add(lookup_name)
        if key != lookup_name:
            _lookup_names.add(key)
    return added


def _df_to_dict_by_key(df: pd.DataFrame, key_col: str) -> dict[str, dict]:
    """Index *df* rows by *key_col*.  One-to-one: last row wins on collision."""
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        key_val = row.get(key_col)
        if pd.isna(key_val):
            continue
        key_str = str(key_val).strip()
        if not key_str:
            continue
        row_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        if key_str in result:
            _merge_row_dicts(result[key_str], row_dict)
        else:
            result[key_str] = row_dict
    return result


def _df_to_list_by_key(df: pd.DataFrame, key_col: str,
                       keep_cols: tuple[str, ...]) -> dict[str, list[dict]]:
    """
    Group *df* rows by *key_col* into lists.
    Only *keep_cols* are retained in each row dict.
    """
    result: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        key_val = row.get(key_col)
        if pd.isna(key_val):
            continue
        key_str = str(key_val).strip()
        if not key_str:
            continue
        entry = {
            col: row[col]
            for col in keep_cols
            if col in df.columns and pd.notna(row.get(col))
        }
        result.setdefault(key_str, []).append(entry)
    return result


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
def load_plant_store(xlsx_path: str | Path) -> int:
    """
    Read all configured sheets from the plant Excel file, merge them
    into unified plant profiles, and populate the in-memory store.

    Returns the number of unique plants loaded.
    """
    global _store, _name_map, _lookup_names

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        logger.warning("Plant Excel not found at %s – store is empty.", xlsx_path)
        return 0

    logger.info("Loading plant data from %s …", xlsx_path)

    # ── Step 1: Read Plants (primary sheet) ───────────────────────
    plants_df = _read_sheet(
        xlsx_path, "Plants",
        probe_cols=(PRIMARY_NAME_KEY, "nameEn", "category", "wateringNeed"),
    )
    if plants_df is None or len(plants_df) == 0:
        logger.error("  'Plants' sheet is missing or empty – aborting load.")
        return 0

    if PRIMARY_NAME_KEY not in plants_df.columns:
        logger.error(
            "  'Plants' sheet has no '%s' column. Found: %s",
            PRIMARY_NAME_KEY, list(plants_df.columns),
        )
        return 0

    # Build initial profiles keyed by nameAr (original text, not normalised)
    profiles_by_name: dict[str, dict] = {}
    profiles_by_norm: dict[str, dict] = {}
    for _, row in plants_df.iterrows():
        name_val = row.get(PRIMARY_NAME_KEY)
        if pd.isna(name_val):
            continue
        name_str = str(name_val).strip()
        if not name_str:
            continue
        profile = {
            k: v for k, v in row.to_dict().items() if pd.notna(v)
        }
        profiles_by_name[name_str] = profile
        norm_key = _join_key(name_str)
        if norm_key:
            profiles_by_norm.setdefault(norm_key, profile)

    logger.info("  Plants base: %d plants", len(profiles_by_name))

    # ── Step 2: Flat-merge Care_Details ──────────────────────────
    care_df = _read_sheet(
        xlsx_path, "Care_Details",
        probe_cols=(SECONDARY_JOIN_KEY, "lightInfoAr", "soilInfoAr",
                    "wateringInfoAr", "plantingStepsAr"),
    )
    if care_df is not None and len(care_df) > 0:
        if SECONDARY_JOIN_KEY in care_df.columns:
            care_by_name = _df_to_dict_by_key(care_df, SECONDARY_JOIN_KEY)
            merged = 0
            for plant_name, care_row in care_by_name.items():
                profile = profiles_by_norm.get(_join_key(plant_name))
                if profile is not None:
                    _merge_row_dicts(profile, care_row)
                    merged += 1
            logger.info("  Care_Details: merged %d/%d plants", merged, len(care_by_name))
        else:
            logger.warning(
                "  Care_Details: '%s' column not found. Columns: %s",
                SECONDARY_JOIN_KEY, list(care_df.columns),
            )

    # ── Step 3: Attach list sheets ────────────────────────────────
    for sheet_name, profile_key in _LIST_SHEETS.items():
        keep_cols = _LIST_SHEET_COLS[sheet_name]
        probe = (SECONDARY_JOIN_KEY,) + keep_cols

        sheet_df = _read_sheet(xlsx_path, sheet_name, probe_cols=probe)
        if sheet_df is None or len(sheet_df) == 0:
            logger.debug("  Sheet '%s' empty or missing – skipping.", sheet_name)
            continue

        if SECONDARY_JOIN_KEY not in sheet_df.columns:
            logger.warning(
                "  Sheet '%s': '%s' column not found. Columns: %s",
                sheet_name, SECONDARY_JOIN_KEY, list(sheet_df.columns),
            )
            continue

        list_by_name = _df_to_list_by_key(sheet_df, SECONDARY_JOIN_KEY, keep_cols)
        attached = 0
        for plant_name, items in list_by_name.items():
            profile = profiles_by_norm.get(_join_key(plant_name))
            if profile is not None:
                profile[profile_key] = items
                attached += 1
        logger.info("  %s → '%s': attached to %d/%d plants",
                    sheet_name, profile_key, attached, len(list_by_name))

    # ── Step 4: Index by normalised names ────────────────────────
    _store = {}
    _name_map = {}
    _lookup_names = set()

    for original_name, profile in profiles_by_name.items():
        _register_lookup_name(profile, original_name, original_name)
        _lookup_names.add(original_name)
        display_variant = _join_key(original_name)
        if display_variant:
            _register_lookup_name(profile, original_name, display_variant)
        if display_variant and display_variant != _lookup_key(original_name):
            _lookup_names.add(display_variant)

        # Also index by English and scientific names when available
        for extra_col in ("nameEn", "nameScientific"):
            extra_val = profile.get(extra_col)
            _register_lookup_name(profile, original_name, str(extra_val or ""))

    # ── Step 5: Register explicit aliases (plant_aliases.py) ─────
    try:
        from src.config.plant_aliases import PLANT_ALIASES
        alias_count = 0
        for official_name, aliases in PLANT_ALIASES.items():
            official_norm = normalize(official_name)
            official_join = _join_key(official_name)
            # The official name may be stored under a slightly different
            # normalised key – try substring fallback to find it.
            profile = _store.get(official_norm.lower()) or profiles_by_norm.get(official_join)
            if profile is None:
                for key, val in _store.items():
                    if official_norm.lower() in key or key in official_norm.lower():
                        profile = val
                        break
            if profile is None:
                logger.debug("  Alias skipped (official name '%s' not in store)", official_name)
                continue
            canonical_name = str(profile.get("nameAr") or official_name).strip()
            for alias in aliases:
                alias_count += _register_lookup_name(profile, canonical_name, alias)
                alias_variant = _join_key(alias)
                if alias_variant:
                    alias_count += _register_lookup_name(profile, canonical_name, alias_variant)
        logger.info("  Aliases registered: %d extra keys from plant_aliases.py", alias_count)
    except ImportError:
        logger.debug("  plant_aliases.py not found – skipping alias registration.")

    unique_count = len(profiles_by_name)
    logger.info(
        "Plant data store loaded: %d unique plants, %d name variants indexed.",
        unique_count, len(_store),
    )
    return unique_count


def is_store_loaded() -> bool:
    """Return True when the store has been populated."""
    return bool(_store)


def get_plant_data(plant_name: str) -> Optional[dict]:
    """
    Look up a plant by name and return its unified profile dict, or
    ``None`` if not found.

    Profile shape
    -------------
    {
      # flat fields from Plants + Care_Details
      "nameAr": "...",
      "nameEn": "...",
      "nameScientific": "...",
      "category": "...",
      "shortDescriptionAr": "...",
      "wateringNeed": "...",
      "minTemp": ...,  "maxTemp": ...,
      "fertilizerType": "...",
      "lightInfoAr": "...",
      "soilInfoAr": "...",
      "wateringInfoAr": "...",
      "careInfoAr": "...",
      "harvestInfoAr": "...",
      "usesInfoAr": "...",
      "plantingStepsAr": "...",
      # ... (all other Plants / Care_Details columns)

      # attached lists
      "suitability_tips": [{"adjustmentTipAr": "..."}, ...],
      "tasks":            [{"taskType": "...", "intervalDays": ...}, ...],
      "month_plants":     [{"monthNumber": ..., "monthName": "...",
                            "season": "...", "plantingNoteAr": "...",
                            "plantingNoteEn": "..."}, ...],
    }
    """
    if not _store:
        return None

    norm = _lookup_key(plant_name)

    # 1. Exact normalised match
    if norm in _store:
        return _store[norm]

    # 2. Try stripping common Arabic prefixes (longest first)
    for prefix in ("بال", "وال", "فال", "كال", "لل", "ال",
                   "نبتة ", "نبات ", "عشبة ", "ب", "ل", "ك"):
        if norm.startswith(prefix) and len(norm) > len(prefix) + 1:
            stripped = _lookup_key(norm[len(prefix):].strip())
            if stripped in _store:
                return _store[stripped]

    # 3. Substring containment fallback
    for key in _store:
        if norm in key or key in norm:
            return _store[key]

    return None


def get_plant_original_name(plant_name: str) -> Optional[str]:
    """Return the original (non-normalised) Arabic name, or None."""
    norm = _lookup_key(plant_name)
    return _name_map.get(norm)


def get_all_plant_names() -> list[str]:
    """Return a list of all original plant names in the store."""
    return list(set(_name_map.values()))


def get_all_plant_lookup_names() -> list[str]:
    """Return Arabic, English, scientific, and alias names for matching."""
    if _lookup_names:
        return sorted(_lookup_names, key=lambda n: len(_lookup_key(n)), reverse=True)
    return get_all_plant_names()


def get_plant_display_name(plant_name: str, language: str = "ar") -> Optional[str]:
    """Return the plant name in the requested display language."""
    profile = get_plant_data(plant_name)
    if not profile:
        return None
    if (language or "").lower().startswith("en"):
        name_en = profile.get("nameEn")
        if name_en is not None and str(name_en).strip():
            return str(name_en).strip()
    name_ar = profile.get("nameAr") or get_plant_original_name(plant_name)
    return str(name_ar).strip() if name_ar else None


def is_store_loaded() -> bool:
    """Check whether the plant store has data."""
    return len(_store) > 0
