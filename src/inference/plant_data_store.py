"""
plant_data_store.py – Structured plant data loaded from multi-sheet Excel.

At startup the Excel is read ONCE.  Data from all sheets is merged
into a unified plant profile keyed by normalised Arabic name, giving
O(1) field lookups at inference time.

Sheets consumed
---------------
- plants_core           → basic info, light, temperature, soil, etc.
- plants_care           → watering, fertilizer, pruning, harvest, etc.
- plants_calendar_pal   → Palestine planting/harvest calendar
- plants_pests_diseases → pests & diseases

Public API
----------
load_plant_store(xlsx_path)   → int          number of plants loaded
get_plant_data(name)          → dict | None  unified profile for a plant
is_store_loaded()             → bool
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.constants import EXCEL_JOIN_KEYS, EXCEL_SHEETS, NAME_COLUMNS
from src.utils.arabic import normalize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Module‑level singleton store
# ─────────────────────────────────────────────────────────────────
_store: dict[str, dict] = {}        # norm_name → unified profile dict
_name_map: dict[str, str] = {}      # norm_name → original Arabic name


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────
def _read_sheet(xlsx_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """Try to read a sheet; return None on failure."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        if len(df) > 0 and "plant_id" in df.columns:
            first = df["plant_id"].iloc[0]
            if pd.isna(first) or isinstance(first, str):
                df = df.iloc[1:].reset_index(drop=True)
        logger.info("  Sheet '%s': %d rows, %d cols", sheet_name, len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.debug("  Sheet '%s' not found or unreadable: %s", sheet_name, exc)
        return None


def _find_join_key(df: pd.DataFrame) -> Optional[str]:
    for key in EXCEL_JOIN_KEYS:
        if key in df.columns:
            return key
    return None


def _merge_row_dicts(base: dict, extra: dict) -> dict:
    """Merge extra into base without overwriting existing non-NaN values."""
    for k, v in extra.items():
        if k not in base or base[k] is None:
            base[k] = v
    return base


def _df_to_dict_by_key(df: pd.DataFrame, key_col: str) -> dict[str, dict]:
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


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
def load_plant_store(xlsx_path: str | Path) -> int:
    """
    Read ALL sheets from the plant Excel file, merge them into
    unified plant profiles, and populate the in-memory store.

    Returns the number of unique plants loaded.
    """
    global _store, _name_map

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        logger.warning("Plant Excel not found at %s – store is empty.", xlsx_path)
        return 0

    logger.info("Loading plant data from %s …", xlsx_path)

    # ── Step 1: Try reading individual sheets ─────────────────────
    sheets: dict[str, pd.DataFrame] = {}
    for sheet_name in EXCEL_SHEETS:
        df = _read_sheet(xlsx_path, sheet_name)
        if df is not None and len(df) > 0:
            sheets[sheet_name] = df

    # ── Fallback: if no named sheets, read the first sheet ────────
    if not sheets:
        logger.info("  No named sheets found – reading default (first) sheet")
        try:
            df = pd.read_excel(xlsx_path)
            if len(df) > 0:
                if "plant_id" in df.columns:
                    first = df["plant_id"].iloc[0]
                    if pd.isna(first) or isinstance(first, str):
                        df = df.iloc[1:].reset_index(drop=True)
                else:
                    df = df.iloc[1:].reset_index(drop=True)
                sheets["default"] = df
                logger.info("  Default sheet: %d rows, %d cols", len(df), len(df.columns))
        except Exception as exc:
            logger.error("  Failed to read default sheet: %s", exc)
            return 0

    if not sheets:
        logger.warning("No usable data found in %s", xlsx_path)
        return 0

    # ── Step 2: Build unified profiles ────────────────────────────
    base_name = "plants_core" if "plants_core" in sheets else list(sheets.keys())[0]
    base_df = sheets.pop(base_name)
    base_key = _find_join_key(base_df)

    if base_key:
        profiles_by_key = _df_to_dict_by_key(base_df, base_key)
    else:
        profiles_by_key = {}
        for idx, row in base_df.iterrows():
            row_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            profiles_by_key[str(idx)] = row_dict

    # Merge additional sheets
    for sheet_name, extra_df in sheets.items():
        join_key = _find_join_key(extra_df)
        if join_key and join_key in extra_df.columns:
            extra_by_key = _df_to_dict_by_key(extra_df, join_key)
            for key_val, extra_row in extra_by_key.items():
                if key_val in profiles_by_key:
                    _merge_row_dicts(profiles_by_key[key_val], extra_row)
                else:
                    profiles_by_key[key_val] = extra_row
            logger.info("  Merged sheet '%s' via key '%s' (%d rows)",
                        sheet_name, join_key, len(extra_by_key))
        else:
            logger.debug("  Skipping sheet '%s' – no matching join key", sheet_name)

    # ── Step 3: Index by normalised Arabic name ───────────────────
    _store = {}
    _name_map = {}

    for profile in profiles_by_key.values():
        primary: Optional[str] = None
        for col in NAME_COLUMNS:
            if col in profile:
                candidate = str(profile[col]).strip()
                if candidate and candidate.lower() not in ("nan", "none", ""):
                    primary = candidate
                    break
        if not primary:
            continue

        norm = normalize(primary)
        _store[norm] = profile
        _name_map[norm] = primary

        # Index by alt names
        for col in NAME_COLUMNS:
            if col in profile:
                alt = str(profile[col]).strip()
                if alt and alt.lower() not in ("nan", "none", ""):
                    alt_norm = normalize(alt)
                    if alt_norm not in _store:
                        _store[alt_norm] = profile
                        _name_map[alt_norm] = primary

        # Index by english / scientific names
        for extra_col in ("english_name_primary", "scientific_name"):
            if extra_col in profile:
                extra_name = str(profile[extra_col]).strip()
                if extra_name and extra_name.lower() not in ("nan", "none", ""):
                    extra_norm = normalize(extra_name).lower()
                    if extra_norm not in _store:
                        _store[extra_norm] = profile
                        _name_map[extra_norm] = primary

    unique_count = len(set(_name_map.values()))
    logger.info(
        "Plant data store loaded: %d unique plants, %d name variants indexed.",
        unique_count, len(_store),
    )
    return unique_count


def get_plant_data(plant_name: str) -> Optional[dict]:
    """
    Look up a plant by name and return its unified profile dict
    (merged from all sheets), or ``None`` if not found.
    """
    if not _store:
        return None

    norm = normalize(plant_name)

    # 1. Exact normalised match
    if norm in _store:
        return _store[norm]

    # 2. Try stripping common prefixes
    for prefix in ("ال", "نبتة ", "نبات ", "عشبة "):
        stripped = norm
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
        if stripped in _store:
            return _store[stripped]

    # 3. Substring containment
    for key in _store:
        if norm in key or key in norm:
            return _store[key]

    return None


def get_plant_original_name(plant_name: str) -> Optional[str]:
    """Return the original (non-normalised) Arabic name, or None."""
    norm = normalize(plant_name)
    return _name_map.get(norm)


def get_all_plant_names() -> list[str]:
    """Return a list of all original plant names in the store."""
    return list(set(_name_map.values()))


def is_store_loaded() -> bool:
    """Check whether the plant store has data."""
    return len(_store) > 0
