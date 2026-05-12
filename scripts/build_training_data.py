"""
build_training_data.py – Generate cards.jsonl and train_pairs.jsonl from Excel.

Offline script: run once after updating the Excel file.
    python -m scripts.build_training_data

Updated to handle plant_data_final_v4.xlsx (5 sheets):
  Plants, Care_Details, Suitability, Tasks, Month_Plants

Removed (no longer in Excel):
  plant_id, arabic_name_primary, plants_core/care/_pal, pests_diseases sheet,
  build_planting_steps(), build_best_month_and_season(), build_pot_info(),
  AR_MONTHS dict, extract_month_numbers(), month_to_season()
"""

import re
from pathlib import Path

import pandas as pd
from datasets import Dataset

from src.config.constants import PRIMARY_NAME_KEY, SECONDARY_JOIN_KEY
from src.config.settings import XLSX_PATH

RAW_XLSX = Path(XLSX_PATH)

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────

def clean_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return re.sub(r"\s+", " ", s)


def _val(row: dict, col: str) -> str:
    """Return a clean string for *col* from *row*, or empty string."""
    v = row.get(col)
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return clean_text(v)


def _find_header_row(xlsx_path: Path, sheet_name: str,
                     probe_cols: tuple[str, ...]) -> int:
    """
    Scan up to 10 rows to find which row is the real column header.
    Returns 0-based index; defaults to 0.
    """
    try:
        df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name,
                               header=None, nrows=10)
        for i, row in df_raw.iterrows():
            row_vals = {str(v).strip() for v in row.values if pd.notna(v)}
            if sum(1 for c in probe_cols if c in row_vals) >= 2:
                return int(i)
    except Exception:
        pass
    return 0


def _read_sheet(xlsx_path: Path, sheet_name: str,
                probe_cols: tuple[str, ...]) -> pd.DataFrame | None:
    """Read *sheet_name*, auto-detecting the header row."""
    try:
        header_row = _find_header_row(xlsx_path, sheet_name, probe_cols)
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Month_Plants → per-plant Arabic summary string
# ─────────────────────────────────────────────────────────────────

def build_months_index(month_df: pd.DataFrame | None) -> dict[str, str]:
    """
    Returns  {nameAr: "مارس (ربيع) - ملاحظة، أبريل (ربيع)"}
    built from the Month_Plants sheet.
    """
    if month_df is None or SECONDARY_JOIN_KEY not in month_df.columns:
        return {}

    grouped: dict[str, list[str]] = {}
    for _, row in month_df.iterrows():
        key = str(row.get(SECONDARY_JOIN_KEY, "")).strip()
        if not key:
            continue
        month_name = _val(row, "monthName")
        season     = _val(row, "season")
        note       = _val(row, "plantingNoteAr")

        parts = []
        if month_name:
            parts.append(month_name)
        if season:
            parts.append(f"({season})")
        if note:
            parts.append(f"- {note}")

        entry = " ".join(parts).strip()
        if entry:
            grouped.setdefault(key, []).append(entry)

    return {k: "، ".join(v) for k, v in grouped.items()}


# ─────────────────────────────────────────────────────────────────
# Card builder
# ─────────────────────────────────────────────────────────────────

def build_fertilizer_text(row: dict) -> str:
    ftype  = _val(row, "fertilizerType")
    stage  = _val(row, "fertilizerStage")
    freq   = _val(row, "fertilizerFrequencyDays")
    notes  = _val(row, "fertilizerNotesAr")

    parts = []
    if ftype:
        parts.append(f"النوع: {ftype}.")
    if stage:
        parts.append(f"المرحلة: {stage}.")
    if freq and freq != "0":
        parts.append(f"التكرار: كل {freq} يوم.")
    if notes:
        parts.append(notes)
    return " ".join(parts).strip()


def row_to_card(row: dict, month_summary: str = "") -> str:
    """
    Build one Arabic text card for a single plant profile.
    *row* is the merged dict from Plants + Care_Details.
    *month_summary* is the pre-built string from Month_Plants.
    """
    name_ar     = _val(row, PRIMARY_NAME_KEY)   # nameAr
    name_en     = _val(row, "nameEn")
    scientific  = _val(row, "nameScientific")
    category    = _val(row, "category")
    difficulty  = _val(row, "difficultyLevel")
    description = _val(row, "shortDescriptionAr")

    # Temperature
    tmin = _val(row, "minTemp")
    tmax = _val(row, "maxTemp")

    # Watering
    watering_need     = _val(row, "wateringNeed")
    watering_interval = _val(row, "wateringIntervalDays")
    watering_info     = _val(row, "wateringInfoAr")    # Care_Details

    # Humidity
    humidity_pref  = _val(row, "humidityPreference")
    humidity_notes = _val(row, "humidityNotesAr")

    # Fertilizer
    fertilizer = build_fertilizer_text(row)

    # Care_Details text blocks
    light_info     = _val(row, "lightInfoAr")
    soil_info      = _val(row, "soilInfoAr")
    care_info      = _val(row, "careInfoAr")
    harvest_info   = _val(row, "harvestInfoAr")
    uses_info      = _val(row, "usesInfoAr")
    planting_steps = _val(row, "plantingStepsAr")

    # Planting meta (Plants sheet)
    planting_method = _val(row, "plantingMethod")
    spacing_min     = _val(row, "plantSpacingCmMin")
    spacing_max     = _val(row, "plantSpacingCmMax")
    germ_days       = _val(row, "germinationDays")
    seed_care       = _val(row, "seedCareInstructionsAr")
    harvest_min     = _val(row, "daysToHarvestMin")
    harvest_max     = _val(row, "daysToHarvestMax")

    parts = []

    # ── Identity ──────────────────────────────────────────────────
    title = " | ".join(p for p in [name_ar, scientific, name_en] if p)
    if title:
        parts.append(f"النبتة: {title}")
    if category:
        parts.append(f"التصنيف: {category}")
    if difficulty:
        parts.append(f"مستوى الصعوبة: {difficulty}")
    if description:
        parts.append(f"وصف: {description}")

    # ── Light ─────────────────────────────────────────────────────
    if light_info:
        parts.append(f"الضوء: {light_info}")

    # ── Watering ─────────────────────────────────────────────────
    if watering_info:
        parts.append(f"الري: {watering_info}")
    elif watering_need or watering_interval:
        segs = [s for s in [
            watering_need,
            f"كل {watering_interval} يوم" if watering_interval else "",
        ] if s]
        parts.append("الري: " + " | ".join(segs))

    # ── Soil ──────────────────────────────────────────────────────
    if soil_info:
        parts.append(f"التربة: {soil_info}")

    # ── Temperature ──────────────────────────────────────────────
    if tmin or tmax:
        parts.append(f"الحرارة المثالية: {tmin}–{tmax} °م")

    # ── Humidity ─────────────────────────────────────────────────
    if humidity_pref or humidity_notes:
        hum = " | ".join(s for s in [humidity_pref, humidity_notes] if s)
        parts.append(f"الرطوبة: {hum}")

    # ── Fertilizer ───────────────────────────────────────────────
    if fertilizer:
        parts.append(f"التسميد: {fertilizer}")

    # ── Planting steps ───────────────────────────────────────────
    if planting_steps:
        parts.append(f"خطوات الزراعة: {planting_steps}")
    else:
        # Fallback: build minimal steps from individual fields
        p_parts = []
        if planting_method:
            p_parts.append(f"طريقة الإكثار: {planting_method}.")
        if spacing_min or spacing_max:
            p_parts.append(f"التباعد: {spacing_min}–{spacing_max} سم.")
        if germ_days:
            p_parts.append(f"مدة الإنبات: {germ_days} يوم.")
        if seed_care:
            p_parts.append(seed_care)
        if p_parts:
            parts.append("خطوات الزراعة: " + " ".join(p_parts))

    # ── Harvest ───────────────────────────────────────────────────
    if harvest_min or harvest_max:
        parts.append(f"أيام حتى الحصاد: {harvest_min}–{harvest_max} يوم")
    if harvest_info:
        parts.append(f"الحصاد: {harvest_info}")

    # ── Uses ──────────────────────────────────────────────────────
    if uses_info:
        parts.append(f"الاستخدامات: {uses_info}")

    # ── General care ─────────────────────────────────────────────
    if care_info:
        parts.append(f"العناية: {care_info}")

    # ── Planting months (from Month_Plants sheet) ─────────────────
    if month_summary:
        parts.append(f"أشهر الزراعة المناسبة: {month_summary}")

    return "\n".join(p for p in parts if p).strip()


# ─────────────────────────────────────────────────────────────────
# Training-pair question builder
# ─────────────────────────────────────────────────────────────────

def build_questions(row: dict, card: str) -> list[tuple[str, str]]:
    name_ar = _val(row, PRIMARY_NAME_KEY)
    if not name_ar or not card:
        return []

    qs = [
        f"كم يحتاج {name_ar} ضوء؟",
        f"كيف أسقي {name_ar}؟",
        f"ما التربة المناسبة لـ {name_ar}؟",
        f"ما الحرارة المثالية لـ {name_ar}؟",
        f"اعطني معلومات عن {name_ar}",
        f"كيف أعتني بـ {name_ar}؟",
        f"كيف أزرع {name_ar}؟",
        f"ما خطوات زراعة {name_ar}؟",
        f"ما موسم زراعة {name_ar}؟",
        f"ما أفضل شهر لزراعة {name_ar}؟",
        f"شو السماد المناسب لـ {name_ar}؟",
        f"كيف تسميد {name_ar}؟",
        f"متى يُحصد {name_ar}؟",
        f"ما فوائد واستخدامات {name_ar}؟",
        f"ما الرطوبة المناسبة لـ {name_ar}؟",
    ]
    return [(q, card) for q in qs]


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"ضع ملف الإكسل الجديد هنا: {RAW_XLSX}"
        )

    # ── 1. Plants (primary sheet) ─────────────────────────────────
    plants_df = _read_sheet(
        RAW_XLSX, "Plants",
        probe_cols=(PRIMARY_NAME_KEY, "nameEn", "category", "wateringNeed"),
    )
    if plants_df is None or len(plants_df) == 0:
        raise ValueError("شيت Plants فارغ أو غير موجود.")
    if PRIMARY_NAME_KEY not in plants_df.columns:
        raise ValueError(
            f"عمود '{PRIMARY_NAME_KEY}' غير موجود في Plants. "
            f"الأعمدة المتاحة: {list(plants_df.columns)}"
        )

    # ── 2. Care_Details ───────────────────────────────────────────
    care_df = _read_sheet(
        RAW_XLSX, "Care_Details",
        probe_cols=(SECONDARY_JOIN_KEY, "lightInfoAr", "wateringInfoAr",
                    "plantingStepsAr"),
    )
    care_index: dict[str, dict] = {}
    if care_df is not None and SECONDARY_JOIN_KEY in care_df.columns:
        for _, row in care_df.iterrows():
            key = str(row.get(SECONDARY_JOIN_KEY, "")).strip()
            if key:
                care_index[key] = {
                    k: v for k, v in row.to_dict().items() if pd.notna(v)
                }

    # ── 3. Month_Plants ───────────────────────────────────────────
    month_df = _read_sheet(
        RAW_XLSX, "Month_Plants",
        probe_cols=(SECONDARY_JOIN_KEY, "monthNumber", "monthName", "season"),
    )
    months_index = build_months_index(month_df)

    # ── 4. Build cards & training pairs ──────────────────────────
    pairs: list[dict] = []
    cards: list[dict] = []

    for _, plants_row in plants_df.iterrows():
        name_ar = str(plants_row.get(PRIMARY_NAME_KEY, "")).strip()
        if not name_ar or name_ar.lower() in ("nan", "none", ""):
            continue

        # Start with Plants columns, then overlay Care_Details
        row: dict = {k: v for k, v in plants_row.to_dict().items() if pd.notna(v)}
        if name_ar in care_index:
            for k, v in care_index[name_ar].items():
                if k not in row:          # don't overwrite Plants values
                    row[k] = v

        month_summary = months_index.get(name_ar, "")
        card = row_to_card(row, month_summary)
        if not card:
            continue

        cards.append({"text": card})
        for q, pos in build_questions(row, card):
            pairs.append({"query": q, "positive": pos})

    # ── 5. Save outputs ───────────────────────────────────────────
    Dataset.from_list(cards).to_json(
        str(OUT_DIR / "cards.jsonl"), orient="records", lines=True
    )
    Dataset.from_list(pairs).to_json(
        str(OUT_DIR / "train_pairs.jsonl"), orient="records", lines=True
    )

    print("Done")
    print(f"plants  : {len(cards)}")
    print(f"cards   : {len(cards)}  -> {OUT_DIR / 'cards.jsonl'}")
    print(f"pairs   : {len(pairs)}  -> {OUT_DIR / 'train_pairs.jsonl'}")


if __name__ == "__main__":
    main()

