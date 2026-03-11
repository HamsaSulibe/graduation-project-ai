"""
build_training_data.py – Generate cards.jsonl and train_pairs.jsonl from Excel.

Offline script: run once after updating the Excel file.
    python -m scripts.build_training_data

Uses shared modules from src.config for column aliases and constants.
"""

import re
from pathlib import Path

import pandas as pd
from datasets import Dataset

from src.config.columns import pick_value
from src.config.settings import XLSX_PATH

RAW_XLSX = Path(XLSX_PATH)
ALT_XLSX = Path("Book 1.xlsx")  # fallback if RAW_XLSX not found

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AR_MONTHS = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "الشتاء"
    if m in (3, 4, 5):
        return "الربيع"
    if m in (6, 7, 8):
        return "الصيف"
    return "الخريف"


def clean_text(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_month_numbers(text: str) -> list[int]:
    """Extract month numbers 1..12 from messy text."""
    if not text or text.strip() in ("مش موجود", "غير متوفر", "غير متوفرة"):
        return []
    nums = re.findall(r"\b(1[0-2]|[1-9])\b", text)
    out = []
    for n in nums:
        try:
            v = int(n)
            if 1 <= v <= 12 and v not in out:
                out.append(v)
        except Exception:
            pass
    return out


def build_planting_steps(row: dict) -> str:
    propagation = pick_value(row, "propagation_method_primary")
    spacing = pick_value(row, "plant_spacing_cm")
    germ_min = pick_value(row, "germination_days_min")
    germ_max = pick_value(row, "germination_days_max")
    transplanting_ok = pick_value(row, "transplanting_ok")
    planting_months = pick_value(row, "planting_months_pal")

    steps = []
    if propagation:
        steps.append(f"طريقة الإكثار: {propagation}.")
    if planting_months and planting_months != "مش موجود":
        steps.append(f"موعد الزراعة (فلسطين): {planting_months}.")
    if spacing:
        steps.append(f"التباعد بين النباتات: {spacing}.")
    if germ_min or germ_max:
        gm = germ_min if germ_min else "?"
        gx = germ_max if germ_max else "?"
        steps.append(f"مدة الإنبات: {gm}–{gx} يوم.")
    if transplanting_ok:
        steps.append(f"النقل/الشتل: {transplanting_ok}.")
    return " ".join(steps).strip()


def build_best_month_and_season(row: dict) -> tuple[str, str]:
    planting_months = pick_value(row, "planting_months_pal")
    months = extract_month_numbers(planting_months)
    if not months:
        return "", ""
    best = months[0]
    best_month = AR_MONTHS.get(best, str(best))
    season = month_to_season(best)
    return best_month, season


def build_pot_info(row: dict) -> str:
    container_possible = pick_value(row, "container_possible")
    pot_diam = pick_value(row, "pot_diameter_cm_min")
    pot_depth = pick_value(row, "pot_depth_cm_min")

    parts = []
    if container_possible:
        parts.append(f"مناسب للأصيص: {container_possible}.")
    if pot_diam:
        parts.append(f"قطر أصيص أدنى (cm): {pot_diam}.")
    if pot_depth:
        parts.append(f"عمق أصيص أدنى (cm): {pot_depth}.")
    return " ".join(parts).strip()


def build_fertilizer_text(row: dict) -> str:
    need = pick_value(row, "fertilizer_need")
    ftype = pick_value(row, "fertilizer_type")
    freq_days = pick_value(row, "fertilizer_frequency_days")

    parts = []
    if need:
        parts.append(f"الاحتياج: {need}.")
    if ftype:
        parts.append(f"النوع: {ftype}.")
    if freq_days and freq_days not in ("0", "غير متوفر", "غير متوفرة"):
        parts.append(f"التكرار التقريبي: كل {freq_days} يوم.")
    return " ".join(parts).strip()


def row_to_card(row: dict) -> str:
    name_ar = pick_value(row, "arabic_name_primary")
    name_en = pick_value(row, "english_name_primary")
    scientific = pick_value(row, "scientific_name")

    light = pick_value(row, "light_level")
    sun_hours_min = pick_value(row, "min_sun_hours")
    sun_hours_max = pick_value(row, "max_sun_hours", "sun_hours_max")

    watering_need = pick_value(row, "watering_need")
    watering_rule = pick_value(row, "watering_rule_text")

    soil = pick_value(row, "soil_texture_preference")
    ph_min = pick_value(row, "soil_ph_min")
    ph_max = pick_value(row, "soil_ph_max")
    drainage = pick_value(row, "drainage_need")

    tmin = pick_value(row, "temperature_optimal_min_c")
    tmax = pick_value(row, "temperature_optimal_max_c")

    pests = pick_value(row, "common_pests")
    diseases = pick_value(row, "common_diseases")

    planting_steps = build_planting_steps(row)
    fertilizer = build_fertilizer_text(row)
    best_month, season = build_best_month_and_season(row)
    pot_info = build_pot_info(row)

    notes = pick_value(
        row,
        "short_summary",
        "beginner_tips",
        "traditional_uses",
        "safety_warning_ar",
        "notes_ar",
        "notes",
    )

    parts = []
    title = " | ".join([p for p in [name_ar, scientific, name_en] if p])
    if title:
        parts.append(f"النبتة: {title}")

    if light or sun_hours_min or sun_hours_max:
        hs = ""
        if sun_hours_min or sun_hours_max:
            hs = f" (ساعات شمس تقريبًا: {sun_hours_min}-{sun_hours_max})"
        parts.append(f"الضوء: {light}{hs}".strip())

    if watering_need or watering_rule:
        rr = " ".join([p for p in [watering_need, watering_rule] if p]).strip()
        if rr:
            parts.append(f"الري: {rr}")

    if soil or ph_min or ph_max or drainage:
        segs = []
        if soil:
            segs.append(f"{soil}")
        if ph_min or ph_max:
            segs.append(f"pH: {ph_min}-{ph_max}".strip())
        if drainage:
            segs.append(f"تصريف: {drainage}")
        parts.append("التربة: " + " | ".join(segs))

    if tmin or tmax:
        parts.append(f"الحرارة المثالية: {tmin}-{tmax}°C".strip())

    if pests:
        parts.append(f"الآفات الشائعة: {pests}")
    if diseases:
        parts.append(f"الأمراض الشائعة: {diseases}")

    if planting_steps:
        parts.append(f"خطوات الزراعة: {planting_steps}")
    if fertilizer:
        parts.append(f"التسميد: {fertilizer}")
    if season:
        parts.append(f"موسم الزراعة: {season}")
    if best_month:
        parts.append(f"أفضل شهر للزراعة في فلسطين: {best_month}")
    if pot_info:
        parts.append(f"الأصيص: {pot_info}")
    if notes:
        parts.append(f"ملاحظات: {notes}")

    return "\n".join([p for p in parts if p]).strip()


def build_questions(row: dict):
    name_ar = pick_value(row, "arabic_name_primary")
    if not name_ar:
        return []

    card = row_to_card(row)
    qs = [
        f"كم يحتاج {name_ar} ضوء؟",
        f"كيف أسقي {name_ar}؟",
        f"ما التربة المناسبة لـ {name_ar}؟",
        f"ما الحرارة المثالية لـ {name_ar}؟",
        f"ما الآفات الشائعة لـ {name_ar}؟",
        f"اعطني معلومات عن {name_ar}",
        f"كيف أعتني بـ {name_ar}؟",
        f"كيف أزرع {name_ar}؟",
        f"ما خطوات زراعة {name_ar}؟",
        f"ما موسم زراعة {name_ar}؟",
        f"ما أفضل شهر لزراعة {name_ar} في فلسطين؟",
        f"شو السماد المناسب لـ {name_ar}؟",
        f"كيف تسميد {name_ar}؟",
        f"هل {name_ar} مناسب للأصيص؟",
        f"شو أفضل أصيص لـ {name_ar}؟",
    ]
    return [(q, card) for q in qs]


def main():
    xlsx_path = RAW_XLSX if RAW_XLSX.exists() else ALT_XLSX
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"ضع ملف الإكسل هنا: {RAW_XLSX} أو بجانب السكربت باسم: {ALT_XLSX}"
        )

    df = pd.read_excel(xlsx_path)

    # تجاهل أول صف إذا كان شرح للأعمدة
    if "plant_id" in df.columns:
        first = df["plant_id"].iloc[0]
        if pd.isna(first) or isinstance(first, str):
            df = df.iloc[1:].reset_index(drop=True)
    else:
        df = df.iloc[1:].reset_index(drop=True)

    rows = df.to_dict(orient="records")

    pairs, cards = [], []
    for r in rows:
        card = row_to_card(r)
        if card:
            cards.append({"text": card})
        for q, pos in build_questions(r):
            pairs.append({"query": q, "positive": pos})

    Dataset.from_list(cards).to_json(
        str(OUT_DIR / "cards.jsonl"), orient="records", lines=True
    )
    Dataset.from_list(pairs).to_json(
        str(OUT_DIR / "train_pairs.jsonl"), orient="records", lines=True
    )

    print("Done")
    print(f"cards: {len(cards)}  -> {OUT_DIR / 'cards.jsonl'}")
    print(f"pairs: {len(pairs)}  -> {OUT_DIR / 'train_pairs.jsonl'}")


if __name__ == "__main__":
    main()
