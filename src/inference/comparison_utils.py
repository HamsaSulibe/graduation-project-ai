"""Comparison helpers for plant-vs-plant questions."""
from __future__ import annotations

from src.config.columns import resolve_column
from src.inference.answer_builders import _is_english
from src.config.assistant_messages import (
    _COMPARISON_CRITERIA_MAP,
    _COMPARISON_CRITERIA_MAP_EN,
    _DIFFICULTY_AR,
    _DIFFICULTY_RANK,
)
from src.utils.arabic import normalize


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
