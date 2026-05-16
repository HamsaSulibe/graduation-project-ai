"""Remaining context-formatting helpers kept outside the compatibility facade."""
from __future__ import annotations

import re
from typing import Optional

from src.config.settings import SAFE_NO_ANSWER
from src.inference.plant_extraction import extract_plant_name


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
