"""
intent_fields.py – Question-intent classification & field-mapping layer.

Design
------
* Intent registry lives in ``src.config.intents`` (pure config).
  Adding a new intent = touching only that file.
* This module contains only logic: classification, type queries,
  field gathering, field resolution, and context building.

Public API
----------
classify_intents(question)                     → list[str]
is_direct_intent(intents)                      → bool
is_multi_field_intent(intents)                 → bool
get_fields_for_intents(intents)                → dict[col, label]
build_field_context_ar(plants, intents)        → str
"""

from __future__ import annotations

from typing import Optional

from src.config.columns import resolve_column
from src.config.intents import INTENTS
from src.utils.arabic import normalize

# Fast name → intent lookup  (derived from config, not hardcoded here)
_INTENT_INDEX: dict[str, dict] = {i["name"]: i for i in INTENTS}


# ─────────────────────────────────────────────────────────────────
# 1.  Intent classification
# ─────────────────────────────────────────────────────────────────
def classify_intents(question: str) -> list[str]:
    """
    Analyse *question* and return matching intent names.

    Multi-field intents are checked FIRST so that broad questions
    like "كيف أعتني بالنعناع" land on care_summary rather than
    a dozen individual intents.

    Falls back to ``["general_summary"]`` if nothing matches.
    """
    q_norm = normalize(question).lower()
    q_words = set(q_norm.split())

    multi_match: list[str] = []
    direct_match: list[str] = []

    for intent in INTENTS:
        for kw in intent["keywords"]:
            kw_norm = normalize(kw).lower()
            if len(kw_norm) <= 2:
                if kw_norm in q_words:
                    target = multi_match if not intent.get("direct", True) else direct_match
                    target.append(intent["name"])
                    break
            else:
                if kw_norm in q_norm:
                    target = multi_match if not intent.get("direct", True) else direct_match
                    target.append(intent["name"])
                    break

    if multi_match:
        result = multi_match
    elif direct_match:
        result = direct_match
    else:
        result = ["general_summary"]

    seen: set[str] = set()
    unique: list[str] = []
    for m in result:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


# ─────────────────────────────────────────────────────────────────
# 2.  Intent type queries
# ─────────────────────────────────────────────────────────────────
def is_direct_intent(intents: list[str]) -> bool:
    """True when ALL matched intents are direct (no LLM needed)."""
    if not intents:
        return False
    return all(_INTENT_INDEX.get(i, {}).get("direct", True) for i in intents)


def is_multi_field_intent(intents: list[str]) -> bool:
    """True when at least one intent is multi-field (LLM rewrite)."""
    return any(not _INTENT_INDEX.get(i, {}).get("direct", True) for i in intents)


def get_fields_for_intents(intents: list[str]) -> dict[str, str]:
    """Collect all {column: arabic_label} needed by the given intents."""
    fields: dict[str, str] = {}
    for intent_name in intents:
        intent_def = _INTENT_INDEX.get(intent_name)
        if intent_def:
            fields.update(intent_def["fields"])
    return fields


# ─────────────────────────────────────────────────────────────────
# 3.  Field resolution
# ─────────────────────────────────────────────────────────────────
def resolve_fields_for_plant(
    plant_data: dict,
    fields: dict[str, str],
) -> dict[str, str]:
    """
    Given a plant profile and a {col: label} dict,
    return {label: resolved_value} for all non-empty columns.
    """
    result: dict[str, str] = {}
    for col_name, ar_label in fields.items():
        val = resolve_column(plant_data, col_name)
        if val is not None:
            result[ar_label] = val
    return result


# ─────────────────────────────────────────────────────────────────
# 4.  Build structured Arabic context (for prompt or direct answer)
# ─────────────────────────────────────────────────────────────────
def build_field_context_ar(
    plants: list[dict],
    intents: list[str],
) -> str:
    """
    Build a clean, labelled Arabic context block from structured
    plant data, filtered to only the fields needed by *intents*.

    Parameters
    ----------
    plants : list[dict]
        Each item: ``{"name": "<Arabic name>", "data": {col: value, …}}``.
    intents : list[str]
        Intent names returned by :func:`classify_intents`.

    Returns
    -------
    str
        Formatted Arabic text ready for injection into the prompt.
    """
    seen_names: set[str] = set()
    sections: list[str] = []

    for plant in plants:
        name = plant["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        data = plant["data"]

        plant_lines: list[str] = [f"=== نبتة: {name} ==="]

        for intent_name in intents:
            intent_def = _INTENT_INDEX.get(intent_name)
            if intent_def is None:
                continue

            present_lines: list[str] = []
            for col_name, ar_label in intent_def["fields"].items():
                value = resolve_column(data, col_name)
                if value is not None:
                    present_lines.append(f"• {ar_label}: {value}")

            if present_lines:
                plant_lines.append(f"\n📌 {intent_def['label_ar']}:")
                plant_lines.extend(present_lines)
            else:
                plant_lines.append(f"\n📌 {intent_def['label_ar']}:")
                plant_lines.append("  لا تتوفر بيانات لهذا الموضوع حاليًا.")

        sections.append("\n".join(plant_lines))

    return "\n\n".join(sections)
