"""
comparison_service.py – Handles plant comparison logic for the Garssa
Smart Plant Assistant.

Exposes a single entry-point: handle_comparison_request().
No Gemini / LLM calls are made here; answers are built purely from
structured Excel data via build_comparison_answer().
No plant names are hardcoded.
"""
import logging
from typing import Optional

from src.inference.context_utils import (
    build_comparison_answer,
    detect_comparison_criterion,
    extract_all_plants_from_question,
)
from src.inference.plant_data_store import (
    get_all_plant_lookup_names,
    get_plant_display_name,
    get_plant_data,
    get_plant_original_name,
    is_store_loaded,
)
from src.inference.conversation_context import (
    extract_pending_comparison_plants_from_history,
)

logger = logging.getLogger(__name__)


def handle_comparison_request(
    question: str,
    effective_question: str,
    history_messages: list[dict[str, str]],
    intents: list[str],
    pending_comparison_plants: Optional[list[str]] = None,
    language: str = "ar",
) -> Optional[dict]:
    """
    Handle a plant-comparison request end-to-end.

    Returns a fully-formed response dict (ready to return from the endpoint)
    if the question is a comparison, or None if it is not.

    Parameters
    ----------
    question:
        The original (unmodified) user question – used in the response body.
    effective_question:
        The context-enriched question used for plant extraction.
    history_messages:
        Normalised conversation history (list of {role, content} dicts).
    intents:
        Already-detected intents for the current turn.
    pending_comparison_plants:
        Optional list of plant names pre-resolved from history (pass the
        result of extract_pending_comparison_plants_from_history if already
        computed, to avoid re-computing it here).
    """
    if "comparison" not in intents:
        return None

    if not is_store_loaded():
        return None

    all_names = get_all_plant_lookup_names()
    cmp_plants = extract_all_plants_from_question(effective_question, all_names)

    # ── Deduplicate: two names → same profile object = same plant ──────────
    seen_pids: set[int] = set()
    deduped: list[str] = []
    for cp in cmp_plants:
        profile = get_plant_data(cp)
        if profile is not None:
            pid = id(profile)
            if pid not in seen_pids:
                seen_pids.add(pid)
                deduped.append(cp)
        else:
            deduped.append(cp)
    cmp_plants = deduped

    # ── Need at least 2 plants ─────────────────────────────────────────────
    if len(cmp_plants) < 2:
        # Try pending comparison plants from conversation history
        fallback_plants = pending_comparison_plants or []
        if len(fallback_plants) < 2:
            fallback_plants = extract_pending_comparison_plants_from_history(
                history_messages, all_names
            )

        if len(fallback_plants) >= 2:
            cmp_plants = fallback_plants
            logger.info(
                "[comparison_service] resolved from context | plants=%s | usedConversationContext=True",
                cmp_plants,
            )
        else:
            logger.info(
                "[comparison_service] fewer than 2 plants found=%r → ask_for_plant_name",
                cmp_plants,
            )
            return {
                "question": question,
                "answer": (
                    "Write the names of the plants you want to compare from Gharsa plants."
                    if language.startswith("en")
                    else "اكتب أسماء النباتات التي تريد مقارنتها من نباتات غرسة."
                ),
                "responseMode": "ask_for_plant_name",
                "plantName": None,
                "retrievedSources": [],
                "suggestedAlternatives": [],
                "detectedIntents": intents,
                "confidence": 0.0,
                "generationUsed": False,
            }

    # ── Resolve canonical names and load data ──────────────────────────────
    data_map: dict = {}
    missing: list[str] = []
    for cp in cmp_plants:
        cpdata = get_plant_data(cp)
        cp_orig = get_plant_original_name(cp)
        cp_display = get_plant_display_name(cp, language) or cp_orig or cp
        if cpdata:
            data_map[cp_display] = cpdata
        else:
            missing.append(cp_display)

    if missing:
        missing_name = missing[0]
        logger.info(
            "[comparison_service] plant not in DB=%r → plant_not_found",
            missing_name,
        )
        return {
            "question": question,
            "answer": (
                "This plant is not currently available in Gharsa."
                if language.startswith("en")
                else "هذا النبات غير متوفر حاليًا في غرسة."
            ),
            "responseMode": "plant_not_found",
            "plantName": missing_name,
            "retrievedSources": [],
            "suggestedAlternatives": [],
            "detectedIntents": intents,
            "confidence": 0.0,
            "generationUsed": False,
        }

    # ── Build answer from Excel data only ─────────────────────────────────
    cmp_answer, cmp_conclusion, cmp_criterion, cmp_used_fields, cmp_missing_fields = (
        build_comparison_answer(
            list(data_map.keys()),
            data_map,
            question=effective_question,
            language=language,
        )
    )

    logger.info(
        "[GHARSIH_COMPARISON_LOG] "
        "comparisonPlants=%s | comparisonCriterion=%s | "
        "comparedFields=%s | missingFields=%s | "
        "comparisonResult=%r | responseMode=comparison_from_data",
        list(data_map.keys()),
        cmp_criterion,
        cmp_used_fields,
        cmp_missing_fields,
        cmp_conclusion,
    )

    return {
        "question": question,
        "answer": cmp_answer,
        "responseMode": "comparison_from_data",
        "plantName": None,
        "retrievedSources": [],
        "suggestedAlternatives": [],
        "detectedIntents": intents,
        "confidence": 1.0,
        "generationUsed": False,
    }
