"""Backward-compatible facade for context/retrieval helper imports.

Implementation now lives in smaller modules; this file only re-exports the
historical API so existing imports continue to work unchanged.
"""
from __future__ import annotations

from src.inference.context_retrieval import (
    get_best_retrieval_score,
    is_retrieval_strong,
    retrieve_relevant_context,
)
from src.inference.plant_extraction import (
    _NON_PLANT_TOKENS,
    extract_all_plants_from_question,
    extract_plant_name,
    extract_target_plant,
)
from src.inference.answer_cleaning import (
    _AR_CHAR_RE,
    _EN_CHAR_RE,
    _EN_CHUNK_RE,
    _EN_OR_DIGIT_RE,
    _clean_english_text,
    _clean_text,
    _has_too_much_english,
    _postprocess_direct,
    _strip_inline_english,
)
from src.config.assistant_messages import (
    _COMPARISON_CRITERIA_MAP,
    _COMPARISON_CRITERIA_MAP_EN,
    _DIFFICULTY_AR,
    _DIFFICULTY_RANK,
    _EN_CANONICAL_OVERRIDES,
    _FIELD_LABELS_EN,
    _TASK_TYPE_AR,
    _TASK_TYPE_EN,
)
from src.inference.answer_builders import (
    _MAX_DIRECT_SENTENCES,
    _NATURALIZERS,
    _TEMPLATE_INTENTS,
    _build_list_context_for_language,
    _build_structured_answer_en,
    _g_lang,
    _is_english,
    _label_for_field,
    _sentences_beginner,
    _sentences_care_summary,
    _sentences_container,
    _sentences_fertilizing,
    _sentences_general_summary,
    _sentences_germination,
    _sentences_harvest_storage,
    _sentences_humidity,
    _sentences_light,
    _sentences_pests_diseases,
    _sentences_plant_names,
    _sentences_planting,
    _sentences_season,
    _sentences_soil,
    _sentences_spacing,
    _sentences_suitability,
    _sentences_tasks,
    _sentences_temperature,
    _sentences_uses,
    _sentences_watering,
    build_answer_draft,
    build_direct_answer,
    has_sufficient_data,
)
from src.inference.comparison_utils import (
    _detect_comparison_criterion,
    build_comparison_answer,
    detect_comparison_criterion,
)
from src.inference.context_helpers import (
    _TEMP_RE,
    _extract_temp_range,
    collect_user_and_climate_context,
    format_fallback_answer,
    suggest_alternative_plants_if_needed,
)

__all__ = [
    "retrieve_relevant_context",
    "get_best_retrieval_score",
    "is_retrieval_strong",
    "extract_plant_name",
    "extract_target_plant",
    "extract_all_plants_from_question",
    "build_direct_answer",
    "build_answer_draft",
    "has_sufficient_data",
    "detect_comparison_criterion",
    "build_comparison_answer",
    "collect_user_and_climate_context",
    "suggest_alternative_plants_if_needed",
    "format_fallback_answer",
]
