"""
assistant_service.py
────────────────────────────────────────────────────────────────────────────
Single entry-point for all /assistant endpoint business logic.

app.py imports ``initialize()`` (called once in startup) and
``handle_assistant_request()`` (called on every /assistant request).

All guard logic, intent classification, routing, plant lookup, comparison
handling, retrieval, answer building, and final validation live here.
app.py itself becomes a pure API routing layer.
"""

import logging
import re
import time
import traceback
from typing import Optional

from src.config.settings import (
    CONFIDENCE_THRESHOLD as SETTINGS_CONFIDENCE_THRESHOLD,
    GENERATION_FALLBACK_PREFIX,
    LEAKED_INTERNAL_PATTERNS,
)
from src.utils.arabic import normalize
from src.inference.rag_generator import generate_grounded_answer, is_model_loaded
from src.inference.prompt_builder import build_rewrite_prompt
from src.inference.context_utils import (
    build_answer_draft,
    build_direct_answer,
    collect_user_and_climate_context,
    detect_comparison_criterion,
    extract_plant_name,
    extract_target_plant,
    format_fallback_answer,
    get_best_retrieval_score,
    has_sufficient_data,
    retrieve_relevant_context,
    suggest_alternative_plants_if_needed,
    _g_lang,
)
from src.inference.intent_fields import (
    classify_intents,
    get_fields_for_intents,
    is_direct_intent,
    is_multi_field_intent,
)
from src.inference.query_router import classify_route
from src.inference.plant_data_store import (
    get_all_plant_lookup_names,
    get_plant_display_name,
    get_plant_data,
    get_plant_original_name,
    is_store_loaded,
)
from src.inference.assistant_guards import (
    _is_arabic,
    _contains_notable_english,
    _is_greeting,
    _is_unclear_plant_expression,
    _is_hallucination_request,
    _contains_out_of_scope_part,
    _contains_domain_part_keywords,
    _contains_hallucination,
    _contains_forbidden_phrase,
)
from src.inference.answer_cleaning import (
    clean_arabic_answer,
    _answer_echoes_question,
    _contains_underscore_artifact,
    _strip_underscore_sentences,
)
from src.inference.conversation_context import (
    normalize_history_messages,
    is_ambiguous_followup_question,
    infer_intents_from_history,
    infer_plant_from_history,
    build_effective_question,
    extract_pending_comparison_plants_from_history,
    _has_strong_domain_intent_signal,
)
from src.inference.plant_resolution import (
    _needs_plant_name,
    _detect_unlisted_plant_mention,
    _find_additional_unlisted_plant,
    suggest_plant_name,
)
from src.inference.comparison_service import handle_comparison_request
from api.schemas import AssistantRequest

logger = logging.getLogger(__name__)
_data_debug_logged = False

CONFIDENCE_THRESHOLD = SETTINGS_CONFIDENCE_THRESHOLD

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


def _detect_answer_language(req: AssistantRequest) -> str:
    """Use explicit app language when available, else infer from the message."""
    hinted = str(
        getattr(req, "language", None)
        or getattr(req, "app_language", None)
        or getattr(req, "appLanguage", None)
        or ""
    ).strip().lower()
    if hinted.startswith("ar"):
        return "ar"
    if hinted.startswith("en"):
        return "en"

    text = req.question or ""
    ar_count = len(_ARABIC_CHAR_RE.findall(text))
    en_count = len(_LATIN_CHAR_RE.findall(text))
    return "en" if en_count > ar_count else "ar"


def _is_en(language: str) -> bool:
    return (language or "").lower().startswith("en")


def _msg(language: str, key: str) -> str:
    messages = {
        "missing_data": {
            "ar": "هذه المعلومة غير متوفرة في بيانات التطبيق.",
            "en": "This information is not available in the app data.",
        },
        "plant_not_found": {
            "ar": "هذا النبات غير متوفر حاليًا في غرسة.",
            "en": "This plant is not currently available in Gharsa.",
        },
        "out_of_scope": {
            "ar": "أنا مساعد غرسة، بقدر أساعدك فقط في العناية بالنباتات الموجودة داخل التطبيق.",
            "en": "I’m Gharsa’s assistant. I can only help with plant-care questions for plants available in the app.",
        },
        "ask_for_plant": {
            "ar": "من فضلك اذكر اسم النبتة التي تريد معلومات عنها.",
            "en": "Please mention the plant you want information about.",
        },
        "greeting": {
            "ar": "أهلاً وسهلاً! أنا مساعد غرسة الذكي. كيف أقدر أساعدك بالعناية بنباتاتك؟",
            "en": "Hello! I’m Gharsa’s smart assistant. How can I help with your plant care today?",
        },
        "unsafe": {
            "ar": "لا أستطيع إضافة معلومات من خارج بيانات غرسة. اسألني عن معلومة موجودة في بيانات التطبيق فقط.",
            "en": "I can’t add information from outside Gharsa’s data. Please ask about information available in the app data only.",
        },
        "unclear_plant": {
            "ar": "مش قادر أحدد النبتة المقصودة من سؤالك. اكتب اسم النبتة الموجودة في تطبيق غرسة بشكل أوضح.",
            "en": "I can’t identify the plant from your question. Please write the name of a plant available in Gharsa more clearly.",
        },
        "insufficient_context": {
            "ar": "هذه المعلومة غير متوفرة في بيانات التطبيق.",
            "en": "This information is not available in the app data.",
        },
    }
    lang_key = "en" if _is_en(language) else "ar"
    return messages[key][lang_key]

# ── Shared runtime state injected by app.py startup ──────────────────────────
_model = None
_index = None
_cards: list = []


def initialize(model, index, cards: list) -> None:
    """
    Inject AI assets loaded during FastAPI startup.

    Must be called once (in ``app.on_event("startup")``) before any
    /assistant request is processed.
    """
    global _model, _index, _cards
    _model = model
    _index = index
    _cards = list(cards)


# ─────────────────────────────────────────────────────────────────────────────
def handle_assistant_request(req: AssistantRequest) -> dict:
    """
    Process a single /assistant request end-to-end.

    Response modes
    ──────────────
    direct_from_data   – single-topic; answer built entirely from Excel data.
    rewrite_from_data  – multi-topic; Excel draft rewritten by Gemini.
    out_of_scope       – question unrelated to the plant database.
    greeting           – Arabic greeting detected.
    fallback           – domain question with insufficient evidence.
    missing_data       – plant found but requested field is empty.
    plant_not_found    – plant token detected but not in the database.
    ask_for_plant_name – plant-specific intent but no plant mentioned.
    unclear_plant      – could not identify which plant the user means.
    unsupported_language – non-Arabic or dominant-English question.
    unsafe_request     – explicit hallucination / outside-DB request.
    """
    global _data_debug_logged

    t0 = time.perf_counter()
    _timings: dict[str, float] = {}

    # Pre-check state flags (populated in GUARDs 0/0.5; used in request log)
    _pre_contains_english: bool = False
    _pre_oos_part: bool = False
    _pre_is_mixed: bool = False
    _pre_blocked_before_plant: bool = False

    # ── GUARD 0: Language detection ──────────────────────────────────────────
    answer_language = _detect_answer_language(req)
    # Accept Arabic and English. We still log the detected script mix.
    _pre_contains_english = _contains_notable_english(req.question)
    _is_arabic_q = _is_arabic(req.question)
    logger.info(
        "[/assistant] PRE-CHECK | containsEnglish=%s | isArabic=%s | answerLanguage=%s",
        _pre_contains_english, _is_arabic_q, answer_language,
    )

    # ── GUARD 0.5: Pure out-of-scope pre-check ───────────────────────────────
    # Fires before plant extraction / intent detection / retrieval / Gemini.
    _pre_oos_part = _contains_out_of_scope_part(req.question)
    _domain_part = _contains_domain_part_keywords(req.question)
    if not _domain_part and is_store_loaded():
        _q_norm_pc = normalize(req.question).lower()
        _domain_part = any(
            normalize(n).lower() in _q_norm_pc for n in get_all_plant_lookup_names()
        )
    _pre_is_mixed = _pre_oos_part and _domain_part

    logger.info(
        "[/assistant] PRE-CHECK | containsOutOfScopePart=%s | "
        "containsPlantDomainPart=%s | isMixedQuestion=%s",
        _pre_oos_part, _domain_part, _pre_is_mixed,
    )

    if _pre_oos_part:
        _pre_blocked_before_plant = True
        logger.info(
            "[/assistant] BLOCKED (GUARD 0.5): out_of_scope | "
            "isMixed=%s | blockedBeforePlantExtraction=True | "
            "responseMode=out_of_scope",
            _pre_is_mixed,
        )
        return {
            "question": req.question,
            "answer": _msg(answer_language, "out_of_scope"),
            "responseMode": "out_of_scope",
            "retrievedSources": [],
            "suggestedAlternatives": [],
            "detectedIntents": [],
            "confidence": 0.0,
            "generationUsed": False,
        }

    # ── GUARD 0.75: Hallucination-request detection ──────────────────────────
    # Fires before plant extraction / intent detection / retrieval / Gemini.
    if _is_hallucination_request(req.question):
        _pre_blocked_before_plant = True
        logger.info(
            "[/assistant] BLOCKED (GUARD 0.75): unsafe_request | "
            "blockedBeforePlantExtraction=True | responseMode=unsafe_request"
        )
        return {
            "question": req.question,
            "answer": _msg(answer_language, "unsafe"),
            "responseMode": "unsafe_request",
            "retrievedSources": [],
            "suggestedAlternatives": [],
            "detectedIntents": [],
            "confidence": 0.0,
            "generationUsed": False,
        }

    # ── GUARD 1: Arabic greeting detection ───────────────────────────────────
    if _is_greeting(req.question):
        logger.info("[/assistant] GREETING detected -> greeting mode")
        return {
            "question": req.question,
            "answer": _msg(answer_language, "greeting"),
            "responseMode": "greeting",
            "retrievedSources": [],
            "suggestedAlternatives": [],
            "detectedIntents": [],
            "confidence": 0.0,
            "generationUsed": False,
        }

    logger.info(
        "[/assistant] ← entered | question=%r | top_k=%d | "
        "climate_data=%s | user_conditions=%s | history=%s",
        req.question, req.top_k,
        bool(req.climate_data), bool(req.user_conditions), bool(req.history),
    )

    # Normalize optional chat history and build a context-aware query.
    history_messages = normalize_history_messages(req.history)
    effective_question = build_effective_question(req.question, history_messages)
    logger.info(
        "[/assistant] context-aware question used=%s | history_turns=%d",
        effective_question != req.question,
        len(history_messages),
    )

    # ── Step 1: Classify intent(s) ───────────────────────────────────────────
    _t = time.perf_counter()
    try:
        intents = classify_intents(effective_question)
    except Exception:
        logger.error("[/assistant] intent classification FAILED\n%s", traceback.format_exc())
        intents = []
    _timings["intent"] = time.perf_counter() - _t
    logger.info("[/assistant] ⏱  Intents: %.3fs  (%s)", _timings["intent"], intents)

    direct = is_direct_intent(intents)
    multi = is_multi_field_intent(intents)

    # Follow-up intent repair: borrow domain intents from recent history when
    # the current question is ambiguous.
    if is_ambiguous_followup_question(req.question) and not _has_strong_domain_intent_signal(intents):
        inferred_intents = infer_intents_from_history(history_messages)
        if inferred_intents:
            intents = inferred_intents
            direct = is_direct_intent(intents)
            multi = is_multi_field_intent(intents)
            logger.info("[/assistant] intents refined from history: %s", intents)

    # ── Comparison-context injection ─────────────────────────────────────────
    # If the previous assistant turn asked for a comparison criterion AND the
    # current question is a criterion-only follow-up, inject "comparison" intent.
    _pending_comparison_plants: list[str] = []
    if "comparison" not in intents and history_messages and is_store_loaded():
        _all_names_inject = get_all_plant_lookup_names()
        _pending_comparison_plants = extract_pending_comparison_plants_from_history(
            history_messages, _all_names_inject
        )
        if _pending_comparison_plants:
            _crit_from_current = detect_comparison_criterion(req.question)
            if _crit_from_current != "unknown" or len(req.question.strip()) <= 40:
                intents = ["comparison"]
                direct = True
                multi = False
                logger.info(
                    "[/assistant] COMPARISON_CONTEXT_INJECTION: injected 'comparison' intent "
                    "| pendingComparisonPlants=%s | resolvedFollowupCriterion=%s "
                    "| usedConversationContext=True",
                    _pending_comparison_plants,
                    _crit_from_current,
                )

    logger.info("[/assistant] direct=%s  multi=%s", direct, multi)

    # ── Step 1.5: Route question (domain_rag vs general_conversation) ────────
    _t = time.perf_counter()
    all_names_for_routing = get_all_plant_lookup_names() if is_store_loaded() else []
    route_decision = classify_route(
        question=effective_question,
        intents=intents,
        all_plant_names=all_names_for_routing,
        conversation_history=history_messages,
        is_followup=is_ambiguous_followup_question(req.question),
    )
    _timings["routing"] = time.perf_counter() - _t
    logger.info(
        "[/assistant] ⏱  Routing: %.3fs  (route=%s, reason=%s)",
        _timings["routing"],
        route_decision.route,
        route_decision.reason,
    )

    if route_decision.route == "general_conversation":
        logger.info(
            "[/assistant] BLOCKED: out_of_scope | reason=%s | question=%r",
            route_decision.reason, req.question,
        )
        return {
            "question": req.question,
            "answer": _msg(answer_language, "out_of_scope"),
            "responseMode": "out_of_scope",
            "retrievedSources": [],
            "suggestedAlternatives": [],
            "detectedIntents": intents,
            "confidence": 0.0,
            "generationUsed": False,
        }

    # ── Step 2: Identify the target plant ────────────────────────────────────
    _t = time.perf_counter()
    plant_name: Optional[str] = None
    plant_data: Optional[dict] = None
    excel_hit = False

    if is_store_loaded():
        all_names = get_all_plant_lookup_names()
        plant_name = extract_target_plant(effective_question, all_names)
        if not plant_name and is_ambiguous_followup_question(req.question):
            plant_name = infer_plant_from_history(history_messages, all_names)
        if plant_name:
            plant_data = get_plant_data(plant_name)
            original = get_plant_original_name(plant_name)
            if original:
                plant_name = original
            display_plant_name = get_plant_display_name(plant_name, answer_language)
            if display_plant_name:
                plant_name = display_plant_name
        if plant_data:
            excel_hit = True
            logger.info("[/assistant] Excel hit: %r", plant_name)

            # Secondary check: detect a second plant-like token NOT in the DB.
            _second_unlisted = _find_additional_unlisted_plant(
                effective_question, all_names, plant_name
            )
            if _second_unlisted:
                logger.info(
                    "[/assistant] SECONDARY_PLANT_CHECK: found unlisted mention=%r "
                    "alongside known plant=%r → plant_not_found",
                    _second_unlisted, plant_name,
                )
                return {
                    "question": req.question,
                    "answer": _msg(answer_language, "plant_not_found"),
                    "responseMode": "plant_not_found",
                    "plantName": plant_name,
                    "retrievedSources": [],
                    "suggestedAlternatives": [],
                    "detectedIntents": intents,
                    "confidence": 0.0,
                    "generationUsed": False,
                }
        else:
            logger.info("[/assistant] No Excel hit via question parsing")
    _timings["excel_lookup"] = time.perf_counter() - _t

    # ── Step 2.5: Comparison mode ─────────────────────────────────────────────
    _t = time.perf_counter()
    comparison_response = handle_comparison_request(
        question=req.question,
        effective_question=effective_question,
        history_messages=history_messages,
        intents=intents,
        pending_comparison_plants=_pending_comparison_plants,
        language=answer_language,
    )
    _timings["comparison"] = time.perf_counter() - _t
    if comparison_response is not None:
        return comparison_response

    # ── Step 3: Retrieval (FAISS semantic search) ─────────────────────────────
    _t = time.perf_counter()
    try:
        retrieved = retrieve_relevant_context(
            query=effective_question,
            model=_model,
            index=_index,
            cards=_cards,
            top_k=req.top_k,
        )
    except Exception:
        logger.error("[/assistant] retrieval FAILED\n%s", traceback.format_exc())
        retrieved = []
    _timings["retrieval"] = time.perf_counter() - _t
    logger.info(
        "[/assistant] ⏱  Retrieval: %.3fs  (%d result(s))",
        _timings["retrieval"], len(retrieved),
    )

    best_score = get_best_retrieval_score(retrieved) if retrieved else 0.0

    # If we didn't find the plant from the question, try from retrieved cards.
    if not excel_hit and retrieved and is_store_loaded():
        for r in retrieved:
            card_name = extract_plant_name(r["card"])
            pdata = get_plant_data(card_name)
            if pdata:
                plant_name = card_name
                original = get_plant_original_name(card_name)
                if original:
                    plant_name = original
                display_plant_name = get_plant_display_name(plant_name, answer_language)
                if display_plant_name:
                    plant_name = display_plant_name
                plant_data = pdata
                excel_hit = True
                logger.info("[/assistant] Excel hit via retrieved card: %r", plant_name)
                break

    # ── Build source references ───────────────────────────────────────────────
    retrieved_sources = [
        {"score": r["score"], "snippet": r["card"][:200] + "…"}
        for r in retrieved
    ]

    # ── Step 4: Climate / user context + alternatives ─────────────────────────
    _t = time.perf_counter()
    try:
        climate_text, user_text = collect_user_and_climate_context(
            climate_data=req.climate_data,
            user_conditions=req.user_conditions,
        )
    except Exception:
        climate_text, user_text = None, None

    user_temp = None
    if req.climate_data and "temperature" in req.climate_data:
        try:
            user_temp = float(req.climate_data["temperature"])
        except (ValueError, TypeError):
            pass

    try:
        alternatives = suggest_alternative_plants_if_needed(
            target_cards=retrieved,
            all_cards=_cards,
            user_temperature=user_temp,
        )
    except Exception:
        alternatives = []
    _timings["context"] = time.perf_counter() - _t

    suggestion = suggest_plant_name(effective_question) if not excel_hit else None

    # ══════════════════════════════════════════════════════════════════════════
    # DECISION: Choose response mode
    # ══════════════════════════════════════════════════════════════════════════
    response_mode: str
    answer: str = ""
    generation_used: bool = False

    if excel_hit and plant_data and direct:
        # ── MODE 1: direct_from_data ──────────────────────────────────────────
        # Single-topic question + Excel data → answer directly from structured
        # data.  NO LLM call.
        _t = time.perf_counter()
        if has_sufficient_data(plant_data, intents, min_fields=1, language=answer_language):
            answer = build_direct_answer(plant_name or "", plant_data, intents, language=answer_language)
            if answer:
                response_mode = "direct_from_data"
                logger.info("[/assistant] MODE: direct_from_data")
            else:
                answer = _msg(answer_language, "missing_data")
                response_mode = "missing_data"
                logger.info("[/assistant] MODE: missing_data (build_direct_answer returned empty)")
        else:
            answer = _msg(answer_language, "missing_data")
            response_mode = "missing_data"
            logger.info("[/assistant] MODE: missing_data (plant found but field empty)")
        _timings["answer_build"] = time.perf_counter() - _t

    elif excel_hit and plant_data and multi:
        # ── MODE 2: rewrite_from_data ─────────────────────────────────────────
        # Multi-topic question + Excel data → build a draft from Excel, then
        # ask the LLM to rewrite it into natural Arabic.
        # The model is NOT a source of information.
        _t = time.perf_counter()
        if has_sufficient_data(plant_data, intents, min_fields=2, language=answer_language):
            draft = build_answer_draft(plant_name or "", plant_data, intents, language=answer_language)
            _timings["answer_build"] = time.perf_counter() - _t

            if draft and is_model_loaded() and not _is_en(answer_language):
                _t2 = time.perf_counter()
                try:
                    prompt = build_rewrite_prompt(
                        user_question=req.question,
                        answer_draft=draft,
                        climate_context=climate_text,
                        user_context=user_text,
                        conversation_history=history_messages if history_messages else None,
                    )
                    _timings["prompt"] = time.perf_counter() - _t2

                    _t3 = time.perf_counter()
                    answer = generate_grounded_answer(prompt)
                    generation_used = True
                    answer = clean_arabic_answer(answer)
                    _timings["generation"] = time.perf_counter() - _t3

                    # Post-check: detect forbidden generalist phrases.
                    if _contains_hallucination(answer):
                        logger.warning(
                            "[/assistant] HALLUCINATION_DETECTED in rewrite answer – retrying once."
                        )
                        try:
                            _answer_retry = clean_arabic_answer(
                                generate_grounded_answer(prompt)
                            )
                            if _contains_hallucination(_answer_retry):
                                logger.warning(
                                    "[/assistant] HALLUCINATION_PERSISTS after retry – "
                                    "falling back to missing_data."
                                )
                                answer = (
                                    _msg(answer_language, "missing_data")
                                )
                                response_mode = "missing_data"
                                logger.info(
                                    "[/assistant] MODE: missing_data "
                                    "(hallucination post-check fallback)"
                                )
                            else:
                                answer = _answer_retry
                                response_mode = "rewrite_from_data"
                                logger.info(
                                    "[/assistant] MODE: rewrite_from_data "
                                    "(LLM rewrote draft – retry passed post-check)"
                                )
                        except Exception:
                            logger.error(
                                "[/assistant] Hallucination retry FAILED\n%s",
                                traceback.format_exc(),
                            )
                            answer = (
                                _msg(answer_language, "missing_data")
                            )
                            response_mode = "missing_data"
                            logger.info(
                                "[/assistant] MODE: missing_data "
                                "(hallucination retry exception)"
                            )
                    else:
                        response_mode = "rewrite_from_data"
                        logger.info(
                            "[/assistant] MODE: rewrite_from_data (LLM rewrote draft)"
                        )
                except Exception:
                    logger.error(
                        "[/assistant] rewrite generation FAILED\n%s", traceback.format_exc()
                    )
                    answer = draft
                    response_mode = "rewrite_from_data"
                    logger.info("[/assistant] MODE: rewrite_from_data (raw draft, LLM failed)")
            elif draft:
                answer = draft
                response_mode = "rewrite_from_data"
                logger.info("[/assistant] MODE: rewrite_from_data (raw draft, model not loaded)")
            else:
                answer = _msg(answer_language, "missing_data")
                response_mode = "missing_data"
        else:
            _timings["answer_build"] = time.perf_counter() - _t
            answer = _msg(answer_language, "missing_data")
            response_mode = "missing_data"
            logger.info("[/assistant] MODE: missing_data (insufficient multi-field data)")

    elif excel_hit and plant_data:
        # ── Intents empty or ambiguous → try direct answer ────────────────────
        _t = time.perf_counter()
        answer = build_direct_answer(
            plant_name or "", plant_data, intents if intents else ["general_summary"],
            language=answer_language,
        )
        _timings["answer_build"] = time.perf_counter() - _t
        if answer:
            response_mode = "direct_from_data"
        else:
            answer = _msg(answer_language, "missing_data")
            response_mode = "missing_data"

    else:
        # ── No Excel hit: safe, specific response.  NO LLM call allowed. ─────
        _t = time.perf_counter()
        _all_names_fb = get_all_plant_lookup_names() if is_store_loaded() else []
        if best_score < CONFIDENCE_THRESHOLD:
            if _is_unclear_plant_expression(effective_question):
                answer = _msg(answer_language, "unclear_plant")
                response_mode = "unclear_plant"
                logger.info(
                    "[/assistant] MODE: unclear_plant | explicit_expression | score=%.4f",
                    best_score,
                )
            else:
                _unlisted = _detect_unlisted_plant_mention(effective_question, _all_names_fb)
                if _unlisted:
                    answer = _msg(answer_language, "plant_not_found")
                    response_mode = "plant_not_found"
                    logger.info(
                        "[/assistant] MODE: plant_not_found | entity=%r | score=%.4f < threshold=%.4f",
                        _unlisted, best_score, CONFIDENCE_THRESHOLD,
                    )
                elif _needs_plant_name(intents):
                    answer = _msg(answer_language, "ask_for_plant")
                    response_mode = "ask_for_plant_name"
                    logger.info(
                        "[/assistant] MODE: ask_for_plant_name | intents=%s | score=%.4f",
                        intents, best_score,
                    )
                else:
                    answer = _msg(answer_language, "insufficient_context")
                    response_mode = "insufficient_context"
                    logger.info(
                        "[/assistant] MODE: insufficient_context | score=%.4f < threshold=%.4f",
                        best_score, CONFIDENCE_THRESHOLD,
                    )
        else:
            answer = _msg(answer_language, "insufficient_context")
            response_mode = "fallback"
            logger.info(
                "[/assistant] MODE: fallback | no Excel hit | score=%.4f | suggestion=%r",
                best_score, suggestion,
            )
        _timings["answer_build"] = time.perf_counter() - _t

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SAFETY NET
    # ══════════════════════════════════════════════════════════════════════════
    if generation_used and answer:
        if len(answer.strip()) < 5:
            logger.warning("[/assistant] SAFETY NET: answer too short → fallback")
            answer = (
                clean_arabic_answer(format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX))
                if retrieved
                else _msg(answer_language, "insufficient_context")
            )
            generation_used = False
            response_mode = "fallback"
        else:
            for pattern in LEAKED_INTERNAL_PATTERNS:
                if pattern in answer:
                    logger.warning(
                        "[/assistant] SAFETY NET: leaked pattern %r → fallback", pattern
                    )
                    answer = (
                        clean_arabic_answer(
                            format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX)
                        )
                        if retrieved
                        else _msg(answer_language, "insufficient_context")
                    )
                    generation_used = False
                    response_mode = "fallback"
                    break

    # ══════════════════════════════════════════════════════════════════════════
    # FORBIDDEN PHRASES FILTER (Rule 11)
    # ══════════════════════════════════════════════════════════════════════════
    if generation_used and answer and _contains_forbidden_phrase(answer):
        logger.warning(
            "[/assistant] RULE-11 SAFETY: forbidden phrase detected → missing_data"
        )
        answer = _msg(answer_language, "missing_data")
        generation_used = False
        response_mode = "missing_data"

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL ANSWER VALIDATION
    # Runs on every answer regardless of responseMode.
    # ══════════════════════════════════════════════════════════════════════════
    if answer and response_mode not in {
        "out_of_scope", "greeting", "general_conversation",
        "unsupported_language", "unsafe_request",
    }:
        if _answer_echoes_question(answer, req.question):
            logger.warning(
                "[/assistant] FINAL_VALIDATION: answer echoes question → missing_data"
            )
            answer = _msg(answer_language, "missing_data")
            response_mode = "missing_data"

        elif _contains_underscore_artifact(answer):
            cleaned = _strip_underscore_sentences(answer)
            if cleaned.strip():
                logger.info("[/assistant] FINAL_VALIDATION: underscore artifact → stripped")
                answer = cleaned
            else:
                logger.warning(
                    "[/assistant] FINAL_VALIDATION: underscore artifact uncleanable → missing_data"
                )
                answer = _msg(answer_language, "missing_data")
                response_mode = "missing_data"

    # ── Performance summary ───────────────────────────────────────────────────
    if not _data_debug_logged:
        try:
            _selected_field_map = get_fields_for_intents(intents) if intents else {}
            _resolved_field_count = 0
            _debug_has_sufficient_data = False
            if plant_data:
                _resolved_field_count = sum(
                    1
                    for _field in _selected_field_map
                    if _g_lang(plant_data, _field, answer_language)
                )
                _debug_has_sufficient_data = has_sufficient_data(
                    plant_data,
                    intents,
                    min_fields=2 if multi else 1,
                    language=answer_language,
                )
            logger.info(
                "[ASSISTANT_DATA_DEBUG] detectedLanguage=%s | detectedIntents=%s | "
                "extractedPlantName=%r | plantDataExists=%s | selectedFields=%s | "
                "resolvedFieldValuesCount=%d | hasSufficientData=%s | responseMode=%s",
                answer_language,
                intents,
                plant_name,
                bool(plant_data),
                list(_selected_field_map.keys()),
                _resolved_field_count,
                _debug_has_sufficient_data,
                response_mode,
            )
            _data_debug_logged = True
        except Exception:
            logger.debug("[ASSISTANT_DATA_DEBUG] failed", exc_info=True)

    elapsed = time.perf_counter() - t0
    _timings["total"] = elapsed

    logger.info(
        "[/assistant] ━━━━━━━━━━━━━━ PERFORMANCE SUMMARY ━━━━━━━━━━━━━━\n"
        "  ⏱  Intent        : %7.3f s\n"
        "  ⏱  Excel lookup  : %7.3f s\n"
        "  ⏱  Retrieval     : %7.3f s\n"
        "  ⏱  Context build : %7.3f s\n"
        "  ⏱  Answer build  : %7.3f s\n"
        "  ⏱  Prompt build  : %7.3f s\n"
        "  ⏱  Generation    : %7.3f s  ← bottleneck if used\n"
        "  ⏱  TOTAL         : %7.3f s\n"
        "  responseMode=%s | generationUsed=%s | best_score=%.4f",
        _timings.get("intent", 0.0),
        _timings.get("excel_lookup", 0.0),
        _timings.get("retrieval", 0.0),
        _timings.get("context", 0.0),
        _timings.get("answer_build", 0.0),
        _timings.get("prompt", 0.0),
        _timings.get("generation", 0.0),
        elapsed,
        response_mode, generation_used, best_score,
    )

    # ── Structured request log (Rule 20) ─────────────────────────────────────
    try:
        from src.inference.intent_fields import get_fields_for_intents as _gfi
        _selected_fields = list(_gfi(intents).keys()) if intents else []
    except Exception:
        _selected_fields = []

    _has_required = (
        excel_hit
        and bool(plant_data)
        and response_mode in {"direct_from_data", "rewrite_from_data"}
    )
    _fallback_reason = (
        response_mode
        if response_mode in {
            "fallback", "missing_data", "plant_not_found",
            "insufficient_context", "unclear_plant", "ask_for_plant_name",
        }
        else "n/a"
    )

    _tasks_found_count = len(plant_data.get("tasks", [])) if plant_data else 0
    _care_details_found = bool(
        plant_data and (
            plant_data.get("careInfoAr") or plant_data.get("wateringInfoAr")
        )
    )
    _selected_table = (
        "Tasks"
        if intents and "tasks" in intents
        else "Care_Details"
        if intents and any(
            i in {"care_summary", "watering", "light", "soil", "humidity", "pests_diseases"}
            for i in intents
        )
        else "Plants"
    )

    logger.info(
        "[GHARSIH_REQUEST_LOG] "
        "userQuestion=%r | detectedLanguage=%s | isInScope=%s | "
        "containsEnglish=%s | containsOutOfScopePart=%s | isMixedQuestion=%s | "
        "blockedBeforePlantExtraction=%s | "
        "extractedPlantName=%r | matchedPlantId=N/A | matchedPlantName=%r | "
        "plantMatchScore=%.4f | detectedIntent=%s | selectedTable=%s | "
        "selectedFields=%s | tasksFoundCount=%d | careDetailsFound=%s | "
        "hasRequiredData=%s | retrievalScore=%.4f | usedGemini=%s | "
        "responseMode=%s | fallbackReason=%s",
        req.question,
        answer_language,
        route_decision.route != "general_conversation",
        _pre_contains_english,
        _pre_oos_part,
        _pre_is_mixed,
        _pre_blocked_before_plant,
        plant_name,
        plant_name,
        best_score,
        intents,
        _selected_table,
        _selected_fields,
        _tasks_found_count,
        _care_details_found,
        _has_required,
        best_score,
        generation_used,
        response_mode,
        _fallback_reason,
    )

    return {
        "question": req.question,
        "answer": answer,
        "responseMode": response_mode,
        "plantName": plant_name,
        "retrievedSources": retrieved_sources,
        "suggestedAlternatives": alternatives,
        "detectedIntents": intents,
        "confidence": best_score,
        "generationUsed": generation_used,
    }
