import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer

# ── RAG generation layer imports ─────────────────────────────────
from src.config.settings import (
    CONFIDENCE_THRESHOLD as SETTINGS_CONFIDENCE_THRESHOLD,
    EMBEDDING_MODEL_DIR,
    GENERATION_FALLBACK_PREFIX,
    INDEX_PATH as SETTINGS_INDEX_PATH,
    LEAKED_INTERNAL_PATTERNS,
    MAX_NEW_TOKENS,
    META_PATH as SETTINGS_META_PATH,
    SAFE_NO_ANSWER,
    TEMPERATURE,
    TOP_K_RETRIEVAL,
    XLSX_PATH as SETTINGS_XLSX_PATH,
)
from src.config.constants import NAME_COLUMNS
from src.utils.arabic import normalize_name, normalize_query
from src.inference.rag_generator import (
    generate_grounded_answer,
    is_model_loaded,
    load_generation_model,
)
from src.inference.prompt_builder import build_assistant_prompt, build_rewrite_prompt
from src.inference.context_utils import (
    build_answer_draft,
    build_direct_answer,
    collect_user_and_climate_context,
    extract_plant_name,
    extract_target_plant,
    format_fallback_answer,
    get_best_retrieval_score,
    has_sufficient_data,
    is_retrieval_strong,
    retrieve_relevant_context,
    suggest_alternative_plants_if_needed,
)
from src.inference.intent_fields import (
    build_field_context_ar,
    classify_intents,
    is_direct_intent,
    is_multi_field_intent,
)
from src.inference.plant_data_store import (
    get_all_plant_names,
    get_plant_data,
    get_plant_original_name,
    is_store_loaded,
    load_plant_store,
)

# ── Logging: emit INFO+ to stdout so every request step is visible ──
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Graduation Project AI API")

# ---------- Paths (from settings.py – single source of truth) ----------
MODEL_DIR = Path(EMBEDDING_MODEL_DIR)
INDEX_PATH = Path(SETTINGS_INDEX_PATH)
META_PATH = Path(SETTINGS_META_PATH)
XLSX_PATH = Path(SETTINGS_XLSX_PATH)

# ---------- Globals ----------
model = None
index = None
cards = None

plant_names = []
plant_names_norm_map = {}  # normalized -> original

# Threshold: sourced from settings.py (overridable via env var).
CONFIDENCE_THRESHOLD = SETTINGS_CONFIDENCE_THRESHOLD


# ---------- Request Models ----------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    include_card: bool = False  # for debugging only


class ChatRequest(BaseModel):
    message: str
    top_k: int = 1


# ── Smart Assistant (RAG) request model ──────────────────────────
class AssistantRequest(BaseModel):
    """Request body for the /assistant endpoint."""
    question: str                              # سؤال المستخدم
    top_k: int = TOP_K_RETRIEVAL               # عدد النتائج المسترجعة
    climate_data: Optional[dict] = None        # بيانات مناخ اختيارية
    user_conditions: Optional[dict] = None     # ظروف المستخدم اختيارية


# ---------- Startup ----------
@app.on_event("startup")
def load_assets():
    global model, index, cards, plant_names, plant_names_norm_map

    model = SentenceTransformer(str(MODEL_DIR))
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    cards = meta["cards"]

    # Load plant names from Excel (if available) for did_you_mean
    plant_names = []
    plant_names_norm_map = {}

    if XLSX_PATH.exists():
        df = pd.read_excel(XLSX_PATH)

        # Ignore first row if it's a "column explanation" row
        if len(df) > 0:
            df = df.iloc[1:].reset_index(drop=True)

        # Use shared NAME_COLUMNS from constants
        name_col = next((c for c in NAME_COLUMNS if c in df.columns), None)

        if name_col:
            plant_names = [
                str(x).strip()
                for x in df[name_col].dropna().tolist()
                if str(x).strip()
            ]

    for n in plant_names:
        nn = normalize_ar_name(n)
        if nn and nn not in plant_names_norm_map:
            plant_names_norm_map[nn] = n

    print("✅ AI assets loaded")
    print(f"✅ cards loaded: {len(cards)}")
    print(f"✅ plant names loaded: {len(plant_names)}")

    # ── Load structured plant data store (for field‑based answers) ──
    plant_count = load_plant_store(XLSX_PATH)
    print(f"✅ Plant data store: {plant_count} plants")

    # ── Load generation model (RAG layer) ────────────────────────
    try:
        load_generation_model()
        print("✅ Generation model loaded: Gemini API")
    except Exception as exc:
        # Non‑fatal: the /assistant endpoint will use fallback mode
        print(f"⚠️  Generation model could not be loaded: {exc}")
        print("   The /assistant endpoint will return retrieval‑only fallback answers.")


# ---------- Arabic answer cleaner ----------
_FIELD_LABEL_RE = re.compile(
    r"(?m)^(?:"
    r"النبتة|الري|الضوء|التربة|الحرارة المثالية|الآفات الشائعة|"
    r"خطوات الزراعة|التسميد|موسم الزراعة|أفضل شهر للزراعة في فلسطين|"
    r"الأصيص|قاعدة الري|نص الري|نتيجة \d+"
    r")\s*:\s*"
)
_SEPARATOR_RE = re.compile(r"(?m)^[-=*_٭]{3,}\s*$")
_CARD_HEADER_RE = re.compile(r"(?m)^-+\s*نتيجة\s*\d+\s*-+\s*$")
_MD_HEADER_RE = re.compile(r"(?m)^#{1,4}\s*")
_PAREN_EN_RE = re.compile(r'\([A-Za-z][A-Za-z\s./%°,;:\'"\x2d\u201c\u201d\u2018\u2019]+\)')


def clean_arabic_answer(text: str) -> str:
    """
    Post-process a generated or fallback Arabic answer:
    - strip Markdown headers
    - remove separator lines
    - remove raw card field labels that leak into generated text
    - collapse multiple blank lines
    - strip surrounding quotes / escape artifacts
    """
    if not text:
        return text
    text = _MD_HEADER_RE.sub("", text)
    text = _SEPARATOR_RE.sub("", text)
    text = _CARD_HEADER_RE.sub("", text)
    text = _FIELD_LABEL_RE.sub("", text)
    # Clean technical artefacts
    text = text.replace(" | ", "، ").replace("|", "، ")
    text = text.replace("• ", "").replace("* ", "")
    text = _PAREN_EN_RE.sub("", text)
    # Collapse 3+ newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    # Strip surrounding quote characters
    text = text.strip('"\'“”‘’`')
    return text.strip()


# ---------- Basic ----------
@app.get("/")
def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Arabic Normalization (uses shared src.utils.arabic) ----------
def normalize_ar_text(s: str) -> str:
    """Wrapper – delegates to the shared normaliser."""
    return normalize_query(s)


def normalize_ar_name(name: str) -> str:
    """Wrapper – delegates to the shared normaliser."""
    return normalize_name(name)


def suggest_plant_name(query: str):
    """Return best fuzzy match from plant_names, ignoring ال/لل/ل and diacritics."""
    if not plant_names_norm_map:
        return None

    q_norm = normalize_ar_name(query)
    norm_names = list(plant_names_norm_map.keys())

    match = process.extractOne(q_norm, norm_names, scorer=fuzz.WRatio)
    if not match:
        return None

    best_norm, score, _ = match
    if score >= 80:
        suggested = plant_names_norm_map.get(best_norm, best_norm)
        return {"suggested": suggested, "score": int(score)}

    return None


# ---------- Answer extraction + formatting ----------
def extract_answer(card_text: str, query: str) -> dict:
    q = normalize_ar_text(query)
    lines = [l.strip() for l in card_text.splitlines() if l.strip()]

    def find_line(prefix: str):
        for line in lines:
            if line.startswith(prefix):
                return line
        return None

    # Core fields
    if any(k in q for k in ["ضوء", "شمس", "ساعات", "sun"]):
        line = find_line("الضوء:")
        return {"field": "light", "raw": line, "short": True}

    if any(k in q for k in ["اسقي", "سقي", "ري", "watering", "water"]):
        line = find_line("الري:")
        return {"field": "watering", "raw": line, "short": True}

    if any(k in q for k in ["تربة", "soil", "ph", "حموضة"]):
        line = find_line("التربة:")
        return {"field": "soil", "raw": line, "short": True}

    if any(k in q for k in ["حرارة", "temperature", "temp"]):
        line = find_line("الحرارة المثالية:")
        return {"field": "temperature", "raw": line, "short": True}

    if any(k in q for k in ["افات", "آفات", "حشرات", "pests"]):
        line = find_line("الآفات الشائعة:")
        return {"field": "pests", "raw": line, "short": True}

    # New requested fields (if they exist in cards)
    if any(k in q for k in ["كيف ازرع", "ازرع", "زراعة", "خطوات"]):
        line = find_line("خطوات الزراعة:")
        return {"field": "planting_steps", "raw": line, "short": True}

    if any(k in q for k in ["سماد", "تسميد", "fertil"]):
        line = find_line("التسميد:")
        return {"field": "fertilizer", "raw": line, "short": True}

    if any(k in q for k in ["موسم", "وقت الزراعة", "season"]):
        line = find_line("موسم الزراعة:")
        return {"field": "season", "raw": line, "short": True}

    if any(k in q for k in ["فلسطين", "افضل شهر", "أفضل شهر", "متى ازرع", "شهر الزراعة"]):
        line = find_line("أفضل شهر للزراعة في فلسطين:")
        return {"field": "best_month_palestine", "raw": line, "short": True}

    if any(k in q for k in ["اصيص", "أصيص", "وعاء", "حوض", "pot", "container"]):
        line = find_line("الأصيص:")
        return {"field": "pot", "raw": line, "short": True}

    # Default: summary (first 3 lines)
    summary = "\n".join(lines[:3]) if lines else card_text
    return {"field": "summary", "raw": summary, "short": False}


def format_answer(field: str, raw_line: str | None) -> str:
    if not raw_line:
        # Field-specific fallback
        fallback = {
            "light": "معلومات الضوء غير متوفرة.",
            "watering": "معلومات الري غير متوفرة.",
            "soil": "معلومات التربة غير متوفرة.",
            "temperature": "معلومات الحرارة غير متوفرة.",
            "pests": "معلومات الآفات غير متوفرة.",
            "planting_steps": "خطوات الزراعة غير متوفرة.",
            "fertilizer": "معلومات التسميد غير متوفرة.",
            "season": "موسم الزراعة غير متوفر.",
            "best_month_palestine": "أفضل شهر للزراعة في فلسطين غير متوفر.",
            "pot": "معلومات الأصيص غير متوفرة.",
        }
        return fallback.get(field, "المعلومة غير متوفرة.")

    # If it's a prefixed line like "الري: ....", extract value after colon
    value = raw_line.split(":", 1)[1].strip() if ":" in raw_line else raw_line.strip()

    templates = {
        "light": f"بالنسبة للضوء: {value} 🌤️",
        "watering": f"بالنسبة للري: {value} 💧",
        "soil": f"بالنسبة للتربة: {value} 🌱",
        "temperature": f"بالنسبة للحرارة المثالية: {value} 🌡️",
        "pests": f"بالنسبة للآفات الشائعة: {value} 🐛",
        "planting_steps": f"خطوات الزراعة: {value} 🪴",
        "fertilizer": f"التسميد: {value} 🧪",
        "season": f"موسم الزراعة: {value} 📅",
        "best_month_palestine": f"أفضل شهر للزراعة في فلسطين: {value} 🇵🇸",
        "pot": f"الأصيص/الوعاء المناسب: {value} 🪴",
        "summary": value,
    }
    return templates.get(field, value)


# ---------- Core Search ----------
@app.post("/search")
def search(req: SearchRequest):
    # Vectorize query
    q_emb = model.encode([req.query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, req.top_k)

    raw_results = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        card_text = cards[idx]
        raw_results.append((float(score), card_text))

    suggestion = suggest_plant_name(req.query)

    # Confidence handling
    best_score = raw_results[0][0] if raw_results else 0.0
    if (not raw_results) or (best_score < CONFIDENCE_THRESHOLD):
        resp = {"query": req.query, "results": []}
        if suggestion:
            resp["did_you_mean"] = suggestion
            resp["message"] = f"{SAFE_NO_ANSWER}\n\nهل تقصد: {suggestion['suggested']}؟"
        else:
            resp["message"] = SAFE_NO_ANSWER
        return resp

    # Build final results
    results = []
    for score, card_text in raw_results:
        extracted = extract_answer(card_text, req.query)
        answer = format_answer(extracted["field"], extracted["raw"])

        item = {
            "score": score,
            "field": extracted["field"],
            "answer": answer,
            "is_short": extracted["short"],
        }
        if req.include_card:
            item["card"] = card_text

        results.append(item)

    return {"query": req.query, "results": results}


# ---------- Chat Endpoint ----------
@app.post("/chat")
def chat(req: ChatRequest):
    # Reuse /search logic but return a single "reply" string for chat UI
    sreq = SearchRequest(query=req.message, top_k=req.top_k, include_card=False)
    res = search(sreq)

    # If no results (safe message)
    if not res.get("results"):
        # Prefer message if exists
        reply = res.get("message", SAFE_NO_ANSWER)
        return {
            "message": req.message,
            "reply": reply,
            "did_you_mean": res.get("did_you_mean"),
            "confidence": 0.0,
        }

    top = res["results"][0]
    return {
        "message": req.message,
        "reply": top["answer"],
        "field": top["field"],
        "confidence": top["score"],
    }


# ---------- Smart Plant Assistant (RAG) Endpoint ----------
@app.post("/assistant")
def assistant(req: AssistantRequest):
    """
    المساعد الذكي للنباتات – Excel-first architecture.

    Three response modes
    --------------------
    1. **direct_from_excel** – single-topic question, answer built
       entirely from structured Excel data.  No LLM call.
    2. **rewrite_from_excel** – multi-topic question, a data draft is
       built from Excel and the LLM only *rewrites* it into natural
       Arabic.  The model is NOT a source of information.
    3. **fallback** – plant not found in Excel or retrieval too weak.
       Returns a safe no-answer message.
    """
    t0 = time.perf_counter()
    _timings: dict[str, float] = {}

    logger.info(
        "[/assistant] ← entered | question=%r | top_k=%d | "
        "climate_data=%s | user_conditions=%s",
        req.question, req.top_k,
        bool(req.climate_data), bool(req.user_conditions),
    )

    # ── Step 1: Classify the question into intent(s) ──────────────
    _t = time.perf_counter()
    try:
        intents = classify_intents(req.question)
    except Exception:
        logger.error("[/assistant] intent classification FAILED\n%s", traceback.format_exc())
        intents = []
    _timings["intent"] = time.perf_counter() - _t
    logger.info("[/assistant] ⏱  Intents: %.3fs  (%s)", _timings["intent"], intents)

    direct = is_direct_intent(intents)
    multi = is_multi_field_intent(intents)
    logger.info("[/assistant] direct=%s  multi=%s", direct, multi)

    # ── Step 2: Try to identify the target plant from the question ─
    _t = time.perf_counter()
    plant_name: Optional[str] = None
    plant_data: Optional[dict] = None
    excel_hit = False

    if is_store_loaded():
        # Try extract_target_plant first (keyword/fuzzy match in question)
        all_names = get_all_plant_names()
        plant_name = extract_target_plant(req.question, all_names)
        if plant_name:
            plant_data = get_plant_data(plant_name)
            # Try to get the canonical display name
            original = get_plant_original_name(plant_name)
            if original:
                plant_name = original
        if plant_data:
            excel_hit = True
            logger.info("[/assistant] Excel hit: %r", plant_name)
        else:
            logger.info("[/assistant] No Excel hit via question parsing")
    _timings["excel_lookup"] = time.perf_counter() - _t

    # ── Step 3: Retrieval (FAISS semantic search) ─────────────────
    _t = time.perf_counter()
    try:
        retrieved = retrieve_relevant_context(
            query=req.question,
            model=model,
            index=index,
            cards=cards,
            top_k=req.top_k,
        )
    except Exception:
        logger.error("[/assistant] retrieval FAILED\n%s", traceback.format_exc())
        retrieved = []
    _timings["retrieval"] = time.perf_counter() - _t
    logger.info("[/assistant] ⏱  Retrieval: %.3fs  (%d result(s))", _timings["retrieval"], len(retrieved))

    best_score = get_best_retrieval_score(retrieved) if retrieved else 0.0

    # If we didn't find the plant from the question, try from retrieved cards
    if not excel_hit and retrieved and is_store_loaded():
        for r in retrieved:
            card_name = extract_plant_name(r["card"])
            pdata = get_plant_data(card_name)
            if pdata:
                plant_name = card_name
                original = get_plant_original_name(card_name)
                if original:
                    plant_name = original
                plant_data = pdata
                excel_hit = True
                logger.info("[/assistant] Excel hit via retrieved card: %r", plant_name)
                break

    # ── Build source references ───────────────────────────────────
    retrieved_sources = [
        {"score": r["score"], "snippet": r["card"][:200] + "…"}
        for r in retrieved
    ]

    # ── Step 4: Climate / user context + alternatives ─────────────
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
            all_cards=cards,
            user_temperature=user_temp,
        )
    except Exception:
        alternatives = []
    _timings["context"] = time.perf_counter() - _t

    suggestion = suggest_plant_name(req.question) if not excel_hit else None

    # ══════════════════════════════════════════════════════════════
    # DECISION: Choose response mode
    # ══════════════════════════════════════════════════════════════
    response_mode: str       # "direct_from_excel" | "rewrite_from_excel" | "fallback"
    answer: str = ""
    generation_used: bool = False

    if excel_hit and plant_data and direct:
        # ── MODE 1: direct_from_excel ─────────────────────────────
        # Single-topic question + Excel data available → answer
        # directly from structured data.  NO LLM call.
        _t = time.perf_counter()
        if has_sufficient_data(plant_data, intents, min_fields=1):
            answer = build_direct_answer(plant_name or "", plant_data, intents)
            response_mode = "direct_from_excel"
            logger.info("[/assistant] MODE: direct_from_excel")
        else:
            # Data exists but specific fields are empty → fallback
            answer = SAFE_NO_ANSWER
            response_mode = "fallback"
            logger.info("[/assistant] MODE: fallback (insufficient direct data)")
        _timings["answer_build"] = time.perf_counter() - _t

    elif excel_hit and plant_data and multi:
        # ── MODE 2: rewrite_from_excel ────────────────────────────
        # Multi-topic question + Excel data → build a draft from
        # Excel, then ask the LLM to rewrite it into natural Arabic.
        _t = time.perf_counter()
        if has_sufficient_data(plant_data, intents, min_fields=2):
            draft = build_answer_draft(plant_name or "", plant_data, intents)
            _timings["answer_build"] = time.perf_counter() - _t

            if draft and is_model_loaded():
                # Build rewrite prompt & call LLM
                _t2 = time.perf_counter()
                try:
                    prompt = build_rewrite_prompt(
                        user_question=req.question,
                        answer_draft=draft,
                        climate_context=climate_text,
                        user_context=user_text,
                    )
                    _timings["prompt"] = time.perf_counter() - _t2

                    _t3 = time.perf_counter()
                    answer = generate_grounded_answer(prompt)
                    generation_used = True
                    answer = clean_arabic_answer(answer)
                    _timings["generation"] = time.perf_counter() - _t3
                    response_mode = "rewrite_from_excel"
                    logger.info("[/assistant] MODE: rewrite_from_excel (LLM rewrote draft)")
                except Exception:
                    logger.error("[/assistant] rewrite generation FAILED\n%s", traceback.format_exc())
                    # Fall back to the raw draft (still from Excel)
                    answer = draft
                    response_mode = "rewrite_from_excel"
                    logger.info("[/assistant] MODE: rewrite_from_excel (raw draft, LLM failed)")
            elif draft:
                # Model not loaded – return the structured draft as-is
                answer = draft
                response_mode = "rewrite_from_excel"
                logger.info("[/assistant] MODE: rewrite_from_excel (raw draft, model not loaded)")
            else:
                answer = SAFE_NO_ANSWER
                response_mode = "fallback"
        else:
            _timings["answer_build"] = time.perf_counter() - _t
            answer = SAFE_NO_ANSWER
            response_mode = "fallback"
            logger.info("[/assistant] MODE: fallback (insufficient multi-field data)")

    elif excel_hit and plant_data:
        # ── Intents are empty or ambiguous → try direct answer ────
        _t = time.perf_counter()
        answer = build_direct_answer(plant_name or "", plant_data, intents if intents else ["general_summary"])
        _timings["answer_build"] = time.perf_counter() - _t
        if answer:
            response_mode = "direct_from_excel"
        else:
            answer = SAFE_NO_ANSWER
            response_mode = "fallback"

    else:
        # ── MODE 3: fallback ──────────────────────────────────────
        # Plant not found in Excel.  Try retrieval-based answer only
        # if retrieval is strong enough; otherwise safe no-answer.
        if retrieved and is_retrieval_strong(retrieved):
            # Build field context from retrieved cards if possible
            _t = time.perf_counter()
            field_context = None
            if is_store_loaded():
                plant_entries: list[dict] = []
                for r in retrieved:
                    cname = extract_plant_name(r["card"])
                    cdata = get_plant_data(cname)
                    if cdata:
                        plant_entries.append({"name": cname, "data": cdata})
                if plant_entries:
                    field_context = build_field_context_ar(plant_entries, intents if intents else ["general_summary"])

            if field_context and is_model_loaded():
                try:
                    card_texts = [r["card"] for r in retrieved]
                    prompt = build_assistant_prompt(
                        user_question=req.question,
                        retrieved_cards=card_texts,
                        field_context=field_context,
                        climate_context=climate_text,
                        user_context=user_text,
                        alternative_suggestions=alternatives if alternatives else None,
                    )
                    _timings["prompt"] = time.perf_counter() - _t

                    _t2 = time.perf_counter()
                    answer = generate_grounded_answer(prompt)
                    generation_used = True
                    answer = clean_arabic_answer(answer)
                    _timings["generation"] = time.perf_counter() - _t2
                    response_mode = "rewrite_from_excel"
                    logger.info("[/assistant] MODE: rewrite_from_excel (via retrieval pipeline)")
                except Exception:
                    logger.error("[/assistant] generation FAILED\n%s", traceback.format_exc())
                    answer = clean_arabic_answer(
                        format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX)
                    )
                    response_mode = "fallback"
            else:
                answer = clean_arabic_answer(
                    format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX)
                )
                response_mode = "fallback"
            _timings["answer_build"] = time.perf_counter() - _t
        else:
            answer = SAFE_NO_ANSWER
            response_mode = "fallback"
            if suggestion:
                answer += f"\n\nهل تقصد: {suggestion['suggested']}؟"

        logger.info("[/assistant] MODE: %s", response_mode)

    # ══════════════════════════════════════════════════════════════
    # FINAL SAFETY NET
    # ══════════════════════════════════════════════════════════════
    if generation_used and answer:
        if len(answer.strip()) < 5:
            logger.warning("[/assistant] SAFETY NET: answer too short → fallback")
            answer = clean_arabic_answer(
                format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX)
            ) if retrieved else SAFE_NO_ANSWER
            generation_used = False
            response_mode = "fallback"
        else:
            for pattern in LEAKED_INTERNAL_PATTERNS:
                if pattern in answer:
                    logger.warning("[/assistant] SAFETY NET: leaked pattern %r → fallback", pattern)
                    answer = clean_arabic_answer(
                        format_fallback_answer(retrieved, GENERATION_FALLBACK_PREFIX)
                    ) if retrieved else SAFE_NO_ANSWER
                    generation_used = False
                    response_mode = "fallback"
                    break

    # ── Performance summary ───────────────────────────────────────
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

    return {
        "question": req.question,
        "answer": answer,
        "responseMode": response_mode,
        "retrievedSources": retrieved_sources,
        "suggestedAlternatives": alternatives,
        "detectedIntents": intents,
        "confidence": best_score,
        "generationUsed": generation_used,
    }