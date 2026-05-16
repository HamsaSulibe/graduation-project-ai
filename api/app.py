"""
app.py – FastAPI application entry-point.

Responsibilities (API layer only):
  • App definition and startup (load AI assets, call assistant_service.initialize)
  • Health / root endpoints
  • /assistant endpoint – thin delegation to assistant_service
"""

import json
import logging
from pathlib import Path

import faiss
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer

from src.config.settings import (
    EMBEDDING_MODEL_DIR,
    INDEX_PATH as SETTINGS_INDEX_PATH,
    META_PATH as SETTINGS_META_PATH,
    XLSX_PATH as SETTINGS_XLSX_PATH,
)
from src.inference.rag_generator import load_generation_model
from src.inference.plant_data_store import (
    get_all_plant_names,
    is_store_loaded,
    load_plant_store,
)
from api.schemas import AssistantRequest
import src.inference.assistant_service as _assistant_svc

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Graduation Project AI API")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(EMBEDDING_MODEL_DIR)
INDEX_PATH = Path(SETTINGS_INDEX_PATH)
META_PATH  = Path(SETTINGS_META_PATH)
XLSX_PATH  = Path(SETTINGS_XLSX_PATH)

# ── Globals ────────────────────────────────────────────────────────────────────
model = None
index = None
cards = None


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_assets():
    global model, index, cards

    model = SentenceTransformer(str(MODEL_DIR))
    index = faiss.read_index(str(INDEX_PATH))
    meta  = json.loads(META_PATH.read_text(encoding="utf-8"))
    cards = meta["cards"]

    plant_count = load_plant_store(XLSX_PATH)
    print(f"✅ Plant data store: {plant_count} plants")

    print("✅ AI assets loaded")
    print(f"✅ cards loaded: {len(cards)}")
    if is_store_loaded():
        print(f"✅ plant names loaded: {len(get_all_plant_names())}")

    try:
        load_generation_model()
        print("✅ Generation model loaded: OpenAI API")
    except Exception as exc:
        print(f"⚠️  Generation model could not be loaded: {exc}")
        print("   The /assistant endpoint will return retrieval‑only fallback answers.")

    # Inject runtime assets into assistant_service
    _assistant_svc.initialize(model, index, cards)


# ── Basic endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── /assistant ─────────────────────────────────────────────────────────────────
@app.post("/assistant")
def assistant(req: AssistantRequest):
    """Smart plant assistant – delegates all logic to assistant_service."""
    return _assistant_svc.handle_assistant_request(req)
