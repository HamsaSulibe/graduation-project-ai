"""
settings.py – Central configuration for the Garssa Smart Plant Assistant (RAG).

All tuneable knobs live here so they can be overridden via environment
variables without touching code.  Defaults are sane for a first run on a
single GPU (or CPU‑only with slower inference).
"""

import os

# ──────────────────────────────────────────────
# 1. Retrieval (embedding model + FAISS index)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_DIR = os.getenv(
    "EMBEDDING_MODEL_DIR",
    "models/garssa-embed-v1",
)
INDEX_PATH = os.getenv("INDEX_PATH", "models/index/cards.faiss")
META_PATH = os.getenv("META_PATH", "models/index/cards_meta.json")

# How many plant cards to retrieve per query
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Minimum cosine‑similarity score to consider a result relevant.
# The backend uses this to decide AT THE PIPELINE LEVEL whether
# retrieval is strong enough to invoke the generation model.
# Below this threshold → fallback answer, NO LLM call.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# ──────────────────────────────────────────────
# 2. Generation model
# ──────────────────────────────────────────────
# Qwen2.5-0.5B-Instruct: smallest model in the Qwen2.5 family (~1 GB).
# Same tokenizer and chat-template as the larger variants, so zero code changes
# are required elsewhere.  Suitable for CPU-only / laptop local testing.
#
# To switch back to a heavier model without editing code:
#   $env:GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
#   $env:GENERATION_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GENERATION_MODEL_NAME = os.getenv(
    "GENERATION_MODEL_NAME",
   "Qwen/Qwen2.5-1.5B-Instruct" ,
)

# 128 tokens is enough for short, focused Arabic plant-care answers.
# Cutting this in half versus 256 roughly halves generation time on CPU.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

# Sampling temperature – lower = more deterministic, slightly faster
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Top‑p nucleus sampling
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Repetition penalty – kept modest; 0.5B models don't loop as badly
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Whether to load the generation model in 4‑bit quantisation
# (requires `bitsandbytes`; keep false for CPU-only machines)
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in ("1", "true", "yes")

# Force CPU so transformers doesn't waste time probing for CUDA on laptops.
# Change to "auto" if you have a compatible GPU.
DEVICE_MAP = os.getenv("DEVICE_MAP", "cpu")

# How many seconds to wait for the generation model before falling back
# to a retrieval-only answer.  Increase this when testing on slow hardware.
# To override without editing code: $env:GENERATION_TIMEOUT_SECONDS = "300"
GENERATION_TIMEOUT_SECONDS = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "180"))

# ──────────────────────────────────────────────
# 3. Data paths
# ──────────────────────────────────────────────
XLSX_PATH = os.getenv("XLSX_PATH", "data/raw/plants.xlsx")

# ──────────────────────────────────────────────
# 4. Fallback / safety messages (Arabic)
# ──────────────────────────────────────────────
SAFE_NO_ANSWER = (
    "عذرًا، لا تتوفر لديّ معلومات كافية في قاعدة البيانات للإجابة بدقة. "
    "جرّب تسأل بصياغة مختلفة أو اكتب اسم النبتة بوضوح 🌿"
)

GENERATION_FALLBACK_PREFIX = (
    "⚠️ لم أتمكن من توليد إجابة مفصّلة الآن. "
    "إليك أفضل ما وجدته من قاعدة البيانات:\n\n"
)

# ──────────────────────────────────────────────
# 5. Safety-net patterns
# ──────────────────────────────────────────────
# If the generated answer contains any of these substrings it means
# internal metadata leaked into the user-visible text.  The API
# layer will replace the whole answer with a safe fallback.
LEAKED_INTERNAL_PATTERNS: list[str] = [
    "retrievedSources",
    "generationUsed",
    "detectedIntents",
    "suggestedAlternatives",
    '"score"',
    '"snippet"',
    '"debug"',
    "CONFIDENCE_THRESHOLD",
    "best_score",
    "field_context",
    "لا أتمكن من توليد",
    "لم أتمكن من توليد",
    "generate_grounded_answer",
    "build_assistant_prompt",
]

# ──────────────────────────────────────────────
# 6. Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
