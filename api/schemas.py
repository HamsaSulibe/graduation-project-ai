"""
schemas.py – Pydantic request models for the Garssa AI API.
"""
from typing import Optional

from pydantic import BaseModel

from src.config.settings import TOP_K_RETRIEVAL


class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    include_card: bool = False  # for debugging only


class ChatRequest(BaseModel):
    message: str
    top_k: int = 1


class HistoryMessage(BaseModel):
    """Single conversation turn used by /assistant for context."""
    role: Optional[str] = None
    sender: Optional[str] = None
    content: str


# ── Smart Assistant (RAG) request model ──────────────────────────
class AssistantRequest(BaseModel):
    """Request body for the /assistant endpoint."""
    question: str                              # سؤال المستخدم
    top_k: int = TOP_K_RETRIEVAL               # عدد النتائج المسترجعة
    language: Optional[str] = None             # لغة التطبيق: ar أو en (اختياري)
    app_language: Optional[str] = None         # Alias for clients that send appLanguage/app_language
    appLanguage: Optional[str] = None
    climate_data: Optional[dict] = None        # بيانات مناخ اختيارية
    user_conditions: Optional[dict] = None     # ظروف المستخدم اختيارية
    history: Optional[list[HistoryMessage]] = None  # سجل المحادثة (اختياري)
