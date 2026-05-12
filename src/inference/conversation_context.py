"""
conversation_context.py – Conversation history helpers for the Garssa
Smart Plant Assistant.

All functions are pure; no FastAPI / app dependencies.
"""
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api.schemas import HistoryMessage

from src.utils.arabic import normalize_query
from src.inference.intent_fields import classify_intents
from src.inference.context_utils import (
    extract_all_plants_from_question,
    extract_target_plant,
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

_HISTORY_ROLE_MAP: dict[str, str] = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}

_FOLLOWUP_PATTERNS: set[str] = {
    "كمل",
    "كمّل",
    "اشرح اكثر",
    "اشرح أكثر",
    "ماذا تقصد",
    "ماذا عنه",
    "ماذا عنها",
    "وهل نفس الشي ينطبق هنا",
    "وهل نفس الشي ينطبق",
    "وهل نفس الشي",
    "ماذا عن",
    "هل يناسبها",
    "هل يناسبه",
    "ما نوع التربة له",
    "ما نوع التربة لها",
    "how often should i water it",
    "how often do i water it",
    "how much light does it need",
    "what soil does it need",
    "when should i harvest it",
    "what about it",
    "what about this plant",
}

_WEAK_HISTORY_INTENTS: set[str] = {
    "general_summary",
    "care_summary",
    "plant_overview",
    "growing_guide",
    "beginner_overview",
}

MAX_HISTORY_MESSAGES: int = 8

# Signal phrase that appears in assistant turn when it asked for a criterion
_ASK_CRITERION_SIGNAL: str = "حدد معيار المقارنة"
_ASK_CRITERION_SIGNAL_EN: str = "Please choose a comparison criterion"


# ──────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────

def normalize_history_messages(
    history: "Optional[list[HistoryMessage]]",
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    """
    Normalize incoming history:
    - accepts mixed role casing (USER/user/Assistant/...)
    - drops empty/invalid turns
    - keeps only the latest *max_messages* turns
    """
    if not history:
        return []

    normalized: list[dict[str, str]] = []
    for msg in history:
        role_raw = (
            getattr(msg, "role", None)
            or getattr(msg, "sender", None)
            or ""
        ).strip().lower()
        content = (getattr(msg, "content", "") or "").strip()
        if not content:
            continue

        role = _HISTORY_ROLE_MAP.get(role_raw)
        if role is None:
            if role_raw in {"user", "assistant", "system"}:
                role = role_raw
            else:
                continue

        normalized.append({"role": role, "content": content})

    if max_messages > 0 and len(normalized) > max_messages:
        return normalized[-max_messages:]
    return normalized


def is_ambiguous_followup_question(question: str) -> bool:
    """Detect short follow-up queries that depend on previous turns."""
    q = normalize_query(question)
    if not q:
        return False

    compact = re.sub(r"\s+", " ", q).strip().lower()
    if compact in _FOLLOWUP_PATTERNS:
        return True

    short_followup_tokens = {
        "كمل", "كمّل", "وضح", "وضّح", "ليش", "لماذا", "طيب", "طيب؟",
        "كيف", "وهل", "هل", "عنه", "عنها", "له", "لها", "هنا", "نفس",
        "it", "its", "this", "that", "about",
    }

    words = compact.split()
    if len(words) <= 4 and any(w in short_followup_tokens for w in words):
        return True

    if len(compact) <= 18:
        return True

    if any(p in compact for p in (" it", "its ", " this plant", " that plant")):
        return True

    return False


def _has_strong_domain_intent_signal(intents: list[str]) -> bool:
    if not intents:
        return False
    return any(intent not in _WEAK_HISTORY_INTENTS for intent in intents)


def infer_intents_from_history(history: list[dict[str, str]]) -> list[str]:
    """Infer likely domain intents from recent user turns for short follow-ups."""
    if not history:
        return []

    for turn in reversed(history[-8:]):
        if turn.get("role") != "user":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue

        candidate = classify_intents(content)
        if _has_strong_domain_intent_signal(candidate):
            return candidate

    return []


def infer_plant_from_history(
    history: list[dict[str, str]],
    all_plant_names: list[str],
) -> Optional[str]:
    """Resolve pronoun-like follow-ups to the latest known plant in history."""
    if not history or not all_plant_names:
        return None

    for turn in reversed(history[-10:]):
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        matched = extract_target_plant(content, all_plant_names)
        if matched:
            return matched

    return None


def build_effective_question(question: str, history: list[dict[str, str]]) -> str:
    """
    Build a context-aware question for retrieval/intent parsing.

    For ambiguous follow-ups, prepend recent conversation context so
    extraction/retrieval can resolve pronouns and references.
    """
    q = (question or "").strip()
    if not q or not history:
        return q

    if not is_ambiguous_followup_question(q):
        return q

    recent_turns = history[-4:]
    context_lines: list[str] = []
    for turn in recent_turns:
        role = turn["role"]
        label = "المستخدم" if role == "user" else "المساعد" if role == "assistant" else "النظام"
        text = turn["content"].strip()
        if text:
            context_lines.append(f"{label}: {text}")

    if not context_lines:
        return q

    conversation_context = "\n".join(context_lines)
    return f"سياق المحادثة:\n{conversation_context}\n\nالسؤال الحالي: {q}"


def extract_pending_comparison_plants_from_history(
    history: list[dict[str, str]],
    all_plant_names: list[str],
) -> list[str]:
    """
    Look through conversation history to find plants from a previous comparison
    question where the assistant replied asking for a criterion.

    Returns a list of ≥2 plant names if found, otherwise [].
    """
    if not history or not all_plant_names:
        return []

    # Walk backwards to find the last assistant message that asked for criterion
    ask_criterion_idx = -1
    for i in range(len(history) - 1, -1, -1):
        turn = history[i]
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            if _ASK_CRITERION_SIGNAL in content or _ASK_CRITERION_SIGNAL_EN in content:
                ask_criterion_idx = i
                break

    if ask_criterion_idx == -1:
        return []

    # Look for the user turn right before that assistant message
    for i in range(ask_criterion_idx - 1, -1, -1):
        turn = history[i]
        if turn.get("role") == "user":
            content = turn.get("content", "")
            plants = extract_all_plants_from_question(content, all_plant_names)
            if len(plants) >= 2:
                return plants
            break

    return []
