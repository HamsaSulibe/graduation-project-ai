"""
query_router.py - Lightweight decision layer before answer generation.

This module routes incoming questions into one of two top-level tracks:
1) domain_rag: plant/agriculture-related questions
2) general_conversation: out-of-domain chat

For domain_rag, downstream code still decides between:
- data-grounded answer
- safe fallback when data is insufficient
"""

from __future__ import annotations

from dataclasses import dataclass

from src.inference.context_utils import extract_target_plant
from src.utils.arabic import normalize


@dataclass(frozen=True)
class RouteDecision:
    """Routing result returned by classify_route()."""

    route: str
    reason: str


_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "نبات",
    "نبتة",
    "نباتات",
    "زراعة",
    "ازرع",
    "أزرع",
    "زرع",
    "ري",
    "اسقي",
    "سقي",
    "تربة",
    "سماد",
    "تسميد",
    "أصيص",
    "اصيص",
    "حصاد",
    "شتلة",
    "شتلات",
    "بذور",
    "زراعي",
    "agriculture",
    "plant",
    "plants",
    "watering",
    "soil",
    "fertiliz",
    "harvest",
)

_WEAK_SIGNAL_INTENTS: set[str] = {
    "general_summary",
    "care_summary",
    "plant_overview",
    "growing_guide",
    "beginner_overview",
}

_FOLLOWUP_REFERENTIAL_TOKENS: tuple[str, ...] = (
    "له",
    "لها",
    "عنه",
    "عنها",
    "هل يناسبها",
    "هل يناسبه",
    "ماذا عنه",
    "ماذا عنها",
    "اشرح أكثر",
    "اشرح اكثر",
    "كمل",
    "كمّل",
    "ماذا تقصد",
)


def _contains_domain_keyword(question: str) -> bool:
    q_norm = normalize(question).lower()
    return any(kw in q_norm for kw in _DOMAIN_KEYWORDS)


def _has_strong_domain_intent(intents: list[str]) -> bool:
    # Some intents are broad linguistic patterns and can appear in non-plant
    # questions (for example "كيف أعتني ..."). Treat them as weak signals.
    if not intents:
        return False
    return any(intent not in _WEAK_SIGNAL_INTENTS for intent in intents)


def _looks_referential_followup(question: str) -> bool:
    q_norm = normalize(question).lower()
    return any(tok in q_norm for tok in _FOLLOWUP_REFERENTIAL_TOKENS)


def _history_has_domain_signal(
    conversation_history: list[dict[str, str]] | None,
    all_plant_names: list[str] | None = None,
) -> bool:
    if not conversation_history:
        return False

    for turn in reversed(conversation_history[-8:]):
        content = (turn.get("content") or "").strip()
        if not content:
            continue

        if _contains_domain_keyword(content):
            return True

        if all_plant_names and extract_target_plant(content, all_plant_names):
            return True

    return False


def classify_route(
    question: str,
    intents: list[str],
    all_plant_names: list[str] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    is_followup: bool = False,
) -> RouteDecision:
    """
    Decide whether a question is plant-domain or general conversation.

    Returns
    -------
    RouteDecision
        route is one of: "domain_rag", "general_conversation"
    """
    q = (question or "").strip()
    if not q:
        return RouteDecision(route="general_conversation", reason="empty_question")

    if _has_strong_domain_intent(intents):
        return RouteDecision(route="domain_rag", reason="strong_intent_match")

    if _contains_domain_keyword(q):
        return RouteDecision(route="domain_rag", reason="domain_keyword_match")

    if all_plant_names:
        matched = extract_target_plant(q, all_plant_names)
        if matched:
            return RouteDecision(route="domain_rag", reason="plant_name_match")

    # Short/ambiguous follow-ups should inherit domain only when
    # recent history carries clear plant-domain signals.
    if is_followup or _looks_referential_followup(q):
        if _history_has_domain_signal(conversation_history, all_plant_names):
            return RouteDecision(route="domain_rag", reason="followup_with_domain_history")

    return RouteDecision(route="general_conversation", reason="out_of_domain")


def build_general_chat_prompt(
    user_question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Build a prompt for open-domain chat with no database-grounding claims."""
    history_lines: list[str] = []
    if conversation_history:
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            role_label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(no prior conversation)"

    return (
        "You are a helpful, natural, and concise general assistant.\n"
        "Rules:\n"
        "1) Answer naturally in the user's language.\n"
        "2) Be flexible with everyday requests (greetings, writing help, explanations, brainstorming).\n"
        "3) Do not claim your answer is retrieved from a plant database.\n"
        "4) Do not mix plant database facts unless explicitly provided in this prompt.\n"
        "5) If the user asks for a plant-specific factual answer without provided evidence, ask them to request a plant-domain grounded answer.\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"User question:\n{user_question}\n\n"
        "Assistant answer:"
    )
