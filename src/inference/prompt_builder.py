"""
prompt_builder.py – Assembles grounded Arabic prompts for the RAG assistant.

Responsibilities (this file):
  - Section assembly and ordering
  - Choosing field_context vs. retrieved_cards
  - Injecting optional climate / user / alternatives sections

What this file does NOT contain:
  - Any Arabic prompt text (lives in src/config/prompts.yaml)
  - Loading logic (lives in src/config/prompt_loader.py)
"""

from __future__ import annotations

from typing import Optional

from src.config.prompt_loader import PROMPTS

# Shorthand references – resolved once at import time
_A = PROMPTS["assistant"]       # assistant prompt config
_R = PROMPTS["rewrite"]         # rewrite prompt config
_AL = _A["labels"]              # assistant section labels
_RL = _R["labels"]              # rewrite section labels


# ─────────────────────────────────────────────────────────────────
# Public builder – grounded RAG assistant prompt
# ─────────────────────────────────────────────────────────────────
def build_assistant_prompt(
    user_question: str,
    retrieved_cards: Optional[list[str]] = None,
    field_context: Optional[str] = None,
    climate_context: Optional[str] = None,
    user_context: Optional[str] = None,
    alternative_suggestions: Optional[list[str]] = None,
) -> str:
    """
    Assemble the full prompt string that will be passed to the generation model.

    Parameters
    ----------
    user_question : str
        The raw user question in Arabic.
    retrieved_cards : list[str] | None
        Raw top‑k plant card texts (used only when *field_context*
        is not available).
    field_context : str | None
        **Preferred.**  Structured Arabic text built by
        ``intent_fields.build_field_context_ar()`` – contains only
        the fields relevant to the detected question intent(s).
    climate_context : str | None
        Free-text summary of climate/weather data for the user's location.
    user_context : str | None
        Known user conditions (e.g. "شرفة مظللة، أصيص 25 سم").
    alternative_suggestions : list[str] | None
        Pre‑computed alternative plant names (may be empty).

    Returns
    -------
    str
        The complete prompt text ready for the generation model.
    """
    sections: list[str] = []

    # -- System preamble -------------------------------------------
    sections.append(_A["system"])

    # -- Plant data section ----------------------------------------
    # Prefer field_context (structured, intent-filtered) over raw cards.
    # IMPORTANT: Only factual plant data is injected here – no scores,
    # no debug info, no metadata.  This keeps the model's context clean.
    if field_context:
        sections.append(_AL["data_available"])
        sections.append(field_context)
    elif retrieved_cards:
        sections.append(_AL["data_available"])
        for i, card in enumerate(retrieved_cards, 1):
            label = _AL["plant_item"].format(n=i)
            sections.append(f"{label}\n{card}")
    else:
        sections.append(_AL["no_data"])

    # -- Climate / weather data (optional) -------------------------
    if climate_context:
        sections.append(f"{_AL['climate']}\n{climate_context}")

    # -- User context (optional) -----------------------------------
    if user_context:
        sections.append(f"{_AL['user_context']}\n{user_context}")

    # -- Pre‑computed alternative suggestions (optional) -----------
    if alternative_suggestions:
        alts = "، ".join(alternative_suggestions)
        sections.append(f"{_AL['alternatives']}\n{alts}")

    # -- User question (always last) -------------------------------
    sections.append(f"{_AL['user_question']}\n{user_question}")

    # -- Final instruction -----------------------------------------
    sections.append(_A["final_instruction"])

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────
# Rewrite-only prompt  (for multi-field intents)
# ─────────────────────────────────────────────────────────────────
def build_rewrite_prompt(
    user_question: str,
    answer_draft: str,
    climate_context: Optional[str] = None,
    user_context: Optional[str] = None,
) -> str:
    """
    Build a prompt that asks the LLM to REWRITE (not generate)
    the structured *answer_draft* into natural Arabic.

    The system prompt explicitly forbids the model from adding
    any information not present in the draft.
    """
    sections: list[str] = [_R["system"]]

    sections.append(f"{_RL['draft_text']}\n{answer_draft}")

    if climate_context:
        sections.append(f"{_RL['climate']}\n{climate_context}")
    if user_context:
        sections.append(f"{_RL['user_context']}\n{user_context}")

    sections.append(f"{_RL['user_question']}\n{user_question}")
    sections.append(_R["final_instruction"])

    return "\n\n".join(sections)
