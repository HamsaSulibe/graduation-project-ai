"""
rag_generator.py – Generation model loader and answer generator.

Delegates text generation to OpenAI via openai_generator.
Exposes the same public interface so the rest of the pipeline is unchanged:

    load_generation_model()
    is_model_loaded() -> bool
    generate_grounded_answer(prompt: str) -> str
"""

from __future__ import annotations

import logging

from src.inference.openai_generator import generate, init_openai, is_openai_ready

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Model loading  (delegates to OpenAI init)
# ─────────────────────────────────────────────────────────────────
def load_generation_model(**_kwargs) -> None:
    """
    Initialise the OpenAI client.

    Accepts (and ignores) any keyword arguments so that existing
    call-sites that pass model_name / device_map / etc. keep working.
    """
    init_openai()


def is_model_loaded() -> bool:
    """Check whether the generation backend is ready."""
    return is_openai_ready()


# ─────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────
def generate_grounded_answer(prompt: str, **_kwargs) -> str:
    """
    Send a fully-constructed prompt to OpenAI and return the response.

    Extra keyword arguments (max_new_tokens, temperature, …) are
    accepted for backward-compatibility but ignored — OpenAI handles
    its own defaults.

    Parameters
    ----------
    prompt : str
        The complete prompt string produced by the prompt builder.

    Returns
    -------
    str
        The model's Arabic response.
    """
    return generate(prompt)
