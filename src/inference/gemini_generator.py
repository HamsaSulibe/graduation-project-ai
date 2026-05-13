"""
gemini_generator.py – Google Gemini API generation layer.

NOT USED – replaced by openai_generator.py.
Kept here for reference only.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────
_client: Optional[genai.Client] = None
_model_name: str = "gemini-2.5-flash"


def init_gemini(model_name: str | None = None) -> None:
    """
    Initialise the Gemini client once using the GEMINI_API_KEY env var.

    Call this at application startup.  Subsequent calls are no-ops.
    """
    global _client, _model_name

    if _client is not None:
        logger.info("Gemini client already initialised – skipping.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before starting the application."
        )

    if model_name:
        _model_name = model_name

    _client = genai.Client(api_key=api_key)
    logger.info("✅ Gemini client initialised (model=%s)", _model_name)


def is_gemini_ready() -> bool:
    """Return True if the Gemini client has been initialised."""
    return _client is not None


def generate(prompt: str) -> str:
    """
    Send *prompt* to Gemini and return the plain-text response.

    Parameters
    ----------
    prompt : str
        The fully-constructed prompt from the prompt builder.

    Returns
    -------
    str
        The model's text response (no metadata).

    Raises
    ------
    RuntimeError
        If the client has not been initialised yet.
    """
    if _client is None:
        raise RuntimeError(
            "Gemini client not initialised. Call init_gemini() first."
        )

    response = _client.models.generate_content(
        model=_model_name,
        contents=prompt,
    )

    return response.text.strip() if response.text else ""
