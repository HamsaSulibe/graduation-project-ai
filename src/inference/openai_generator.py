"""
openai_generator.py – OpenAI generation layer.

Provides text generation via the OpenAI API.
Exposes the same generate() interface used by the RAG pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────
_client: Optional[OpenAI] = None
_model_name: str = "gpt-4.1-mini"


def init_openai(model_name: str | None = None) -> None:
    """
    Initialise the OpenAI client once using the OPENAI_API_KEY env var.

    Call this at application startup.  Subsequent calls are no-ops.
    """
    global _client, _model_name

    if _client is not None:
        logger.info("OpenAI client already initialised – skipping.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before starting the application."
        )

    env_model = os.environ.get("OPENAI_MODEL")
    if env_model:
        _model_name = env_model
    if model_name:
        _model_name = model_name

    _client = OpenAI(api_key=api_key)
    logger.info("✅ OpenAI client initialised (model=%s)", _model_name)


def is_openai_ready() -> bool:
    """Return True if the OpenAI client has been initialised."""
    return _client is not None


def generate(prompt: str) -> str:
    """
    Send *prompt* to OpenAI and return the plain-text response.

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
            "OpenAI client not initialised. Call init_openai() first."
        )

    response = _client.chat.completions.create(
        model=_model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""
