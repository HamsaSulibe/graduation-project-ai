"""
rag_generator.py – Generation model loader and answer generator.

Loads a Hugging Face instruction‑following model (default: Qwen2.5‑7B‑Instruct)
and exposes a single public function:

    generate_grounded_answer(prompt: str) -> str

The module is designed to be:
  • imported by the API layer for online inference, OR
  • imported by a notebook / CLI for offline testing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.settings import (
    DEVICE_MAP,
    GENERATION_MODEL_NAME,
    LOAD_IN_4BIT,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)

logger = logging.getLogger(__name__)

# ── Module‑level singletons (loaded once) ────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


# ─────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────
def load_generation_model(
    model_name: str = GENERATION_MODEL_NAME,
    device_map: str = DEVICE_MAP,
    load_in_4bit: bool = LOAD_IN_4BIT,
) -> None:
    """
    Load the generation model and tokenizer into module‑level singletons.

    Call this once at application startup (e.g. inside FastAPI's `on_event("startup")`).
    Subsequent calls are no‑ops if the model is already loaded.
    """
    global _tokenizer, _model

    if _model is not None:
        logger.info("Generation model already loaded – skipping.")
        return

    logger.info("Loading generation model: %s (4‑bit=%s) …", model_name, load_in_4bit)

    # ── Build quantisation config if requested ────────────────────
    model_kwargs: dict = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        except ImportError:
            logger.warning(
                "bitsandbytes not installed – falling back to fp16."
            )

    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    logger.info("✅ Generation model loaded successfully.")


def is_model_loaded() -> bool:
    """Check whether the generation model is ready."""
    return _model is not None and _tokenizer is not None


# ─────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────
def generate_grounded_answer(
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
) -> str:
    """
    Send a fully‑constructed prompt to the generation model and return
    the generated text (assistant reply only, without the prompt echo).

    Parameters
    ----------
    prompt : str
        The complete prompt string produced by `build_assistant_prompt()`.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability mass.
    repetition_penalty : float
        Penalty for repeated tokens.

    Returns
    -------
    str
        The model's Arabic response.

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet (call `load_generation_model()` first).
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Generation model not loaded. Call load_generation_model() first."
        )

    # ── Qwen chat‑template expects list of messages ──────────────
    # We already have the full prompt from the prompt builder, but
    # Qwen works best when we pass structured messages through
    # apply_chat_template.  If the caller already formatted the
    # prompt as a single string we wrap it in a simple user message.
    messages = [{"role": "user", "content": prompt}]

    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )

    # Decode only the newly generated tokens (skip the input)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return answer
