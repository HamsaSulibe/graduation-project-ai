"""
prompt_loader.py – Load and validate prompt templates from prompts.yaml.

Loads once at import time. All consumers import the module-level
``PROMPTS`` dict directly; no repeated file I/O at request time.

Fallback: if prompts.yaml is missing or malformed, hard-coded minimal
defaults are used so the server never crashes on startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_YAML_PATH = Path(__file__).parent / "prompts.yaml"

# ─────────────────────────────────────────────────────────────────
# Minimal fallback templates (kept here – NOT in prompt_builder.py)
# Only used when prompts.yaml cannot be read.
# ─────────────────────────────────────────────────────────────────
_FALLBACK: dict[str, Any] = {
    "assistant": {
        "system": (
            "أنت مساعد زراعي ذكي. أجب فقط بناءً على البيانات المقدمة. "
            "إذا لم تجد المعلومة قل: «لا أملك معلومات كافية في قاعدة البيانات.»"
        ),
        "labels": {
            "data_available": "البيانات المتوفرة:",
            "no_data": "(لا توجد نتائج مطابقة في قاعدة البيانات)",
            "plant_item": "نبتة {n}:",
            "climate": "بيانات المناخ:",
            "user_context": "سياق المستخدم:",
            "alternatives": "نباتات بديلة قد تناسب الظروف:",
            "user_question": "سؤال المستخدم:",
        },
        "final_instruction": (
            "أجب الآن بإجابة قصيرة ومفيدة بالعربية اعتمادًا فقط على البيانات أعلاه."
        ),
    },
    "rewrite": {
        "system": (
            "أنت أداة إعادة صياغة فقط. "
            "أعد صياغة النص المزوّد إلى فقرة عربية سلسة بدون إضافة أي معلومة جديدة."
        ),
        "labels": {
            "draft_text": "النص المطلوب إعادة صياغته:",
            "climate": "بيانات المناخ:",
            "user_context": "سياق المستخدم:",
            "user_question": "سؤال المستخدم:",
        },
        "final_instruction": (
            "أعد صياغة النص أعلاه فقط بجمل عربية طبيعية مترابطة. "
            "لا تضف أي معلومة غير موجودة في النص."
        ),
    },
}

# Required top-level keys in prompts.yaml
_REQUIRED_KEYS = {"assistant", "rewrite"}
# Required sub-keys per prompt type
_REQUIRED_SUB = {
    "assistant": {"system", "labels", "final_instruction"},
    "rewrite": {"system", "labels", "final_instruction"},
}


def _validate(data: dict[str, Any]) -> None:
    missing_top = _REQUIRED_KEYS - data.keys()
    if missing_top:
        raise ValueError(f"prompts.yaml missing top-level keys: {missing_top}")
    for section, sub_keys in _REQUIRED_SUB.items():
        missing_sub = sub_keys - data[section].keys()
        if missing_sub:
            raise ValueError(f"prompts.yaml[{section}] missing keys: {missing_sub}")


def _load() -> dict[str, Any]:
    if not _YAML_PATH.exists():
        log.warning("prompts.yaml not found at %s – using built-in fallback", _YAML_PATH)
        return _FALLBACK

    try:
        with _YAML_PATH.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        _validate(data)
        log.debug("Loaded prompt templates from %s", _YAML_PATH)
        return data
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load prompts.yaml (%s) – using built-in fallback", exc)
        return _FALLBACK


# Module-level singleton – loaded once at import time
PROMPTS: dict[str, Any] = _load()
