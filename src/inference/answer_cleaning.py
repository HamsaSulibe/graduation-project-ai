"""
answer_cleaning.py – Arabic answer post-processing utilities for the
Garssa Smart Plant Assistant.

All functions are pure (no FastAPI / app dependencies).
"""
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Regex constants
# ──────────────────────────────────────────────────────────────────

_FIELD_LABEL_RE = re.compile(
    r"(?m)^(?:"
    r"النبتة|التصنيف|مستوى الصعوبة|وصف|"
    r"الضوء|الري|التربة|الحرارة المثالية|الرطوبة|"
    r"التسميد|خطوات الزراعة|أيام حتى الحصاد|الحصاد|"
    r"الاستخدامات|العناية|أشهر الزراعة المناسبة|"
    r"نتيجة \d+"
    r")\s*:\s*"
)
_SEPARATOR_RE = re.compile(r"(?m)^[-=*_٭]{3,}\s*$")
_CARD_HEADER_RE = re.compile(r"(?m)^-+\s*نتيجة\s*\d+\s*-+\s*$")
_MD_HEADER_RE = re.compile(r"(?m)^#{1,4}\s*")
_PAREN_EN_RE = re.compile(r'\([A-Za-z][A-Za-z\s./%°,;:\'"\x2d\u201c\u201d\u2018\u2019]+\)')

_UNDERSCORE_SENTENCE_RE = re.compile(
    r'[^\n.!؟،]*(?:\s+ضمن\s+_|\s+في\s+_|\s+من\s+_|يُصنَّف\s+ضمن\s+_|يصنف\s+ضمن\s+_)[^\n.!؟،]*[.!؟،]?'
)
_BARE_UNDERSCORE_RE = re.compile(r'(?<!\w)_(?!\w)')


# ──────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────

def clean_arabic_answer(text: str) -> str:
    """
    Post-process a generated or fallback Arabic answer:
    - strip Markdown headers
    - remove separator lines
    - remove raw card field labels that leak into generated text
    - collapse multiple blank lines
    - strip surrounding quotes / escape artifacts
    """
    if not text:
        return text
    text = _MD_HEADER_RE.sub("", text)
    text = _SEPARATOR_RE.sub("", text)
    text = _CARD_HEADER_RE.sub("", text)
    text = _FIELD_LABEL_RE.sub("", text)
    # Clean technical artefacts
    text = text.replace(" | ", "، ").replace("|", "، ")
    text = text.replace("• ", "").replace("* ", "")
    text = _PAREN_EN_RE.sub("", text)
    # Collapse 3+ newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    # Strip surrounding quote characters
    text = text.strip('"\'""''`')
    return text.strip()


def _answer_echoes_question(answer: str, question: str) -> bool:
    """True when the answer text is essentially the same as the question."""
    if not answer or not question:
        return False
    from src.utils.arabic import normalize as _normalize

    a_norm = re.sub(r'\s+', ' ', _normalize(answer.strip())).lower()
    q_norm = re.sub(r'\s+', ' ', _normalize(question.strip())).lower()
    if a_norm == q_norm:
        return True
    if len(a_norm) <= len(q_norm) + 5 and q_norm in a_norm:
        return True
    return False


def _contains_underscore_artifact(text: str) -> bool:
    """True when the answer contains a bare underscore placeholder."""
    return bool(_BARE_UNDERSCORE_RE.search(text))


def _strip_underscore_sentences(text: str) -> str:
    """Remove sentences containing bare underscore placeholders."""
    cleaned = _UNDERSCORE_SENTENCE_RE.sub('', text)
    cleaned = _BARE_UNDERSCORE_RE.sub('', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

# ---------------------------------------------------------------------------
# Direct-answer cleaning helpers moved from context_utils.py
# ---------------------------------------------------------------------------

_EN_CHUNK_RE = re.compile(r'[A-Za-z][A-Za-z\-\'\/]*(?:\s+[A-Za-z][A-Za-z\-\'\/]*)+|[A-Za-z]{3,}')

_EN_CHAR_RE  = re.compile(r'[A-Za-z]')

_AR_CHAR_RE  = re.compile(r'[\u0600-\u06FF]')

_EN_OR_DIGIT_RE = re.compile(r'[A-Za-z0-9]')

def _clean_text(s: str) -> str:
    """Strip technical artefacts: |, *, English parens, junk substrings, excess whitespace."""
    if not s:
        return s
    s = s.replace(" | ", "، ").replace("|", "، ")
    s = s.replace(" + ", " و")
    s = s.replace("• ", "").replace("* ", "")
    s = s.replace("\\n", " ").replace("\n", " ")
    s = _PAREN_EN_RE.sub("", s)
    # Remove junk placeholders embedded in longer text
    s = re.sub(r'مش موجود|غير متوفر|غير متوفرة|غير معروف|غير مذكور', '', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = s.replace("..", ".").replace("،.", ".").replace(".،", "،")
    return s

def _clean_english_text(value: str) -> Optional[str]:
    """Return an English-only display value, or None if no usable English remains."""
    if value is None:
        return None
    text = _clean_text(str(value))
    if not text:
        return None

    text = re.sub(r'(?<=\d)\s*سم(?![\u0600-\u06FF])', ' cm', text)
    text = re.sub(r'(?<=\d)\s*سنتيمتر(?![\u0600-\u06FF])', ' cm', text)
    text = re.sub(r'(?<=\d)\s*(?:إنش|انش|بوصة)(?![\u0600-\u06FF])', ' inch', text)
    text = re.sub(r'(?<![\u0600-\u06FF])نعم(?![\u0600-\u06FF])', 'yes', text)
    text = re.sub(r'(?<![\u0600-\u06FF])لا(?![\u0600-\u06FF])', 'no', text)
    text = text.replace("°م", "C").replace("م°", "C")

    text = re.sub(r'\bsource data\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u0600-\u06FF]+', ' ', text)
    text = re.sub(r'\s+([,.;:!?%)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace(" ,", ",").replace("..", ".").strip(" -،,.;:")

    if not text or _AR_CHAR_RE.search(text):
        return None
    if not _EN_OR_DIGIT_RE.search(text):
        return None
    return text

def _strip_inline_english(text: str) -> str:
    """Remove inline English word sequences from Arabic text.

    Removes: multi-word English phrases and standalone English words ≥3 chars.
    Preserves: degree symbols (°C/°F), % numbers, and short abbreviations like pH.
    """
    if not text:
        return text
    result = _EN_CHUNK_RE.sub('', text)
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r'\s+([،,.:؛])', r'\1', result)
    # Remove orphaned leading/trailing punctuation and stray digits at boundaries
    result = re.sub(r'^[\s.،,]+', '', result)
    result = re.sub(r'[\s.،,]+$', '', result)
    # Clean up double periods from stripping
    result = re.sub(r'\.\.+', '.', result)
    result = re.sub(r'\s{2,}', ' ', result)
    return result.strip()

def _has_too_much_english(text: str, threshold: int = 10) -> bool:
    """Return True if *text* contains more than *threshold* Latin characters."""
    return bool(text) and len(_EN_CHAR_RE.findall(text)) > threshold

def _postprocess_direct(text: str) -> str:
    """Final cleanup for direct answers: paragraph form, no technical artefacts."""
    if not text:
        return text
    text = _clean_text(text)
    # Remove stray bullet markers
    text = re.sub(r'(?:^|\n)\s*[-•]\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
