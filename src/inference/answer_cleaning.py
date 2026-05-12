"""
answer_cleaning.py – Arabic answer post-processing utilities for the
Garssa Smart Plant Assistant.

All functions are pure (no FastAPI / app dependencies).
"""
import re


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
