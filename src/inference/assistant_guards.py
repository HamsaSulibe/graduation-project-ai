"""
assistant_guards.py – Guard functions and their associated constants for
the Garssa Smart Plant Assistant (/assistant endpoint).

All functions are pure (no FastAPI / app dependencies).
"""
import re

from src.utils.arabic import normalize, normalize_query


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# ---------- Greeting detection (Rule 18) ----------
_GREETING_WORDS: frozenset[str] = frozenset({
    "مرحبا", "مرحباً", "هلا", "أهلا", "أهلاً", "سلام",
    "هاي", "يسلمو", "تسلم", "تسلمي", "يعطيك", "يعطيكي",
    "hello", "hi", "hey",
})
_GREETING_PHRASES: tuple[str, ...] = (
    "صباح الخير", "مساء الخير", "صباح النور", "مساء النور",
    "كيف حالك", "كيف الحال", "السلام عليكم", "وعليكم السلام",
    "أهلاً وسهلاً", "أهلا وسهلا", "مساء الورد", "صباح الورد",
    "good morning", "good evening", "how are you",
)
_GREETING_DOMAIN_WORDS: frozenset[str] = frozenset({
    "نبات", "نبتة", "زراعة", "ري", "سقي", "تربة",
    "سماد", "ضوء", "حرارة", "بذور", "حصاد",
    "plant", "plants", "watering", "water", "soil", "fertilizer",
    "light", "sun", "sunlight", "harvest", "herb", "herbs",
})

# ---------- Unclear-plant detection (Rule 7) ----------
_UNCLEAR_PLANT_EXPRESSIONS: tuple[str, ...] = (
    "اسمه غير واضح",
    "اسمها غير واضح",
    "مش عارف اسمه",
    "مش عارف اسمها",
    "مش عارف اسم",
    "لا أعرف اسمه",
    "لا أعرف اسمها",
    "ما أعرف اسمه",
    "ما أعرف اسمها",
    "اسمه مش واضح",
    "اسمها مش واضح",
    "نبتة مجهولة",
    "نبات مجهول",
    "مش متأكد اسمه",
    "مش متأكدة اسمه",
    "اسمها غير معروف",
    "اسمه غير معروف",
)

# ---------- Hallucination-request detection (GUARD 0.75) ----------
_HALLUCINATION_REQUEST_PHRASES: tuple[str, ...] = (
    "من عندك",
    "من معرفتك",
    "من برا الداتا",
    "خارج الداتا",
    "مش لازم من الداتا",
    "لازم تكون من الداتا",
    "حتى لو مش موجودة",
    "حتى لو مش عندك",
    "حتى لو ما عندك",
    "حتى لو مش في الداتا",
    "حتى لو مش في البيانات",
    "حتى لو مش موجود",
    "خمنلي",
    "خمن لي",
    "خمنها",
    "خمنه",
    "خمن",
    "أي معلومة عامة",
    "معلومة عامة",
    "معلومات عامة",
    "معلومات من عندك",
    "نصائح من عندك",
    "من الإنترنت",
    "من انترنت",
    "من الويب",
    "من معلوماتك",
    "بناءا على معرفتك",
    "بناء على معرفتك",
    "outside the data",
    "outside app data",
    "from your knowledge",
    "from the internet",
    "guess",
    "even if not available",
    "general knowledge",
)

# ---------- Out-of-scope topic detection ----------
_OUT_OF_SCOPE_TOPIC_KEYWORDS: tuple[str, ...] = (
    # Geography / general knowledge
    "عاصمة", "عواصم", "دولة", "قارة",
    # Politics / governance
    "سياسة", "سياسي", "سياسية", "انتخابات", "برلمان", "حكومة",
    # Programming / software
    "برمجة", "خوارزمية", "جافا", "بايثون", "سكريبت", "لغة برمجة",
    # Mathematics
    "رياضيات", "معادلة", "جبر", "إحصاء", "احصاء",
    # News / media
    "أخبار", "اخبار",
    # Cooking / food dishes (distinct from plant-use queries)
    "وصفة", "وصفات", "طبخة", "طبخ", "أكلة", "اكلة",
    "مقلوبة", "شوربة", "حلويات", "منسف", "كبسة", "مجدرة", "كنافة",
    # Human medicine (distinct from plant medicinal uses)
    "طبيب", "مستشفى", "علاج بشري", "دواء بشري",
    # Sports
    "كرة القدم", "كرة السلة", "ملعب",
    "capital", "country", "politics", "election", "programming",
    "python", "java", "math", "news", "recipe", "cooking",
    "football", "basketball", "medicine", "doctor",
)

# Lightweight domain keywords for fast pre-check (no DB lookup).
_DOMAIN_PART_KEYWORDS: tuple[str, ...] = (
    "نبات", "نبتة", "نباتات", "زراعة", "ازرع", "أزرع", "زرع",
    "ري", "اسقي", "سقي", "تربة", "سماد", "حصاد", "شتلة",
    "بذور", "أعشاب", "عشبة", "غرسة", "نخل", "شجرة",
    "plant", "plants", "herb", "herbs", "watering", "water",
    "soil", "fertilizer", "fertilizing", "light", "sun", "sunlight",
    "harvest", "planting", "grow", "growing", "gharsa",
)

# ---------- Hallucination post-check ----------
_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "عادةً",
    "عادة ",
    "غالبًا",
    "غالبا ",
    "بشكل عام",
    "حسب معرفتي",
    "من الإنترنت",
    "في العادة",
    "ينصح بـ",
    "ينصح ب",
    "من الأفضل",
    "حسب الخبراء",
    "علميًا معروف",
    "علميا معروف",
    "من المعروف أن",
    "من الطبيعي أن",
)

# ---------- Forbidden phrases in generated output (Rule 11) ----------
_FORBIDDEN_GENERATED_PHRASES: tuple[str, ...] = (
    "عادةً",
    "عادة ",
    "غالبًا",
    "غالبا ",
    "ينصح بـ",
    "ينصح ب",
    "من الأفضل",
    "بشكل عام",
    "يمكن استخدامه لعلاج",
    "يساعد على علاج",
    "حسب معرفتي",
    "في العادة",
    "النباتات المشابهة",
    "حسب الخبراء",
    "من الإنترنت",
    "علميًا معروف أن",
    "علميا معروف أن",
)

# ---------- English-presence detection ----------
_ENGLISH_WORD_RE = re.compile(r'[A-Za-z]{2,}')


# ──────────────────────────────────────────────────────────────────
# Guard functions
# ──────────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    """Return True when the text is predominantly Arabic (>= 30 % Arabic codepoints)."""
    if not text or not text.strip():
        return False
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return (arabic_chars / max(len(text.strip()), 1)) >= 0.30


def _contains_notable_english(text: str) -> bool:
    """
    True when the text contains 3+ English words (each ≥ 2 Latin chars).
    Used to catch mixed-language questions like 'How do I water [plant]?'
    that pass the 30 % Arabic threshold but are primarily in English.
    """
    if not text:
        return False
    english_words = _ENGLISH_WORD_RE.findall(text)
    return len(english_words) >= 3


def _is_greeting(text: str) -> bool:
    """True when the message is purely a greeting with no plant-domain content."""
    if not text or not text.strip():
        return False
    t = normalize_query(text).strip()
    if not t:
        return False
    # Reject if any domain word is present
    if any(w in t for w in _GREETING_DOMAIN_WORDS):
        return False
    # Check whole-phrase greetings
    for phrase in _GREETING_PHRASES:
        if normalize_query(phrase) in t:
            return True
    # Check first word against known greeting words (≤ 6 words total)
    words = t.split()
    if len(words) <= 6:
        first = words[0]
        if any(first == normalize_query(g) for g in _GREETING_WORDS):
            return True
    return False


def _is_unclear_plant_expression(question: str) -> bool:
    """True when the user explicitly states the plant name is unclear/unknown."""
    q_norm = normalize(question).lower()
    return any(normalize(expr).lower() in q_norm for expr in _UNCLEAR_PLANT_EXPRESSIONS)


def _is_hallucination_request(text: str) -> bool:
    """True when the user explicitly asks for data outside the plant database."""
    if not text:
        return False
    q_norm = normalize(text).lower()
    return any(normalize(phrase).lower() in q_norm for phrase in _HALLUCINATION_REQUEST_PHRASES)


def _contains_out_of_scope_part(text: str) -> bool:
    """True if the text contains any clear out-of-scope topic keyword."""
    if not text:
        return False
    q_norm = normalize(text).lower()
    return any(normalize(kw).lower() in q_norm for kw in _OUT_OF_SCOPE_TOPIC_KEYWORDS)


def _contains_domain_part_keywords(text: str) -> bool:
    """
    Fast domain-keyword check (no DB lookup).
    True if any plant-domain keyword appears in the text.
    """
    if not text:
        return False
    q_norm = normalize(text).lower()
    return any(normalize(kw).lower() in q_norm for kw in _DOMAIN_PART_KEYWORDS)


def _contains_hallucination(text: str) -> bool:
    """Return True if the answer contains any forbidden generalist phrase."""
    if not text:
        return False
    return any(phrase in text for phrase in _HALLUCINATION_PHRASES)


def _contains_forbidden_phrase(text: str) -> bool:
    """True if LLM-generated text contains a hallucination-risk phrase (Rule 11)."""
    if not text:
        return False
    return any(phrase in text for phrase in _FORBIDDEN_GENERATED_PHRASES)
