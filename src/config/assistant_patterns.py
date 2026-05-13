"""
assistant_patterns.py - Text patterns used by the Smart Assistant guardrails.

These are not plant answers. They are routing/context rules for detecting
comparison requests and follow-up questions.
"""

EXPLICIT_COMPARISON_PATTERNS: tuple[str, ...] = (
    "قارن",
    "مقارنة",
    "الفرق بين",
    "مين أكثر",
    "مين اكثر",
    "مين أسهل",
    "مين اسهل",
    "مين أفضل",
    "مين افضل",
    "أيهما",
    "ايهما",
    "أيهم أفضل",
    "ايهم افضل",
    "أيهما أفضل",
    "ايهما افضل",
    "compare",
    "comparison",
    "difference between",
    "which is better",
    "which is easier",
    "which needs more",
    "better",
    "easier",
    "harder",
    "needs less",
    "needs more sun",
    "needs more water",
)

HISTORY_FOLLOWUP_PATTERNS: tuple[str, ...] = (
    "ماذا عن",
    "وماذا عن",
    "شو عن",
    "طيب و",
    "هل بده",
    "هل بدها",
    "هل يحتاج",
    "هل تحتاج",
    "كم مرة اسقيه",
    "كم مرة اسقيها",
    "كم مرة أسقيه",
    "كم مرة أسقيها",
    "متى ازرعه",
    "متى ازرعها",
    "متى أزرعه",
    "متى أزرعها",
    "متى احصده",
    "متى احصدها",
    "متى أحصده",
    "متى أحصدها",
    "شو استخداماته",
    "شو استخداماتها",
    "شو استخدامها",
    "شو تربته",
    "شو تربتها",
    "التربة المناسبة اله",
    "التربة المناسبة إله",
    "كيف اعتني فيه",
    "كيف اعتني فيها",
    "كيف أعتني فيه",
    "كيف أعتني فيها",
    "what about",
    "and what about",
    "what about watering",
    "does it need sun",
    "how often should i water it",
    "how often do i water it",
    "what soil does it need",
    "what are its uses",
    "how much light does it need",
    "when should i harvest it",
    "when can i harvest it",
)

HISTORY_PRONOUN_PATTERN: str = (
    r"\b(it|its|this plant|that plant)\b|"
    r"(اسقيه|اسقيها|أسقيه|أسقيها|ازرعه|ازرعها|أزرعه|أزرعها|"
    r"احصده|احصدها|أحصده|أحصدها|استخداماته|استخداماتها|استخدامها|"
    r"تربته|تربتها|فيه|فيها|له|لها|اله|إله|عنه|عنها|بده|بدها)"
)