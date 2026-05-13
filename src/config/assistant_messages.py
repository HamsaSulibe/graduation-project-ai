"""
assistant_messages.py - Fixed Smart Assistant guardrail messages.

These messages are safe fallback/guardrail responses, not plant-care answers.
Plant answers must still come from the app plant data.
"""

ASSISTANT_MESSAGES: dict[str, dict[str, str]] = {
    "missing_data": {
        "ar": "هذه المعلومة غير متوفرة في بيانات التطبيق.",
        "en": "This information is not available in the app data.",
    },
    "plant_not_found": {
        "ar": "هذا النبات غير متوفر حاليًا في غرسة.",
        "en": "This plant is not currently available in Gharsa.",
    },
    "out_of_scope": {
        "ar": "أنا مساعد غرسة، بقدر أساعدك فقط في العناية بالنباتات الموجودة داخل التطبيق.",
        "en": "I’m Gharsa’s assistant. I can only help with plant-care questions for plants available in the app.",
    },
    "ask_for_plant": {
        "ar": "من فضلك اذكر اسم النبتة التي تريد معلومات عنها.",
        "en": "Please mention the plant you want information about.",
    },
    "greeting": {
        "ar": "أهلاً وسهلاً! أنا مساعد غرسة الذكي. كيف أقدر أساعدك بالعناية بنباتاتك؟",
        "en": "Hello! I’m Gharsa’s smart assistant. How can I help with your plant care today?",
    },
    "unsafe": {
        "ar": "لا أستطيع إضافة معلومات من خارج بيانات غرسة. اسألني عن معلومة موجودة في بيانات التطبيق فقط.",
        "en": "I can’t add information from outside Gharsa’s data. Please ask about information available in the app data only.",
    },
    "unclear_plant": {
        "ar": "مش قادر أحدد النبتة المقصودة من سؤالك. اكتب اسم النبتة الموجودة في تطبيق غرسة بشكل أوضح.",
        "en": "I can’t identify the plant from your question. Please write the name of a plant available in Gharsa more clearly.",
    },
    "insufficient_context": {
        "ar": "هذه المعلومة غير متوفرة في بيانات التطبيق.",
        "en": "This information is not available in the app data.",
    },
}
