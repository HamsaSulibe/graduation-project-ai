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


# ---------------------------------------------------------------------------
# Answer-building labels and comparison maps moved from context_utils.py
# ---------------------------------------------------------------------------

_EN_CANONICAL_OVERRIDES: dict[str, str] = {
    "arabic_name_primary": "english_name_primary",
    "short_summary": "short_summary_en",
    "watering_rule_text": "watering_info_en",
    "light_level": "light_info_en",
    "soil_texture_preference": "soil_info_en",
    "fertilizer_notes_ar": "fertilizer_notes_en",
    "harvest_method": "harvest_info_en",
    "care_steps_json": "planting_steps_en",
    "planting_steps_ar": "planting_steps_en",
    "seed_care_ar": "seed_care_en",
    "care_info_ar": "care_info_en",
    "beginner_tips": "care_info_en",
    "uses_info_ar": "uses_info_en",
    "humidity_notes_ar": "humidity_notes_en",
    "adjustment_tip_ar": "adjustment_tip_en",
    "planting_months_pal": "planting_note_en",
    "season_notes_pal": "planting_note_en",
}

_FIELD_LABELS_EN: dict[str, str] = {
    "arabic_name_primary": "Arabic name",
    "english_name_primary": "English name",
    "scientific_name": "Scientific name",
    "short_summary": "Description",
    "short_summary_en": "Description",
    "category": "Category",
    "difficulty_level": "Difficulty",
    "watering_need": "Watering need",
    "watering_rule_text": "Watering",
    "watering_interval_days_min": "Watering interval (days)",
    "light_level": "Light",
    "temperature_optimal_min_c": "Minimum temperature",
    "temperature_optimal_max_c": "Maximum temperature",
    "temperature_sensitivity": "Temperature sensitivity",
    "soil_texture_preference": "Soil",
    "soil_moisture_min": "Minimum soil moisture",
    "soil_moisture_max": "Maximum soil moisture",
    "fertilizer_type": "Fertilizer type",
    "fertilizer_stage": "Fertilizer stage",
    "fertilizer_frequency_days": "Fertilizing frequency (days)",
    "fertilizer_notes_ar": "Fertilizing notes",
    "harvest_method": "Harvest",
    "harvest_after_days_min": "Minimum days to harvest",
    "harvest_after_days_max": "Maximum days to harvest",
    "care_steps_json": "Planting steps",
    "propagation_method_primary": "Propagation method",
    "germination_days_min": "Germination days",
    "seed_to_seedling_days": "Seed to seedling days",
    "plant_spacing_cm_min": "Minimum spacing (cm)",
    "plant_spacing_cm_max": "Maximum spacing (cm)",
    "seed_care_ar": "Seed care",
    "establishment_days": "Establishment days",
    "care_info_ar": "Care information",
    "beginner_tips": "Care tips",
    "uses_info_ar": "Uses",
    "humidity_preference": "Humidity preference",
    "air_humidity_min": "Minimum air humidity (%)",
    "air_humidity_max": "Maximum air humidity (%)",
    "humidity_notes_ar": "Humidity notes",
    "adjustment_tip_ar": "Adjustment tip",
    "planting_months_pal": "Planting months and notes",
    "season_notes_pal": "Season",
    "task_type": "Task",
    "interval_days": "Every (days)",
}

_TASK_TYPE_EN: dict[str, str] = {
    "WATERING": "Watering",
    "FERTILIZING": "Fertilizing",
    "PRUNING": "Pruning",
    "HARVEST": "Harvest",
    "CARE": "Care",
    "watering": "Watering",
    "fertilizing": "Fertilizing",
    "pruning": "Pruning",
    "harvest": "Harvest",
    "care": "Care",
}

_TASK_TYPE_AR: dict[str, str] = {
    "WATERING":    "الري",
    "FERTILIZING": "التسميد",
    "PRUNING":     "التقليم",
    "HARVEST":     "الحصاد",
    "CARE":        "العناية",
    "watering":    "الري",
    "fertilizing": "التسميد",
    "pruning":     "التقليم",
    "harvest":     "الحصاد",
    "care":        "العناية",
}

_COMPARISON_CRITERIA_MAP: dict[str, tuple[str, str, bool, str]] = {
    "difficulty":  ("difficulty_level",           "مستوى الصعوبة",  False, "الأسهل للمبتدئين"),
    "watering":    ("watering_interval_days_min",  "فترة الري (أيام)", False, "يحتاج ري أكثر تكرارًا"),
    "heat":        ("temperature_optimal_max_c",   "أقصى حرارة (°C)", True,  "يتحمل حرارة أعلى"),
    "cold":        ("temperature_optimal_min_c",   "أدنى حرارة (°C)", False, "يتحمل برودة أشد"),
    "harvest":     ("harvest_after_days_min",       "أيام الحصاد",   False, "أسرع حصادًا"),
}

_COMPARISON_CRITERIA_MAP_EN: dict[str, tuple[str, str, bool, str]] = {
    "difficulty": ("difficulty_level", "difficulty", False, "is easier for beginners"),
    "watering": ("watering_interval_days_min", "watering interval", False, "needs more frequent watering"),
    "heat": ("temperature_optimal_max_c", "maximum temperature", True, "tolerates higher heat"),
    "cold": ("temperature_optimal_min_c", "minimum temperature", False, "tolerates colder conditions"),
    "harvest": ("harvest_after_days_min", "days to harvest", False, "is faster to harvest"),
}

_DIFFICULTY_RANK: dict[str, int] = {
    "EASY": 1, "MEDIUM": 2, "HARD": 3,
    "easy": 1, "medium": 2, "hard": 3,
    "سهل": 1,  "متوسط": 2,  "صعب": 3,
    "1": 1, "2": 2, "3": 3,
}

_DIFFICULTY_AR: dict[str, str] = {
    "EASY": "سهل", "MEDIUM": "متوسط", "HARD": "صعب",
    "easy": "سهل", "medium": "متوسط", "hard": "صعب",
}
