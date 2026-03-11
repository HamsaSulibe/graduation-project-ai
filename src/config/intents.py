"""
intents.py – Static intent registry (pure config, no logic).

Each intent defines:
    name       – internal code used by classify_intents()
    label_ar   – Arabic section header shown in formatted answers
    keywords   – trigger words matched against the normalised question
    fields     – {canonical_column: arabic_label} pairs
    direct     – True  → answer directly from Excel, no LLM needed
                 False → multi-field, LLM rewrite recommended
"""

from __future__ import annotations

INTENTS: list[dict] = [
    # ── 1. Watering ───────────────────────────────────────────────
    {
        "name": "watering",
        "label_ar": "الري والسقي",
        "keywords": [
            "اسقي", "سقي", "ري", "ماء", "ماي", "رطوبة",
            "عطش", "عطشان", "غرق", "افراط بالري", "نقص ري",
            "watering", "water", "overwater", "underwater",
        ],
        "fields": {
            "watering_need":               "حاجة الري",
            "watering_rule_text":          "قاعدة الري",
            "watering_interval_days_min":  "أقل فترة ري (أيام)",
            "watering_interval_days_max":  "أقصى فترة ري (أيام)",
            "fix_underwatering":           "علاج نقص الري",
            "fix_overwatering":            "علاج الإفراط بالري",
        },
        "direct": True,
    },
    # ── 2. Light ──────────────────────────────────────────────────
    {
        "name": "light",
        "label_ar": "الضوء والشمس",
        "keywords": [
            "ضوء", "شمس", "ظل", "اضاءة", "ساعات شمس",
            "نافذة", "اتجاه",
            "light", "sun", "shade", "window",
        ],
        "fields": {
            "light_level":              "مستوى الضوء",
            "min_sun_hours":            "أقل ساعات شمس",
            "indoor_window_direction":  "اتجاه النافذة المناسب",
        },
        "direct": True,
    },
    # ── 3. Temperature ────────────────────────────────────────────
    {
        "name": "temperature",
        "label_ar": "الحرارة والتحمل",
        "keywords": [
            "حرارة", "برد", "حر", "درجة", "تحمل",
            "صقيع", "تجمد",
            "temperature", "temp", "cold", "heat", "frost",
        ],
        "fields": {
            "temperature_optimal_min_c": "الحرارة المثالية الدنيا (°م)",
            "temperature_optimal_max_c": "الحرارة المثالية العليا (°م)",
            "heat_tolerance":            "تحمّل الحرارة",
            "frost_tolerance":           "تحمّل الصقيع",
        },
        "direct": True,
    },
    # ── 4. Soil / drainage ────────────────────────────────────────
    {
        "name": "soil",
        "label_ar": "التربة والتصريف",
        "keywords": [
            "تربة", "حموضة", "ph", "تصريف", "رملي", "طيني",
            "soil", "drainage", "ph",
        ],
        "fields": {
            "soil_texture_preference": "نوع التربة المفضل",
            "soil_ph_min":             "أقل حموضة (pH)",
            "soil_ph_max":             "أعلى حموضة (pH)",
            "drainage_need":           "حاجة التصريف",
            "soil_amendments":         "تحسينات التربة",
        },
        "direct": True,
    },
    # ── 5. Fertilizing ────────────────────────────────────────────
    {
        "name": "fertilizing",
        "label_ar": "التسميد",
        "keywords": [
            "سماد", "تسميد", "سمد", "كمبوست", "تحسين التربة",
            "fertiliz", "compost", "amendment",
        ],
        "fields": {
            "fertilizer_need":           "الحاجة للتسميد",
            "fertilizer_type":           "نوع السماد",
            "fertilizer_frequency_days": "تكرار التسميد (بالأيام)",
            "compost_recommended":       "هل يُنصح بالكمبوست",
        },
        "direct": True,
    },
    # ── 6. Season / Palestine timing ──────────────────────────────
    {
        "name": "season",
        "label_ar": "الموسم والتوقيت في فلسطين",
        "keywords": [
            "موسم", "وقت", "متى", "شهر", "فصل",
            "فلسطين", "منطقة", "حصاد",
            "season", "month", "when",
        ],
        "fields": {
            "planting_months_pal": "أشهر الزراعة في فلسطين",
            "harvest_months_pal":  "أشهر الحصاد في فلسطين",
            "season_notes_pal":    "ملاحظات موسمية",
            "pal_region":          "المنطقة في فلسطين",
        },
        "direct": True,
    },
    # ── 7. Harvest / storage ──────────────────────────────────────
    {
        "name": "harvest_storage",
        "label_ar": "الحصاد والتجفيف والتخزين",
        "keywords": [
            "حصاد", "احصد", "جمع", "قطف",
            "تجفيف", "جفف", "تخزين", "خزن", "احفظ",
            "harvest", "dry", "store", "storage",
        ],
        "fields": {
            "harvest_method":          "طريقة الحصاد",
            "harvest_after_days_min":  "أقل أيام حتى الحصاد",
            "harvest_after_days_max":  "أقصى أيام حتى الحصاد",
            "drying_method":           "طريقة التجفيف",
            "storage_method":          "طريقة التخزين",
            "storage_duration_months": "مدة التخزين (أشهر)",
        },
        "direct": True,
    },
    # ── 8. Pests & diseases ───────────────────────────────────────
    {
        "name": "pests_diseases",
        "label_ar": "الآفات والأمراض",
        "keywords": [
            "افات", "آفات", "حشرات", "امراض", "أمراض",
            "مرض", "افة", "عفن", "فطر",
            "pest", "disease", "insect", "fungus",
        ],
        "fields": {
            "common_pests":    "الآفات الشائعة",
            "common_diseases": "الأمراض الشائعة",
        },
        "direct": True,
    },
    # ── 9. Beginner / difficulty ──────────────────────────────────
    {
        "name": "beginner",
        "label_ar": "نصائح المبتدئين وسهولة العناية",
        "keywords": [
            "مبتدئ", "سهل", "صعب", "بسيط",
            "beginner", "easy", "difficult",
        ],
        "fields": {
            "difficulty_level": "مستوى الصعوبة",
            "time_commitment":  "الوقت المطلوب",
            "beginner_tips":    "نصائح للمبتدئين",
        },
        "direct": True,
    },
    # ── 10. Container / pot ───────────────────────────────────────
    {
        "name": "container",
        "label_ar": "الأصيص والزراعة في الحاويات",
        "keywords": [
            "اصيص", "أصيص", "وعاء", "حوض", "حاوية",
            "pot", "container",
        ],
        "fields": {
            "container_possible":   "هل تصلح للأصيص",
            "pot_diameter_cm_min":  "أقل قطر أصيص (سم)",
            "pot_depth_cm_min":     "أقل عمق أصيص (سم)",
            "drainage_need":        "حاجة التصريف",
        },
        "direct": True,
    },
    # ── 11. Planting steps ────────────────────────────────────────
    {
        "name": "planting",
        "label_ar": "الزراعة وخطواتها",
        "keywords": [
            "ازرع", "زراعة", "خطوات", "اكثار", "بذر",
            "نقل", "شتل", "تفريد", "انبات", "بذور",
            "plant", "propagat", "sow", "germinat", "transplant",
        ],
        "fields": {
            "propagation_method_primary": "طريقة الإكثار",
            "plant_spacing_cm":           "التباعد بين النباتات (سم)",
            "germination_days_min":       "أقل مدة إنبات (يوم)",
            "germination_days_max":       "أقصى مدة إنبات (يوم)",
            "transplanting_ok":           "إمكانية النقل / الشتل",
            "care_steps_json":            "خطوات العناية",
        },
        "direct": True,
    },
    # ── 12. General summary ───────────────────────────────────────
    {
        "name": "general_summary",
        "label_ar": "ملخص عام عن النبتة",
        "keywords": [
            "معلومات", "عرفني", "وصف", "تعريف",
            "ما هي", "ما هذه", "فوائد", "فايدة",
            "info", "about", "what is", "describe",
        ],
        "fields": {
            "short_summary":     "ملخص",
            "category":          "التصنيف",
            "growth_habit":      "نمط النمو",
            "life_cycle":        "دورة الحياة",
            "fragrance_level":   "مستوى العطر",
            "edible_parts":      "الأجزاء الصالحة للأكل",
        },
        "direct": True,
    },

    # ══════════════════════════════════════════════════════════════
    # MULTI-FIELD intents  (LLM rewrite recommended)
    # ══════════════════════════════════════════════════════════════
    {
        "name": "care_summary",
        "label_ar": "ملخص العناية الشاملة",
        "keywords": [
            "عناية", "اعتني", "رعاية", "كيف اعتني",
            "care", "maintain",
        ],
        "fields": {
            "short_summary":              "ملخص",
            "watering_rule_text":         "قاعدة الري",
            "light_level":               "مستوى الضوء",
            "temperature_optimal_min_c":  "الحرارة المثالية الدنيا",
            "temperature_optimal_max_c":  "الحرارة المثالية العليا",
            "soil_texture_preference":    "نوع التربة",
            "drainage_need":              "التصريف",
            "fertilizer_type":            "نوع السماد",
            "beginner_tips":              "نصائح للمبتدئين",
            "care_steps_json":            "خطوات العناية",
        },
        "direct": False,
    },
    {
        "name": "plant_overview",
        "label_ar": "نظرة عامة عن النبتة",
        "keywords": [
            "اعطني", "اخبرني", "كل شي", "لخص",
            "اهم معلومات", "كل المعلومات",
            "overview", "tell me", "everything",
        ],
        "fields": {
            "short_summary":              "ملخص",
            "category":                   "التصنيف",
            "difficulty_level":           "مستوى الصعوبة",
            "watering_rule_text":         "قاعدة الري",
            "light_level":               "مستوى الضوء",
            "temperature_optimal_min_c":  "الحرارة الدنيا",
            "temperature_optimal_max_c":  "الحرارة العليا",
            "planting_months_pal":        "أشهر الزراعة",
            "common_pests":               "الآفات الشائعة",
            "beginner_tips":              "نصائح للمبتدئين",
        },
        "direct": False,
    },
    {
        "name": "growing_guide",
        "label_ar": "دليل الزراعة الكامل",
        "keywords": [
            "كيف ازرع", "دليل زراعة", "ازرعها", "ازرعه",
            "guide", "grow",
        ],
        "fields": {
            "propagation_method_primary": "طريقة الإكثار",
            "care_steps_json":            "خطوات العناية",
            "soil_texture_preference":    "نوع التربة",
            "light_level":               "مستوى الضوء",
            "watering_rule_text":         "قاعدة الري",
            "planting_months_pal":        "أشهر الزراعة",
            "container_possible":         "مناسب للأصيص",
            "beginner_tips":              "نصائح للمبتدئين",
        },
        "direct": False,
    },
    {
        "name": "beginner_overview",
        "label_ar": "دليل المبتدئين الشامل",
        "keywords": [
            "مناسب للمبتدئين", "ينفع للمبتدئين",
            "احتياجات", "ماذا تحتاج", "ماذا يحتاج",
        ],
        "fields": {
            "difficulty_level":           "مستوى الصعوبة",
            "time_commitment":            "الوقت المطلوب",
            "beginner_tips":              "نصائح للمبتدئين",
            "watering_rule_text":         "قاعدة الري",
            "light_level":               "مستوى الضوء",
            "container_possible":         "مناسب للأصيص",
            "short_summary":              "ملخص",
        },
        "direct": False,
    },
]
