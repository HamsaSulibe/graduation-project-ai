"""
intents.py â€“ Static intent registry (pure config, no logic).

Each intent defines:
    name       â€“ internal code used by classify_intents()
    label_ar   â€“ Arabic section header shown in formatted answers
    keywords   â€“ trigger words matched against the normalised question
    fields     â€“ {canonical_column: arabic_label} pairs
    direct     â€“ True  â†’ answer directly from Excel, no LLM needed
                 False â†’ multi-field, LLM rewrite recommended

Updated to use plant_data_final_v4.xlsx columns only.
Canonical field names match columns.py COLUMN_ALIASES so the alias
resolution layer bridges them to the actual Excel column names.
"""

from __future__ import annotations

INTENTS: list[dict] = [
    # â”€â”€ 1. Watering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # wateringNeed + wateringIntervalDays (Plants)
    # wateringInfoAr (Care_Details) via alias watering_rule_text
    {
        "name": "watering",
        "label_ar": "Ø§Ù„Ø±ÙŠ ÙˆØ§Ù„Ø³Ù‚ÙŠ",
        "keywords": [
            "Ø§Ø³Ù‚ÙŠ", "Ø³Ù‚ÙŠ", "Ø±ÙŠ", "Ù…Ø§Ø¡", "Ù…Ø§ÙŠ", "Ø±Ø·ÙˆØ¨Ø©",
            "Ø¹Ø·Ø´", "Ø¹Ø·Ø´Ø§Ù†", "ØºØ±Ù‚", "Ø§ÙØ±Ø§Ø· Ø¨Ø§Ù„Ø±ÙŠ", "Ù†Ù‚Øµ Ø±ÙŠ",
            "watering", "water", "overwater", "underwater",
            "اروي",
            "أروي",
            "روي",
            "ارو",
            "ارويه",
            "ارويها",
            "اسقيه",
            "اسقيها",
            "كم مرة اسقي",
            "كم مره اسقي",
            "سقي نبتة",
            "سقي نبات",
            "اسقي",
            "سقي",
            "ري",
            "ماء",
            "عطش",
            "افراط بالري",
            "نقص ري",
        ],
        "fields": {
            "watering_need":              "Ø­Ø§Ø¬Ø© Ø§Ù„Ø±ÙŠ",
            "watering_rule_text":         "Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„Ø±ÙŠ",        # â†’ wateringInfoAr
            "watering_interval_days_min": "ÙØªØ±Ø© Ø§Ù„Ø±ÙŠ (Ø£ÙŠØ§Ù…)",    # â†’ wateringIntervalDays
        },
        "direct": True,
    },
    # â”€â”€ 2. Light â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # lightInfoAr (Care_Details) via alias light_level
    {
        "name": "light",
        "label_ar": "Ø§Ù„Ø¶ÙˆØ¡ ÙˆØ§Ù„Ø´Ù…Ø³",
        "keywords": [
            "Ø¶ÙˆØ¡", "Ø´Ù…Ø³", "Ø¸Ù„", "Ø§Ø¶Ø§Ø¡Ø©", "Ø³Ø§Ø¹Ø§Øª Ø´Ù…Ø³",
            "Ù†Ø§ÙØ°Ø©", "Ø§ØªØ¬Ø§Ù‡",
            "light", "sun", "sunlight", "shade", "window",
            "إضاءة",
            "اضاءة",
            "الضوء المناسب",
            "ضوء مناسب",
            "شمس مباشرة",
            "شمس غير مباشرة",
            "ضوء",
            "شمس",
            "ظل",
            "ساعات شمس",
            "نافذة",
            "اتجاه",
        ],
        "fields": {
            "light_level": "Ù…ØªØ·Ù„Ø¨Ø§Øª Ø§Ù„Ø¶ÙˆØ¡",   # â†’ lightInfoAr
        },
        "direct": True,
    },
    # â”€â”€ 3. Temperature â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # minTemp / maxTemp / temperatureSensitivity (Plants)
    {
        "name": "temperature",
        "label_ar": "Ø§Ù„Ø­Ø±Ø§Ø±Ø© ÙˆØ§Ù„ØªØ­Ù…Ù„",
        "keywords": [
            "Ø­Ø±Ø§Ø±Ø©", "Ø¨Ø±Ø¯", "Ø­Ø±", "Ø¯Ø±Ø¬Ø©", "ØªØ­Ù…Ù„",
            "ØµÙ‚ÙŠØ¹", "ØªØ¬Ù…Ø¯",
            "temperature", "temp", "cold", "heat", "frost",
            "درجة الحرارة",
            "درجات الحرارة",
            "دفي",
            "دافئ",
            "حرارة",
            "برد",
            "حر",
            "درجة",
            "تحمل",
            "صقيع",
            "تجمد",
        ],
        "fields": {
            "temperature_optimal_min_c": "Ø§Ù„Ø­Ø±Ø§Ø±Ø© Ø§Ù„Ù…Ø«Ø§Ù„ÙŠØ© Ø§Ù„Ø¯Ù†ÙŠØ§ (Â°Ù…)",  # â†’ minTemp
            "temperature_optimal_max_c": "Ø§Ù„Ø­Ø±Ø§Ø±Ø© Ø§Ù„Ù…Ø«Ø§Ù„ÙŠØ© Ø§Ù„Ø¹Ù„ÙŠØ§ (Â°Ù…)",  # â†’ maxTemp
            "temperature_sensitivity":   "Ø­Ø³Ø§Ø³ÙŠØ© Ø¯Ø±Ø¬Ø© Ø§Ù„Ø­Ø±Ø§Ø±Ø©",           # â†’ temperatureSensitivity
        },
        "direct": True,
    },
    # â”€â”€ 4. Soil â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # soilInfoAr (Care_Details) via alias soil_texture_preference
    # soilMoistureMin/Max (Plants)
    {
        "name": "soil",
        "label_ar": "Ø§Ù„ØªØ±Ø¨Ø© ÙˆØ§Ù„Ø±Ø·ÙˆØ¨Ø©",
        "keywords": [
            "ØªØ±Ø¨Ø©", "Ø­Ù…ÙˆØ¶Ø©", "ph", "ØªØµØ±ÙŠÙ", "Ø±Ù…Ù„ÙŠ", "Ø·ÙŠÙ†ÙŠ",
            "Ø±Ø·ÙˆØ¨Ø© Ø§Ù„ØªØ±Ø¨Ø©",
            "soil", "drainage", "ph",
            "نوع التربة",
            "التربة المناسبة",
            "تربة مناسبة",
            "تربة",
            "حموضة",
            "تصريف",
            "رملي",
            "طيني",
        ],
        "fields": {
            "soil_texture_preference": "Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„ØªØ±Ø¨Ø©",      # â†’ soilInfoAr
            "soil_moisture_min":       "Ø±Ø·ÙˆØ¨Ø© Ø§Ù„ØªØ±Ø¨Ø© Ø§Ù„Ø¯Ù†ÙŠØ§", # â†’ soilMoistureMin
            "soil_moisture_max":       "Ø±Ø·ÙˆØ¨Ø© Ø§Ù„ØªØ±Ø¨Ø© Ø§Ù„Ø¹Ù„ÙŠØ§", # â†’ soilMoistureMax
        },
        "direct": True,
    },
    # â”€â”€ 5. Fertilizing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # fertilizerType / fertilizerStage / fertilizerFrequencyDays / fertilizerNotesAr (Plants)
    {
        "name": "fertilizing",
        "label_ar": "Ø§Ù„ØªØ³Ù…ÙŠØ¯",
        "keywords": [
            "Ø³Ù…Ø§Ø¯", "ØªØ³Ù…ÙŠØ¯", "Ø³Ù…Ø¯", "ÙƒÙ…Ø¨ÙˆØ³Øª", "ØªØ­Ø³ÙŠÙ† Ø§Ù„ØªØ±Ø¨Ø©",
            "fertiliz", "fertilizer", "fertilizing", "compost", "amendment",
            "اسمد",
            "أسمد",
            "سمدت",
            "سماد نبتة",
            "سماد",
            "تسميد",
            "سمد",
            "كمبوست",
        ],
        "fields": {
            "fertilizer_type":           "Ù†ÙˆØ¹ Ø§Ù„Ø³Ù…Ø§Ø¯",             # â†’ fertilizerType
            "fertilizer_stage":          "Ù…Ø±Ø­Ù„Ø© Ø§Ù„ØªØ³Ù…ÙŠØ¯",          # â†’ fertilizerStage
            "fertilizer_frequency_days": "ØªÙƒØ±Ø§Ø± Ø§Ù„ØªØ³Ù…ÙŠØ¯ (Ø£ÙŠØ§Ù…)",   # â†’ fertilizerFrequencyDays
            "fertilizer_notes_ar":       "Ù…Ù„Ø§Ø­Ø¸Ø§Øª Ø§Ù„ØªØ³Ù…ÙŠØ¯",        # â†’ fertilizerNotesAr
        },
        "direct": True,
    },
    # â”€â”€ 6. Season / planting calendar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Month_Plants sheet: plantingNoteAr / monthName / season
    # via alias planting_months_pal â†’ plantingNoteAr
    {
        "name": "season",
        "label_ar": "Ø§Ù„Ù…ÙˆØ³Ù… ÙˆØ£Ø´Ù‡Ø± Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",
        "keywords": [
            "Ù…ÙˆØ³Ù…", "ÙˆÙ‚Øª", "Ù…ØªÙ‰", "Ø´Ù‡Ø±", "ÙØµÙ„",
            "ÙÙ„Ø³Ø·ÙŠÙ†", "Ù…Ù†Ø·Ù‚Ø©", "Ø­ØµØ§Ø¯",
            "season", "month", "when",
            "موسم",
            "شهر",
            "فصل",
            "فلسطين",
        ],
        "fields": {
            "planting_months_pal": "Ø£Ø´Ù‡Ø± ÙˆÙ…Ù„Ø§Ø­Ø¸Ø§Øª Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",  # â†’ plantingNoteAr (Month_Plants)
            "season_notes_pal":    "Ø§Ù„ÙØµÙ„ Ø§Ù„Ù…Ù†Ø§Ø³Ø¨",           # â†’ plantingNoteAr (approximate)
        },
        "direct": True,
    },
    # â”€â”€ 7. Harvest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # daysToHarvestMin/Max (Plants) + harvestInfoAr (Care_Details)
    # NOTE: drying/storage fields no longer exist in the new Excel.
    {
        "name": "harvest_storage",
        "label_ar": "Ø§Ù„Ø­ØµØ§Ø¯",
        "keywords": [
            "Ø­ØµØ§Ø¯", "Ø§Ø­ØµØ¯", "Ø¬Ù…Ø¹", "Ù‚Ø·Ù",
            "ØªØ¬ÙÙŠÙ", "Ø¬ÙÙ", "ØªØ®Ø²ÙŠÙ†", "Ø®Ø²Ù†", "Ø§Ø­ÙØ¸",
            "harvest", "harvesting", "dry", "store", "storage",
            "متى احصد",
            "متى أحصد",
            "وقت الحصاد",
            "حصاد",
            "احصد",
            "جمع",
            "قطف",
            "تجفيف",
            "تخزين",
            "خزن",
        ],
        "fields": {
            "harvest_method":         "Ø·Ø±ÙŠÙ‚Ø© Ø§Ù„Ø­ØµØ§Ø¯",          # â†’ harvestInfoAr
            "harvest_after_days_min": "Ø£Ù‚Ù„ Ø£ÙŠØ§Ù… Ø­ØªÙ‰ Ø§Ù„Ø­ØµØ§Ø¯",  # â†’ daysToHarvestMin
            "harvest_after_days_max": "Ø£Ù‚ØµÙ‰ Ø£ÙŠØ§Ù… Ø­ØªÙ‰ Ø§Ù„Ø­ØµØ§Ø¯", # â†’ daysToHarvestMax
        },
        "direct": True,
    },
    # â”€â”€ 8. Pests & diseases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # NO EQUIVALENT â€” plants_pests_diseases sheet removed entirely.
    # Intent is kept so routing does not break; will return empty fields
    # until data is re-added. careInfoAr is used as a soft fallback.
    {
        "name": "pests_diseases",
        "label_ar": "Ø§Ù„Ø¢ÙØ§Øª ÙˆØ§Ù„Ø£Ù…Ø±Ø§Ø¶",
        "keywords": [
            "Ø§ÙØ§Øª", "Ø¢ÙØ§Øª", "Ø­Ø´Ø±Ø§Øª", "Ø§Ù…Ø±Ø§Ø¶", "Ø£Ù…Ø±Ø§Ø¶",
            "Ù…Ø±Ø¶", "Ø§ÙØ©", "Ø¹ÙÙ†", "ÙØ·Ø±",
            "pest", "disease", "insect", "fungus",
            "مشاكل",
            "مشكلة",
            "يصفر",
            "يجف",
            "يموت",
            "تموت",
            "اوراق صفراء",
            "أوراق صفراء",
            "تساقط الأوراق",
            "افات",
            "آفات",
            "حشرات",
            "امراض",
            "أمراض",
            "مرض",
            "افة",
            "عفن",
            "فطر",
        ],
        "fields": {
            "care_info_ar": "Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„Ø¹Ù†Ø§ÙŠØ© (Ù‚Ø¯ ØªØªØ¶Ù…Ù† ØªÙ†Ø¨ÙŠÙ‡Ø§Øª)",  # â†’ careInfoAr (fallback)
        },
        "direct": True,
    },
    # â”€â”€ 9. Beginner / difficulty â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # difficultyLevel (Plants) + careInfoAr as beginner_tips proxy
    {
        "name": "beginner",
        "label_ar": "Ù†ØµØ§Ø¦Ø­ Ø§Ù„Ù…Ø¨ØªØ¯Ø¦ÙŠÙ† ÙˆØ³Ù‡ÙˆÙ„Ø© Ø§Ù„Ø¹Ù†Ø§ÙŠØ©",
        "keywords": [
            "Ù…Ø¨ØªØ¯Ø¦", "Ø³Ù‡Ù„", "ØµØ¹Ø¨", "Ø¨Ø³ÙŠØ·",
            "beginner", "easy", "difficult",
            "مبتدئ",
            "سهل",
            "صعب",
            "بسيط",
        ],
        "fields": {
            "difficulty_level": "Ù…Ø³ØªÙˆÙ‰ Ø§Ù„ØµØ¹ÙˆØ¨Ø©",    # â†’ difficultyLevel
            "beginner_tips":    "Ù†ØµØ§Ø¦Ø­ Ø§Ù„Ø¹Ù†Ø§ÙŠØ©",     # â†’ careInfoAr (proxy)
        },
        "direct": True,
    },
    # â”€â”€ 10. Container / pot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # NO EQUIVALENT â€” container_possible / pot dimensions removed.
    # Kept for routing compatibility; soilInfoAr used as soft fallback.
    {
        "name": "container",
        "label_ar": "Ø§Ù„Ø£ØµÙŠØµ ÙˆØ§Ù„Ø²Ø±Ø§Ø¹Ø© ÙÙŠ Ø§Ù„Ø­Ø§ÙˆÙŠØ§Øª",
        "keywords": [
            "Ø§ØµÙŠØµ", "Ø£ØµÙŠØµ", "ÙˆØ¹Ø§Ø¡", "Ø­ÙˆØ¶", "Ø­Ø§ÙˆÙŠØ©",
            "pot", "container",
            "اصيص",
            "وعاء",
            "حوض",
            "حاوية",
        ],
        "fields": {
            "planting_steps_ar": "Ø®Ø·ÙˆØ§Øª Ø§Ù„Ø²Ø±Ø§Ø¹Ø© (ØªØ´Ù…Ù„ Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„ØªØ¨Ø§Ø¹Ø¯)",  # â†’ plantingStepsAr
            "soil_texture_preference": "Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø§Ù„ØªØ±Ø¨Ø©",                    # â†’ soilInfoAr
        },
        "direct": True,
    },
    # â”€â”€ 11. Planting steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # plantingStepsAr (Care_Details) via alias care_steps_json
    # + plantingMethod / germinationDays / spacing (Plants)
    {
        "name": "planting",
        "label_ar": "Ø§Ù„Ø²Ø±Ø§Ø¹Ø© ÙˆØ®Ø·ÙˆØ§ØªÙ‡Ø§",
        "keywords": [
            "Ø§Ø²Ø±Ø¹", "Ø²Ø±Ø§Ø¹Ø©", "Ø®Ø·ÙˆØ§Øª", "Ø§ÙƒØ«Ø§Ø±", "Ø¨Ø°Ø±",
            "Ù†Ù‚Ù„", "Ø´ØªÙ„", "ØªÙØ±ÙŠØ¯", "Ø§Ù†Ø¨Ø§Øª", "Ø¨Ø°ÙˆØ±",
            "plant", "planting", "propagat", "sow", "germinat", "transplant",
            "كيف ازرع",
            "كيف أزرع",
            "طريقة الزراعة",
            "خطوات الزراعة",
            "ازرع",
            "زراعة",
            "خطوات",
            "اكثار",
            "بذر",
            "شتل",
            "بذور",
        ],
        "fields": {
            "care_steps_json":            "Ø®Ø·ÙˆØ§Øª Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",             # â†’ plantingStepsAr
            "propagation_method_primary": "Ø·Ø±ÙŠÙ‚Ø© Ø§Ù„Ø¥ÙƒØ«Ø§Ø±",             # â†’ plantingMethod
            "germination_days_min":       "Ù…Ø¯Ø© Ø§Ù„Ø¥Ù†Ø¨Ø§Øª (Ø£ÙŠØ§Ù…)",        # â†’ germinationDays
            "seed_to_seedling_days":      "الأيام من البذرة للشتلة",   # → seedToSeedlingDays
            "plant_spacing_cm_min":       "Ø§Ù„ØªØ¨Ø§Ø¹Ø¯ Ø§Ù„Ø£Ø¯Ù†Ù‰ (Ø³Ù…)",       # â†’ plantSpacingCmMin
            "plant_spacing_cm_max":       "Ø§Ù„ØªØ¨Ø§Ø¹Ø¯ Ø§Ù„Ø£Ù‚ØµÙ‰ (Ø³Ù…)",       # â†’ plantSpacingCmMax
            "seed_care_ar":               "ØªØ¹Ù„ÙŠÙ…Ø§Øª Ø§Ù„Ø¹Ù†Ø§ÙŠØ© Ø¨Ø§Ù„Ø¨Ø°ÙˆØ±",   # â†’ seedCareInstructionsAr
            "establishment_days":         "Ø£ÙŠØ§Ù… Ø§Ù„ØªØ£Ø³ÙŠØ³",              # â†’ establishmentDays
        },
        "direct": True,
    },
    # â”€â”€ 12. General summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # shortDescriptionAr (Plants) + usesInfoAr (Care_Details)
    {
        "name": "uses",
        "label_ar": "الاستخدامات",
        "keywords": [
            "استخدامات", "استخدام", "فوائد", "فوايد", "فائدة", "فايدة",
            "uses", "use", "what is it used for", "what are its uses",
        ],
        "fields": {
            "uses_info_ar": "الاستخدامات",
        },
        "direct": True,
    },
    {
        "name": "general_summary",
        "label_ar": "Ù…Ù„Ø®Øµ Ø¹Ø§Ù… Ø¹Ù† Ø§Ù„Ù†Ø¨ØªØ©",
        "keywords": [
            "Ù…Ø¹Ù„ÙˆÙ…Ø§Øª", "Ø¹Ø±ÙÙ†ÙŠ", "ÙˆØµÙ", "ØªØ¹Ø±ÙŠÙ",
            "Ù…Ø§ Ù‡ÙŠ", "Ù…Ø§ Ù‡Ø°Ù‡", "ÙÙˆØ§Ø¦Ø¯", "ÙØ§ÙŠØ¯Ø©",
            "info", "about", "what is", "describe", "uses", "use",
            "احكيلي",
            "حكيلي",
            "قولي",
            "اخبرني",
            "خبرني",
            "اعطيني معلومات",
            "شو هو",
            "شو هي",
            "ايش هو",
            "ايش هي",
            "بدي اعرف",
            "بدي أعرف",
            "معلومات",
            "عرفني",
            "وصف",
            "تعريف",
            "فوائد",
            "فائدة",
            "فوايد",
            "استخدامات",
            "استخدام",
        ],
        "fields": {
            "short_summary":  "ÙˆØµÙ Ù…Ø®ØªØµØ±",    # â†’ shortDescriptionAr
            "category":       "Ø§Ù„ØªØµÙ†ÙŠÙ",       # â†’ category
            "uses_info_ar":   "Ø§Ù„Ø§Ø³ØªØ®Ø¯Ø§Ù…Ø§Øª",   # â†’ usesInfoAr
        },
        "direct": True,
    },

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # MULTI-FIELD intents  (LLM rewrite recommended)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    {
        "name": "care_summary",
        "label_ar": "Ù…Ù„Ø®Øµ Ø§Ù„Ø¹Ù†Ø§ÙŠØ© Ø§Ù„Ø´Ø§Ù…Ù„Ø©",
        "keywords": [
            "Ø¹Ù†Ø§ÙŠØ©", "Ø§Ø¹ØªÙ†ÙŠ", "Ø±Ø¹Ø§ÙŠØ©", "ÙƒÙŠÙ Ø§Ø¹ØªÙ†ÙŠ",
            "care", "maintain",
            "كيف اعتني",
            "كيف أعتني",
            "كيف يزرع",
            "روتين",
            "طريقة العناية",
            "كيفية العناية",
            "عناية",
            "اعتني",
            "رعاية",
        ],
        "fields": {
            "short_summary":             "وصف مختصر",          # → shortDescriptionAr
            "difficulty_level":          "مستوى الصعوبة",      # → difficultyLevel
            "care_info_ar":              "معلومات العناية",    # → careInfoAr (Care_Details)
            "watering_rule_text":        "الري",               # → wateringInfoAr
            "light_level":               "الضوء",             # → lightInfoAr
            "temperature_optimal_min_c": "الحرارة الدنيا",    # → minTemp
            "temperature_optimal_max_c": "الحرارة العليا",    # → maxTemp
            "soil_texture_preference":   "التربة",            # → soilInfoAr
            "fertilizer_type":           "نوع السماد",        # → fertilizerType
            "fertilizer_notes_ar":       "ملاحظات التسميد",   # → fertilizerNotesAr
            "humidity_preference":       "تفضيل الرطوبة",     # → humidityPreference
            "humidity_notes_ar":         "ملاحظات الرطوبة",   # → humidityNotesAr
            "beginner_tips":             "نصائح العناية",     # → careInfoAr (proxy)
            "care_steps_json":           "خطوات الزراعة",     # → plantingStepsAr
        },
        "direct": True ,
    },
    {
        "name": "plant_overview",
        "label_ar": "Ù†Ø¸Ø±Ø© Ø¹Ø§Ù…Ø© Ø¹Ù† Ø§Ù„Ù†Ø¨ØªØ©",
        "keywords": [
            "Ø§Ø¹Ø·Ù†ÙŠ", "Ø§Ø®Ø¨Ø±Ù†ÙŠ", "ÙƒÙ„ Ø´ÙŠ", "Ù„Ø®Øµ",
            "Ø§Ù‡Ù… Ù…Ø¹Ù„ÙˆÙ…Ø§Øª", "ÙƒÙ„ Ø§Ù„Ù…Ø¹Ù„ÙˆÙ…Ø§Øª",
            "overview", "tell me", "everything",
            "احكيلي عن",
            "حكيلي عن",
            "قولي عن",
            "اخبرني عن",
            "معلومات كاملة عن",
            "كل شي عن",
            "كل شيء عن",
            "اعطني",
            "اخبرني",
            "كل شي",
            "لخص",
            "اهم معلومات",
        ],
        "fields": {
            "short_summary":              "ÙˆØµÙ Ù…Ø®ØªØµØ±",          # â†’ shortDescriptionAr
            "category":                   "Ø§Ù„ØªØµÙ†ÙŠÙ",
            "difficulty_level":           "Ù…Ø³ØªÙˆÙ‰ Ø§Ù„ØµØ¹ÙˆØ¨Ø©",     # â†’ difficultyLevel
            "watering_rule_text":         "Ø§Ù„Ø±ÙŠ",               # â†’ wateringInfoAr
            "light_level":                "Ø§Ù„Ø¶ÙˆØ¡",              # â†’ lightInfoAr
            "temperature_optimal_min_c":  "Ø§Ù„Ø­Ø±Ø§Ø±Ø© Ø§Ù„Ø¯Ù†ÙŠØ§",    # â†’ minTemp
            "temperature_optimal_max_c":  "Ø§Ù„Ø­Ø±Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§",    # â†’ maxTemp
            "planting_months_pal":        "Ø£Ø´Ù‡Ø± Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",      # â†’ plantingNoteAr
            "humidity_preference": "تفضيل الرطوبة",   # → humidityPreference
            "uses_info_ar":               "Ø§Ù„Ø§Ø³ØªØ®Ø¯Ø§Ù…Ø§Øª",        # â†’ usesInfoAr
        },
        "direct": False,
    },
    {
        "name": "growing_guide",
        "label_ar": "Ø¯Ù„ÙŠÙ„ Ø§Ù„Ø²Ø±Ø§Ø¹Ø© Ø§Ù„ÙƒØ§Ù…Ù„",
        "keywords": [
            "ÙƒÙŠÙ Ø§Ø²Ø±Ø¹", "Ø¯Ù„ÙŠÙ„ Ø²Ø±Ø§Ø¹Ø©", "Ø§Ø²Ø±Ø¹Ù‡Ø§", "Ø§Ø²Ø±Ø¹Ù‡",
            "guide", "grow", "growing",
            "كيف ازرع",
            "دليل زراعة",
            "ازرعها",
            "ازرعه",
        ],
        "fields": {
            "propagation_method_primary": "Ø·Ø±ÙŠÙ‚Ø© Ø§Ù„Ø¥ÙƒØ«Ø§Ø±",      # â†’ plantingMethod
            "care_steps_json":            "Ø®Ø·ÙˆØ§Øª Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",      # â†’ plantingStepsAr
            "soil_texture_preference":    "Ø§Ù„ØªØ±Ø¨Ø©",              # â†’ soilInfoAr
            "light_level":                "Ø§Ù„Ø¶ÙˆØ¡",               # â†’ lightInfoAr
            "watering_rule_text":         "Ø§Ù„Ø±ÙŠ",                # â†’ wateringInfoAr
            "planting_months_pal":        "Ø£Ø´Ù‡Ø± Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",       # â†’ plantingNoteAr
            "seed_care_ar":               "ØªØ¹Ù„ÙŠÙ…Ø§Øª Ø§Ù„Ø¨Ø°ÙˆØ±",      # â†’ seedCareInstructionsAr
            "beginner_tips":              "Ù†ØµØ§Ø¦Ø­ Ø§Ù„Ø¹Ù†Ø§ÙŠØ©",       # â†’ careInfoAr
        },
        "direct": False,
    },
    {
        "name": "beginner_overview",
        "label_ar": "Ø¯Ù„ÙŠÙ„ Ø§Ù„Ù…Ø¨ØªØ¯Ø¦ÙŠÙ† Ø§Ù„Ø´Ø§Ù…Ù„",
        "keywords": [
            "Ù…Ù†Ø§Ø³Ø¨ Ù„Ù„Ù…Ø¨ØªØ¯Ø¦ÙŠÙ†", "ÙŠÙ†ÙØ¹ Ù„Ù„Ù…Ø¨ØªØ¯Ø¦ÙŠÙ†",
            "Ø§Ø­ØªÙŠØ§Ø¬Ø§Øª", "Ù…Ø§Ø°Ø§ ØªØ­ØªØ§Ø¬", "Ù…Ø§Ø°Ø§ ÙŠØ­ØªØ§Ø¬",
        ],
        "fields": {
            "difficulty_level":   "Ù…Ø³ØªÙˆÙ‰ Ø§Ù„ØµØ¹ÙˆØ¨Ø©",   # â†’ difficultyLevel
            "beginner_tips":      "Ù†ØµØ§Ø¦Ø­ Ø§Ù„Ø¹Ù†Ø§ÙŠØ©",    # â†’ careInfoAr
            "watering_rule_text": "Ø§Ù„Ø±ÙŠ",              # â†’ wateringInfoAr
            "light_level":        "Ø§Ù„Ø¶ÙˆØ¡",             # â†’ lightInfoAr
            "short_summary":      "ÙˆØµÙ Ù…Ø®ØªØµØ±",        # â†’ shortDescriptionAr
            "care_steps_json":    "Ø®Ø·ÙˆØ§Øª Ø§Ù„Ø²Ø±Ø§Ø¹Ø©",    # â†’ plantingStepsAr
        },
        "direct": False,
    },
    # ── 13. Humidity ──────────────────────────────────────────────────────
    # humidityPreference / airHumidityMin / airHumidityMax / humidityNotesAr (Plants)
    {
        "name": "humidity",
        "label_ar": "الرطوبة",
        "keywords": [
            "رطوبة", "رطب", "جاف", "جفاف",
            "رطوبة الهواء", "رطوبة التربة",
            "humidity", "moist", "dry",
            "رطوبة محيطية",
            "جاف جدا",
        ],
        "fields": {
            "humidity_preference": "تفضيل الرطوبة",           # → humidityPreference
            "air_humidity_min":    "رطوبة الهواء الدنيا (%)",  # → airHumidityMin
            "air_humidity_max":    "رطوبة الهواء القصوى (%)",  # → airHumidityMax
            "humidity_notes_ar":   "ملاحظات الرطوبة",          # → humidityNotesAr
        },
        "direct": True,
    },
    # ── 14. Suitability ─────────────────────────────────────────────────────
    # adjustmentTipAr (Suitability sheet)
    {
        "name": "suitability",
        "label_ar": "الملاءمة والتعديلات",
        "keywords": [
            "يناسب", "مناسب", "ملاءمة", "تعديل", "تعديلات",
            "يصلح", "مناسب لي",
            "suitab", "adjust",
            "لبيتي",
            "لمنزلي",
            "لبيتنا",
            "ملائم",
            "يلائم",
            "هل يناسب",
            "هل تناسب",
            "مناسب لبيتي",
            "يصلح لبيتي",
        ],
        "fields": {
            "adjustment_tip_ar": "نصيحة التعديل",   # → adjustmentTipAr (Suitability)
        },
        "direct": True,
    },
    # ── 15. Tasks / care schedule ────────────────────────────────────────────
    # taskType / intervalDays (Tasks sheet)
    {
        "name": "tasks",
        "label_ar": "مهام العناية",
        "keywords": [
            # Single-word high-signal keywords
            "مهمة", "مهام", "جدول", "روتين", "تذكير", "تذكيرات",
            "مواعيد",
            # Multi-word high-signal patterns (used for preemption in classify_intents)
            "جدول العناية", "جدول المهام", "جدول رعاية", "مهام العناية",
            "كل كم يوم", "كم يوم مهمة", "مواعيد العناية",
            "كل كم", "عندي مهمة",
            # New: schedule/routine phrasing variants
            "روتين المهام", "روتين العناية",
            "المطلوب أعمل", "شو المطلوب", "ايش المطلوب",
            "ايش لازم أعمل", "ايش لازم اعمل",
            "شو لازم أعمل", "شو لازم اعمل",
            "مهام الرعاية",
            # Reminder/schedule phrasing
            "متى أسمد", "متى اسمد", "متى أروي", "متى اروي",
            "task", "tasks", "schedule", "routine",
            "جدول زراعي",
            "عناية يومية", "عناية أسبوعية",
        ],
        "fields": {
            "task_type":     "نوع المهمة",   # → taskType (Tasks)
            "interval_days": "كل كم يوم",   # → intervalDays (Tasks)
        },
        "direct": True,
    },
    # ── 16. Plant names ──────────────────────────────────────────────────────
    # nameAr / nameEn / nameScientific / category (Plants)
    {
        "name": "plant_names",
        "label_ar": "أسماء النبتة",
        "keywords": [
            "اسمها", "اسمه", "اسم", "بالإنجليزي", "بالانجليزي",
            "الاسم العلمي", "اسمه العلمي", "اسمها العلمي",
            "scientific", "english name",
            "الاسم",
            "اسمه بالعربي",
            "اسمه بالانجليزي",
            "اسم علمي",
            "بالعربي",
            "ايش اسمه",
            "ايش اسمها",
        ],
        "fields": {
            "arabic_name_primary":  "الاسم العربي",     # → nameAr
            "english_name_primary": "الاسم الإنجليزي",  # → nameEn
            "scientific_name":      "الاسم العلمي",     # → nameScientific
            "category":             "التصنيف",          # → category
        },
        "direct": True,
    },
    # ── 17. Comparison ──────────────────────────────────────────────────────────
    # Compares two or more plants on a specific criterion (difficulty, watering,
    # temperature, harvest speed).  Detected and handled entirely in app.py
    # via the comparison_from_data response mode – no LLM call.
    {
        "name": "comparison",
        "label_ar": "مقارنة بين نباتين",
        "keywords": [
            # Explicit comparison triggers
            "مين أسهل", "أيهما أسهل", "مين أفضل", "أيهما أفضل",
            "أي نبات أسهل", "أي نبات أفضل", "أي نبات أسرع",
            "مين يحتاج ري", "أي نبات يحتاج",
            "مين يتحمل", "مين يتحمل حرارة", "مين يتحمل برد",
            "مين أسرع", "مين أسرع حصاد",
            "مين مناسب أكثر",
            "قارن", "قارن بين", "الفرق بين", "مقارنة",
            "أيهما أسرع", "أيهما يحتاج",
            "compare", "comparison", "difference between",
            "which is easier", "which is better", "which is faster",
            "which needs more water", "which tolerates heat", "which tolerates cold",
        ],
        "fields": {
            "difficulty_level":           "مستوى الصعوبة",
            "watering_interval_days_min": "فترة الري (أيام)",
            "temperature_optimal_max_c":  "الحرارة العليا",
            "temperature_optimal_min_c":  "الحرارة الدنيا",
            "harvest_after_days_min":     "أيام الحصاد",
            "harvest_after_days_max":     "أيام الحصاد (الأقصى)",
        },
        "direct": True,
    },
    # ── 18. Germination ──────────────────────────────────────────────────────────
    # germinationDays (Plants) — strict single-field intent.
    # Handles questions like "كم يوم يحتاج لينبت؟" / "أيام الإنبات".
    {
        "name": "germination",
        "label_ar": "الإنبات",
        "keywords": [
            "ينبت", "انبات", "إنبات", "أيام الإنبات", "مدة الإنبات",
            "كم يوم لينبت", "متى ينبت", "يطلع البذر", "طلوع البذر",
        ],
        "fields": {
            "germination_days_min": "أيام الإنبات",   # → germinationDays
        },
        "direct": True,
    },
    # ── 19. Spacing ──────────────────────────────────────────────────────────────
    # plantSpacingCmMin / plantSpacingCmMax (Plants) — strict dual-field intent.
    # Handles questions like "كم المسافة بين النباتات؟" / "التباعد".
    {
        "name": "spacing",
        "label_ar": "التباعد بين النباتات",
        "keywords": [
            "مسافة", "تباعد", "تباعد النباتات", "المسافة بين",
            "كم مسافة", "المسافة المناسبة",
        ],
        "fields": {
            "plant_spacing_cm_min": "التباعد الأدنى (سم)",   # → plantSpacingCmMin
            "plant_spacing_cm_max": "التباعد الأقصى (سم)",   # → plantSpacingCmMax
        },
        "direct": True,
    },

]

