"""
add_intent_keywords.py – Adds missing Arabic keywords to intents.py.
Run once: .venv\Scripts\python.exe scripts/add_intent_keywords.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

INTENTS_PATH = "src/config/intents.py"

with open(INTENTS_PATH, "r", encoding="utf-8-sig") as f:
    content = f.read()

# ── Extra keywords to merge into each intent ─────────────────────
# Format: intent_name → list of new keywords to ADD (skipped if already present)
EXTRA_KEYWORDS: dict[str, list[str]] = {
    "watering": [
        "اروي", "أروي", "روي", "ارو", "ارويه", "ارويها",
        "اسقيه", "اسقيها", "كم مرة اسقي", "كم مره اسقي",
        "سقي نبتة", "سقي نبات",
    ],
    "light": [
        "إضاءة", "اضاءة", "الضوء المناسب", "ضوء مناسب",
        "شمس مباشرة", "شمس غير مباشرة",
    ],
    "temperature": [
        "درجة الحرارة", "درجات الحرارة", "دفي", "دافئ",
    ],
    "soil": [
        "نوع التربة", "التربة المناسبة", "تربة مناسبة",
    ],
    "fertilizing": [
        "اسمد", "أسمد", "سمدت", "سماد نبتة",
    ],
    "suitability": [
        "لبيتي", "لمنزلي", "لبيتنا", "ملائم", "يلائم",
        "هل يناسب", "هل تناسب", "مناسب لي",
        "مناسب لبيتي", "يصلح لبيتي",
    ],
    "tasks": [
        "جدول العناية", "روتين العناية", "مهام العناية",
        "جدول رعاية", "عناية يومية", "عناية أسبوعية",
    ],
    "general_summary": [
        "احكيلي", "حكيلي", "قولي", "اخبرني", "خبرني",
        "اعطيني معلومات", "شو هو", "شو هي", "ايش هو", "ايش هي",
        "بدي اعرف", "بدي أعرف",
    ],
    "plant_overview": [
        "احكيلي عن", "حكيلي عن", "قولي عن", "اخبرني عن",
        "معلومات كاملة عن", "كل شي عن", "كل شيء عن",
    ],
    "care_summary": [
        "كيف اعتني", "كيف أعتني", "كيف يزرع", "روتين",
        "طريقة العناية", "كيفية العناية",
    ],
    "pests_diseases": [
        "مشاكل", "مشكلة", "يصفر", "يجف", "يموت", "تموت",
        "اوراق صفراء", "أوراق صفراء", "تساقط الأوراق",
    ],
    "harvest_storage": [
        "متى احصد", "متى أحصد", "وقت الحصاد",
    ],
    "planting": [
        "كيف ازرع", "كيف أزرع", "طريقة الزراعة", "خطوات الزراعة",
    ],
    "plant_names": [
        "الاسم", "اسمه بالعربي", "اسمه بالانجليزي",
        "اسم علمي", "بالعربي", "ايش اسمه", "ايش اسمها",
    ],
    "humidity": [
        "رطوبة الهواء", "رطوبة محيطية", "رطب", "جاف جدا",
    ],
}

import re
changes = 0

for intent_name, new_kws in EXTRA_KEYWORDS.items():
    # Find the keywords list for this intent
    pattern = rf'("name":\s*"{re.escape(intent_name)}".*?"keywords":\s*\[)(.*?)(\])'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"[WARN] intent '{intent_name}' not found")
        continue

    pre, kw_block, post = match.group(1), match.group(2), match.group(3)
    # Extract existing keywords
    existing = re.findall(r'"([^"]+)"', kw_block)
    existing_set = set(existing)

    to_add = [kw for kw in new_kws if kw not in existing_set]
    if not to_add:
        print(f"[SKIP] '{intent_name}': all keywords already present")
        continue

    # Append new keywords to the block (before the closing bracket)
    additions = "".join(f'\n            "{kw}",' for kw in to_add)
    new_kw_block = kw_block.rstrip() + additions + "\n        "
    new_content = content[:match.start()] + pre + new_kw_block + post + content[match.end():]
    content = new_content
    print(f"[OK]  '{intent_name}': added {len(to_add)} keywords: {to_add}")
    changes += 1

if changes:
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nWrote updated intents.py ({changes} intents modified)")
else:
    print("No changes needed.")
