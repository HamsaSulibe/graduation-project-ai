"""
add_bare_keywords.py – Fix intent matching by adding proper-Unicode bare keywords.
The original intents.py contains double-encoded (mojibake) Arabic keywords that
don't match against proper-Unicode user input. This script adds correct forms.

Run: .venv\Scripts\python.exe scripts/add_bare_keywords.py
"""
import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')

# Resolve path relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTENTS_PATH = os.path.join(PROJECT_ROOT, "src", "config", "intents.py")
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Bare proper-Unicode keywords (replacing mojibake old ones)
BARE_KEYWORDS: dict[str, list[str]] = {
    "watering":       ["اسقي", "سقي", "ري", "ماء", "عطش", "افراط بالري", "نقص ري"],
    "light":          ["ضوء", "شمس", "ظل", "اضاءة", "ساعات شمس", "نافذة", "اتجاه"],
    "temperature":    ["حرارة", "برد", "حر", "درجة", "تحمل", "صقيع", "تجمد"],
    "soil":           ["تربة", "حموضة", "تصريف", "رملي", "طيني"],
    "fertilizing":    ["سماد", "تسميد", "سمد", "كمبوست"],
    "season":         ["موسم", "شهر", "فصل", "فلسطين"],
    "harvest_storage":["حصاد", "احصد", "جمع", "قطف", "تجفيف", "تخزين", "خزن"],
    "pests_diseases": ["افات", "آفات", "حشرات", "امراض", "أمراض", "مرض", "افة", "عفن", "فطر"],
    "beginner":       ["مبتدئ", "سهل", "صعب", "بسيط"],
    "container":      ["اصيص", "وعاء", "حوض", "حاوية"],
    "planting":       ["ازرع", "زراعة", "خطوات", "اكثار", "بذر", "شتل", "بذور"],
    "general_summary":["معلومات", "عرفني", "وصف", "تعريف", "فوائد", "فائدة"],
    "plant_overview": ["اعطني", "اخبرني", "كل شي", "لخص", "اهم معلومات"],
    "care_summary":   ["عناية", "اعتني", "رعاية"],
    "growing_guide":  ["كيف ازرع", "دليل زراعة", "ازرعها", "ازرعه"],
    "plant_names":    ["اسمها", "اسمه", "اسم", "بالانجليزي", "الاسم العلمي"],
    "humidity":       ["رطوبة", "رطب", "جاف", "جفاف"],
}

changes = 0
for intent_name, new_kws in BARE_KEYWORDS.items():
    pattern = (
        r'("name":\s*"' + re.escape(intent_name) + r'"'
        r'.*?"keywords":\s*\[)(.*?)(\])'
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"[WARN] intent '{intent_name}' not found in file")
        continue
    pre, kw_block, post = match.group(1), match.group(2), match.group(3)
    existing = re.findall(r'"([^"]+)"', kw_block)
    existing_set = set(existing)
    to_add = [kw for kw in new_kws if kw not in existing_set]
    if not to_add:
        print(f"[SKIP] '{intent_name}': already present")
        continue
    additions = "".join(f'\n            "{kw}",' for kw in to_add)
    new_kw_block = kw_block.rstrip() + additions + "\n        "
    new_content = content[:match.start()] + pre + new_kw_block + post + content[match.end():]
    content = new_content
    print(f"[OK]  '{intent_name}': +{len(to_add)} → {to_add}")
    changes += 1

if changes:
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nUpdated intents.py — {changes} intents modified.")
else:
    print("Nothing to add.")
