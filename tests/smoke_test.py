"""Quick smoke-test for the refactored project."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.arabic import normalize, normalize_name, normalize_query
from src.config.constants import is_junk, NAME_COLUMNS
from src.config.columns import get_column, resolve_column
from src.inference.plant_data_store import load_plant_store, get_plant_data, get_all_plant_names
from src.inference.intent_fields import classify_intents, is_direct_intent, is_multi_field_intent
from src.inference.context_utils import (
    build_direct_answer, build_answer_draft, has_sufficient_data, extract_target_plant,
)

PASS = 0
FAIL = 0

def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  V {label}")
    else:
        FAIL += 1
        print(f"  X {label}")

print("=" * 50)
print("1. Arabic normaliser")
check("normalize hamza", normalize("أَلْحَبَق") == "الحبق")
check("normalize_name strips al", normalize_name("الريحان") == "ريحان")
check("normalize_query replaces ?", "اسقي" in normalize_query("كيف أسقي؟"))

print("\n2. Constants")
check("is_junk(ghyr mtofr)", is_junk("غير متوفر") is True)
check("is_junk(mint)", is_junk("نعناع") is False)
check("NAME_COLUMNS has 7", len(NAME_COLUMNS) == 7)

print("\n3. Column resolution")
d = {"watering_rule": "اسقِ كل 3 أيام"}
check("resolve alias", resolve_column(d, "watering_rule_text") is not None)
check("get_column alias", get_column(d, "watering_rule_text") is not None)
check("resolve_column(None)", resolve_column(None, "x") is None)

print("\n4. Plant data store")
n = load_plant_store("data/raw/plants.xlsx")
check(f"loaded {n} plants", n > 0)
names = get_all_plant_names()
check(f"got {len(names)} names", len(names) > 0)
sample = names[0]
data = get_plant_data(sample)
check(f"get_plant_data({sample})", data is not None)

print("\n5. Intent classification")
i_w = classify_intents("كيف أسقي النعناع؟")
check(f"watering: {i_w}", "watering" in i_w)
check("is_direct", is_direct_intent(i_w))
i_m = classify_intents("كيف أعتني بالنعناع؟")
check(f"care: {i_m}", "care_summary" in i_m)
check("is_multi", is_multi_field_intent(i_m))

print("\n6. Direct answer")
pn = extract_target_plant("كيف أسقي الألوفيرا؟", names)
if pn:
    pd_ = get_plant_data(pn)
    if pd_:
        ans = build_direct_answer(pn, pd_, ["watering"])
        check(f"direct len={len(ans)}", len(ans) > 10)
    else:
        check("plant data None", False)
else:
    check("extract failed", False)

print("\n7. Multi-field draft")
pn2 = extract_target_plant("اعطني كل المعلومات عن الكمون", names)
if pn2:
    pd2 = get_plant_data(pn2)
    if pd2:
        draft = build_answer_draft(pn2, pd2, ["plant_overview"])
        check(f"draft len={len(draft)}", len(draft) > 20)
    else:
        check("plant data None", False)
else:
    check("extract failed", False)

print("\n" + "=" * 50)
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
