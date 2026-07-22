import json, os

_DIR = os.path.dirname(os.path.abspath(__file__))

base_path   = os.path.join(_DIR, "tutti_inglese.json")
source_path = os.path.join(_DIR, "all_italian_translated.json")

with open(base_path, "r", encoding="utf-8") as f:
    base = json.load(f)

with open(source_path, "r", encoding="utf-8") as f:
    source = json.load(f)

converted = []
for group in source:
    comments = group.get("comments", group.get("commenti", []))
    commenti = [
        {"commento": c.get("comment", c.get("commento", "")), "like": c.get("like", 0)}
        for c in comments
    ]
    converted.append({"date": group["date"], "commenti": commenti})

merged = base + converted

with open(base_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

total_src  = sum(len(g.get("comments", g.get("commenti", []))) for g in source)
total_base = sum(len(g.get("commenti", [])) for g in base)
print(f"Base comments  : {total_base}")
print(f"Added comments : {total_src}")
print(f"Total groups   : {len(merged)}")
