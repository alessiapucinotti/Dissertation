import json
import os
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

INPUT_FILE   = "reddit_comments_cleaned.json"
OUT_ITALIAN  = "reddit_italian.json"
OUT_ENGLISH  = "reddit_english.json"
MIN_LENGTH   = 10
MIN_CONFIDENCE = 0.85


def detect_lang(text: str) -> str | None:
    if not text or len(text) < MIN_LENGTH:
        return None
    try:
        results = detect_langs(text)
        top = results[0]
        if top.prob >= MIN_CONFIDENCE:
            return top.lang
        return None
    except LangDetectException:
        return None


def main() -> None:
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, INPUT_FILE)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    ita: dict[str, list] = {}
    eng: dict[str, list] = {}

    total = count_ita = count_eng = count_other = count_skip = 0

    for group in data:
        date = group["date"]
        for c in group["comments"]:
            total += 1
            lang = detect_lang(c["comment"])

            if lang is None:
                count_skip += 1
            elif lang == "it":
                ita.setdefault(date, []).append(c)
                count_ita += 1
            elif lang == "en":
                eng.setdefault(date, []).append(c)
                count_eng += 1
            else:
                count_other += 1

        if total % 2000 == 0:
            print(f"  Processed: {total}...")

    def to_list(groups: dict) -> list:
        return [{"date": d, "comments": v} for d, v in sorted(groups.items())]

    for out_name, groups, label in [
        (OUT_ITALIAN, ita, "italian"),
        (OUT_ENGLISH, eng, "english"),
    ]:
        result = to_list(groups)
        out_path = os.path.join(base_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        n = sum(len(g["comments"]) for g in result)
        print(f"[{label:8s}] {n:5d} comments in {len(result):3d} groups -> {out_name}")

    print()
    print(f"Total input     : {total}")
    print(f"  Italian       : {count_ita}")
    print(f"  English       : {count_eng}")
    print(f"  Other language: {count_other}  (discarded)")
    print(f"  Too short     : {count_skip}  (discarded)")


if __name__ == "__main__":
    main()
