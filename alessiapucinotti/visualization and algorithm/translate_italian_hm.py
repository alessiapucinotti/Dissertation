"""
H&M: split the mixed-language corpus and translate Italian -> English.

Reads all_comments_hm.json (mixed EN/IT comments), detects the language of
each comment and produces:
  - all_comments_hm_english.json             (native English + everything else)
  - all_comments_hm_italian_translated.json  (Italian, machine-translated to EN)

Both files are then loaded by analisi_premiumization_hm.py as a single
unified English corpus — the same design used for Zara
(translate_italian_to_english.py + merged corpus).

Design choice: a comment goes to the Italian file ONLY if langdetect is
confident (>= 0.85) that it is Italian. Everything else (English, short
texts, emojis, uncertain cases) stays in the English corpus. This way NO
comment is discarded — unlike the old EN/IT split, which dropped more than
half of the corpus and biased the sentiment estimates.

Requirements: pip install transformers sentencepiece torch langdetect
Run:
  python translate_italian_hm.py
"""

import json
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0

MODEL   = "Helsinki-NLP/opus-mt-it-en"
INPUT   = "all_comments_hm.json"
OUT_EN  = "all_comments_hm_english.json"
OUT_IT  = "all_comments_hm_italian_translated.json"
BATCH   = 8      # small batches -> lower peak RAM on CPU
MAX_LEN = 256    # trim very long comments to save memory

MIN_LANG_CONFIDENCE = 0.85
MIN_LANG_LEN        = 10

_DIR = os.path.dirname(os.path.abspath(__file__))


def is_italian(text):
    """True only if langdetect is CONFIDENT the comment is Italian."""
    if not text or len(text) < MIN_LANG_LEN:
        return False
    try:
        top = detect_langs(text)[0]
        return top.lang == 'it' and top.prob >= MIN_LANG_CONFIDENCE
    except LangDetectException:
        return False


def translate_all(texts, tokenizer, model):
    """Translate a list of strings in batches, preserving order."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [str(t)[:MAX_LEN] for t in texts[i:i + BATCH]]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LEN).to(device)
            translated = model.generate(**inputs)
            out.extend(tokenizer.decode(t, skip_special_tokens=True)
                       for t in translated)
        except Exception as e:
            print(f"  WARNING: batch {i}-{i + len(batch)} failed ({e}), keeping original text")
            out.extend(batch)
        print(f"  translated {min(i + BATCH, len(texts))}/{len(texts)}")
    return out


def main():
    in_path = os.path.join(_DIR, INPUT)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split by language, preserving the {date, comments:[...]} grouping.
    groups_en, groups_it = [], []
    n_en = n_it = 0
    for group in data:
        clk = 'comments' if 'comments' in group else 'commenti'
        en_comments, it_comments = [], []
        for c in group[clk]:
            ctk  = 'comment' if 'comment' in c else 'commento'
            text = c.get(ctk, '')
            row  = {"comment": text, "like": c.get("like", 0)}
            if is_italian(text):
                it_comments.append(row)
                n_it += 1
            else:
                en_comments.append(row)
                n_en += 1
        if en_comments:
            groups_en.append({"date": group["date"], "comments": en_comments})
        if it_comments:
            groups_it.append({"date": group["date"], "comments": it_comments})

    print(f"Corpus H&M: {n_en + n_it} comments | EN/other: {n_en} | IT (confident): {n_it}")

    with open(os.path.join(_DIR, OUT_EN), "w", encoding="utf-8") as f:
        json.dump(groups_en, f, ensure_ascii=False, indent=2)
    print(f"Saved {n_en} comments -> {OUT_EN}")

    if not groups_it:
        with open(os.path.join(_DIR, OUT_IT), "w", encoding="utf-8") as f:
            json.dump([], f)
        print("No confident-Italian comments found; wrote empty file.")
        return

    # Flatten the Italian comments, translate, write back in place.
    flat_texts, index = [], []
    for gi, group in enumerate(groups_it):
        for ci, c in enumerate(group["comments"]):
            flat_texts.append(c["comment"])
            index.append((gi, ci))

    print(f"Loading translation model {MODEL} (first run downloads weights)...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL)
    model     = MarianMTModel.from_pretrained(MODEL)

    print(f"Translating {len(flat_texts)} Italian comments into English...")
    translated = translate_all(flat_texts, tokenizer, model)

    for (gi, ci), en_text in zip(index, translated):
        groups_it[gi]["comments"][ci]["comment"] = en_text

    with open(os.path.join(_DIR, OUT_IT), "w", encoding="utf-8") as f:
        json.dump(groups_it, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(translated)} translated comments -> {OUT_IT}")
    print("Both files are loaded by analisi_premiumization_hm.py as one corpus.")


if __name__ == "__main__":
    main()
