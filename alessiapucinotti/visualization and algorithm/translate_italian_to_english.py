"""
Translate Italian comments into English (Helsinki-NLP/opus-mt-it-en).

Rationale: instead of discarding the Italian-language comments, they are
machine-translated into English so the whole study runs on a single, unified
English corpus. This is a standard preprocessing step in multilingual social
media NLP and is documented in the Methodology chapter.

Input  : all_italian.json   (schema: [{date, comments:[{comment, like}]}])
Output : all_italian_translated.json  (same schema, 'comment' translated to EN)

The output is then merged with the native English corpus (all_english.json)
by the main pipeline (analisi_premiumization.py).

Requirements: pip install transformers sentencepiece torch
Run (same python as the rest):
  & C:/Users/aless/AppData/Local/Programs/Python/Python313/python.exe translate_italian_to_english.py
"""

import json
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL    = "Helsinki-NLP/opus-mt-it-en"
# Accept either the new English name or the legacy Italian one
INPUT_CANDIDATES = ["all_italian.json", "tutti_italiano.json"]
OUTPUT   = "all_italian_translated.json"
BATCH    = 8      # small batches → lower peak RAM on CPU
MAX_LEN  = 256    # trim very long comments to save memory

_DIR = os.path.dirname(os.path.abspath(__file__))


def comment_list_key(group):
    return 'comments' if 'comments' in group else 'commenti'


def comment_text_key(comment):
    return 'comment' if 'comment' in comment else 'commento'


def resolve_input():
    for name in INPUT_CANDIDATES:
        path = os.path.join(_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No Italian corpus found. Expected one of: " + ", ".join(INPUT_CANDIDATES))


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
            out.extend(tokenizer.decode(t, skip_special_tokens=True) for t in translated)
        except Exception as e:
            print(f"  WARNING: batch {i}–{i+len(batch)} failed ({e}), keeping original text")
            out.extend(batch)
        print(f"  translated {min(i + BATCH, len(texts))}/{len(texts)}")
    return out


def main():
    in_path  = resolve_input()
    out_path = os.path.join(_DIR, OUTPUT)
    print(f"Input: {os.path.basename(in_path)}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all comments into one list (keeps a pointer back to its slot).
    # The output is normalised to the English schema: {date, comments:[{comment, like}]}.
    flat_texts = []
    index = []         # (group_idx, comment_idx)
    out_data = []      # normalised structure to fill with translations
    for gi, group in enumerate(data):
        clk = comment_list_key(group)
        out_comments = []
        for ci, c in enumerate(group[clk]):
            ctk = comment_text_key(c)
            flat_texts.append(c[ctk])
            index.append((gi, ci))
            out_comments.append({"comment": c[ctk], "like": c.get("like", 0)})
        out_data.append({"date": group["date"], "comments": out_comments})

    print(f"Loading translation model {MODEL} (first run downloads weights)...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL)
    model     = MarianMTModel.from_pretrained(MODEL)

    print(f"Translating {len(flat_texts)} Italian comments into English...")
    translated = translate_all(flat_texts, tokenizer, model)

    # Write the translations back into the normalised structure
    for (gi, ci), en_text in zip(index, translated):
        out_data[gi]["comments"][ci]["comment"] = en_text

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(translated)} translated comments -> {out_path}")
    print("It will be merged with the English corpus by the main pipeline.")


if __name__ == "__main__":
    main()
