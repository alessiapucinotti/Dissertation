"""
SHAP interpretation of the RoBERTa sentiment model – H&M version.
Shows WHICH WORDS drive the sentiment on H&M comments.

Input  : commenti_sentiment_hm_en.json  (output of analisi_premiumization_hm.py)
Output : shap_examples_hm.html  -> highlighted tokens (red = pushes negative,
         blue = pushes positive) for representative H&M comments.

Requirements: pip install shap
Run:
  python shap_interpretation_hm.py
"""

import os
import pandas as pd
import shap
from transformers import pipeline

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(_DIR, "commenti_sentiment_hm_en.json")
N_MAX = 40

pipe      = pipeline("sentiment-analysis", model=MODEL,
                     top_k=None, truncation=True, max_length=512)
explainer = shap.Explainer(pipe)

df = pd.read_json(INPUT).dropna(subset=["testo_pulito"])

most_pos = list(df.nlargest(10, "sentiment_score")["testo_pulito"])
most_neg = list(df.nsmallest(10, "sentiment_score")["testo_pulito"])

# Parole chiave rilevanti per H&M: collaborazioni, prezzo, qualità, sostenibilità
hm_kw = (
    "mugler|stella mccartney|price|quality|cheap|expensive|"
    "fabric|material|sustainable|organic|premium|luxury"
)
ptq = df[df["testo_pulito"].str.contains(hm_kw, case=False, na=False)]
ptq = list(ptq.sample(min(20, len(ptq)), random_state=42)["testo_pulito"]) if len(ptq) else []

sample = [str(t)[:300] for t in (most_pos + most_neg + ptq) if str(t).strip()][:N_MAX]
print(f"Computing SHAP on {len(sample)} H&M comments (this may take a few minutes)...")

shap_values = explainer(sample)

blocks = []
for i in range(len(sample)):
    blocks.append(f"<hr><p style='font-family:sans-serif;color:#666'>Comment {i+1}</p>")
    blocks.append(shap.plots.text(shap_values[i], display=False))

out_path = os.path.join(_DIR, "shap_examples_hm.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'></head><body>" + "".join(blocks) + "</body></html>")

print(f"Saved: {out_path}")
print("Open it in the browser to see which words drive sentiment on H&M comments.")
