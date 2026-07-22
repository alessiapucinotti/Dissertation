"""
SHAP interpretation of the RoBERTa sentiment model.
Shows WHICH WORDS drive the sentiment (interpretability requested by the supervisor).

Input  : comments_sentiment.json  (output of analisi_premiumization.py)
Output : shap_examples.html  -> highlighted tokens (red = pushes negative,
         blue = pushes positive) for representative comments.

Requirements: pip install shap
Run (same python as the rest):
  & C:/Users/aless/AppData/Local/Programs/Python/Python313/python.exe shap_interpretation.py
"""

import os
import pandas as pd
import shap
from transformers import pipeline

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # same champion as the thesis
_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(_DIR, "comments_sentiment.json")
N_MAX = 40                                                    # SHAP is slow: small sample

# pipeline returning ALL classes (required by SHAP)
pipe = pipeline("sentiment-analysis", model=MODEL,
                top_k=None, truncation=True, max_length=512)
explainer = shap.Explainer(pipe)

df = pd.read_json(INPUT).dropna(subset=["cleaned_text"])

# representative examples: most positive, most negative, and price/quality related
most_pos = list(df.nlargest(10, "sentiment_score")["cleaned_text"])
most_neg = list(df.nsmallest(10, "sentiment_score")["cleaned_text"])
ptq = df[df["cleaned_text"].str.contains("price|quality|cheap|expensive|fabric|material",
                                         case=False, na=False)]
ptq = list(ptq.sample(min(20, len(ptq)), random_state=42)["cleaned_text"]) if len(ptq) else []

sample = [str(t)[:300] for t in (most_pos + most_neg + ptq) if str(t).strip()][:N_MAX]
print(f"Computing SHAP on {len(sample)} comments (this may take a few minutes)...")

shap_values = explainer(sample)

# save the highlighted tokens to HTML (open in browser / insert as a figure)
blocks = []
for i in range(len(sample)):
    blocks.append(f"<hr><p style='font-family:sans-serif;color:#666'>Comment {i+1}</p>")
    blocks.append(shap.plots.text(shap_values[i], display=False))

with open(os.path.join(_DIR, "shap_examples.html"), "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'></head><body>" + "".join(blocks) + "</body></html>")

print("Saved: shap_examples.html  -> open it in the browser to see the words that")
print("drive the sentiment (useful as an interpretability figure in Chapter 5).")
