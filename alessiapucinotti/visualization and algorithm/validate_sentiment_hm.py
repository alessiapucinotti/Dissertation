"""
Sentiment model VALIDATION on LABELLED data – H&M version.

Same benchmark logic as validate_sentiment.py; outputs go to results_hm/
so the two brand analyses stay independent.

Models compared:
  - VADER                (lexicon baseline)
  - SVM / Naive Bayes / Logistic Regression on TF-IDF  (classic ML)
  - RoBERTa Cardiff      (transformer, zero-shot)
  - Fashion fine-tuned   (Cardiff + fine-tuning sul dominio fashion:
                          il CHAMPION richiesto dal prof. Caricato da
                          ./fashion-sentiment-model-hm se presente)

Gold labels : rating <= 2 -> negative (0); == 3 -> neutral (1); >= 4 -> positive (2)
Output      : results_hm/sentiment_model_comparison_hm.pdf + CSV

Run:
  python validate_sentiment_hm.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib.pyplot as plt

_DIR       = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(_DIR, 'results_hm')
HF_DATASET = "Censius-AI/ECommerce-Women-Clothing-Reviews"
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
N_TOTAL    = 8000
TEST_SIZE  = 0.25


def rating_to_label(r):
    return 0 if r <= 2 else (1 if r == 3 else 2)


def load_labelled():
    from datasets import load_dataset
    df   = load_dataset(HF_DATASET, split="train").to_pandas()
    tcol = next(c for c in df.columns if c.lower() in ("review text", "text", "review"))
    rcol = next(c for c in df.columns if c.lower() in ("rating", "stars", "score"))
    df   = df[[tcol, rcol]].dropna()
    df   = df[df[tcol].astype(str).str.strip().str.len() > 0]
    df   = df.rename(columns={tcol: "text"})
    df["label"] = df[rcol].astype(int).apply(rating_to_label)
    parts = [g.sample(min(len(g), N_TOTAL // 3), random_state=42)
             for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def vader_labels(texts):
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = sia.polarity_scores(str(t))['compound']
        out.append(2 if c >= 0.05 else (0 if c <= -0.05 else 1))
    return np.array(out)


def transformer_labels(texts, model_name, tag):
    from transformers import pipeline as hf_pipeline
    pipe     = hf_pipeline("sentiment-analysis", model=model_name,
                           truncation=True, max_length=512)
    name2id  = {"negative": 0, "neutral": 1, "positive": 2}
    out = []
    for i in range(0, len(texts), 64):
        for r in pipe([str(t)[:512] for t in texts[i:i+64]]):
            out.append(name2id.get(r["label"].lower(), 1))
        print(f"    {tag} {min(i+64, len(texts))}/{len(texts)}")
    return np.array(out)


def roberta_labels(texts):
    return transformer_labels(
        texts, "cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa")


def fashion_labels(texts):
    """Modello fashion fine-tuned (base Cardiff), se già addestrato."""
    model_dir = os.path.join(_DIR, "fashion-sentiment-model-hm")
    if not os.path.isdir(model_dir):
        return None
    return transformer_labels(texts, model_dir, "Fashion FT")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_labelled()
    print(f"[H&M] Labelled sample: {len(df)} | balance {df['label'].value_counts().sort_index().to_dict()}")

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42,
                                         stratify=df["label"])
    y_test = test_df["label"].values

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
    Xtr   = tfidf.fit_transform(train_df["text"])
    Xte   = tfidf.transform(test_df["text"])
    classic = {
        "SVM (TF-IDF)":          LinearSVC(),
        "Naive Bayes (TF-IDF)":  MultinomialNB(),
        "Logistic Reg (TF-IDF)": LogisticRegression(max_iter=1000),
    }
    results = {}
    for name, clf in classic.items():
        clf.fit(Xtr, train_df["label"])
        results[name] = clf.predict(Xte)

    texts = test_df["text"].tolist()
    print("  Scoring VADER...")
    results["VADER"] = vader_labels(texts)
    print("  Scoring RoBERTa (Cardiff)...")
    results["RoBERTa (zero-shot)"] = roberta_labels(texts)
    print("  Scoring Fashion fine-tuned (Cardiff base)...")
    fashion_pred = fashion_labels(texts)
    if fashion_pred is not None:
        results["Fashion FT (champion)"] = fashion_pred
    else:
        print("    -> fashion-sentiment-model-hm non trovato: esegui prima "
              "finetune_fashion_sentiment_hm.py")

    rows = []
    print("\n" + "=" * 60)
    for name, pred in results.items():
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="macro")
        rows.append({"model": name, "accuracy": acc, "macro_f1": f1})
        print(f"\n### {name}  |  accuracy={acc:.3f}  macro-F1={f1:.3f}")
        print(classification_report(y_test, pred,
              target_names=[ID2LABEL[i] for i in (0, 1, 2)], zero_division=0))

    res = pd.DataFrame(rows)
    deployable = {"VADER", "RoBERTa (zero-shot)", "Fashion FT (champion)"}
    res["deployable"] = res["model"].isin(deployable)
    res = res.sort_values(["deployable", "macro_f1"],
                          ascending=[True, True]).reset_index(drop=True)
    res.to_csv(os.path.join(OUT_DIR, "sentiment_model_comparison_hm.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    x  = np.arange(len(res)); bw = 0.4
    zidx = [i for i, d in enumerate(res["deployable"]) if d]
    if zidx:
        ax.axvspan(min(zidx) - 0.5, max(zidx) + 0.5, color='#fdf0e6', zorder=0)
    ax.bar(x - bw/2, res["accuracy"], bw, color='#9ecae1', label='Accuracy', zorder=3)
    ax.bar(x + bw/2, res["macro_f1"], bw, color='#185FA5', label='Macro-F1', zorder=3)
    for i, (a, f) in enumerate(zip(res["accuracy"], res["macro_f1"])):
        ax.text(i - bw/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=8)
        ax.text(i + bw/2, f + 0.01, f'{f:.2f}', ha='center', fontsize=8)
    n_sup = len(res) - len(zidx)
    if n_sup:
        ax.text((n_sup - 1) / 2, 0.97,
                'Supervised, trained on this dataset\n(in-domain reference — NOT usable on unlabelled H&M comments)',
                ha='center', va='top', fontsize=8.5, color='#444')
    if zidx:
        ax.text((min(zidx) + max(zidx)) / 2, 0.97,
                'Zero-shot\n(usable on H&M comments)',
                ha='center', va='top', fontsize=8.5, color='#993c1d')
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Sentiment model validation on labelled fashion reviews\n'
                 'H&M analysis – Fashion fine-tuned (Cardiff base) as champion',
                 fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, 'sentiment_model_comparison_hm.pdf')
    plt.savefig(out)
    plt.close()
    print(f"\nSaved: {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
