"""
Domain adaptation: fine-tune a sentiment model on FASHION reviews.

Why: there is no off-the-shelf fashion-domain sentiment classifier. The standard,
citable approach is to take a sentiment-capable base model and fine-tune it on a
labelled fashion-review corpus, so it reads clothing language ('the fabric feels
cheap', 'true to size', 'sheer') better than a generic model. This addresses the
supervisor's request for a fashion-trained model.

Base model : cardiffnlp/twitter-roberta-base-sentiment-latest (Cardiff).
             Pre-addestrato su ~124M tweet e già fine-tuned per il sentiment
             sui social, con la STESSA testa a 3 classi (0=neg/1=neu/2=pos).
             Partendo da Cardiff, il fine-tuning sul fashion è una
             specializzazione di dominio leggera: il modello impara il
             lessico fashion SENZA perdere la gestione di ironia/sarcasmo
             tipica dei commenti social (che un base generico addestrato
             solo su recensioni e-commerce non ha).
Dataset    : "Women's E-Commerce Clothing Reviews" (~23.5k reviews). Downloaded
             automatically from HuggingFace if no local CSV is given.
Labels     : Rating <= 2 -> negative (0); == 3 -> neutral (1); >= 4 -> positive (2)
Output     : ./fashion-sentiment-model/   (the main pipeline loads it automatically)

HOW TO RUN (Windows PowerShell):
  1) Install the libraries once:
       & C:/Users/aless/AppData/Local/Programs/Python/Python313/python.exe -m pip install transformers datasets scikit-learn accelerate torch
  2) Train (dataset is downloaded automatically, no CSV needed):
       & C:/Users/aless/AppData/Local/Programs/Python/Python313/python.exe finetune_fashion_sentiment.py
  3) When it finishes, re-run analisi_premiumization_zara.py — it uses the new model.

Faster:    add  --sample 4000
Full data: add  --sample 0
Local CSV: add  --csv "Womens Clothing E-Commerce Reviews.csv"
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fashion-sentiment-model")
HF_DATASET = "Censius-AI/ECommerce-Women-Clothing-Reviews"  # same data as the Kaggle file
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}


def rating_to_label(r):
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2


def load_reviews(csv):
    """Use a local CSV if it exists; otherwise download the same dataset from
    HuggingFace (no Kaggle login, works from any folder)."""
    if csv and os.path.exists(csv):
        print(f"Loading local CSV: {csv}")
        return pd.read_csv(csv)
    print(f"No local CSV -> downloading from HuggingFace: {HF_DATASET} ...")
    from datasets import load_dataset
    return load_dataset(HF_DATASET, split="train").to_pandas()


def find_column(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Optional local CSV. If omitted, the dataset is downloaded "
                         "automatically from HuggingFace.")
    # Base = Cardiff (twitter-roberta): pre-addestrato sui social, stessa
    # testa a 3 classi (0=neg/1=neu/2=pos). Il fine-tuning sul dataset
    # fashion è una specializzazione di dominio leggera che NON cancella
    # la capacità di gestire ironia/sarcasmo dei commenti social.
    ap.add_argument("--base", default="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    help="Base model (default: Cardiff twitter-roberta)")
    ap.add_argument("--sample", type=int, default=6000,
                    help="Max reviews per class (0 = use all). Lower = faster.")
    # Iperparametri conservativi (1 epoca, lr basso) per evitare il
    # catastrophic forgetting del pre-training social.
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | base model: {args.base}")

    df = load_reviews(args.csv)
    text_col   = find_column(df, ["Review Text", "review_text", "Text", "review"])
    rating_col = find_column(df, ["Rating", "rating", "stars", "score"])

    df = df[[text_col, rating_col]].dropna()
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df["label"] = df[rating_col].astype(int).apply(rating_to_label)
    df = df.rename(columns={text_col: "text"})[["text", "label"]]

    # BILANCIAMENTO REALE DELLE CLASSI: il dataset è ~77% positivo, quindi
    # "fino a N per classe" lascerebbe il positivo sovra-rappresentato e il
    # modello imparerebbe un bias positivo. Si campiona esattamente lo
    # stesso numero di esempi per classe (= classe minoritaria).
    n_min = int(df["label"].value_counts().min())
    n_per_class = (min(n_min, args.sample)
                   if args.sample and args.sample > 0 else n_min)
    parts = [g.sample(n_per_class, random_state=42)
             for _, g in df.groupby("label")]
    df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Training on {len(df)} reviews | class balance:\n"
          f"{df['label'].value_counts().sort_index().to_string()}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42,
                                        stratify=df["label"])

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tok, batched=True)
    val_ds   = Dataset.from_pandas(val_df,   preserve_index=False).map(tok, batched=True)

    # Cardiff ha già una testa a 3 classi con lo stesso mapping: nessun
    # mismatch atteso, quindi niente ignore_mismatched_sizes (nasconderebbe
    # problemi reali con altri checkpoint).
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro")}

    targs = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    common = dict(model=model, args=targs,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=metrics)
    try:
        # transformers >= 4.46 renamed 'tokenizer' -> 'processing_class'
        trainer = Trainer(processing_class=tokenizer, **common)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **common)

    t0 = time.time()
    trainer.train()
    print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")
    print("Validation:", trainer.evaluate())

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved fashion-domain sentiment model -> {OUT_DIR}")
    print("Now just re-run analisi_premiumization_zara.py — it will use this model automatically.")


if __name__ == "__main__":
    main()
