"""
Training di un classificatore di sentiment fine-tuned sul dominio fashion,
da usare al posto dei modelli RoBERTa generici in analisi_premiumization_hm.py.

Il modello viene salvato in una cartella "fashion-sentiment-model-hm" accanto
a questo script: se analisi_premiumization_hm.py si trova nella STESSA
cartella, lo rileva e lo usa automaticamente al posto dei modelli generici.

CHANGELOG correzioni:
  - Split train/val/test (invece di solo train/val): la validation viene
    usata per il model selection, il test per una stima finale onesta.
  - load_best_model_at_end=True (con save_strategy allineata a
    eval_strategy): salva il checkpoint migliore, non semplicemente
    quello dell'ultima epoca.
  - Rimosso ignore_mismatched_sizes=True: non necessario per una testa di
    classificazione nuova (nessun mismatch atteso); tenerlo di default
    poteva nascondere problemi reali con altri checkpoint in futuro.
  - --sample rinominato in --sample_per_class per chiarire che il numero
    indicato è PER CLASSE, non il totale del dataset.
  - BASE MODEL = Cardiff (cardiffnlp/twitter-roberta-base-sentiment-latest)
    invece di distilbert-base-multilingual-cased. Motivazione: il base
    multilingue, addestrato solo su recensioni e-commerce (testi educati
    e letterali), non coglieva ironia/sarcasmo tipici dei commenti social
    e produceva risultati sfasati. Cardiff è pre-addestrato su ~124M di
    tweet e già fine-tuned sul sentiment social: partendo da lì e facendo
    una LEGGERA specializzazione sul dominio fashion si ottiene un modello
    che (a) è specializzato sul fashion come richiesto, (b) conserva la
    capacità di gestire ironia e linguaggio informale dei social.
    NB: la testa di classificazione di Cardiff ha già 3 classi con lo
    stesso mapping (0=negative, 1=neutral, 2=positive), quindi il
    fine-tuning parte da una testa già sensata, non da pesi casuali.
  - Iperparametri conservativi di default (lr=1e-5, 1 epoca): l'obiettivo
    è ADATTARE il modello al lessico fashion, non riaddestrarlo da zero.
    Un learning rate alto o troppe epoche causerebbero catastrophic
    forgetting del pre-training social (e quindi dell'ironia).
  - Un base mono-lingua inglese è ora sufficiente: la pipeline principale
    lavora su un corpus unico in inglese (i commenti italiani vengono
    tradotti con Helsinki-NLP/opus-mt-it-en, come per Zara).
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
                          "fashion-sentiment-model-hm")
HF_DATASET = "Censius-AI/ECommerce-Women-Clothing-Reviews"
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}

# Base = Cardiff: pre-addestrato su ~124M tweet e già fine-tuned per il
# sentiment sui social (stesso mapping 0=neg/1=neu/2=pos). Il fine-tuning
# sul dataset fashion è quindi una specializzazione di dominio LEGGERA,
# non un riaddestramento da zero: il modello conserva la gestione di
# ironia/sarcasmo dei social e impara il lessico fashion/e-commerce.
# Il corpus della pipeline è tutto in inglese (l'italiano viene tradotto
# da translate_italian_hm.py), quindi un base mono-lingua EN va bene.
DEFAULT_BASE = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def rating_to_label(r):
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2


def load_reviews(csv):
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
    ap.add_argument("--csv",             default=None)
    ap.add_argument("--base",            default=DEFAULT_BASE,
                    help="Modello base HuggingFace. Di default Cardiff "
                         "(twitter-roberta), pre-addestrato sui social.")
    ap.add_argument("--sample_per_class", type=int, default=6000,
                    help="Numero massimo di esempi PER CLASSE (0/1/2) da "
                         "usare per il training, non il totale del dataset.")
    # Iperparametri conservativi: 1 epoca, lr basso. L'obiettivo è adattare
    # Cardiff al lessico fashion senza cancellare (catastrophic forgetting)
    # il pre-training social che gli permette di riconoscere l'ironia.
    ap.add_argument("--epochs",  type=int, default=1)
    ap.add_argument("--batch",   type=int, default=16)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--lr",      type=float, default=1e-5)
    args = ap.parse_args()

    print(f"[H&M] Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | base model: {args.base}")

    df = load_reviews(args.csv)
    text_col   = find_column(df, ["Review Text", "review_text", "Text", "review"])
    rating_col = find_column(df, ["Rating", "rating", "stars", "score"])

    df = df[[text_col, rating_col]].dropna()
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df["label"] = df[rating_col].astype(int).apply(rating_to_label)
    df = df.rename(columns={text_col: "text"})[["text", "label"]]

    # BILANCIAMENTO REALE DELLE CLASSI. Il dataset è ~77% positivo (rating
    # 4-5): campionare "FINO A N per classe" non bilancia nulla, perché le
    # classi negativa (~2.400) e neutra (~2.900) restano molto più piccole
    # dei 6.000 positivi -> il modello impara un bias positivo (causa del
    # sentiment troppo positivo nelle run precedenti). Qui si campiona
    # ESATTAMENTE lo stesso numero di esempi per classe, pari alla classe
    # minoritaria (o a --sample_per_class se inferiore).
    n_min = int(df["label"].value_counts().min())
    n_per_class = (min(n_min, args.sample_per_class)
                   if args.sample_per_class and args.sample_per_class > 0
                   else n_min)
    parts = [g.sample(n_per_class, random_state=42)
             for _, g in df.groupby("label")]
    df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training su {len(df)} recensioni totali ({args.sample_per_class} per classe richiesti) "
          f"| class balance:\n{df['label'].value_counts().sort_index().to_string()}")

    # Split a tre vie: 70% train / 15% val / 15% test.
    # La validation guida il model selection durante il training,
    # il test è tenuto completamente da parte per una stima finale onesta
    # delle performance (mai visto né dal training né dalle decisioni di
    # model selection).
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

    print(f"Split -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tok, batched=True)
    val_ds   = Dataset.from_pandas(val_df,   preserve_index=False).map(tok, batched=True)
    test_ds  = Dataset.from_pandas(test_df,  preserve_index=False).map(tok, batched=True)

    # Cardiff ha già una testa a 3 classi con lo stesso mapping: nessun
    # mismatch atteso.
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
        save_strategy="epoch",          # allineata a eval_strategy: richiesto da load_best_model_at_end
        save_total_limit=1,             # tiene solo il checkpoint migliore, non tutte le epoche
        load_best_model_at_end=True,    # il modello salvato è il migliore su validation, non l'ultimo
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    common = dict(model=model, args=targs,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=metrics)
    try:
        trainer = Trainer(processing_class=tokenizer, **common)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **common)

    t0 = time.time()
    trainer.train()
    print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")

    print("\nValidation (usata per il model selection):", trainer.evaluate())

    print("\nTest set (mai visto durante training/model selection):")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(test_metrics)

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved H&M fashion-domain sentiment model -> {OUT_DIR}")
    print("Se analisi_premiumization_hm.py si trova nella STESSA cartella di questo "
          "script, lo rileverà e lo userà automaticamente al posto dei modelli RoBERTa generici.")


if __name__ == "__main__":
    main()
