"""
H&M — Premiumization Tolerance Index ONLY (calculation + time-series plot).

Minimal version of the pipeline: no LDA/BERTopic/wordcloud/ABSA.
It only:
  1. loads the H&M English corpus
  2. runs sentiment with the fashion fine-tuned model (fashion-sentiment-model-hm,
     falls back to Cardiff if the folder does not exist)
  3. PTI(t) = 100 + 100*(tau_t - tau_0)  [professor's formula]
     tau = engagement-weighted mean sentiment of the price/quality comments
     in the quarter; a quarter is valid if it has >= 5 price/quality comments;
     baseline tau_0 = first observed quarter (2019)
  4. saves: results_hm/pti_time_series.pdf + pti_results_hm.json + console log

Run:
  python pti_hm.py            (recomputes sentiment with the model)
  python pti_hm.py --reuse    (reuses comments_sentiment_hm.json if present,
                               skips recomputation: useful to regenerate only
                               the plot)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.format':    'pdf',
    'legend.frameon':    False,
    'legend.fontsize':   9,
})

C_POS = '#2166ac'

_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, 'results_hm')

CORPUS_FILES = ['all_comments_hm.json']
SENTIMENT_CACHE = os.path.join(_DIR, 'comments_sentiment_hm.json')

DATE_FROM = '2019-01-01'
MIN_DOC   = 5   # minimum price/quality comments to validate a quarter

EVENTS = {
    '2021-03-01': 'H&M Conscious\nCollection',
    '2022-09-01': 'Mugler\ncollaboration',
    '2023-03-01': 'Stella McCartney\ncollaboration',
    '2023-09-01': 'Price increase\n+10%',
    '2024-03-01': 'H&M Studio\nSS24',
}

# Price/value keywords + quality judgments (identical to Zara).
KW_PTQ = sorted({
    'price', 'pricing', 'priced', 'cost', 'value', 'worth', 'worthless',
    'affordable', 'price point', 'price increase', 'price tag', 'good value',
    'value for money', 'expensive', 'overpriced', 'pricey', 'too much',
    'too pricey', 'high price', 'raised price', 'price hike',
    'quality', 'good quality', 'bad quality', 'poor quality', 'high quality',
    'low quality', 'fast fashion', 'substandard', 'shoddy',
})


def load_corpus():
    rows = []
    for name in CORPUS_FILES:
        path = os.path.join(_DIR, name)
        if not os.path.exists(path):
            print(f"  WARNING: {name} missing")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for group in data:
            for c in group['comments']:
                rows.append({'date': group['date'],
                             'cleaned_text': c['comment'],
                             'like': c.get('like', 0)})
        print(f"  loaded: {name}")
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[(df['date'] >= pd.Timestamp(DATE_FROM)) &
            (df['date'] <= pd.Timestamp.now())]
    df['cleaned_text'] = df['cleaned_text'].astype(str).str.strip()
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    return df


def load_cached_sentiment():
    with open(SENTIMENT_CACHE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # to_json saves dates as epoch milliseconds
    if np.issubdtype(df['date'].dtype, np.number):
        df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[(df['date'] >= pd.Timestamp(DATE_FROM))]
    return df.reset_index(drop=True)


def run_sentiment(df):
    from transformers import pipeline as hf_pipeline
    # USE_FASHION_MODEL=False: si usa SEMPRE il Cardiff generico zero-shot.
    # I modelli fine-tuned sulle recensioni e-commerce (registro educato,
    # ~77% positivo) ammorbidiscono il segnale negativo su prezzo/qualita'
    # -> sentiment tutto positivo. Il Cardiff zero-shot preserva il negativo.
    USE_FASHION_MODEL = False
    fashion_model = os.path.join(_DIR, 'fashion-sentiment-model-hm')
    sent_model = (fashion_model
                  if USE_FASHION_MODEL and os.path.isdir(fashion_model)
                  else 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    print(f"  Sentiment model: {os.path.basename(sent_model)}")
    pipe = hf_pipeline('sentiment-analysis', model=sent_model,
                       truncation=True, max_length=512)

    scores, labels = [], []
    texts = df['cleaned_text'].tolist()
    for i in range(0, len(texts), 64):
        for r in pipe([t[:512] for t in texts[i:i + 64]]):
            lbl  = r['label'].lower()
            conf = float(r['score'])
            labels.append(lbl)
            scores.append(conf if 'pos' in lbl else (-conf if 'neg' in lbl else 0.0))
        print(f"    sentiment {min(i + 64, len(texts))}/{len(texts)}")
    df['sentiment_label'] = labels
    df['sentiment_score'] = scores
    df.to_json(SENTIMENT_CACHE, orient='records', force_ascii=False, indent=4)
    print(f"  Saved: {SENTIMENT_CACHE}")
    return df


def weighted_mean(scores, likes):
    likes = likes.fillna(0).clip(lower=0)
    if (likes > 0).mean() >= 0.05:
        return float(np.average(scores, weights=1.0 + np.log1p(likes)))
    return float(scores.mean())


def pti_index_temporal(df):
    df = df.copy()
    df['period'] = df['date'].dt.to_period('Q')
    pattern = '|'.join(KW_PTQ)
    tau = {}
    for period, group in df.groupby('period'):
        sub = group[group['cleaned_text'].str.contains(pattern, case=False, na=False)]
        if len(sub) < MIN_DOC:
            continue
        tau[period.to_timestamp()] = weighted_mean(sub['sentiment_score'], sub['like'])
    tau_series = pd.Series(tau).sort_index()
    if tau_series.empty:
        return tau_series
    tau0 = tau_series.iloc[0]
    return (100 + 100 * (tau_series - tau0)).round(2)


def plot_pti(pti_ts):
    fig, ax = plt.subplots(figsize=(14, 7))
    if len(pti_ts) >= 12:
        ax.plot(pti_ts.index, pti_ts.values, color=C_POS, linewidth=1,
                marker='o', markersize=4, alpha=0.35, label='PTI', zorder=3)
        trend = pti_ts.rolling(4, center=True, min_periods=2).mean()
        ax.plot(trend.index, trend.values, color=C_POS, linewidth=2.6,
                label='PTI (rolling trend)', zorder=4)
    else:
        ax.plot(pti_ts.index, pti_ts.values, color=C_POS, linewidth=2.6,
                marker='o', markersize=7, label='PTI', zorder=4)
    ax.axhline(100, color='black', linewidth=0.8, linestyle='--',
               alpha=0.6, label='Base 100', zorder=2)

    xform = ax.get_xaxis_transform()
    for i, (date_str, label) in enumerate(EVENTS.items()):
        date_dt = pd.to_datetime(date_str)
        ax.axvline(date_dt, color='#777777', linewidth=0.8,
                   linestyle=':', alpha=0.65, zorder=2)
        y_frac = 0.96 if i % 2 == 0 else 0.74
        ax.text(date_dt, y_frac, label, transform=xform, rotation=90,
                va='top', ha='right', fontsize=8, color='#444444',
                linespacing=1.4)

    ax.set_xlabel('Quarter')
    ax.set_ylabel('PTI (base 100)')
    ax.set_title('Premiumization Tolerance Index – Temporal Evolution\n'
                 'H&M 2019–2026', pad=12)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'pti_time_series.pdf')
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true',
                    help='reuse comments_sentiment_hm.json without recomputing')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.reuse and os.path.exists(SENTIMENT_CACHE):
        print("Reusing already computed sentiment (comments_sentiment_hm.json)...")
        df = load_cached_sentiment()
    else:
        print("Loading the H&M corpus...")
        df = load_corpus()
        print(f"  Corpus: {len(df)} comments (from {DATE_FROM})")
        print("Computing sentiment...")
        df = run_sentiment(df)

    dist = df['sentiment_label'].value_counts()
    print("\nSentiment distribution:")
    for lbl, cnt in dist.items():
        print(f"  {lbl:<12}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")
    print(f"  Mean score : {df['sentiment_score'].mean():+.4f}")

    print("\nPer-year diagnostic (price/quality comments):")
    pattern = '|'.join(KW_PTQ)
    sub = df[df['cleaned_text'].str.contains(pattern, case=False, na=False)].copy()
    sub['year'] = sub['date'].dt.year
    for y, g in sub.groupby('year'):
        w = 1.0 + np.log1p(g['like'].fillna(0).clip(lower=0))
        print(f"  {y}: n={len(g):5d} | sentiment={g['sentiment_score'].mean():+.3f} "
              f"| weighted tau={np.average(g['sentiment_score'], weights=w):+.3f}")

    pti_ts = pti_index_temporal(df)
    pti_last = float(pti_ts.iloc[-1]) if not pti_ts.empty else None
    print(f"\n{'='*55}\n  PREMIUMIZATION TOLERANCE INDEX H&M (base 2019 = 100)\n{'='*55}")
    print(pti_ts.to_string())
    print(f"\n  Last period: {pti_last}")
    print("  PTI > 100 -> tolerance above baseline | PTI < 100 -> erosion")

    with open(os.path.join(_DIR, 'pti_results_hm.json'), 'w', encoding='utf-8') as f:
        json.dump({'PTI_last': pti_last,
                   'PTI_series': {str(k.date()): float(v)
                                  for k, v in pti_ts.items()}},
                  f, indent=4, ensure_ascii=False)
    print("  Saved: pti_results_hm.json")

    plot_pti(pti_ts)


if __name__ == '__main__':
    main()
