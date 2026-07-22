"""
Full pipeline - Zara Premiumization Analysis (English-only, unified corpus)
Champion-vs-Challenger: LDA (baseline) -> BERTopic (champion)
-> RoBERTa sentiment -> ABSA -> visualisations -> PTI time-series

The corpus is a single English-language dataset: native English comments plus
Italian comments machine-translated into English (see translate_italian_to_english.py).

Input : all_english.json + all_italian_translated.json
        (schema: [{date, comments:[{comment, like}]}])
Output: vector PDFs + CSV of the results
"""

import json
import re
import os
import sys
import copy
import warnings
from collections import Counter
from multiprocessing import freeze_support
warnings.filterwarnings("ignore")


class _Tee:
    """Write to both the terminal and a log file at the same time."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.corpus import stopwords as _nltk_sw
    _NLTK_STOP = set(_nltk_sw.words('english'))
except Exception:
    _NLTK_STOP = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'will', 'just', 'now', 'ain', 'aren',
        'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    }
    print("  WARNING: NLTK stopwords not available offline, using built-in fallback.")


class _StopWrapper:
    """Drop-in replacement for nltk.corpus.stopwords."""
    def words(self, lang='english'):
        return list(_NLTK_STOP)


stopwords = _StopWrapper()

from sklearn.feature_extraction.text import CountVectorizer

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from transformers import pipeline as hf_pipeline

import spacy
_NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
_NLP.max_length = 10_000_000

# Generic fashion/e-commerce words: too frequent to distinguish topics.
GENERIC = {
    'everyone', 'comment', 'good', 'really', 'help', 'bad', 'hoodie', 'long',
    'buying', 'totally', 'cap', 'work', 'www', 'already', 'shop', 'trouser',
    'over', 'much', 'actually', 'feel', 'time', 'sneaker', 'wardrobe', 'nope',
    'point', 'wanna', 'item', 'collar', 'yeah', 'thing', 'let', 'please',
    'kinda', 'anna', 'jean', 'yes', 'nonessential', 'type', 'right', 'belt',
    'one', 'think', 'shoe', 'made', 'etc', 'http', 'blazer', 'still',
    'anonymized', 'anything', 'aspect', 'wear', 'kind', 'person', 'buy',
    'yet', 'number', 'end', 'hat', 'post', 'then', 'back', 'cloth', 'gonna',
    'thank', 'want', 'anyone', 'getting', 'last', 'sock', 'ok', 'say',
    'okay', 'piece', 'blouse', 'get', 'cardigan', 'something', 'cause',
    'know', 'dress', 'shirt', 'pretty', 'try', 'new', 'look', 'seem',
    'show', 'day', 'very', 'week', 'sleeve', 'yesterday', 'love', 'year',
    'fact', 'removed', 'clothing', 'like', 'honestly', 'place', 'outfit',
    'coat', 'zara', 'everything', 'find', 'start', 'would', 'sandal',
    'skirt', 'scarf', 'keep', 'amp', 'just', 'top', 'share', 'deleted',
    'sweater', 'bag', 'too', 'used', 'same', 'heel', 'stuff', 'form',
    'pocket', 'today', 'always', 'things', 'even', 'could', 'redacted',
    'bought', 'now', 'every', 'month', 'someone', 'wearing', 'also',
    'watch', 'store', 'said', 'going', 'line', 'gotta', 'put', 'set',
    'though', 'follow', 'page', 'take', 'use', 'redact', 'tell', 'come',
    'give', 'side', 'sort', 'sure', 'bit', 'shein', 'ask', 'often',
    'reply', 'thanks', 'way', 'lol', 'ago', 'soon', 'video', 'boot',
    'omg', 'case', 'never', 'old', 'next', 'well', 'pant', 'reason',
    'make', 'literally', 'little', 'brand', 'lot', 'purse', 'around',
    'people', 'tshirt', 'mean', 'legging', 'need', 'jacket', 'see', 'go',
    'clothes', 'got', 'either', 'part', 'because', 'collection', 'basically',
}

# Contractions -> expanded forms (applied before phrase normalisation).
_CONTRACTIONS = [
    (re.compile(r"\bisn't\b", re.I), 'is not'),
    (re.compile(r"\baren't\b", re.I), 'are not'),
    (re.compile(r"\bwasn't\b", re.I), 'was not'),
    (re.compile(r"\bweren't\b", re.I), 'were not'),
    (re.compile(r"\bcannot\b", re.I), 'can not'),
    (re.compile(r"\bcan't\b", re.I), 'can not'),
    (re.compile(r"\bcouldn't\b", re.I), 'could not'),
    (re.compile(r"\bwon't\b", re.I), 'will not'),
    (re.compile(r"\bwouldn't\b", re.I), 'would not'),
    (re.compile(r"\bdoesn't\b", re.I), 'does not'),
    (re.compile(r"\bdon't\b", re.I), 'do not'),
    (re.compile(r"\bdidn't\b", re.I), 'did not'),
    (re.compile(r"\bhaven't\b", re.I), 'have not'),
    (re.compile(r"\bhasn't\b", re.I), 'has not'),
    (re.compile(r"\bhadn't\b", re.I), 'had not'),
    (re.compile(r"\bshouldn't\b", re.I), 'should not'),
    (re.compile(r"\bmustn't\b", re.I), 'must not'),
    (re.compile(r"\bneedn't\b", re.I), 'need not'),
]

# Multi-word sentiment phrases -> single polarity-bearing tokens.
# Keeps negation scope ("not good quality" -> "substandard") so that the
# keyword-based ABSA and the topic models see one unambiguous token.
_PHRASE_NORMS = [
    (re.compile(r"\bnot\s+(?:very\s+)?(?:high|good)\s+quality\b", re.I), 'substandard'),
    (re.compile(r"\bnot\s+(?:look|seem|feel|appear)\s+(?:good|great|nice)\b", re.I), 'poor'),
    (re.compile(r"\bnot\s+(?:very\s+)?(?:good|great|nice|impressive|wonderful|amazing|perfect|fine)\b", re.I), 'poor'),
    (re.compile(r"\bnot\s+worth\b", re.I), 'worthless'),
    (re.compile(r"\bnot\s+afford\b", re.I), 'overpriced'),
    (re.compile(r"\bnot\s+(?:affordable|cheap|inexpensive)\b", re.I), 'overpriced'),
    (re.compile(r"\bnot\s+(?:satisfied|happy|pleased|content)\b", re.I), 'dissatisfied'),
    (re.compile(r"\bnot\s+(?:impressed|impressive)\b", re.I), 'disappointing'),
    (re.compile(r"\bnot\s+recommend\b", re.I), 'disappointing'),
    (re.compile(r"\bnot\s+(?:bad|terrible|awful|horrible|dreadful)\b", re.I), 'decent'),
    (re.compile(r"\bnot\s+(?:expensive|overpriced|pricey)\b", re.I), 'affordable'),
    (re.compile(r"\b(?:very|really|so|too|extremely|absolutely)\s+bad\b", re.I), 'terrible'),
    (re.compile(r"\b(?:very|really|so|too|extremely|absolutely)\s+expensive\b", re.I), 'overpriced'),
    (re.compile(r"\b(?:very|really|so|too|extremely)\s+poor\b", re.I), 'terrible'),
    (re.compile(r"\b(?:very|really|so|too|extremely)\s+cheap\b", re.I), 'shoddy'),
    (re.compile(r"\b(?:very|really|so|extremely|absolutely)\s+(?:good|great|nice|wonderful|amazing|perfect)\b", re.I), 'excellent'),
    (re.compile(r"\b(?:very|really|so|extremely)\s+(?:high|good)\s+quality\b", re.I), 'premium'),
    (re.compile(r"\b(?:very|really|so|extremely)\s+affordable\b", re.I), 'affordable'),
    (re.compile(r"\b(?:do not|never|stop|avoid|would not|could not)\s+(?:buy|buying|purchase|purchasing|shop|shopping)\b", re.I), 'boycott'),
    (re.compile(r"\btoo\s+much\s+(?:expensive|pricey|costly)\b", re.I), 'overpriced'),
    (re.compile(r"\btoo\s+much\s+money\b", re.I), 'overpriced'),
    (re.compile(r"\b(?:costs?|paying|paid|spend|spent)\s+too\s+much\b", re.I), 'overpriced'),
    (re.compile(r"\bwaste\s+of\s+money\b", re.I), 'worthless'),
    (re.compile(r"\brip\s*-?\s*off\b", re.I), 'overpriced'),
]


def normalize_phrases(text):
    """Expand contractions, then replace multi-word sentiment phrases with single tokens."""
    text = str(text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _PHRASE_NORMS:
        text = pattern.sub(replacement, text)
    return text


def make_lemma_tokenizer():
    def tokenizer(doc):
        doc = normalize_phrases(doc)
        return [t.lemma_.lower() for t in _NLP(doc)
                if t.is_alpha and len(t.lemma_) > 2
                and t.lemma_.lower() not in GENERIC]
    return tokenizer


# ── Global style ─────────────────────────────────────────────
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

FMT   = 'pdf'
C_POS = '#2166ac'
C_NEG = '#d6604d'
C_NEU = '#d9d9d9'

_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, 'results')

CORPUS_FILES = [
    ['../file finale/English_cleaned_comments.json'],
    ['../file finale/English_comments_Youtube.json'],
    ['../file finale/English_comments_reddit.json'],
    ['all_italian_translated.json'],
]

DATE_FROM = '2019-01-01'

# Keep also the comments that do not name the brand explicitly: they were
# scraped from brand-specific videos/threads, so they still talk about Zara.
RELEVANCE_FILTER = False

TIME_FREQ = 'Q'

# Numero massimo di topic SOLO per il grafico Champion vs Challenger.
# Il modello "ufficiale" usato ovunque altrove (df['topic'], topic_info.csv,
# eventuali analisi/plot futuri per topic) NON viene toccato da questo cap:
# mantiene il numero di topic scelto naturalmente da HDBSCAN.
BERT_TOPIC_CAP = 5

EVENTS = {
    '2022-01-01': 'Marta Ortega\nas Chair',
    '2022-09-01': 'Zara Studio\nlaunch',
    '2023-03-01': 'Galliano\ncollaboration',
    '2023-09-01': 'Price increase\n+15%',
    '2024-03-01': 'Narciso Rodriguez\ncollaboration',
    '2026-05-01': 'Bad Bunny\ncollaboration',
}

# Cumulative Zara price increase (%) vs 2019, used by the tolerance analysis.
PRICE_INCREASE = {
    '2019-01-01': 0,
    '2020-01-01': 5,
    '2021-01-01': 12,
    '2022-01-01': 20,
    '2023-01-01': 28,
    '2024-01-01': 35,
    '2025-01-01': 45,
}


# ============================================================
# FUNCTIONS
# ============================================================

def _resolve(candidates):
    """Return the first existing path from a list of candidate names (relative to _DIR)."""
    for name in candidates:
        path = os.path.normpath(os.path.join(_DIR, name))
        if os.path.exists(path):
            return path
    return None


def load_and_flatten(corpus_files):
    """Load one or more grouped JSON files and flatten into a single DataFrame.
Each item in corpus_files is a list of candidate filenames. Comment-list and
comment-text keys are resolved flexibly ('comments'/'commenti', 'comment'/'commento')
so both the new English schema and the legacy Italian one are accepted."""
    rows = []
    for candidates in corpus_files:
        path = _resolve(candidates)
        if path is None:
            print(f"  WARNING: none found, skipped: {candidates}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for group in data:
            clk = 'comments' if 'comments' in group else 'commenti'
            for c in group[clk]:
                ctk = 'comment' if 'comment' in c else 'commento'
                rows.append({'date': group['date'],
                             'cleaned_text': c[ctk],
                             'like': c.get('like', 0)})
        print(f"  loaded: {os.path.basename(path)}")
    return pd.DataFrame(rows)


def preprocess_for_lda(documents, stop):
    result = []
    for doc in _NLP.pipe((normalize_phrases(d) for d in documents), batch_size=256):
        tokens = [t.lemma_.lower() for t in doc
                  if t.is_alpha and len(t.lemma_) > 2
                  and t.lemma_.lower() not in stop]
        if tokens:
            result.append(tokens)
    bigram = Phraser(Phrases(result, min_count=5, threshold=10))
    return [bigram[doc] for doc in result]


def run_lda(documents, stop):
    print(f"\n{'─'*55}\n  LDA Baseline  ({len(documents)} doc)\n{'─'*55}")
    if len(documents) < 20:
        print("  Too few documents. Skip.")
        return None, None, None, None

    tokens     = preprocess_for_lda(documents, stop)
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokens]

    best_cv, best_n, best_model = -1.0, None, None
    for n in [5, 7, 10, 12, 15]:
        m  = LdaModel(corpus, num_topics=n, id2word=dictionary,
                      passes=15, random_state=42, alpha='auto')
        cv = round(CoherenceModel(model=m, texts=tokens, dictionary=dictionary,
                                  coherence='c_v').get_coherence(), 4)
        print(f"    n_topics={n:2d} -> c_v={cv:.4f}")
        if cv > best_cv:
            best_cv, best_n, best_model = cv, n, m

    print(f"\n  Best LDA: n_topics={best_n}, c_v={best_cv:.4f}")
    for idx, topic in best_model.print_topics(num_words=5):
        print(f"    Topic {idx}: {topic}")
    return best_model, best_cv, best_n, (tokens, dictionary)


def compute_bertopic_coherence(topic_model, tokens, dictionary):
    topic_words = []
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid)[:10] if w]
        if words:
            topic_words.append(words)
    if not topic_words:
        return None
    try:
        return round(CoherenceModel(topics=topic_words, texts=tokens,
                                    dictionary=dictionary,
                                    coherence='c_v').get_coherence(), 4)
    except Exception:
        return None


def run_bertopic(documents, stop, lda_aux):
    print(f"\n{'─'*55}\n  BERTopic Champion  ({len(documents)} doc)\n{'─'*55}")
    if len(documents) < 10:
        print("  Too few documents. Skip.")
        return None, [-1] * len(documents), None

    n_neighbors = min(15, len(documents) - 1)
    min_cluster = max(50, len(documents) // 150)
    stop_words  = list(stop)

    def new_vectorizer():
        return CountVectorizer(tokenizer=make_lemma_tokenizer(),
                               stop_words=stop_words,
                               min_df=5, ngram_range=(1, 2))

    model = BERTopic(
        language='english', nr_topics=None,
        umap_model=UMAP(n_components=5, n_neighbors=n_neighbors,
                        min_dist=0.0, random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster, metric='euclidean',
                              cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=new_vectorizer(),
        calculate_probabilities=False, verbose=False
    )
    try:
        topics, _ = model.fit_transform(documents)
        topics = model.reduce_outliers(documents, topics, strategy='c-tf-idf')
        model.update_topics(documents, topics=topics,
                            vectorizer_model=new_vectorizer())
    except Exception as e:
        print(f"  BERTopic failed: {e}")
        return None, [-1] * len(documents), None

    info  = model.get_topic_info()
    valid = info[info['Topic'] != -1]
    print(f"  Topics found: {len(valid)}")
    for _, row in valid.iterrows():
        top3 = ', '.join([w for w, _ in model.get_topic(row['Topic'])[:3]])
        print(f"    Topic {row['Topic']:2d} ({row['Count']:4d} doc) → {top3}")

    bert_cv = None
    if lda_aux:
        tokens, dictionary = lda_aux
        bert_cv = compute_bertopic_coherence(model, tokens, dictionary)
        print(f"\n  BERTopic c_v = {bert_cv}")
    return model, topics, bert_cv


def analyze_sentiment(text, pipe):
    try:
        res   = pipe(str(text)[:512])[0]
        label = res['label'].lower()
        conf  = float(res['score'])
        if 'positive' in label or label == 'pos':
            return label, conf, conf
        elif 'negative' in label or label == 'neg':
            return label, conf, -conf
        return label, conf, 0.0
    except Exception:
        return 'neutral', 0.0, 0.0


def _weighted_mean(scores, likes):
    """Engagement-weighted mean sentiment.
Falls back to a simple mean when fewer than 5 % of comments have a positive
like count — this happens when the scraper did not capture like data (all zeros),
making the weighting formula meaningless."""
    likes = likes.fillna(0).clip(lower=0)
    if (likes > 0).mean() >= 0.05:
        return float(np.average(scores, weights=1.0 + np.log1p(likes)))
    return float(scores.mean())


def compute_absa(df, aspects, min_volume=30):
    rows = []
    for asp, kw in aspects.items():
        pattern = '|'.join(kw)
        sub = df[df['cleaned_text'].str.contains(pattern, case=False, na=False)]
        if len(sub) < min_volume:
            continue
        if 'like' in sub.columns:
            sentiment_w = _weighted_mean(sub['sentiment_score'], sub['like'])
        else:
            sentiment_w = float(sub['sentiment_score'].mean())
        rows.append({
            'aspect':    asp,
            'sentiment': round(float(sentiment_w), 4),
            'pct_pos':   round((sub['sentiment_score'] >  0.1).mean() * 100, 1),
            'pct_neg':   round((sub['sentiment_score'] < -0.1).mean() * 100, 1),
            'pct_neu':   round((sub['sentiment_score'].abs() <= 0.1).mean() * 100, 1),
            'volume':    int(len(sub)),
            'pct':       round(len(sub) / len(df) * 100, 1),
        })
    if not rows:
        return pd.DataFrame(columns=['aspect', 'sentiment', 'pct_pos',
                                     'pct_neg', 'pct_neu', 'volume', 'pct'])
    return pd.DataFrame(rows).sort_values('sentiment', ascending=False)


def compute_pti(absa_df, positive_aspects, negative_aspects):
    pos = absa_df[absa_df['aspect'].isin(positive_aspects)]['sentiment'].mean()
    neg = absa_df[absa_df['aspect'].isin(negative_aspects)]['sentiment'].mean()
    if pd.isna(pos) or pd.isna(neg) or neg >= 0:
        return None
    return round(float(pos) / abs(float(neg)), 4)


def tau_ptq(df, kw_ptq):
    """Price-to-quality tolerance signal (tau in [-1, 1]): ENGAGEMENT-WEIGHTED
mean sentiment of the comments that evaluate price and/or value
(quality, materials, design, collaborations)."""
    pattern = '|'.join(kw_ptq)
    sub = df[df['cleaned_text'].str.contains(pattern, case=False, na=False)]
    if len(sub) == 0:
        return None
    if 'like' in sub.columns:
        return _weighted_mean(sub['sentiment_score'], sub['like'])
    return float(sub['sentiment_score'].mean())


def pti_index_temporal(df, kw_ptq, freq='Q', min_doc=15):
    """PTI(t) = 100 + 100*(tau_t - tau_baseline).
A period is kept only if it has at least `min_doc` price/value comments.
tau0 = first observed period (professor's formula)."""
    df = df.copy()
    df['period'] = df['date'].dt.to_period(freq)
    pattern = '|'.join(kw_ptq)
    tau = {}
    for period, group in df.groupby('period'):
        sub = group[group['cleaned_text'].str.contains(pattern, case=False, na=False)]
        if len(sub) < min_doc:
            continue
        t = tau_ptq(group, kw_ptq)
        if t is None:
            continue
        tau[period.to_timestamp()] = t
    tau_series = pd.Series(tau).sort_index()
    if tau_series.empty:
        return tau_series
    tau0 = tau_series.iloc[0]
    return (100 + 100 * (tau_series - tau0)).round(2)


def relative_tolerance(df_zara, df_other, kw_ptq, freq='Q', min_doc=5):
    """Cross-brand (e.g. Zara vs H&M): RT(t) = tau_Zara(t) - tau_other(t).
RT > 0 -> Zara more tolerated. df_other must share the same schema
(date, cleaned_text, like, sentiment_score)."""
    def tau_series(df):
        out = {}
        df = df.copy()
        df['period'] = df['date'].dt.to_period(freq)
        for p, g in df.groupby('period'):
            if len(g) < min_doc:
                continue
            t = tau_ptq(g, kw_ptq)
            if t is not None:
                out[p.to_timestamp()] = t
        return pd.Series(out).sort_index()

    a   = tau_series(df_zara)
    b   = tau_series(df_other)
    idx = a.index.union(b.index)
    return (a.reindex(idx) - b.reindex(idx)).dropna().round(4)


def tolerance_threshold(pti_series, price_increase, threshold=100.0,
                        search_from='2022-01-01'):
    """BREAKING POINT: 'up to what price increase does tolerance hold?'.
Finds the first period (from `search_from` onward, to skip the pre-pivot and the
2020 COVID anomaly) with PTI < threshold and returns (period, p*, PTI), where
p* = cumulative price increase (% vs 2019) at that period."""
    if pti_series is None or pti_series.empty:
        return None, None, None
    pr = pd.Series({pd.to_datetime(k): v
                    for k, v in price_increase.items()}).sort_index()
    cutoff = pd.to_datetime(search_from)
    for period, pti in pti_series.items():
        if period < cutoff:
            continue
        if pti < threshold:
            valid  = pr[pr.index <= period]
            p_star = float(valid.iloc[-1]) if len(valid) else None
            return period, p_star, float(pti)
    return None, None, None


def normalize_base100(series):
    if series.empty:
        return series
    return series / series.iloc[0] * 100


# ── Visualisations ───────────────────────────────────────────

def plot_champion_vs_challenger(lda_model, lda_n, lda_cv, bert_model, bert_cv):
    """Side-by-side topic word table: reader judges quality visually."""
    lda_topics = []
    if lda_model:
        for _, topic_str in lda_model.print_topics(num_words=6):
            lda_topics.append(re.findall(r'"([^"]+)"', topic_str))

    bert_topics = []
    if bert_model:
        info = bert_model.get_topic_info()
        info = info[info['Topic'] != -1].nlargest(10, 'Count')
        for _, row in info.iterrows():
            words = [w for w, _ in bert_model.get_topic(row['Topic'])[:6] if w]
            if words:
                bert_topics.append(words)

    n_rows = max(len(lda_topics), len(bert_topics), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 1.8 + n_rows * 0.52))

    panels = [
        (axes[0], lda_topics,
         f'LDA  ·  Baseline\n{lda_n} topics' +
         (f'  ·  c_v={lda_cv:.3f}' if lda_cv else ''),
         '#d0e4f7', '#1a1a1a'),
        (axes[1], bert_topics,
         f'BERTopic  ·  Champion\n{len(bert_topics)} topics' +
         (f'  ·  c_v={bert_cv:.3f}' if bert_cv else ''),
         '#2166ac', 'white'),
    ]
    for ax, topics, header, bg, fg in panels:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_facecolor('#f7f9fc')
        for i, words in enumerate(topics):
            y = n_rows - 1 - i
            ax.text(0.03, y, f'{i + 1}.', fontsize=9, fontweight='bold',
                    va='center', color='#555')
            ax.text(0.12, y, '  ·  '.join(words), fontsize=9,
                    va='center', color='#111')
        ax.set_title(header, fontsize=10, fontweight='bold', pad=10, color=fg,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor=bg,
                               edgecolor='none'))
        ax.axis('off')
    plt.suptitle('Champion vs Challenger – Representative Topic Words',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'champion_vs_challenger.{FMT}')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def real_examples_test(examples, lda_model, dictionary, stop, bert_model, roberta_pipe):
    """Real-comment comparison test: how VADER, RoBERTa, LDA and BERTopic
    each treat the same verbatim comments from the corpus."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # LDA: same preprocessing used in training (lemma + stopwords)
    toks = preprocess_for_lda(examples, stop)

    # BERTopic: topic assignment for the examples
    if bert_model is not None:
        bert_topics, _ = bert_model.transform(examples)
    else:
        bert_topics = [None] * len(examples)

    rows = []
    for i, text in enumerate(examples):
        vc = sia.polarity_scores(text)['compound']
        vlab = ('positive' if vc >= 0.05 else
                'negative' if vc <= -0.05 else 'neutral')

        r = roberta_pipe(text)[0]
        rlab = r['label'].lower()

        # Only compute bow if a gensim dictionary is available
        bow = []
        if dictionary is not None and i < len(toks) and toks[i]:
            bow = dictionary.doc2bow(toks[i])
        if bow and lda_model is not None:
            ldist = lda_model.get_document_topics(bow)
            ltop, lp = max(ldist, key=lambda x: x[1])
            lda_cell = f"{ltop} ({lp:.0%})"
        else:
            lda_cell = "—"

        rows.append({
            'comment':        text[:80] + ('…' if len(text) > 80 else ''),
            'VADER':          f"{vlab} ({vc:+.2f})",
            'RoBERTa':        f"{rlab} ({r['score']:.2f})",
            'LDA topic':      lda_cell,
            'BERTopic topic': (int(bert_topics[i])
                               if bert_topics[i] is not None else '—'),
        })

    out = pd.DataFrame(rows)
    print(f"\n{'='*55}\n  REAL-COMMENT TEST  "
          f"(VADER vs RoBERTa vs LDA vs BERTopic)\n{'='*55}")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(_DIR, 'real_examples_test.csv'),
               index=False, encoding='utf-8-sig')
    print("  Saved: real_examples_test.csv")


def diverging_bar(absa_df, title, output_path):
    df_plot = absa_df.sort_values('sentiment').reset_index(drop=True)
    aspects = df_plot['aspect'].tolist()
    y_pos   = np.arange(len(aspects))
    fig, ax = plt.subplots(figsize=(10, max(5, len(aspects) * 0.65)))
    ax.barh(y_pos, -df_plot['pct_neg'], color=C_NEG, label='Negative (%)',
            edgecolor='white', linewidth=0.5, zorder=3)
    ax.barh(y_pos, df_plot['pct_neu'],
            left=-(df_plot['pct_neg'] + df_plot['pct_neu']),
            color=C_NEU, label='Neutral (%)', edgecolor='white',
            linewidth=0.5, zorder=3)
    ax.barh(y_pos, df_plot['pct_pos'], color=C_POS, label='Positive (%)',
            edgecolor='white', linewidth=0.5, zorder=3)
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(51, i, f"mean score: {row['sentiment']:+.2f}",
                va='center', fontsize=8, color='#555555')
    ax.axvline(0, color='black', linewidth=0.8, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(aspects, fontsize=10)
    ax.set_xlabel('Comment distribution (%)')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{abs(x):.0f}%'))
    ax.set_xlim(-105, 115)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def network_graph(absa_df, title, output_path):
    """Star network: Zara at centre, aspect nodes on outer ring, coloured by sentiment."""
    from matplotlib import colors as mcolors

    center   = 'Zara'
    sent_map = absa_df.set_index('aspect')['sentiment'].to_dict()
    vol_map  = absa_df.set_index('aspect')['volume'].to_dict()

    G = nx.Graph()
    G.add_node(center, type='brand')
    for _, row in absa_df.iterrows():
        G.add_node(row['aspect'], sentiment=float(row['sentiment']),
                   volume=int(row['volume']))
        G.add_edge(center, row['aspect'], weight=row['volume'])

    graphml_path = output_path.replace(f'.{FMT}', '.graphml')
    nx.write_graphml(G, graphml_path)
    print(f"  Saved GraphML: {graphml_path}")

    shells = [[center], list(absa_df['aspect'])]
    pos    = nx.shell_layout(G, shells)

    cmap = plt.cm.RdBu
    norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)

    node_colors, node_sizes = [], []
    for node in G.nodes():
        if node == center:
            node_colors.append('#333333')
            node_sizes.append(6000)
        else:
            s   = float(sent_map.get(node, 0))
            vol = int(vol_map.get(node, 0))
            node_colors.append(cmap(norm(s)))
            node_sizes.append(1200 + vol * 5)

    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.93)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_family='serif',
                            font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#aaaaaa',
                           width=1.4, alpha=0.65)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_label('Mean Sentiment Score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def _lemma_counts(docs, stop):
    c = Counter()
    for doc in _NLP.pipe((normalize_phrases(d) for d in docs), batch_size=256):
        c.update(t.lemma_.lower() for t in doc
                 if t.is_alpha and len(t.lemma_) > 2
                 and t.lemma_.lower() not in stop)
    return c


def wordcloud_sentiment(df, positive, title, output_path):
    stop = set(stopwords.words('english')) | GENERIC
    pos_docs = df[df['sentiment_score'] >  0.1]['cleaned_text'].dropna().astype(str).tolist()
    neg_docs = df[df['sentiment_score'] < -0.1]['cleaned_text'].dropna().astype(str).tolist()
    if not pos_docs or not neg_docs:
        print(f"  Not enough data for '{title}'. Skip.")
        return

    cpos = _lemma_counts(pos_docs, stop)
    cneg = _lemma_counts(neg_docs, stop)
    tot_pos = sum(cpos.values()) or 1
    tot_neg = sum(cneg.values()) or 1
    alpha     = 1.0
    MIN_COUNT = 30

    # Log-odds: keep only the words genuinely DISTINCTIVE of one polarity,
    # weighted by how often they occur.
    scored = {}
    for w in set(cpos) | set(cneg):
        total = cpos[w] + cneg[w]
        if total < MIN_COUNT:
            continue
        log_odds = (np.log((cpos[w] + alpha) / tot_pos)
                    - np.log((cneg[w] + alpha) / tot_neg))
        if (positive and log_odds > 0) or (not positive and log_odds < 0):
            scored[w] = abs(log_odds) * np.sqrt(total)
    if not scored:
        print(f"  No distinctive words for '{title}'. Skip.")
        return

    freq = dict(sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:80])
    wc = WordCloud(width=1400, height=700, background_color='white',
                   colormap='Blues' if positive else 'Reds',
                   prefer_horizontal=0.9, max_font_size=80,
                   collocations=False).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_lda_topics(lda_model, lda_n, output_path):
    """2-column grid: each cell is a topic with its top-word probability bars."""
    if lda_model is None:
        return
    TOPN  = 8
    NCOLS = 2
    nrows = (lda_n + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(13, 2.2 + nrows * 1.2))
    axes = np.array(axes).flatten()
    base_colors = plt.cm.tab10.colors

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= lda_n:
            ax.set_visible(False)
            continue
        topic = lda_model.show_topic(idx, topn=TOPN)
        words = [w for w, _ in topic][::-1]
        probs = [p for _, p in topic][::-1]
        color = base_colors[idx % len(base_colors)]
        bars  = ax.barh(words, probs, color=color, alpha=0.78, height=0.6)
        for bar, p in zip(bars, probs):
            ax.text(p + max(probs) * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f'{p:.3f}', va='center', fontsize=7.5, color='#444')
        ax.set_xlim(0, max(probs) * 1.45)
        ax.set_title(f'Topic {idx + 1}', fontsize=10, fontweight='bold',
                     loc='left', pad=3, color=color)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='y', labelsize=8.5)

    plt.suptitle(f'LDA Baseline – Top Words per Topic  (n = {lda_n})',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_vader_vs_roberta(df, output_path):
    """Champion-vs-challenger for sentiment. Left: how each method labels the comments
(grouped bars) — shows RoBERTa is more decisive while VADER leaves more neutral.
Right: score distributions. Agreement % and Cohen's kappa are reported in the title."""
    from sklearn.metrics import cohen_kappa_score, accuracy_score

    print("  Computing VADER scores for comparison...")
    sia = SentimentIntensityAnalyzer()
    df  = df.copy()
    df['vader_compound'] = df['cleaned_text'].apply(
        lambda t: sia.polarity_scores(str(t))['compound'])
    df['vader_label'] = df['vader_compound'].apply(
        lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral'))

    order = ('negative', 'neutral', 'positive')
    agree = accuracy_score(df['vader_label'], df['sentiment_label']) * 100
    kappa = cohen_kappa_score(df['vader_label'], df['sentiment_label'])

    vader_pct   = [(df['vader_label'] == c).mean() * 100 for c in order]
    roberta_pct = [(df['sentiment_label'] == c).mean() * 100 for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x  = np.arange(len(order))
    bw = 0.38
    b1 = ax.bar(x - bw / 2, vader_pct,   bw, color='#9ecae1', label='VADER (baseline)')
    b2 = ax.bar(x + bw / 2, roberta_pct, bw, color=C_POS,     label='RoBERTa (champion)')
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f'{bar.get_height():.0f}%',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in order])
    ax.set_ylabel('% of comments')
    ax.set_title(f'How each method labels the comments\n'
                 f'{agree:.0f}% agree · Cohen’s κ = {kappa:.2f}', fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    ax = axes[1]
    ax.hist(df['vader_compound'],  bins=60, alpha=0.6, color='#aec7e8',
            label='VADER compound', density=True)
    ax.hist(df['sentiment_score'], bins=60, alpha=0.6, color=C_POS,
            label='RoBERTa score', density=True)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.axvline(0, color='black', linewidth=0.4, alpha=0.6)
    ax.legend()

    plt.suptitle('Baseline (VADER) vs Champion (RoBERTa) – Sentiment Comparison',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sentiment_heatmap(df, aspects, output_path):
    """Heatmap: rows = aspects, columns = quarters, cells = mean sentiment score."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    MIN_CELL = 10
    rows = []
    for asp, keywords in aspects.items():
        pattern = '|'.join(re.escape(k) for k in keywords)
        sub = df[df['cleaned_text'].str.contains(pattern, case=False, na=False)]
        if sub.empty:
            continue
        for y, g in sub.groupby('year'):
            if len(g) < MIN_CELL:
                continue
            rows.append({'aspect': asp, 'year': int(y),
                         'sentiment': g['sentiment_score'].mean()})
    if not rows:
        print("  Heatmap: no data. Skip.")
        return
    pivot = (pd.DataFrame(rows)
             .pivot(index='aspect', columns='year', values='sentiment')
             .sort_index())
    fig, ax = plt.subplots(
        figsize=(max(8, 1.1 * len(pivot.columns)), 0.7 * len(pivot) + 1.5))
    sns.heatmap(pivot, ax=ax, cmap='RdBu', center=0, vmin=-0.5, vmax=0.5,
                annot=True, fmt='.2f', linewidths=0.4, linecolor='#dddddd',
                cbar_kws={'label': 'Mean Sentiment Score', 'shrink': 0.8})
    ax.set_title('Sentiment Heatmap by Attribute and Year', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('')
    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pti_time_series(pti_ts):
    fig, ax = plt.subplots(figsize=(14, 7))
    if not pti_ts.empty:
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

    ax.set_xlabel('Year' if TIME_FREQ == 'Y' else 'Quarter')
    ax.set_ylabel('PTI (base 100)')
    ax.set_title('Premiumization Tolerance Index – Temporal Evolution\n'
                 'Zara 2019–2026', pad=12)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'pti_time_series.{FMT}')
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    freeze_support()
    os.chdir(_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Reproducibility
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    try:
        import torch
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    _log_file  = open(os.path.join(OUT_DIR, 'pipeline_log.txt'), 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    print("Loading data...")
    df = load_and_flatten(CORPUS_FILES)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[df['date'] >= pd.Timestamp(DATE_FROM)]
    df = df[df['date'] <= pd.Timestamp.now()]
    df['cleaned_text'] = (df['cleaned_text'].astype(str).str.strip()
                          .str.replace(r'\bzadar\b', 'zara', case=False, regex=True))
    df = df[df['cleaned_text'].str.len() > 0]

    # Relevance filter (disabled: the comments come from brand-specific
    # videos/threads, so even those that do not name the brand still
    # talk about it. Enable only to drop comments that mention ONLY a
    # competitor and never the brand).
    if RELEVANCE_FILTER:
        competitors = r'\b(uniqlo|shein|h ?& ?m|mango|primark|asos|boohoo|zalando|arket|& ?other ?stories|weekday|monki)\b'
        has_zara = df['cleaned_text'].str.contains('zara|inditex', case=False, na=False)
        has_comp = df['cleaned_text'].str.contains(competitors,   case=False, na=False)
        before_rel = len(df)
        df = df[has_zara | ~has_comp]
        print(f"  Relevance filter: removed {before_rel - len(df)} comments about other brands only")

    df = df.reset_index(drop=True)
    print(f"  Corpus: {len(df)} comments (from {DATE_FROM})")

    stop = set(stopwords.words('english')) | GENERIC

    # ── 1. LDA baseline ──────────────────────────────────────
    lda, lda_cv, lda_n, lda_aux = run_lda(df['cleaned_text'].tolist(), stop)

    # ── 2. BERTopic champion ─────────────────────────────────
    topic_model, topics, bert_cv = run_bertopic(df['cleaned_text'].tolist(),
                                                stop, lda_aux)
    df['topic'] = topics
    if topic_model:
        topic_model.get_topic_info().to_csv(
            os.path.join(_DIR, 'topic_info.csv'), index=False)

    # --- Copia ridotta SOLO per il confronto Champion vs Challenger ---
    # topic_model resta il modello "ufficiale" (usato per df['topic'],
    # topic_info.csv ed eventuali altre analisi per topic): il numero di
    # topic continua ad essere quello scelto naturalmente da HDBSCAN.
    # bert_model_for_plot è una copia indipendente, ridotta a un numero
    # massimo di topic, usata SOLO dentro plot_champion_vs_challenger.
    bert_model_for_plot = topic_model
    bert_cv_for_plot    = bert_cv
    if topic_model is not None:
        n_topics_found = topic_model.get_topic_info()['Topic'].nunique() - 1  # esclude outlier (-1)
        print(f"\n  Riduzione BERTopic a {BERT_TOPIC_CAP} topic (valore fisso) "
              f"SOLO per il grafico Champion vs Challenger "
              f"(modello ufficiale invariato: {n_topics_found} topic)")
        # reduce_topics conta anche il topic outlier (-1) dentro nr_topics,
        # quindi per ottenere BERT_TOPIC_CAP topic REALI nel grafico bisogna
        # chiedergliene uno in piu.
        bert_model_for_plot = copy.deepcopy(topic_model)
        bert_model_for_plot.reduce_topics(df['cleaned_text'].tolist(),
                                          nr_topics=BERT_TOPIC_CAP + 1)
        if lda_aux:
            tokens, dictionary = lda_aux
            bert_cv_for_plot = compute_bertopic_coherence(
                bert_model_for_plot, tokens, dictionary)
            print(f"  BERTopic c_v (ridotto a {BERT_TOPIC_CAP}, solo per il plot) = {bert_cv_for_plot}")

    # ── 3. Champion vs Challenger ────────────────────────────
    print(f"\n{'='*55}\n  CHAMPION vs CHALLENGER\n{'='*55}")
    comparison = pd.DataFrame([
        {'Model': 'LDA (Baseline)', 'n_topics': lda_n, 'c_v': lda_cv},
        {'Model': 'BERTopic (Champion)',
         'n_topics': (topic_model.get_topic_info()['Topic'].nunique() - 1
                      if topic_model else None),
         'c_v': bert_cv},
    ])
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(_DIR, 'champion_vs_challenger.csv'), index=False)
    plot_champion_vs_challenger(lda, lda_n, lda_cv, bert_model_for_plot, bert_cv_for_plot)
    plot_lda_topics(lda, lda_n, os.path.join(OUT_DIR, f'lda_topics.{FMT}'))

    # ── 4. Sentiment (fashion fine-tuned model, Cardiff base) ─
    # USE_FASHION_MODEL=True: usa il modello fine-tuned sul dominio fashion
    # prodotto da finetune_fashion_sentiment.py (base Cardiff twitter-roberta,
    # quindi conserva la gestione dell'ironia dei social). Se la cartella
    # non esiste, ricade sul Cardiff generico.
    USE_FASHION_MODEL = False
    fashion_model = os.path.join(_DIR, 'fashion-sentiment-model')
    sent_model = (fashion_model
                  if USE_FASHION_MODEL and os.path.isdir(fashion_model)
                  else 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    print(f"\n{'─'*55}\n  Loading sentiment model: {os.path.basename(sent_model)}\n{'─'*55}")
    roberta = hf_pipeline('sentiment-analysis', model=sent_model,
                          truncation=True, max_length=512)

    print("\n  Running sentiment...")
    df[['sentiment_label', 'confidence', 'sentiment_score']] = \
        df['cleaned_text'].apply(lambda t: pd.Series(analyze_sentiment(t, roberta)))

    dist = df['sentiment_label'].value_counts()
    print("\n  Distribution:")
    for lbl, cnt in dist.items():
        print(f"    {lbl:<12}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")
    print(f"    Mean score : {df['sentiment_score'].mean():.4f}")

    # --- REAL-EXAMPLES TEST (VADER vs RoBERTa vs LDA vs BERTopic)
    REAL_EXAMPLES = [
        "Looks like Chinese stuff with luxury prices",
        "zara used to be good now its just expensive shein",
        "This happened a long time ago! Zara used to be good quality 20 years ago…",
        "you folks should buy second hand clothes. they are cheap and good for ya",
        "I do not get Zara… the appeal… to me the fabrics in the women's section all look cheap",
        "But making the Stores looking \"expensive\" lol",
        "I love massimo but it's expensive",
        "Zara's quality keeps getting worse and worse each year :/",
    ]
    try:
        dict_arg = (lda_aux[1] if lda_aux else None)
        real_examples_test(REAL_EXAMPLES, lda, dict_arg, stop, topic_model, roberta)
    except Exception as e:
        print(f"  REAL_EXAMPLES test failed: {e}")

    # Phrase normalisation AFTER sentiment (the transformer wants raw text),
    # BEFORE ABSA (keyword matching wants the normalised polarity tokens).
    df['cleaned_text'] = df['cleaned_text'].apply(normalize_phrases)

    df.to_json(os.path.join(_DIR, 'comments_sentiment.json'),
               orient='records', force_ascii=False, indent=4)
    plot_vader_vs_roberta(df, os.path.join(OUT_DIR, f'vader_vs_roberta.{FMT}'))

    # ── 5. ABSA ──────────────────────────────────────────────
    aspects = {
        'Collaborations':      ['galliano', 'collaboration', 'collab',
                                'john galliano', 'galliano collection',
                                'bad bunny', 'badbunny', 'narciso',
                                'narciso rodriguez'],
        'Studio Collection':   ['studio collection', 'zara studio'],
        'Design':              ['design', 'aesthetic', 'style', 'visual',
                                'color', 'colour', 'pattern', 'cut',
                                'silhouette'],
        'Luxury':              ['luxury', 'premium', 'high-end', 'upscale',
                                'exclusive', 'high end', 'luxury brand',
                                'premium brand'],
        'Quality / Materials': ['quality', 'material', 'leather', 'fabric',
                                'durable', 'good quality', 'bad quality',
                                'poor quality', 'high quality', 'low quality',
                                'fast fashion', 'substandard', 'shoddy'],
        'Price / Value':       ['price', 'pricing', 'priced', 'cost', 'value',
                                'worth', 'worthless', 'affordable',
                                'price point', 'price increase', 'price tag',
                                'good value', 'value for money', 'expensive',
                                'overpriced', 'pricey', 'too much',
                                'too pricey', 'high price', 'raised price',
                                'price hike'],
    }

    absa = compute_absa(df, aspects)
    print(f"\n{'─'*55}\n  ABSA\n{'─'*55}")
    print(absa.to_string(index=False))
    absa.to_json(os.path.join(_DIR, 'absa_results.json'),
                 orient='records', force_ascii=False, indent=4)

    # ── 6. Visualisations ────────────────────────────────────
    diverging_bar(absa, 'Sentiment Distribution by Attribute',
                  os.path.join(OUT_DIR, f'diverging_absa.{FMT}'))
    network_graph(absa, 'Semantic Network – Zara Attributes',
                  os.path.join(OUT_DIR, f'network.{FMT}'))
    wordcloud_sentiment(df, True,  'Positive Word Cloud',
                        os.path.join(OUT_DIR, f'wc_pos.{FMT}'))
    wordcloud_sentiment(df, False, 'Negative Word Cloud',
                        os.path.join(OUT_DIR, f'wc_neg.{FMT}'))
    plot_sentiment_heatmap(df, aspects,
                           os.path.join(OUT_DIR, f'sentiment_heatmap.{FMT}'))

    # ── 7. PTI time-series ───────────────────────────────────
    _kw_quality_judgment = {'quality', 'good quality', 'bad quality',
                            'poor quality', 'high quality', 'low quality',
                            'fast fashion', 'substandard', 'shoddy'}
    kw_ptq = sorted(set(aspects['Price / Value']) | _kw_quality_judgment)
    pti_ts = pti_index_temporal(df, kw_ptq, freq=TIME_FREQ, min_doc=5)

    print("\n--- PTI DIAGNOSTIC (price-to-quality comments per year) ---")
    import numpy as _np
    _pat = '|'.join(kw_ptq)
    _sub = df[df['cleaned_text'].str.contains(_pat, case=False, na=False)].copy()
    _sub['year'] = _sub['date'].dt.year
    for _y, _g in _sub.groupby('year'):
        _w   = 1.0 + _np.log1p(_g['like'].fillna(0).clip(lower=0))
        _unw = _g['sentiment_score'].mean()
        _wt  = _np.average(_g['sentiment_score'], weights=_w)
        print(f"  {_y}: n={len(_g):5d} | sentiment non pesato={_unw:+.3f} | tau pesato={_wt:+.3f}")

    pti_last = float(pti_ts.iloc[-1]) if not pti_ts.empty else None
    print(f"\n{'='*55}\n  PREMIUMIZATION TOLERANCE INDEX (base 2019 = 100)\n{'='*55}")
    print(f"  Last period: {pti_last}")
    print("  PTI > 100 -> tolerance above baseline | PTI < 100 -> erosion")
    with open(os.path.join(_DIR, 'pti_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'PTI_last': pti_last,
                   'PTI_series': {str(k.date()): float(v)
                                  for k, v in pti_ts.items()}},
                  f, indent=4, ensure_ascii=False)

    plot_pti_time_series(pti_ts)

    # ── 8. Tolerance threshold (breaking point) ──────────────
    print(f"\n{'='*55}\n  TOLERANCE THRESHOLD (price breaking point)\n{'='*55}")
    t_star, p_star, pti_val = tolerance_threshold(pti_ts, PRICE_INCREASE,
                                                  threshold=100.0)
    if t_star is not None:
        print(f"  breaking point: {t_star.date()} (PTI={pti_val:.1f}) "
              f"at a price increase of ~{p_star}% vs 2019")
    else:
        print("  no breaking point in the observed period (PTI >= baseline)")

    print(f"\n{'='*55}")
    print("  Analysis complete. Files generated:")
    print(f"{'='*55}")
    print("  CSV     : champion_vs_challenger | topic_info")
    print("  JSON    : comments_sentiment | absa_results | pti_results")
    print("  PDF     : champion_vs_challenger | diverging_absa")
    print("            wc_pos/neg | topic_sentiment")
    print("            pti_time_series   <- main thesis visualisation")
    print("  GRAPHML : network   <- to open in Gephi")

    sys.stdout = sys.__stdout__
    _log_file.close()
    print(f"  Log saved → {os.path.join(OUT_DIR, 'pipeline_log.txt')}")
