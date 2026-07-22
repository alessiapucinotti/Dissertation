"""
Pipeline H&M Premiumization Analysis – PTI only
RoBERTa Sentiment → ABSA → PTI (globale + time series)

Input : all_comments_hm.json  (struttura: [{"date": "...", "comments": [{"comment": "...", "like": N}]}])
Output: JSON/CSV dei risultati + PDF della time series del PTI  (_hm suffix)

NB: PTI ora calcolato con formula ADDITIVA (non più rapporto pos/|neg|):

    PTI = 100 + 100 * (pos_t - neg_t)

dove pos_t/neg_t sono le medie pesate (per volume) del sentiment_score
([-1, 1]) degli aspetti positivi/negativi. Essendo pos_t - neg_t vincolato
in [-2, 2], il PTI è vincolato in [-100, 300]: niente più esplosioni verso
+-infinito quando il sentiment medio degli aspetti negativi si avvicina a
zero (come succedeva con pos/|neg|, che è un rapporto tra due medie e
diventa instabile quando il denominatore tende a 0).
"""

import argparse
import json
import re
import os
import warnings
from multiprocessing import freeze_support
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
from tqdm import tqdm

from transformers import pipeline as hf_pipeline

from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0

# ── Stile globale ────────────────────────────────────────────
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

_DIR = os.path.dirname(os.path.abspath(__file__))


def outpath(nome):
    """Tutti gli output vengono scritti sempre accanto allo script,
    indipendentemente dalla cwd da cui viene lanciato."""
    return os.path.join(_DIR, nome)


# Path del modello fine-tuned prodotto da train_fashion_sentiment_hm.py
MODELLO_FINE_TUNED = outpath('fashion-sentiment-model-hm')

# Se True, usa SEMPRE i modelli generici RoBERTa/XLM-RoBERTa, anche quando
# esiste un modello fine-tuned locale in MODELLO_FINE_TUNED. Utile per
# confrontare i due approcci o quando si vuole evitare deliberatamente il
# modello fine-tuned (es. per sospetto bias, come discusso in precedenza).
FORZA_MODELLO_GENERICO = True

# Pipeline a doppio check: prima dell'analisi di sentiment, ogni commento
# viene passato a un modello di irony detection. I modelli di sentiment
# fine-tuned su recensioni "pulite" (come quello fashion) associano parole
# positive/negative al sentiment, ma non colgono il sarcasmo tipico dei
# social (es. "wow che qualità pazzesca" detto con disprezzo). I commenti
# rilevati come ironici NON vengono passati al modello fashion: restano con
# sentiment_score = NaN (escluso automaticamente da ABSA/PTI, che usano
# medie che ignorano i NaN) e vengono segnalati per revisione manuale.
MODELLO_IRONIA = "cardiffnlp/twitter-roberta-base-irony"
# Il modello cardiffnlp restituisce tipicamente LABEL_0 (non ironico) /
# LABEL_1 (ironico). Se usi un modello che restituisce etichette testuali
# diverse, aggiorna questo set di conseguenza.
LABEL_IRONICO = {"irony", "ironic", "label_1"}

EVENTI = {
    '2021-03-01': 'H&M Conscious\nCollection',
    '2022-09-01': 'Collaborazione\nMugler',
    '2023-03-01': 'Collaborazione\nStella McCartney',
    '2023-09-01': 'Aumento prezzi\n+10%',
    '2024-03-01': 'H&M Studio\nSS24',
}

MIN_LANG_CONFIDENCE = 0.85
MIN_LANG_LEN        = 10

# Soglia minima di commenti che devono citare un aspetto perché venga
# considerato affidabile in ABSA/PTI (evita che 2-3 commenti rumorosi
# determinino un intero punteggio di aspetto).
MIN_VOLUME_ASPETTO    = 15
MIN_VOLUME_ASPETTO_TS = 5   # soglia più permissiva per la serie temporale trimestrale
MIN_DOC_PER_PERIODO   = 20  # minimo di commenti nel trimestre perché sia calcolato il PTI


# ============================================================
# FUNZIONI
# ============================================================

def detect_lang(text):
    if not text or len(text) < MIN_LANG_LEN:
        return None
    try:
        top = detect_langs(text)[0]
        return top.lang if top.prob >= MIN_LANG_CONFIDENCE else None
    except LangDetectException:
        return None


def carica_e_separa(path):
    """
    Carica all_comments_hm.json (chiavi: comments / comment),
    rileva la lingua di ogni commento e restituisce df_en (solo inglese).
    Logga anche quanti commenti vengono scartati e perché (lunghezza
    insufficiente, confidenza sotto soglia, lingua diversa da EN).
    """
    with open(path, 'r', encoding='utf-8') as f:
        dati = json.load(f)

    rows_en = []
    total = 0
    scartati_corti = 0
    scartati_altra_lingua = 0

    for gruppo in dati:
        data = gruppo['date']
        for c in gruppo['comments']:
            testo = c.get('comment', '')
            like  = c.get('like', 0)
            total += 1

            if not testo or len(testo) < MIN_LANG_LEN:
                scartati_corti += 1
                continue

            lang = detect_lang(testo)
            row  = {'date': data, 'testo_pulito': testo, 'like': like}
            if lang == 'en':
                rows_en.append(row)
            else:
                scartati_altra_lingua += 1

    tenuti = len(rows_en)
    print(f"  Totale commenti: {total} | EN: {len(rows_en)}")
    print(f"  Scartati troppo corti (<{MIN_LANG_LEN} char): {scartati_corti}")
    print(f"  Scartati (non EN o confidenza < {MIN_LANG_CONFIDENCE}): {scartati_altra_lingua}")
    if total:
        print(f"  Tasso di copertura EN: {tenuti/total*100:.1f}%")

    return pd.DataFrame(rows_en)


def analizza_sentiment_batch(testi, pipe, batch_size=32):
    """Versione con batching (usa la GPU se disponibile) al posto di
    applicare la pipeline riga per riga con .apply(), molto più lenta e
    che ignorava completamente eventuale accelerazione hardware."""
    testi_puliti = [str(t)[:512] if pd.notna(t) and t else '' for t in testi]
    if not testi_puliti:
        return []
    try:
        # iter(...) forza la pipeline a restituire un generatore invece di
        # accumulare tutto internamente: cosi' tqdm mostra avanzamento/ETA
        # reali invece di bloccare senza output fino alla fine del batch.
        risultati = list(tqdm(
            pipe(iter(testi_puliti), batch_size=batch_size, truncation=True),
            total=len(testi_puliti), desc="  Sentiment"))
    except Exception:
        # fallback: elabora uno a uno se il batching fallisce per qualche input
        risultati = []
        for t in tqdm(testi_puliti, desc="  Sentiment (fallback)"):
            try:
                risultati.append(pipe(t, truncation=True)[0])
            except Exception:
                risultati.append({'label': 'neutral', 'score': 0.0})

    out = []
    for res in risultati:
        label = res['label'].lower()
        conf  = float(res['score'])
        if 'positive' in label or label == 'pos':
            out.append((label, conf, conf))
        elif 'negative' in label or label == 'neg':
            out.append((label, conf, -conf))
        else:
            out.append((label, conf, 0.0))
    return out


def rileva_ironia_batch(testi, pipe_irony, batch_size=32):
    """Restituisce una lista di booleani is_ironic, elaborando in batch
    (stesso principio di analizza_sentiment_batch: niente .apply() riga
    per riga, che sarebbe molto più lento e non sfrutterebbe la GPU)."""
    testi_puliti = [str(t)[:512] if pd.notna(t) and t else '' for t in testi]
    if not testi_puliti:
        return []
    try:
        risultati = list(tqdm(
            pipe_irony(iter(testi_puliti), batch_size=batch_size, truncation=True),
            total=len(testi_puliti), desc="  Irony detection"))
    except Exception:
        risultati = []
        for t in tqdm(testi_puliti, desc="  Irony detection (fallback)"):
            try:
                risultati.append(pipe_irony(t, truncation=True)[0])
            except Exception:
                risultati.append({'label': 'label_0', 'score': 0.0})
    return [res['label'].lower() in LABEL_IRONICO for res in risultati]


def esporta_campione_aspetti(df, aspects, lingua, n_campione=15):
    """Per ogni aspetto ABSA, esporta un campione casuale dei commenti
    effettivamente matchati (con relativo sentiment_score). Serve a
    verificare A OCCHIO che il keyword-matching non stia catturando
    commenti fuori tema (es. 'self-conscious' matchato dalla keyword
    'conscious' pensata per 'H&M Conscious Collection') o dal significato
    ribaltato (come 'inexpensive' matchato da 'expensive' prima del fix
    del word-boundary). Non sostituisce una validazione statistica, ma è
    un controllo di sanità rapido ed essenziale prima di fidarsi dei numeri.
    """
    for asp, kw in aspects.items():
        pattern = '|'.join(r'\b' + re.escape(k) + r'\b' for k in kw)
        sub = df[df['testo_pulito'].str.contains(pattern, case=False, na=False, regex=True)]
        if sub.empty:
            continue
        campione = sub[['testo_pulito', 'sentiment_score']].sample(
            min(n_campione, len(sub)), random_state=42)
        nome_file = asp.lower().replace(' ', '_').replace('/', '-').replace('à', 'a')
        campione.to_csv(outpath(f'campione_absa_{nome_file}_hm_{lingua}.csv'), index=False)
    print(f"  [ABSA] Campioni di verifica manuale salvati (campione_absa_*_hm_{lingua}.csv)")


def sentiment_pesato_per_engagement(scores, likes):
    """Media del sentiment pesata per engagement (1 + log(1+like)) invece
    della media semplice. Un commento con 500 like riflette un'opinione
    vista/condivisa da molti più utenti di uno con 0 like: pesarlo di più
    fa sì che 'sentiment' rifletta meglio cosa il pubblico ha davvero
    percepito, non solo quanti commenti sono stati scritti. Se nel
    sottoinsieme meno del 5% dei commenti ha almeno un like (dato di
    engagement troppo scarso/assente per essere affidabile), si ricade
    sulla media semplice."""
    likes = likes.fillna(0).clip(lower=0)
    if (likes > 0).mean() >= 0.05:
        return float(np.average(scores, weights=1.0 + np.log1p(likes)))
    return float(scores.mean())


def calcola_absa(df, aspects, min_volume=MIN_VOLUME_ASPETTO):
    """Calcola sentiment/volume per aspetto. Gli aspetti con un volume di
    commenti inferiore a min_volume vengono esclusi: con pochissimi
    commenti la percentuale/sentiment medio è troppo rumoroso e instabile
    per essere interpretato in modo affidabile.

    Il sentiment per aspetto è pesato per engagement (vedi
    sentiment_pesato_per_engagement), non più una media semplice."""
    righe = []
    scartati = []
    for asp, kw in aspects.items():
        # \b...\b = word boundary: senza questo, keyword come 'expensive'
        # matchavano anche 'inexpensive' (che significa l'OPPOSTO, economico)
        # e 'price' matchava 'priceless' (di valore inestimabile, lodativo),
        # inquinando gli aspetti con commenti dal significato ribaltato.
        pattern = '|'.join(r'\b' + re.escape(k) + r'\b' for k in kw)
        sub     = df[df['testo_pulito'].str.contains(pattern, case=False, na=False, regex=True)]
        if len(sub) == 0:
            continue
        if len(sub) < min_volume:
            scartati.append((asp, len(sub)))
            continue
        righe.append({
            'aspect':    asp,
            'sentiment': round(sentiment_pesato_per_engagement(sub['sentiment_score'], sub['like']), 4),
            'pct_pos':   round((sub['sentiment_score'] >  0.1).mean() * 100, 1),
            'pct_neg':   round((sub['sentiment_score'] < -0.1).mean() * 100, 1),
            'pct_neu':   round((sub['sentiment_score'].abs() <= 0.1).mean() * 100, 1),
            'volume':    int(len(sub)),
            'pct':       round(len(sub)/len(df)*100, 1),
        })
    if scartati:
        dettaglio = ', '.join(f"{a} (n={v})" for a, v in scartati)
        print(f"  [ABSA] Aspetti esclusi per volume < {min_volume}: {dettaglio}")
    if not righe:
        return pd.DataFrame(columns=['aspect','sentiment','pct_pos','pct_neg','pct_neu','volume','pct'])
    return pd.DataFrame(righe).sort_values('sentiment', ascending=False)


def calcola_pti(absa_df, positive_aspects, negative_aspects, min_volume=MIN_VOLUME_ASPETTO, verbose=True):
    """PTI = 100 + 100 * (pos_t - neg_t)

    pos_t = media pesata (per volume di commenti) del sentiment_score degli
    aspetti positivi (es. Mugler, Stella McCartney, Design, Luxury/Premium).
    neg_t = media pesata del sentiment_score degli aspetti negativi
    (es. Price/Expensive).

    Formula ADDITIVA, non un rapporto: sentiment_score è in [-1, 1], quindi
    pos_t - neg_t è vincolato in [-2, 2] e il PTI risultante è vincolato in
    [-100, 300]. A differenza della vecchia formula pos/|neg|, qui non c'è
    nessuna divisione: il risultato non esplode mai anche se il sentiment
    medio degli aspetti negativi si avvicina a zero in un trimestre con
    pochi commenti su prezzo/costo.

    Interpretazione: PTI = 100 è il punto neutro (pos e neg si bilanciano
    esattamente). PTI > 100 -> il pubblico premia gli aspetti "premium"
    più di quanto penalizzi il prezzo (tolleranza sopra baseline).
    PTI < 100 -> il prezzo pesa più del contenuto premium (erosione).
    """
    pos_df = absa_df[absa_df['aspect'].isin(positive_aspects) & (absa_df['volume'] >= min_volume)]
    neg_df = absa_df[absa_df['aspect'].isin(negative_aspects) & (absa_df['volume'] >= min_volume)]
    if pos_df.empty:
        if verbose:
            print(f"  [PTI] None: nessun aspetto positivo con volume >= {min_volume} "
                  f"tra {positive_aspects}")
        return None
    if neg_df.empty:
        if verbose:
            print(f"  [PTI] None: nessun aspetto negativo con volume >= {min_volume} "
                  f"tra {negative_aspects}")
        return None
    pos = np.average(pos_df['sentiment'], weights=pos_df['volume'])
    neg = np.average(neg_df['sentiment'], weights=neg_df['volume'])
    if pd.isna(pos) or pd.isna(neg):
        return None
    if verbose and neg >= 0:
        # Non blocchiamo più il calcolo (niente divisione = niente segno
        # "fuorviante"), ma segnaliamo comunque il caso: se gli aspetti
        # 'negativi' hanno in media sentiment positivo, vuol dire che in
        # questo dataset il prezzo non è percepito come problema, e il PTI
        # va letto tenendo conto di questo.
        print(f"  [PTI] Nota: gli aspetti 'negativi' {negative_aspects} hanno "
              f"sentiment medio pesato POSITIVO ({neg:+.4f}) in questo periodo.")
    return round(100 + 100 * (float(pos) - float(neg)), 4)


# ── Visualizzazioni ──────────────────────────────────────────

def plot_pti_time_series(pti_ts_en):
    fig, ax = plt.subplots(figsize=(14, 7))
    if pti_ts_en.empty:
        pass
    elif len(pti_ts_en) >= 8:
        # Con molti trimestri, la serie grezza trimestre-su-trimestre è
        # rumorosa (ogni punto dipende da poche decine di commenti).
        # Mostriamo i punti grezzi tenui e sopra una rolling mean a 4
        # periodi (1 anno) più marcata, che rende leggibile il trend
        # senza nascondere la variabilità sottostante.
        ax.plot(pti_ts_en.index, pti_ts_en.values, color=C_POS, linewidth=1,
                marker='o', markersize=4, alpha=0.35, label='PTI (trimestrale)', zorder=3)
        trend = pti_ts_en.rolling(4, center=True, min_periods=2).mean()
        ax.plot(trend.index, trend.values, color=C_POS, linewidth=2.6,
                label='PTI (trend, media mobile 4 trim.)', zorder=4)
    else:
        ax.plot(pti_ts_en.index, pti_ts_en.values,
                color=C_POS, linewidth=2.6, marker='o', markersize=7,
                label='EN – Pubblico internazionale', zorder=4)
    ax.axhline(100, color='black', linewidth=0.8, linestyle='--',
               alpha=0.6, label='Neutro (100)', zorder=2)

    # Etichette eventi posizionate in coordinate degli assi (non dei dati),
    # con altezza alternata: evita che si sovrappongano tutte in cima al
    # grafico indipendentemente dalla scala/range del PTI in quel momento.
    xform = ax.get_xaxis_transform()
    for i, (data_str, etichetta) in enumerate(EVENTI.items()):
        data_dt = pd.to_datetime(data_str)
        ax.axvline(data_dt, color='#777777', linewidth=0.8,
                   linestyle=':', alpha=0.65, zorder=2)
        y_frac = 0.96 if i % 2 == 0 else 0.74
        ax.text(data_dt, y_frac, etichetta, transform=xform, rotation=90,
                va='top', ha='right', fontsize=8, color='#444444',
                linespacing=1.4)

    ax.set_xlabel('Trimestre')
    ax.set_ylabel('PTI = 100 + 100·(pos − neg)')
    ax.set_title('Premiumization Tolerance Index – Evoluzione Temporale\nH&M 2019–2026 (formula additiva)', pad=12)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(outpath(f'pti_time_series_hm.{FMT}'))
    plt.close()
    print(f"  Salvato: pti_time_series_hm.{FMT}")


# ============================================================
# MAIN
# ============================================================

SENTIMENT_CACHE = outpath('commenti_sentiment_hm_en.json')


def carica_da_cache():
    """Ricarica df_en (con sentiment/is_ironic già calcolati) dalla cache
    JSON salvata da una run precedente, saltando modelli/irony/sentiment.
    Utile quando si vuole solo aggiustare ABSA/PTI/grafico senza rilanciare
    RoBERTa + irony detection su tutto il corpus (operazione lenta)."""
    with open(SENTIMENT_CACHE, 'r', encoding='utf-8') as f:
        dati = json.load(f)
    df = pd.DataFrame(dati)
    if np.issubdtype(df['date'].dtype, np.number):
        df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    return df


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action='store_true',
                        help=f'riusa {os.path.basename(SENTIMENT_CACHE)} se presente, '
                             'saltando modelli/irony/sentiment (utile per iterare solo su ABSA/PTI/grafico)')
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Device per i modelli di sentiment: {'GPU' if device == 0 else 'CPU'}")

    if args.reuse and os.path.exists(SENTIMENT_CACHE):
        print(f"--reuse: carico sentiment già calcolato da {SENTIMENT_CACHE} (skip modelli/irony/sentiment)...")
        df_en = carica_da_cache()
        n_scartate = int(df_en['date'].isna().sum())
        df_en = df_en.dropna(subset=['date'])
        if n_scartate:
            print(f"  Scartate {n_scartate} righe con data non valida.")
        print(f"  Commenti ricaricati dalla cache: {len(df_en)}")
        _skip_sentiment_pipeline = True
    else:
        _skip_sentiment_pipeline = False

    if not _skip_sentiment_pipeline:
        # ── 0. Caricamento e split lingua ────────────────────────
        print("Caricamento e rilevamento lingua...")
        df_en = carica_e_separa(outpath('all_comments_hm.json'))
        df_en['date'] = pd.to_datetime(df_en['date'], errors='coerce')
        n_date_invalide = int(df_en['date'].isna().sum())
        n_future = int((df_en['date'] > pd.Timestamp.now()).sum())
        if n_date_invalide or n_future:
            print(f"  Scartate {n_date_invalide} righe con data non valida "
                  f"e {n_future} righe con data futura (probabili errori nei dati).")
        df_en = df_en[df_en['date'].notna() & (df_en['date'] <= pd.Timestamp.now())].reset_index(drop=True)

    if not _skip_sentiment_pipeline:
        # ── 1. Sentiment (fine-tuned locale se disponibile, altrimenti RoBERTa) ──
        print(f"\n{'─'*55}\n  Caricamento modelli di sentiment...\n{'─'*55}")
        if os.path.isdir(MODELLO_FINE_TUNED) and not FORZA_MODELLO_GENERICO:
            print(f"  Trovato modello fine-tuned locale: {MODELLO_FINE_TUNED}")
            roberta_en = hf_pipeline(
                "sentiment-analysis", model=MODELLO_FINE_TUNED,
                tokenizer=MODELLO_FINE_TUNED, truncation=True, max_length=512,
                device=device)
        else:
            if FORZA_MODELLO_GENERICO and os.path.isdir(MODELLO_FINE_TUNED):
                print("  FORZA_MODELLO_GENERICO=True: ignoro il modello fine-tuned locale.")
            else:
                print("  Nessun modello fine-tuned trovato, uso il modello RoBERTa generico.")
            print("  Uso il modello RoBERTa generico (non fine-tuned).")
            roberta_en = hf_pipeline("sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True, max_length=512, device=device)

        # ── 2. Irony detection (doppio check prima del sentiment) ─────────
        print(f"\n{'─'*55}\n  Irony Detection [{MODELLO_IRONIA}]\n{'─'*55}")
        pipe_irony = hf_pipeline("text-classification", model=MODELLO_IRONIA,
                                 tokenizer=MODELLO_IRONIA, truncation=True,
                                 max_length=512, device=device)

        for lingua, df in [('EN', df_en)]:
            if df.empty:
                df['is_ironic'] = pd.Series(dtype=bool)
                continue
            print(f"  Irony detection [{lingua}] su {len(df)} commenti...")
            df['is_ironic'] = rileva_ironia_batch(df['testo_pulito'].tolist(), pipe_irony)
            n_iron = int(df['is_ironic'].sum())
            print(f"    Ironici: {n_iron} ({n_iron/len(df)*100:.1f}%) -> trattati come sentiment negativo")

        # ── 3. Sentiment: RoBERTa sui commenti non ironici, i commenti ironici
        # vengono forzati a negativo (il sarcasmo tipico dei social, es. "wow
        # che qualità pazzesca" detto con disprezzo, è un segnale di critica
        # mascherata da elogio, non di sentiment neutro/assente).
        df_en['sentiment_label'] = pd.NA
        df_en['confidence']      = pd.NA
        df_en['sentiment_score'] = np.nan

        mask_en_ok = ~df_en['is_ironic'] if not df_en.empty else pd.Series(dtype=bool)

        if not df_en.empty and mask_en_ok.any():
            print(f"\n  Sentiment EN (batch) su {int(mask_en_ok.sum())} commenti non ironici...")
            ris_en = analizza_sentiment_batch(df_en.loc[mask_en_ok, 'testo_pulito'].tolist(), roberta_en)
            df_en.loc[mask_en_ok, ['sentiment_label', 'confidence', 'sentiment_score']] = \
                pd.DataFrame(ris_en, index=df_en.loc[mask_en_ok].index).values

        if not df_en.empty and df_en['is_ironic'].any():
            mask_ironic = df_en['is_ironic']
            df_en.loc[mask_ironic, 'sentiment_label'] = 'negative'
            df_en.loc[mask_ironic, 'confidence']      = 1.0
            df_en.loc[mask_ironic, 'sentiment_score']  = -1.0

        df_en['sentiment_score'] = pd.to_numeric(df_en['sentiment_score'], errors='coerce')

        for lingua, df in [('EN', df_en)]:
            if df.empty:
                continue
            dist = df['sentiment_label'].value_counts()
            print(f"\n  Distribuzione [{lingua}] (ironici forzati a negativo):")
            for lbl, cnt in dist.items():
                print(f"    {lbl:<12}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")
            print(f"    Score medio : {df['sentiment_score'].mean():.4f}")
            print(f"    Di cui ironici (forzati a negativo): {int(df['is_ironic'].sum())}")

        df_en.to_json(outpath('commenti_sentiment_hm_en.json'), orient='records', force_ascii=False, indent=4)

        # Esporta separatamente i commenti ironici per la revisione manuale
        # (sono comunque inclusi come negativi in ABSA/PTI, ma vale la pena
        # controllare a campione che l'irony detection non abbia falsi positivi,
        # specialmente sull'inglese non nativo dei social).
        for lingua, df in [('en', df_en)]:
            if df.empty:
                continue
            da_rivedere = df[df['is_ironic']][['date', 'testo_pulito', 'like']]
            if not da_rivedere.empty:
                da_rivedere.to_csv(outpath(f'commenti_da_revisionare_ironia_hm_{lingua}.csv'), index=False)
                print(f"  Salvato: commenti_da_revisionare_ironia_hm_{lingua}.csv "
                      f"({len(da_rivedere)} commenti ironici, forzati a negativo, da rivedere a campione)")

    # ── 4. ABSA – aspetti H&M ────────────────────────────────
    aspects_en = {
        'Mugler':              ['mugler', 'mugler collection', 'thierry mugler'],
        'Stella McCartney':    ['stella mccartney', 'stella', 'mccartney'],
        'Design':              ['design', 'aesthetic', 'style', 'look'],
        'Luxury / Premium':    ['luxury', 'premium', 'high-end', 'upscale', 'conscious collection'],
        'Quality / Materials': ['quality', 'material', 'fabric', 'sustainable', 'organic'],
        'Price / Expensive':   ['price', 'pricing', 'cost', 'expensive', 'overpriced', 'pricey'],
    }


    absa_en = calcola_absa(df_en, aspects_en)

    print(f"\n{'─'*55}\n  ABSA [EN]  (aspetti con volume >= {MIN_VOLUME_ASPETTO})\n{'─'*55}")
    print(absa_en.to_string(index=False) if not absa_en.empty else "  (nessun aspetto sopra soglia)")

    absa_en.to_json(outpath('absa_results_hm_en.json'), orient='records', force_ascii=False, indent=4)


    esporta_campione_aspetti(df_en, aspects_en, 'en')

    # ── 5. PTI globale ───────────────────────────────────────
    pti_en = calcola_pti(absa_en,
                         ['Mugler', 'Stella McCartney', 'Design', 'Luxury / Premium'],
                         ['Price / Expensive'])

    print(f"\n{'='*55}\n  PREMIUMIZATION TOLERANCE INDEX – H&M\n{'='*55}")
    print(f"  EN : {pti_en}")
    print("  PTI = 100 + 100*(pos - neg)  |  PTI > 100 -> tolleranza sopra baseline  |  PTI < 100 -> erosione")
    with open(outpath('pti_results_hm.json'), 'w', encoding='utf-8') as f:
        json.dump({'PTI_EN': pti_en}, f, indent=4)

    # ── 6. PTI Time-Series ───────────────────────────────────
    def pti_temporale(df, aspects, pos_asp, neg_asp, freq='Q'):
        df = df.copy()
        df['periodo'] = df['date'].dt.to_period(freq)
        serie = {}
        for periodo, gruppo in df.groupby('periodo'):
            if len(gruppo) < MIN_DOC_PER_PERIODO:
                continue
            absa = calcola_absa(gruppo, aspects, min_volume=MIN_VOLUME_ASPETTO_TS)
            pti  = calcola_pti(absa, pos_asp, neg_asp, min_volume=MIN_VOLUME_ASPETTO_TS, verbose=False)
            if pti is not None:
                serie[periodo.to_timestamp()] = pti
        return pd.Series(serie).sort_index()

    pti_ts_en = pti_temporale(df_en, aspects_en,
                               ['Mugler', 'Stella McCartney', 'Design', 'Luxury / Premium'],
                               ['Price / Expensive'])
    # NB: niente più normalizza_base100() qui. La formula additiva è già in
    # scala "base 100" per costruzione (100 = pos e neg che si bilanciano
    # esattamente): rinormalizzare di nuovo dividendo per il primo
    # trimestre reintrodurrebbe lo stesso rischio di instabilità che questa
    # formula è pensata per evitare (se il primo trimestre avesse un
    # valore vicino a 0, dividerci sopra amplificherebbe tutto il resto).
    plot_pti_time_series(pti_ts_en)

    print(f"\n{'='*55}")
    print("  Analisi H&M completata. File generati in:")
    print(f"  {_DIR}")
    print(f"{'='*55}")
    print("  CSV     : commenti_da_revisionare_ironia_hm_en (revisione manuale)")
    print("  JSON    : commenti_sentiment_hm_en (con colonna is_ironic) | absa_results_hm_en | pti_results_hm")
    print("  PDF     : pti_time_series_hm")
