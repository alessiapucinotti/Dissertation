# -*- coding: utf-8 -*-
"""
Figure 5.1 - Topic map (treemap) for Chapter 5.

Reads topic_info.csv (native 7-topic BERTopic solution from the definitive
run). If the CSV in the folder is the degraded one (4 topics from the re-run
in a different environment), it falls back to the embedded data of the
definitive 9 July run.

Usage:
    pip install squarify   (once)
    python plot_topic_map.py
Output: results/topic_map.pdf (+ .png)
"""
import os
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import squarify
except ImportError:
    raise SystemExit("Missing squarify: run  pip install squarify")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_DIR, "results")
os.makedirs(OUT, exist_ok=True)

LABELS = {
    0: "General discussion",
    1: "Price-quality ratio",
    2: "Competitors",
    3: "Ethics & boycott",
    4: "Social & industry",
    5: "Tailoring",
    6: "Menswear quality",
}

# Tableau-10 muted palette: one distinct colour per topic
COLORS = {
    0: "#4E79A7",   # blue      - general discussion
    1: "#E15759",   # red       - price-quality (the critical core)
    2: "#76B7B2",   # teal      - competitors
    3: "#59A14F",   # green     - ethics & boycott
    4: "#EDC948",   # yellow    - social & industry
    5: "#B07AA1",   # purple    - tailoring
    6: "#F28E2B",   # orange    - menswear
}
DARK_TEXT = {4}     # topics whose fill needs dark text

# Definitive run data (results/pipeline_log.txt, 9 July) as fallback
FALLBACK = [
    (0, 17440, "clothe, fashion, quality"),
    (1, 3986, "quality, price, fashion"),
    (2, 375, "uniqlo, quality, recommend"),
    (3, 371, "free, boycott, support"),
    (4, 535, "cos, industry, social"),
    (5, 467, "tailor, fit, spend"),
    (6, 412, "men, woman, section"),
]


def load_topics():
    path = os.path.join(_DIR, "topic_info.csv")
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df["Topic"] = pd.to_numeric(df["Topic"], errors="coerce")
        df = df.dropna(subset=["Topic"])
        df["Topic"] = df["Topic"].astype(int)
        df = df[df["Topic"] != -1]
        if len(df) != 7:
            print(f"  WARNING: topic_info.csv has {len(df)} topics "
                  f"(expected 7, definitive run). Using embedded data.")
            return FALLBACK
        rows = []
        for _, r in df.iterrows():
            try:
                words = ", ".join(ast.literal_eval(str(r["Representation"]))[:3])
            except Exception:
                words = " ".join(str(r["Name"]).split("_")[1:4])
            rows.append((int(r["Topic"]), int(r["Count"]), words))
        return rows
    except Exception as e:
        print(f"  CSV not readable ({e}). Using embedded data.")
        return FALLBACK


def main():
    data = sorted(load_topics(), key=lambda r: -r[1])
    total = sum(c for _, c, _ in data)
    sizes = [c for _, c, _ in data]

    fig, ax = plt.subplots(figsize=(12, 7))
    rects = squarify.squarify(squarify.normalize_sizes(sizes, 100, 100),
                              0, 0, 100, 100)

    for rect, (tid, count, words) in zip(rects, data):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        color = COLORS.get(tid, "#999999")
        ax.add_patch(plt.Rectangle((x, y), dx, dy, facecolor=color,
                                   edgecolor="white", linewidth=2.5,
                                   alpha=0.92))
        pct = count / total * 100
        label = LABELS.get(tid, f"Topic {tid}")
        cx, cy = x + dx / 2, y + dy / 2
        txtcol = "#333333" if tid in DARK_TEXT else "white"
        if dx * dy > 400:
            ax.text(cx, cy + 6, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txtcol)
            ax.text(cx, cy, f"{count:,} comments ({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, color=txtcol)
            ax.text(cx, cy - 6, words, ha="center", va="center",
                    fontsize=10, style="italic", color=txtcol)
        else:
            fs = 8 if dx > 16 else 6.5
            short = label if dx > 14 else label.replace(" & ", "\n& ")
            ax.text(cx, cy, f"{short}\n{count:,}", ha="center", va="center",
                    fontsize=fs, color=txtcol, clip_on=True)

    minors = [f"{LABELS.get(t, t)}: {w} ({c:,})" for t, c, w in data if c < 1000]
    mid = (len(minors) + 1) // 2
    fig.text(0.5, 0.055, "Minor topics —  " + "  |  ".join(minors[:mid]),
             ha="center", fontsize=8, style="italic", color="#444444")
    fig.text(0.5, 0.025, "  |  ".join(minors[mid:]),
             ha="center", fontsize=8, style="italic", color="#444444")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("Thematic Structure of the Zara Conversation — "
                 "BERTopic, Native Solution (7 topics, N = 24,083)", pad=14)
    plt.subplots_adjust(top=0.91, bottom=0.11, left=0.03, right=0.97)

    for ext in ("pdf", "png"):
        out = os.path.join(OUT, f"topic_map.{ext}")
        plt.savefig(out, dpi=200)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
