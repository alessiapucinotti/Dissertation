# -*- coding: utf-8 -*-
"""
Figure 5.5 - PTI comparison: Zara vs H&M on the same axes.

Reads the official outputs of the two pipelines:
    pti_results.json      (Zara   - analisi_premiumization_zara.py)
    pti_results_hm.json   (H&M    - pti_hm.py)
Run AFTER re-running pti_hm.py, so both series come from the
zero-shot Cardiff sentiment model.

Usage:  python plot_pti_confronto.py
Output: results/pti_comparison.pdf (+ .png)
"""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_DIR, "results")
os.makedirs(OUT, exist_ok=True)

SERIES = [
    ("Zara", "pti_results.json", "#2166ac"),
    ("H&M", "pti_results_hm.json", "#b2182b"),
]


def load(fname):
    with open(os.path.join(_DIR, fname), encoding="utf-8") as f:
        data = json.load(f)
    s = pd.Series({pd.Timestamp(k): v for k, v in data["PTI_series"].items()})
    return s.sort_index()


def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, fname, color in SERIES:
        try:
            s = load(fname)
        except FileNotFoundError:
            print(f"  {fname} not found - skipped ({name})")
            continue
        except KeyError:
            print(f"  {fname} has no 'PTI_series' (old/wrong pipeline output)."
                  f" Re-run pti_hm.py to regenerate it - skipped ({name})")
            continue
        trend = s.rolling(4, center=True, min_periods=2).mean()
        ax.plot(s.index, s.values, color=color, linewidth=0.9,
                marker="o", markersize=3.5, alpha=0.25, zorder=3)
        ax.plot(trend.index, trend.values, color=color, linewidth=2.6,
                label=f"{name} (4-quarter trend)", zorder=4)
        last = s.iloc[-1]
        ax.annotate(f"{name}: {last:.1f}", xy=(s.index[-1], last),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=color,
                    fontweight="bold")

    ax.axhline(100, color="black", linewidth=0.8, linestyle="--",
               alpha=0.6, label="Base 100", zorder=2)
    ax.set_title("Premiumization Tolerance Index — Zara vs H&M, 2019–2026")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("PTI (base 2019 = 100)")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.04)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(OUT, f"pti_comparison.{ext}")
        plt.savefig(out, dpi=200)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
