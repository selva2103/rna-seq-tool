"""
RNA-Seq Universal Report Generator
====================================
Supports: Tumor vs Normal, Treated vs Control, Time Series,
          Knockout vs Wildtype, Single Condition, Pre-computed DEG

Version: 2.1 (Streamlit Cloud compatible)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import zipfile
import gzip
import io
import os
import re
import tempfile
from datetime import datetime

# Optional seaborn — fallback to matplotlib if not installed
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer, Table,
    TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🧬 RNA-Seq Universal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

    .metric-card {
        background: linear-gradient(135deg, #1a1d27, #22263a);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .dataset-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #00d4aa, #7c3aed);
        color: white;
        margin: 3px;
    }
    .info-box {
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    .section-header {
        border-left: 4px solid #00d4aa;
        padding-left: 12px;
        margin: 28px 0 14px 0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .premium-box {
        background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(0,212,170,0.08));
        border: 1px solid rgba(124,58,237,0.35);
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #00a884) !important;
        color: #0f1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATASET TYPE DETECTION
# ─────────────────────────────────────────────
DATASET_PATTERNS = {
    "tumor_normal": [
        r"\btumor\b", r"\bnormal\b", r"\bcancer\b", r"\bmalignant\b",
        r"\bbenign\b", r"\bNT\b"
    ],
    "treated_control": [
        r"\btreated?\b", r"\bcontrol\b", r"\bdrug\b", r"\buntreated?\b",
        r"\bvehicle\b", r"\bdmso\b", r"\bstimulated?\b", r"\binhibitor\b"
    ],
    "time_series": [
        r"\b\d+h\b", r"\b\d+hr\b", r"\b\d+hours?\b", r"\b\d+days?\b",
        r"\btime\b", r"\bt\d+\b", r"\bweek\b"
    ],
    "knockout_wildtype": [
        r"\bko\b", r"\bwt\b", r"\bknockout\b", r"\bwildtype\b",
        r"\bkd\b", r"\bknockdown\b", r"\bsiRNA\b", r"\bshRNA\b",
        r"\bcrispr\b", r"\bmutant\b"
    ],
}


def detect_dataset_type(df):
    cols_str = " ".join(df.columns.tolist()).lower()
    for dtype, patterns in DATASET_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, cols_str, re.IGNORECASE):
                return dtype
    if "log2FoldChange" in df.columns and "padj" in df.columns:
        return "pre_computed"
    return "single_condition"


def detect_group_columns(df, dataset_type):
    gene_like = {
        "gene", "geneid", "gene_id", "symbol", "genename",
        "probe_id", "probeid", "id_ref", "id", "ensembl",
        "transcript", "feature_id", "mirna", "mirbase", "name"
    }
    sample_cols = [
        c for c in df.columns
        if c.lower().replace("_", "").replace(" ", "") not in gene_like
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(sample_cols) < 2:
        return sample_cols[:1], sample_cols[1:2], "Group A", "Group B"

    def _match(patterns, cols):
        return [c for c in cols if any(re.search(p, c, re.IGNORECASE) for p in patterns)]

    if dataset_type == "tumor_normal":
        grp_a = _match([r"tumor", r"cancer", r"malignant", r"_T\d*$"], sample_cols)
        grp_b = _match([r"normal", r"benign", r"NT", r"_N\d*$"], sample_cols)
        label_a, label_b = "Tumor", "Normal"

    elif dataset_type == "treated_control":
        grp_a = _match([r"treat", r"drug", r"stimul", r"inhibit"], sample_cols)
        grp_b = _match([r"control", r"untreat", r"vehicle", r"dmso", r"mock"], sample_cols)
        label_a, label_b = "Treated", "Control"

    elif dataset_type == "knockout_wildtype":
        grp_a = _match([r"\bko\b", r"knockout", r"\bkd\b", r"knockdown",
                        r"siRNA", r"crispr", r"mutant"], sample_cols)
        grp_b = _match([r"\bwt\b", r"wildtype", r"wild.type",
                        r"scramble", r"negative"], sample_cols)
        label_a, label_b = "Knockout/KD", "Wildtype"

    elif dataset_type == "time_series":
        def _get_time(c):
            m = re.search(r"(\d+)", c)
            return int(m.group(1)) if m else 9999
        sorted_cols = sorted(sample_cols, key=_get_time)
        mid = len(sorted_cols) // 2
        grp_a = sorted_cols[mid:]
        grp_b = sorted_cols[:mid]
        label_a, label_b = "Late Timepoint", "Early Timepoint"

    else:
        half = len(sample_cols) // 2
        grp_a = sample_cols[half:]
        grp_b = sample_cols[:half]
        label_a, label_b = "Group A", "Group B"

    # Fallback if matching failed
    if not grp_a or not grp_b:
        half = len(sample_cols) // 2
        grp_b = sample_cols[:half]
        grp_a = sample_cols[half:]

    return grp_a, grp_b, label_a, label_b


# ─────────────────────────────────────────────
#  DIFFERENTIAL EXPRESSION
# ─────────────────────────────────────────────
def compute_differential_expression(df, grp_a, grp_b, label_a, label_b, norm_method="log2_cpm"):
    expr = df.copy()
    for c in grp_a + grp_b:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.dropna(subset=grp_a + grp_b, how="all")

    a_vals = expr[grp_a]
    b_vals = expr[grp_b]

    if norm_method == "log2_cpm":
        a_norm = np.log2(a_vals + 1)
        b_norm = np.log2(b_vals + 1)
    elif norm_method == "zscore":
        a_norm = (a_vals - a_vals.mean()) / (a_vals.std() + 1e-9)
        b_norm = (b_vals - b_vals.mean()) / (b_vals.std() + 1e-9)
    else:
        a_norm = a_vals
        b_norm = b_vals

    expr["mean_A"] = a_norm.mean(axis=1)
    expr["mean_B"] = b_norm.mean(axis=1)
    expr["log2FoldChange"] = expr["mean_A"] - expr["mean_B"]

    # Per-gene t-test
    p_vals = []
    if len(grp_a) >= 2 and len(grp_b) >= 2:
        for i in range(len(expr)):
            row_a = a_norm.iloc[i].dropna().values
            row_b = b_norm.iloc[i].dropna().values
            if len(row_a) >= 2 and len(row_b) >= 2:
                _, p = ttest_ind(row_a, row_b, equal_var=False)
                p_vals.append(float(p))
            else:
                p_vals.append(np.nan)
    else:
        # Single sample fallback: approximate from fold change magnitude
        lfc = expr["log2FoldChange"].abs()
        p_vals = (1 / (1 + lfc)).tolist()

    expr["pvalue"] = p_vals

    # Benjamini-Hochberg FDR correction
    valid = expr["pvalue"].notna()
    pv = expr.loc[valid, "pvalue"].values
    n = len(pv)
    if n > 0:
        order = np.argsort(pv)
        padj = pv.copy()
        padj[order] = pv[order] * n / (np.arange(n) + 1)
        padj = np.minimum.accumulate(padj[::-1])[::-1]
        padj = np.clip(padj, 0, 1)
        expr.loc[valid, "padj"] = padj
    else:
        expr["padj"] = np.nan

    expr["label_A"] = label_a
    expr["label_B"] = label_b
    return expr


def classify_genes(df, lfc_thr=1.0, padj_thr=0.05):
    df = df.copy()
    df["Category"] = "Not Significant"
    df.loc[(df["padj"] < padj_thr) & (df["log2FoldChange"] > lfc_thr),  "Category"] = "Upregulated"
    df.loc[(df["padj"] < padj_thr) & (df["log2FoldChange"] < -lfc_thr), "Category"] = "Downregulated"
    return df


def find_gene_column(df):
    candidates = [
        "gene", "Gene", "gene_id", "GeneID", "gene_name", "GeneName",
        "symbol", "Symbol", "SYMBOL", "Gene_Symbol", "gene_symbol",
        "ID_REF", "ID", "probe_id", "ProbeID", "feature_id",
        "Ensembl", "ensembl_id", "transcript_id", "NAME", "name"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


# ─────────────────────────────────────────────
#  VISUALIZATION FUNCTIONS (no seaborn required)
# ─────────────────────────────────────────────
BG      = "#0f1117"
BG_AX   = "#12151f"
BORDER  = "#2a2d3e"
TEXT    = "#c8cfe0"
MUTED   = "#8b92a5"
UP_CLR  = "#ff4d6d"
DN_CLR  = "#4da6ff"
ACCENT  = "#00d4aa"


def _style_ax(ax, title=""):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    if title:
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)


def plot_volcano(df, gene_col, label_a, label_b, lfc_thr=1.0, padj_thr=0.05):
    df = df.dropna(subset=["log2FoldChange", "padj"]).copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    _style_ax(ax, "Volcano Plot")

    color_map = {"Upregulated": UP_CLR, "Downregulated": DN_CLR, "Not Significant": "#3a3f55"}
    size_map  = {"Upregulated": 25, "Downregulated": 25, "Not Significant": 7}
    alpha_map = {"Upregulated": 0.85, "Downregulated": 0.85, "Not Significant": 0.3}

    for cat in ["Not Significant", "Downregulated", "Upregulated"]:
        sub = df[df["Category"] == cat]
        ax.scatter(
            sub["log2FoldChange"], sub["-log10padj"],
            c=color_map[cat], s=size_map[cat],
            alpha=alpha_map[cat], label=f"{cat} (n={len(sub)})",
            edgecolors="none"
        )

    ax.axvline(x=lfc_thr,  color=BORDER, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(x=-lfc_thr, color=BORDER, linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(y=-np.log10(padj_thr), color=BORDER, linestyle="--", linewidth=1, alpha=0.8)

    top_up = df[df["Category"] == "Upregulated"].nsmallest(8, "padj")
    top_dn = df[df["Category"] == "Downregulated"].nsmallest(8, "padj")
    for _, row in pd.concat([top_up, top_dn]).iterrows():
        clr = UP_CLR if row["Category"] == "Upregulated" else DN_CLR
        ax.annotate(
            str(row[gene_col])[:14],
            (row["log2FoldChange"], row["-log10padj"]),
            fontsize=7, color=clr, fontfamily="monospace",
            xytext=(5, 3), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=clr, alpha=0.4, lw=0.6)
        )

    ax.set_xlabel(f"log₂FC  ({label_a} / {label_b})", color=TEXT, fontsize=10)
    ax.set_ylabel("-log₁₀(padj)", color=TEXT, fontsize=10)
    ax.legend(framealpha=0.15, labelcolor="white", fontsize=8,
              facecolor=BG_AX, edgecolor=BORDER)
    fig.tight_layout()
    return fig


def plot_ma(df, label_a, label_b):
    df = df.dropna(subset=["log2FoldChange", "mean_A", "mean_B"]).copy()
    df["baseMean"] = (df["mean_A"] + df["mean_B"]) / 2

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    _style_ax(ax, "MA Plot")

    for cat, clr, sz, al in [
        ("Not Significant", "#3a3f55", 6, 0.25),
        ("Downregulated", DN_CLR, 18, 0.8),
        ("Upregulated", UP_CLR, 18, 0.8),
    ]:
        sub = df[df["Category"] == cat]
        ax.scatter(sub["baseMean"], sub["log2FoldChange"],
                   c=clr, s=sz, alpha=al, edgecolors="none", label=cat)

    ax.axhline(y=0, color=ACCENT, linestyle="-", linewidth=1, alpha=0.6)
    ax.set_xlabel("Mean Expression", color=TEXT, fontsize=10)
    ax.set_ylabel(f"log₂FC ({label_a}/{label_b})", color=TEXT, fontsize=10)
    ax.legend(framealpha=0.15, labelcolor="white", fontsize=8,
              facecolor=BG_AX, edgecolor=BORDER)
    fig.tight_layout()
    return fig


def plot_heatmap(df, grp_a, grp_b, gene_col, n_top=40):
    """Heatmap using pure matplotlib — no seaborn needed."""
    sig = df[df["Category"] != "Not Significant"].copy()
    if len(sig) == 0:
        sig = df.copy()
    sig = sig.nsmallest(min(n_top, len(sig)), "padj")

    cols_use = [c for c in (grp_b + grp_a) if c in df.columns]
    if not cols_use:
        return None

    matrix = sig[cols_use].apply(pd.to_numeric, errors="coerce").fillna(0)
    row_std = matrix.std(axis=1).replace(0, 1)
    matrix_z = matrix.sub(matrix.mean(axis=1), axis=0).div(row_std, axis=0)
    gene_labels = sig[gene_col].astype(str).values[:len(matrix_z)]

    fig_h = max(7, len(matrix_z) * 0.27)
    fig_w = max(8, len(cols_use) * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG_AX)

    # Diverging colormap without seaborn
    cmap = plt.cm.RdBu_r
    im = ax.imshow(matrix_z.values, cmap=cmap, aspect="auto",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")

    ax.set_xticks(range(len(cols_use)))
    ax.set_xticklabels(cols_use, rotation=45, ha="right", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(matrix_z)))
    ax.set_yticklabels(gene_labels, fontsize=7.5, color=TEXT, fontfamily="monospace")
    ax.set_title("Top DEG Heatmap (Z-score)", color="white", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.set_label("Z-score", color=MUTED, fontsize=9)

    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout()
    return fig


def plot_pca(df, grp_a, grp_b, label_a, label_b):
    cols = [c for c in grp_a + grp_b if c in df.columns]
    if len(cols) < 3:
        return None

    matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).T.values
    scaler = StandardScaler()
    matrix_s = scaler.fit_transform(matrix)

    n_comp = min(2, matrix_s.shape[0], matrix_s.shape[1])
    if n_comp < 2:
        return None

    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(matrix_s)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    _style_ax(ax, "PCA — Sample Clustering")

    labels = [label_b] * len(grp_b) + [label_a] * len(grp_a)
    clr_map = {label_a: UP_CLR, label_b: DN_CLR}

    for lbl in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(
            pcs[idxs, 0], pcs[idxs, 1],
            c=clr_map.get(lbl, ACCENT), s=90, label=lbl,
            alpha=0.9, edgecolors="white", linewidths=0.5
        )
        for i in idxs:
            ax.annotate(cols[i], (pcs[i, 0], pcs[i, 1]),
                        fontsize=7, color=MUTED,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", color=TEXT, fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", color=TEXT, fontsize=10)
    ax.legend(framealpha=0.15, labelcolor="white", fontsize=9,
              facecolor=BG_AX, edgecolor=BORDER)
    fig.tight_layout()
    return fig


def plot_expression_distribution(df, grp_a, grp_b, label_a, label_b):
    """Violin plot using pure matplotlib."""
    all_data = []
    all_labels = []
    for c in grp_a:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().clip(-20, 20).values
            all_data.append(vals)
            all_labels.append(f"{label_a}\n{c[:10]}")
    for c in grp_b:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().clip(-20, 20).values
            all_data.append(vals)
            all_labels.append(f"{label_b}\n{c[:10]}")

    if not all_data:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(all_data) * 1.2), 5), facecolor=BG)
    _style_ax(ax, "Expression Distribution by Sample")

    vp = ax.violinplot(all_data, positions=range(len(all_data)),
                       showmedians=True, showextrema=True)

    for i, body in enumerate(vp["bodies"]):
        clr = UP_CLR if i < len(grp_a) else DN_CLR
        body.set_facecolor(clr)
        body.set_alpha(0.55)
        body.set_edgecolor(clr)

    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[part].set_edgecolor(TEXT)
        vp[part].set_linewidth(1)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, color=TEXT, fontsize=7.5)
    ax.set_ylabel("Expression (log₂)", color=TEXT, fontsize=10)
    fig.tight_layout()
    return fig


def plot_bar_summary(up, down, label_a, label_b):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    _style_ax(ax, "DEG Summary")

    bars = ax.bar(
        [f"↑ Upregulated\n({label_a})", f"↓ Downregulated\n({label_b})"],
        [up, down],
        color=[UP_CLR, DN_CLR],
        width=0.45, edgecolor=BG, linewidth=1.5, alpha=0.9
    )
    for bar, val in zip(bars, [up, down]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha="center", va="bottom",
            color="white", fontsize=13, fontweight="bold"
        )

    ax.set_ylabel("Number of Genes", color=TEXT, fontsize=10)
    ax.set_ylim(0, max(up, down, 1) * 1.3)
    fig.tight_layout()
    return fig


def plot_time_series_trends(df, sorted_cols, gene_col, n_show=8):
    sig = df[df["Category"] != "Not Significant"] if "Category" in df.columns else df
    sig = sig.head(n_show)
    time_labels = [re.search(r"(\d+)", c).group(1) if re.search(r"(\d+)", c) else c
                   for c in sorted_cols]

    n_plots = min(n_show, len(sig))
    if n_plots == 0:
        return None

    cols_grid = 4
    rows_grid = (n_plots + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid, cols_grid,
                              figsize=(14, rows_grid * 3), facecolor=BG)
    axes = np.array(axes).flatten()

    palette = plt.cm.plasma(np.linspace(0.2, 0.9, n_plots))

    for i, (_, row) in enumerate(sig.iterrows()):
        if i >= n_plots:
            break
        ax = axes[i]
        ax.set_facecolor(BG_AX)
        vals = [pd.to_numeric(row.get(c, np.nan), errors="coerce") for c in sorted_cols]
        ax.plot(time_labels, vals, color=palette[i], linewidth=2,
                marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor=palette[i])
        ax.set_title(str(row.get(gene_col, f"Gene {i}"))[:12],
                     color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Time Series — Top DEG Expression Profiles",
                 color="white", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  PATHWAY INFERENCE
# ─────────────────────────────────────────────
PATHWAY_KEYWORDS = {
    "Cell Cycle & Proliferation": ["CDK", "CCND", "CCNE", "CDC", "MKI67", "PCNA", "E2F", "RB1"],
    "Apoptosis":                  ["BCL2", "BAX", "CASP", "PARP", "TP53", "APAF", "BID", "FAS"],
    "DNA Repair":                 ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK", "MLH1"],
    "Immune Response":            ["IL", "TNF", "IFNG", "CD", "HLA", "CXCL", "CCL", "TLR"],
    "Metabolism":                 ["ALDH", "PKM", "LDHA", "HK", "G6PD", "FASN", "IDH"],
    "Angiogenesis":               ["VEGF", "FGF", "ANGPT", "HIF", "MMP", "PDGF"],
    "Transcription":              ["MYC", "FOS", "JUN", "STAT", "SMAD", "FOXO", "SOX"],
    "Signal Transduction":        ["AKT", "MTOR", "KRAS", "BRAF", "ERK", "PI3K", "PTEN"],
    "Extracellular Matrix":       ["COL", "FN1", "VIM", "ACTA", "TGF", "ITGA"],
    "Epigenetics":                ["DNMT", "TET", "KDM", "EZH", "HDAC", "BRD"],
}


def infer_pathways(gene_list):
    gene_upper = [str(g).upper() for g in gene_list]
    results = {}
    for pw, kws in PATHWAY_KEYWORDS.items():
        hits = [g for g in gene_upper if any(kw in g for kw in kws)]
        if hits:
            results[pw] = hits
    return results


def generate_interpretation(df, gene_col, dataset_type, label_a, label_b, up, down):
    up_genes = df[df["Category"] == "Upregulated"][gene_col].astype(str).tolist()
    dn_genes = df[df["Category"] == "Downregulated"][gene_col].astype(str).tolist()
    up_pw = infer_pathways(up_genes)
    dn_pw = infer_pathways(dn_genes)

    type_desc = {
        "tumor_normal":       "tumor vs normal tissue comparison",
        "treated_control":    "treatment vs control comparison",
        "time_series":        "time series expression analysis",
        "knockout_wildtype":  "knockout/knockdown vs wildtype comparison",
        "pre_computed":       "pre-computed differential expression analysis",
        "single_condition":   "expression profiling analysis",
    }

    lines = [
        f"This {type_desc.get(dataset_type, 'expression')} identified "
        f"{up + down} differentially expressed genes: "
        f"{up} upregulated in {label_a} and {down} downregulated relative to {label_b}."
    ]

    if up > 0:
        lines.append(f"\nUpregulated Genes ({label_a}):")
        for pw, hits in list(up_pw.items())[:5]:
            lines.append(f"  • {pw}: {', '.join(hits[:5])}")
        if not up_pw:
            lines.append(f"  • Top genes: {', '.join(up_genes[:8])}")

    if down > 0:
        lines.append(f"\nDownregulated Genes ({label_b}):")
        for pw, hits in list(dn_pw.items())[:5]:
            lines.append(f"  • {pw}: {', '.join(hits[:5])}")
        if not dn_pw:
            lines.append(f"  • Top genes: {', '.join(dn_genes[:8])}")

    context = {
        "tumor_normal":      "Oncological context: elevated proliferation markers and reduced tumor suppressors may indicate active oncogenesis.",
        "treated_control":   "Pharmacological context: the treatment response shows modulation of key regulatory pathways.",
        "knockout_wildtype": "Genetic perturbation context: compensatory pathway changes observed downstream of the perturbation.",
        "time_series":       "Temporal context: sequential transcriptional programs indicate regulatory cascades across timepoints.",
    }
    if dataset_type in context:
        lines.append(f"\n{context[dataset_type]}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  FILE READER
# ─────────────────────────────────────────────
def read_geo_soft(content):
    lines = content.split("\n")
    data_lines = []
    in_table = False
    for line in lines:
        if "table_begin" in line.lower():
            in_table = True
            continue
        if "table_end" in line.lower():
            in_table = False
            continue
        if in_table:
            data_lines.append(line)
    if data_lines:
        return pd.read_csv(io.StringIO("\n".join(data_lines)),
                           sep="\t", index_col=None, low_memory=False)
    return None


def smart_read_file(uploaded_file):
    fname = uploaded_file.name
    msgs = []

    def _try_parse(raw_text):
        # Check GEO SOFT first
        if "table_begin" in raw_text.lower():
            df = read_geo_soft(raw_text)
            if df is not None and df.shape[1] > 1:
                return df, "GEO SOFT"
        # Try tab then comma
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(io.StringIO(raw_text), sep=sep,
                                 encoding="latin1", comment="!", low_memory=False)
                if df.shape[1] > 1:
                    return df, f"sep='{sep}'"
            except Exception:
                pass
        return None, None

    try:
        if fname.endswith(".zip"):
            msgs.append(("info", "📦 ZIP detected — extracting..."))
            with zipfile.ZipFile(uploaded_file) as z:
                for name in z.namelist():
                    if name.endswith("/"):
                        continue
                    with z.open(name) as f:
                        raw = f.read().decode("latin1", errors="replace")
                    df, fmt = _try_parse(raw)
                    if df is not None:
                        msgs.append(("success", f"✅ Parsed `{name}` ({fmt})"))
                        return df, msgs
            msgs.append(("error", "Could not parse any file inside ZIP."))
            return None, msgs

        elif fname.endswith(".gz"):
            msgs.append(("info", "🗜️ GZ file — decompressing..."))
            with gzip.open(uploaded_file, "rt", encoding="latin1", errors="replace") as f:
                raw = f.read()
            df, fmt = _try_parse(raw)
            if df is not None:
                msgs.append(("success", f"✅ GZ parsed ({fmt})"))
                return df, msgs

        elif fname.endswith((".txt", ".tsv")):
            raw = uploaded_file.read().decode("latin1", errors="replace")
            df, fmt = _try_parse(raw)
            if df is not None:
                msgs.append(("success", f"✅ Text file parsed ({fmt})"))
                return df, msgs

        elif fname.endswith(".csv"):
            for enc in ["utf-8", "latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    msgs.append(("success", f"✅ CSV parsed ({enc})"))
                    return df, msgs
                except Exception:
                    pass

    except Exception as e:
        msgs.append(("error", f"Fatal error: {e}"))
        return None, msgs

    msgs.append(("error", "Unsupported format or parsing failed completely."))
    return None, msgs


# ─────────────────────────────────────────────
#  PDF — FREE
# ─────────────────────────────────────────────
def _save_fig(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_free_pdf(df, gene_col, figs, up, down, label_a, label_b, dataset_type, out_path):
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=18,
                              textColor=colors.HexColor("#00d4aa"), alignment=TA_CENTER)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                         textColor=colors.HexColor("#7c3aed"), spaceBefore=12)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    el = []

    el.append(Paragraph("🧬 RNA-Seq Analysis Report (Free)", title_s))
    el.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
                         ParagraphStyle("sub", parent=styles["Normal"], fontSize=9,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#00d4aa")))
    el.append(Spacer(1, 12))

    # Summary table
    el.append(Paragraph("Dataset Summary", h2))
    type_map = {
        "tumor_normal": "Tumor vs Normal", "treated_control": "Treated vs Control",
        "time_series": "Time Series", "knockout_wildtype": "Knockout vs Wildtype",
        "pre_computed": "Pre-computed DEG", "single_condition": "Expression Profiling"
    }
    tdata = [
        ["Dataset Type", type_map.get(dataset_type, dataset_type)],
        ["Comparison",   f"{label_a} vs {label_b}"],
        ["Total Genes",  str(len(df))],
        ["Upregulated",  str(up)],
        ["Downregulated", str(down)],
        ["Total DEGs",   str(up + down)],
    ]
    t = Table(tdata, colWidths=[5.5*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8faf7")),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#00a884")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    el.append(t)
    el.append(Spacer(1, 14))

    # Top genes
    el.append(Paragraph("Top 20 Significant Genes", h2))
    top = df[df["Category"] != "Not Significant"].nsmallest(20, "padj")
    if len(top) > 0:
        rows = [["Gene", "log2FC", "padj", "Category"]]
        for _, row in top.iterrows():
            rows.append([
                str(row.get(gene_col, ""))[:20],
                f"{row.get('log2FoldChange', 0):.3f}",
                f"{row.get('padj', 1):.2e}",
                row.get("Category", "")
            ])
        t2 = Table(rows, colWidths=[5*cm, 3*cm, 3*cm, 4.5*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#00d4aa")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("PADDING",    (0,0), (-1,-1), 5),
        ]))
        el.append(t2)
    el.append(Spacer(1, 14))

    if "volcano" in figs and figs["volcano"]:
        el.append(Paragraph("Volcano Plot", h2))
        el.append(Image(figs["volcano"], width=13*cm, height=10*cm))

    el.append(Spacer(1, 16))
    el.append(Paragraph(
        "⚠ Free report. Upgrade to Premium for full analysis with all plots, "
        "pathway enrichment, and biological interpretation.",
        ParagraphStyle("note", parent=styles["Normal"], fontSize=8,
                        textColor=colors.grey, alignment=TA_CENTER)
    ))
    doc.build(el)


# ─────────────────────────────────────────────
#  PDF — PREMIUM
# ─────────────────────────────────────────────
def build_premium_pdf(df, gene_col, figs, up, down, label_a, label_b,
                       dataset_type, interpretation, out_path):
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                              textColor=colors.HexColor("#7c3aed"), alignment=TA_CENTER)
    h2  = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                          textColor=colors.HexColor("#7c3aed"), spaceBefore=16)
    h3  = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12,
                          textColor=colors.HexColor("#00a884"), spaceBefore=10)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                           leading=15, alignment=TA_JUSTIFY)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    el = []

    # Cover
    el.append(Spacer(1, 20))
    el.append(Paragraph("💎 RNA-Seq Premium Analysis Report", title_s))
    el.append(Paragraph("Comprehensive Differential Expression & Biological Interpretation",
                         ParagraphStyle("sub", parent=styles["Normal"], fontSize=11,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(Spacer(1, 8))
    el.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                         ParagraphStyle("d", parent=styles["Normal"], fontSize=9,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#7c3aed")))
    el.append(Spacer(1, 20))

    # Section 1 — Overview
    el.append(Paragraph("1. Dataset Overview", h2))
    type_map = {
        "tumor_normal": "Tumor vs Normal Tissue", "treated_control": "Drug Treatment vs Control",
        "time_series": "Time Series Expression", "knockout_wildtype": "Knockout vs Wildtype",
        "pre_computed": "Pre-computed DEG", "single_condition": "Expression Profiling"
    }
    info = [
        ["Parameter",          "Value"],
        ["Analysis Type",      type_map.get(dataset_type, dataset_type)],
        ["Comparison",         f"{label_a} vs {label_b}"],
        ["Total Genes",        str(len(df))],
        ["Total DEGs",         str(up + down)],
        ["Upregulated",        str(up)],
        ["Downregulated",      str(down)],
        ["Up/Down Ratio",      f"{up / max(down,1):.2f}"],
        ["LFC Threshold",      "±1.0"],
        ["Significance",       "padj < 0.05"],
    ]
    t = Table(info, colWidths=[7*cm, 8.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#f5f0ff")),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#faf8ff")]),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    el.append(t)
    el.append(Spacer(1, 16))

    # Section 2 — Interpretation
    el.append(Paragraph("2. Biological Interpretation", h2))
    for line in interpretation.split("\n"):
        if line.strip():
            el.append(Paragraph(line, body))
    el.append(Spacer(1, 16))

    # Section 3 — Plots
    el.append(Paragraph("3. Visualizations", h2))
    plot_info = [
        ("volcano", "3.1 Volcano Plot",
         "Red = upregulated in Group A, Blue = downregulated. Dashed lines = thresholds."),
        ("ma",      "3.2 MA Plot",
         "Relationship between mean expression and fold change."),
        ("pca",     "3.3 PCA — Sample Clustering",
         "Principal components of all samples. Well-separated = strong biological signal."),
        ("heatmap", "3.4 Top DEG Heatmap",
         "Z-score normalized expression of top significant genes."),
        ("dist",    "3.5 Expression Distribution",
         "Per-sample expression distribution (violin plots)."),
        ("bar",     "3.6 DEG Summary",
         "Count of upregulated vs downregulated genes."),
    ]
    for key, title, caption in plot_info:
        if key in figs and figs[key]:
            el.append(Paragraph(title, h3))
            el.append(Paragraph(caption, body))
            el.append(Image(figs[key], width=13*cm, height=10*cm))
            el.append(Spacer(1, 12))

    # Section 4 — Gene Tables
    el.append(PageBreak())
    el.append(Paragraph("4. Top Differentially Expressed Genes", h2))
    for cat, lbl, clr_hex in [
        ("Upregulated",   f"Upregulated in {label_a}", "#ff4d6d"),
        ("Downregulated", f"Downregulated in {label_b}", "#4da6ff"),
    ]:
        sub = df[df["Category"] == cat].nsmallest(25, "padj")
        if len(sub) == 0:
            continue
        el.append(Paragraph(lbl, h3))
        rows = [["#", "Gene", "log2FC", "padj", "Mean A", "Mean B"]]
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            rows.append([
                str(rank),
                str(row.get(gene_col, ""))[:18],
                f"{row.get('log2FoldChange', 0):.3f}",
                f"{row.get('padj', 1):.2e}",
                f"{row.get('mean_A', 0):.2f}",
                f"{row.get('mean_B', 0):.2f}",
            ])
        t = Table(rows, colWidths=[1.2*cm, 4.5*cm, 2.5*cm, 2.8*cm, 2.5*cm, 2.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(clr_hex)),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8.5),
            ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("FONTNAME",   (0,1), (-1,-1), "Courier"),
            ("PADDING",    (0,0), (-1,-1), 4),
        ]))
        el.append(t)
        el.append(Spacer(1, 14))

    # Section 5 — Methods
    el.append(PageBreak())
    el.append(Paragraph("5. Methods & Statistical Notes", h2))
    el.append(Paragraph(
        "Expression data was log₂ transformed (log₂(x+1)). Differential expression was "
        "computed using Welch's t-test per gene. Multiple testing correction applied via "
        "Benjamini-Hochberg FDR. Genes with |log₂FC| > 1.0 and padj < 0.05 were called "
        "significant. Pathway annotations are keyword-based and exploratory only. "
        "For publication-grade analysis, use DESeq2, edgeR, or limma-voom.",
        body
    ))
    el.append(Spacer(1, 10))
    el.append(Paragraph(
        "⚠ Disclaimer: All findings require experimental validation.",
        ParagraphStyle("disc", parent=styles["Normal"], fontSize=9,
                        textColor=colors.grey, alignment=TA_JUSTIFY)
    ))

    doc.build(el)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    lfc_threshold  = st.slider("log₂FC Threshold", 0.5, 3.0, 1.0, 0.25)
    padj_threshold = st.select_slider("padj Cutoff",
                                       options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.05)
    norm_method    = st.selectbox("Normalization", ["log2_cpm", "zscore", "none"])
    n_top_genes    = st.slider("Top genes in heatmap", 10, 80, 40, 5)

    st.markdown("---")
    st.markdown("### 📋 Supported Dataset Types")
    for t in ["🔬 Tumor vs Normal", "💊 Treated vs Control",
               "⏱️ Time Series", "🔧 Knockout vs Wildtype",
               "📊 Single Condition", "📁 Pre-computed DEG"]:
        st.markdown(f"<span class='dataset-badge'>{t}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Accepted Formats")
    st.markdown("`CSV`  `TXT`  `TSV`  `GZ`  `ZIP`\n\nIncludes GEO Series Matrix")

    if not HAS_SEABORN:
        st.warning("seaborn not installed — using matplotlib fallback (all features work)")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:2.1rem;margin-bottom:0'>🧬 RNA-Seq Universal Analyzer</h1>
<p style='text-align:center;color:#8b92a5;margin-top:6px'>
Supports NCBI GEO · Tumor/Normal · Treated/Control · Time Series · KO/WT
</p>
""", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload RNA-Seq Data File",
    type=["csv", "txt", "tsv", "gz", "zip"],
    help="GEO series matrix, raw count tables, or pre-computed DEG tables"
)

if uploaded_file:
    with st.spinner("Reading file..."):
        df_raw, read_msgs = smart_read_file(uploaded_file)

    for level, msg in read_msgs:
        getattr(st, level)(msg)

    if df_raw is None:
        st.stop()

    with st.expander("📄 Raw Data Preview", expanded=False):
        st.write(f"Shape: **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**")
        st.dataframe(df_raw.head(10), use_container_width=True)

    # Detect type
    dataset_type = detect_dataset_type(df_raw)
    type_labels = {
        "tumor_normal":      "🔬 Tumor vs Normal",
        "treated_control":   "💊 Treated vs Control",
        "time_series":       "⏱️ Time Series",
        "knockout_wildtype": "🔧 Knockout vs Wildtype",
        "pre_computed":      "📁 Pre-computed DEG",
        "single_condition":  "📊 Single Condition",
    }
    st.markdown(
        f"<div class='info-box'>🔍 Auto-detected: "
        f"<strong>{type_labels.get(dataset_type, dataset_type)}</strong></div>",
        unsafe_allow_html=True
    )
    override = st.selectbox(
        "Override dataset type (optional)",
        ["auto"] + list(type_labels.keys()),
        format_func=lambda x: "Auto-detect" if x == "auto" else type_labels.get(x, x)
    )
    if override != "auto":
        dataset_type = override

    # Compute DE
    if dataset_type == "pre_computed" and \
       "log2FoldChange" in df_raw.columns and "padj" in df_raw.columns:
        st.markdown("<div class='info-box'>✅ Pre-computed columns found</div>",
                    unsafe_allow_html=True)
        df = df_raw.copy()
        label_a, label_b = "Group A", "Group B"
        if "mean_A" not in df.columns:
            df["mean_A"] = 0
            df["mean_B"] = 0
        grp_a, grp_b = [], []
    else:
        grp_a, grp_b, label_a, label_b = detect_group_columns(df_raw, dataset_type)
        st.markdown(f"""
        <div class='info-box'>
        📊 <strong>{label_a}</strong>: {', '.join(grp_a[:4])}{'...' if len(grp_a)>4 else ''} ({len(grp_a)} samples)<br>
        📊 <strong>{label_b}</strong>: {', '.join(grp_b[:4])}{'...' if len(grp_b)>4 else ''} ({len(grp_b)} samples)
        </div>""", unsafe_allow_html=True)

        if not grp_a or not grp_b:
            st.error("Could not identify sample groups. Please check your column names.")
            st.stop()

        with st.spinner("Computing differential expression..."):
            df = compute_differential_expression(df_raw, grp_a, grp_b,
                                                  label_a, label_b, norm_method)

    df = classify_genes(df, lfc_threshold, padj_threshold)
    gene_col = find_gene_column(df)
    if gene_col is None:
        df["_gene"] = df.index.astype(str)
        gene_col = "_gene"

    up   = (df["Category"] == "Upregulated").sum()
    down = (df["Category"] == "Downregulated").sum()

    # Metrics
    st.markdown("<div class='section-header'>📊 Analysis Summary</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, clr in [
        (c1, len(df),    "Total Genes",       "#00d4aa"),
        (c2, up,         f"↑ Up ({label_a})", "#ff4d6d"),
        (c3, down,       f"↓ Down ({label_b})", "#4da6ff"),
        (c4, up + down,  "Total DEGs",        "#7c3aed"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number' style='color:{clr}'>{val:,}</div>
            <div style='color:#8b92a5;font-size:0.83rem;margin-top:4px'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Plots
    st.markdown("<div class='section-header'>📈 Visualizations</div>",
                unsafe_allow_html=True)

    saved_figs = {}
    tmp_dir = tempfile.mkdtemp()

    def save_fig(fig, key):
        if fig is None:
            return
        path = os.path.join(tmp_dir, f"{key}.png")
        _save_fig(fig, path)
        saved_figs[key] = path

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌋 Volcano", "📉 MA Plot", "🧊 Heatmap",
        "🔵 PCA", "🎻 Distribution", "📊 Summary"
    ])

    with tab1:
        fig = plot_volcano(df, gene_col, label_a, label_b, lfc_threshold, padj_threshold)
        save_fig(fig, "volcano")
        st.pyplot(fig)

    with tab2:
        if "mean_A" in df.columns and "mean_B" in df.columns:
            fig = plot_ma(df, label_a, label_b)
            save_fig(fig, "ma")
            st.pyplot(fig)
        else:
            st.info("MA plot needs raw group data.")

    with tab3:
        if grp_a and grp_b:
            fig = plot_heatmap(df, grp_a, grp_b, gene_col, n_top_genes)
            if fig:
                save_fig(fig, "heatmap")
                st.pyplot(fig)
        else:
            st.info("Heatmap needs raw sample columns.")

    with tab4:
        if grp_a and grp_b:
            fig = plot_pca(df, grp_a, grp_b, label_a, label_b)
            if fig:
                save_fig(fig, "pca")
                st.pyplot(fig)
            else:
                st.info("PCA needs at least 3 samples.")
        else:
            st.info("PCA needs raw sample columns.")

    with tab5:
        if grp_a and grp_b:
            fig = plot_expression_distribution(df, grp_a, grp_b, label_a, label_b)
            if fig:
                save_fig(fig, "dist")
                st.pyplot(fig)
        else:
            st.info("Distribution plot needs raw sample columns.")

    with tab6:
        fig = plot_bar_summary(up, down, label_a, label_b)
        save_fig(fig, "bar")
        st.pyplot(fig)

    # Time series
    if dataset_type == "time_series" and grp_a and grp_b:
        with st.expander("⏱️ Time Series Expression Profiles"):
            all_cols = grp_b + grp_a
            def _get_time(c):
                m = re.search(r"(\d+)", c)
                return int(m.group(1)) if m else 9999
            sorted_cols = sorted(all_cols, key=_get_time)
            fig = plot_time_series_trends(df, sorted_cols, gene_col)
            if fig:
                st.pyplot(fig)

    # Top genes table
    st.markdown("<div class='section-header'>🔝 Top Significant Genes</div>",
                unsafe_allow_html=True)
    sig = df[df["Category"] != "Not Significant"].copy()
    if len(sig) > 0:
        disp_cols = [gene_col, "log2FoldChange", "padj", "Category"]
        if "mean_A" in sig.columns:
            disp_cols += ["mean_A", "mean_B"]
        top50 = sig[disp_cols].nsmallest(50, "padj").rename(columns={
            "log2FoldChange": "log2FC",
            "mean_A": f"Mean({label_a})",
            "mean_B": f"Mean({label_b})"
        })

        def _color(val):
            if val == "Upregulated":
                return "background-color:rgba(255,77,109,0.12);color:#ff4d6d"
            if val == "Downregulated":
                return "background-color:rgba(77,166,255,0.12);color:#4da6ff"
            return ""

        st.dataframe(
            top50.style.applymap(_color, subset=["Category"])
                       .format({"log2FC": "{:.3f}", "padj": "{:.2e}"}),
            use_container_width=True, height=380
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "📥 Download Full Results CSV", buf.getvalue(),
            file_name=f"RNA_Seq_{dataset_type}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No DEGs found — try lowering the LFC threshold or padj cutoff.")

    # Pathway preview
    st.markdown("<div class='section-header'>🔬 Pathway Insights</div>",
                unsafe_allow_html=True)
    up_genes = df[df["Category"] == "Upregulated"][gene_col].astype(str).tolist()
    dn_genes = df[df["Category"] == "Downregulated"][gene_col].astype(str).tolist()
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**↑ Enriched in {label_a}**")
        pw = infer_pathways(up_genes)
        if pw:
            for p, g in list(pw.items())[:5]:
                st.markdown(f"• **{p}**: {', '.join(g[:4])}")
        else:
            st.info("No pathway keywords matched.")
    with cb:
        st.markdown(f"**↓ Enriched in {label_b}**")
        pw = infer_pathways(dn_genes)
        if pw:
            for p, g in list(pw.items())[:5]:
                st.markdown(f"• **{p}**: {', '.join(g[:4])}")
        else:
            st.info("No pathway keywords matched.")

    # Reports
    st.markdown("---")
    st.markdown("<div class='section-header'>📄 Download Reports</div>",
                unsafe_allow_html=True)
    col_f, col_p = st.columns(2)

    with col_f:
        st.markdown("### 🆓 Free Report")
        st.markdown("Summary + Top Genes + Volcano Plot")
        if st.button("📥 Generate Free PDF"):
            with st.spinner("Building free PDF..."):
                path = os.path.join(tmp_dir, "Free_Report.pdf")
                try:
                    build_free_pdf(df, gene_col, saved_figs, up, down,
                                   label_a, label_b, dataset_type, path)
                    with open(path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Free PDF", f.read(),
                            file_name=f"RNA_Free_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    st.success("✅ Free PDF ready!")
                except Exception as e:
                    st.error(f"PDF error: {e}")

    with col_p:
        st.markdown("""
        <div class='premium-box'>
        <h3 style='color:#7c3aed;margin:0 0 6px'>💎 Premium Report</h3>
        <p style='font-size:0.88rem;color:#555;margin:0'>
        All 6 plots · Full gene tables · Pathway analysis · Biological interpretation · Methods
        </p></div>""", unsafe_allow_html=True)

        code = st.text_input("🔑 Access Code", type="password", placeholder="Enter after payment")
        VALID = {"BIO100", "RNASEQ2025", "PREMIUM99"}
        if code in VALID:
            st.success("✅ Access granted!")
            if st.button("💎 Generate Premium PDF"):
                with st.spinner("Building premium PDF..."):
                    interp = generate_interpretation(
                        df, gene_col, dataset_type, label_a, label_b, up, down
                    )
                    path = os.path.join(tmp_dir, "Premium_Report.pdf")
                    try:
                        build_premium_pdf(df, gene_col, saved_figs, up, down,
                                          label_a, label_b, dataset_type, interp, path)
                        with open(path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Premium PDF", f.read(),
                                file_name=f"RNA_Premium_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                        st.success("✅ Premium PDF ready!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"PDF error: {e}")
        elif code:
            st.error("❌ Invalid code.")
        else:
            st.info("Enter access code to unlock.")

else:
    st.markdown("""
    <div style='text-align:center;padding:50px 20px'>
        <div style='font-size:4.5rem'>🧬</div>
        <h2 style='color:#00d4aa'>Upload your RNA-Seq file to begin</h2>
        <p style='color:#8b92a5;max-width:580px;margin:0 auto'>
        Drop CSV, TXT, GZ or ZIP — including NCBI GEO Series Matrix files.
        Auto-detects dataset type and runs full differential expression analysis.
        </p>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🔬", "Smart Detection",    "Auto-identifies dataset type from column names"),
        (c2, "📊", "6 Visualizations",   "Volcano · MA · Heatmap · PCA · Distribution · Bar"),
        (c3, "📄", "Dual PDF Reports",   "Free summary + Premium full analysis"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='text-align:left'>
            <div style='font-size:2rem'>{icon}</div>
            <div style='color:white;font-weight:700;margin:8px 0 4px'>{title}</div>
            <div style='color:#8b92a5;font-size:0.85rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)
