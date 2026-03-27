"""
analysis.py — Differential expression analysis, group detection, and plots
Extracted from app.py v7.0 for modular structure.
"""

import re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ── Dataset type detection ────────────────────────────────────────────────────
DTYPE_PATS = {
    "tumor_normal":      [r"\btumor\b", r"\bnormal\b", r"\bcancer\b", r"\bmalignant\b"],
    "treated_control":   [r"\btreated?\b", r"\bcontrol\b", r"\bdrug\b", r"\bstress\b",
                          r"\bcus\b", r"\bctrl\b", r"\bmdd\b", r"\bdepressed?\b"],
    "time_series":       [r"\b\d+h\b", r"\b\d+hr\b", r"\b\d+days?\b", r"\btime\b"],
    "knockout_wildtype": [r"\bko\b", r"\bwt\b", r"\bknockout\b", r"\bknockdown\b"],
}


def detect_type_meta(geo_meta: dict, gsm_groups: dict) -> str:
    combined = " ".join(gsm_groups.values()).lower() if gsm_groups else ""
    for dtype, pats in DTYPE_PATS.items():
        for p in pats:
            if re.search(p, combined, re.IGNORECASE):
                return dtype
    return "single_condition"


def detect_type_cols(df: pd.DataFrame) -> str:
    cols = " ".join(df.columns).lower()
    for dtype, pats in DTYPE_PATS.items():
        for p in pats:
            if re.search(p, cols, re.IGNORECASE):
                return dtype
    if "log2FoldChange" in df.columns and "padj" in df.columns:
        return "pre_computed"
    return "single_condition"


# ── Group clustering ─────────────────────────────────────────────────────────
def cluster_gsm_groups(gsm_groups: dict) -> tuple:
    """Auto-assign GSMs to Group A / B based on label keywords."""
    if not gsm_groups:
        return [], [], "Group A", "Group B"

    gsm_list   = list(gsm_groups.keys())
    label_list = list(gsm_groups.values())

    POS = [r"\btumor\b", r"\bcancer\b", r"\bmalignant\b", r"\btreated?\b",
           r"\btreatment\b", r"\bdrug\b", r"\bstimulat\b", r"\binfected\b",
           r"\bko\b", r"\bknockout\b", r"\bkd\b", r"\bknockdown\b", r"\bsiRNA\b",
           r"\bmutant\b", r"\bcrispr\b", r"\bstress\b", r"\bdisease\b",
           r"\bcase\b", r"\bpatient\b", r"\bmdd\b", r"\bdepressed?\b", r"\bsi\b"]
    NEG = [r"\bnormal\b", r"\bbenign\b", r"\bcontrol\b", r"\bctrl\b", r"\buntreat\b",
           r"\bvehicle\b", r"\bdmso\b", r"\bmock\b", r"\bwt\b", r"\bwildtype\b",
           r"\bscramble\b", r"\bhealthy\b", r"\badjacent\b", r"\bnaive\b", r"\bsham\b"]

    grp_a, grp_b = [], []
    for gsm, lbl in zip(gsm_list, label_list):
        ll  = lbl.lower()
        pos = any(re.search(p, ll) for p in POS)
        neg = any(re.search(p, ll) for p in NEG)
        if pos and not neg:
            grp_a.append(gsm)
        elif neg and not pos:
            grp_b.append(gsm)

    if not grp_a or not grp_b:
        unique = list(dict.fromkeys(label_list))
        if len(unique) >= 2:
            mid   = len(unique) // 2
            set_a = set(unique[mid:]); set_b = set(unique[:mid])
            grp_a = [g for g, l in zip(gsm_list, label_list) if l in set_a]
            grp_b = [g for g, l in zip(gsm_list, label_list) if l in set_b]
        else:
            mid   = len(gsm_list) // 2
            grp_a = gsm_list[mid:]
            grp_b = gsm_list[:mid]

    def _label(subset):
        vals = [gsm_groups[g] for g in subset if g in gsm_groups]
        if not vals:
            return "Group"
        words = re.findall(r"[A-Za-z]{3,}", " ".join(vals))
        if words:
            from collections import Counter
            noise = {"and", "the", "for", "with", "from", "that", "this",
                     "sample", "ctrl"}
            words = [w for w in words if w.lower() not in noise]
            if words:
                return Counter(words).most_common(1)[0][0].capitalize()
        return vals[0][:20]

    la = _label(grp_a); lb = _label(grp_b)
    if la == lb:
        la, lb = "Group A", "Group B"
    return grp_a, grp_b, la, lb


def detect_groups_non_geo(df: pd.DataFrame, dtype: str) -> tuple:
    gene_like = {"gene", "geneid", "gene_id", "symbol", "genename", "probe_id",
                 "probeid", "id_ref", "id", "ensembl", "transcript", "name"}
    scols = [c for c in df.columns
             if c.lower().replace("_", "") not in gene_like
             and pd.api.types.is_numeric_dtype(df[c])]
    if len(scols) < 2:
        return scols[:1], scols[1:2], "Group A", "Group B"

    def _m(pats, cols):
        return [c for c in cols if any(re.search(p, c, re.IGNORECASE) for p in pats)]

    if dtype == "tumor_normal":
        a = _m([r"tumor", r"cancer", r"malignant"], scols)
        b = _m([r"normal", r"benign"], scols)
        la, lb = "Tumor", "Normal"
    elif dtype == "treated_control":
        a = _m([r"treat", r"drug", r"stimul"], scols)
        b = _m([r"control", r"untreat", r"dmso"], scols)
        la, lb = "Treated", "Control"
    elif dtype == "knockout_wildtype":
        a = _m([r"\bko\b", r"knockout", r"\bkd\b"], scols)
        b = _m([r"\bwt\b", r"wildtype"], scols)
        la, lb = "KO", "WT"
    elif dtype == "time_series":
        def _t(c):
            m = re.search(r"(\d+)", c)
            return int(m.group(1)) if m else 9999
        sc  = sorted(scols, key=_t)
        mid = len(sc) // 2
        a, b, la, lb = sc[mid:], sc[:mid], "Late", "Early"
    else:
        half = len(scols) // 2
        a, b, la, lb = scols[half:], scols[:half], "Group A", "Group B"

    if not a or not b:
        half = len(scols) // 2
        b, a = scols[:half], scols[half:]
    return a, b, la, lb


# ── Differential expression ──────────────────────────────────────────────────
def compute_de(df: pd.DataFrame, grp_a: list, grp_b: list,
               label_a: str, label_b: str, norm: str = "log2_cpm") -> pd.DataFrame:
    expr = df.copy()
    for c in grp_a + grp_b:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.dropna(subset=grp_a + grp_b, how="all")

    av = expr[grp_a]; bv = expr[grp_b]
    if norm == "log2_cpm":
        an = np.log2(av.clip(lower=0) + 1)
        bn = np.log2(bv.clip(lower=0) + 1)
    elif norm == "zscore":
        an = (av - av.mean()) / (av.std() + 1e-9)
        bn = (bv - bv.mean()) / (bv.std() + 1e-9)
    else:
        an, bn = av, bv

    expr["mean_A"] = an.mean(axis=1)
    expr["mean_B"] = bn.mean(axis=1)
    expr["log2FoldChange"] = expr["mean_A"] - expr["mean_B"]

    pv = []
    if len(grp_a) >= 2 and len(grp_b) >= 2:
        for i in range(len(expr)):
            ra = an.iloc[i].dropna().values
            rb = bn.iloc[i].dropna().values
            if len(ra) >= 2 and len(rb) >= 2:
                _, p = ttest_ind(ra, rb, equal_var=False)
                pv.append(float(p) if np.isfinite(p) else 1.0)
            else:
                pv.append(1.0)
    else:
        pv = (1 / (1 + expr["log2FoldChange"].abs())).tolist()

    expr["pvalue"] = pv
    pvn = np.array(pv); n = len(pvn); o = np.argsort(pvn)
    pa  = pvn.copy()
    pa[o] = pvn[o] * n / (np.arange(n) + 1)
    pa  = np.minimum.accumulate(pa[::-1])[::-1]
    expr["padj"]    = np.clip(pa, 0, 1)
    expr["label_A"] = label_a
    expr["label_B"] = label_b
    return expr


def classify_genes(df: pd.DataFrame, lfc: float = 1.0,
                   padj: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df["Category"] = "Not Significant"
    df.loc[(df["padj"] < padj) & (df["log2FoldChange"] >  lfc), "Category"] = "Upregulated"
    df.loc[(df["padj"] < padj) & (df["log2FoldChange"] < -lfc), "Category"] = "Downregulated"
    return df


def find_gene_col(df: pd.DataFrame) -> str | None:
    for c in ["gene", "Gene", "gene_id", "GeneID", "gene_name", "symbol", "Symbol",
              "ID_REF", "probe_id", "NAME", "name", "Ensembl", "feature_id"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


# ── Plot helpers ─────────────────────────────────────────────────────────────
BG = "#0f1117"; AX = "#12151f"; BD = "#2a2d3e"
TX = "#c8cfe0"; MT = "#8b92a5"; UP = "#ff4d6d"; DN = "#4da6ff"; AC = "#00d4aa"


def _ax(ax, t=""):
    ax.set_facecolor(AX)
    ax.tick_params(colors=MT, labelsize=9)
    for s in ax.spines.values():
        s.set_color(BD)
    if t:
        ax.set_title(t, color="white", fontsize=13, fontweight="bold", pad=10)


def _save(fig, path: str):
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_volcano(df, gc, la, lb, lt, pt):
    df = df.dropna(subset=["log2FoldChange", "padj"]).copy()
    df["lp"] = -np.log10(df["padj"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    _ax(ax, "Volcano Plot")
    for cat, clr, sz, al in [("Not Significant", "#3a3f55", 7, 0.3),
                               ("Downregulated", DN, 25, 0.85),
                               ("Upregulated", UP, 25, 0.85)]:
        sub = df[df["Category"] == cat]
        ax.scatter(sub["log2FoldChange"], sub["lp"], c=clr, s=sz, alpha=al,
                   label=f"{cat} (n={len(sub)})", edgecolors="none")
    ax.axvline(x=lt,  color=BD, ls="--", lw=1, alpha=0.8)
    ax.axvline(x=-lt, color=BD, ls="--", lw=1, alpha=0.8)
    ax.axhline(y=-np.log10(pt), color=BD, ls="--", lw=1, alpha=0.8)
    top = pd.concat([df[df["Category"] == "Upregulated"].nsmallest(8, "padj"),
                     df[df["Category"] == "Downregulated"].nsmallest(8, "padj")])
    for _, row in top.iterrows():
        clr = UP if row["Category"] == "Upregulated" else DN
        ax.annotate(str(row[gc])[:14], (row["log2FoldChange"], row["lp"]),
                    fontsize=7, color=clr, fontfamily="monospace",
                    xytext=(5, 3), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=clr, alpha=0.4, lw=0.6))
    ax.set_xlabel(f"log2FC ({la}/{lb})", color=TX, fontsize=10)
    ax.set_ylabel("-log10(padj)", color=TX, fontsize=10)
    ax.legend(framealpha=0.15, labelcolor="white", fontsize=8, facecolor=AX, edgecolor=BD)
    fig.tight_layout()
    return fig


def plot_ma(df, la, lb):
    df = df.dropna(subset=["log2FoldChange", "mean_A", "mean_B"]).copy()
    df["A"] = (df["mean_A"] + df["mean_B"]) / 2
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _ax(ax, "MA Plot")
    for cat, clr, sz, al in [("Not Significant", "#3a3f55", 6, 0.3),
                               ("Downregulated", DN, 20, 0.8),
                               ("Upregulated", UP, 20, 0.8)]:
        sub = df[df["Category"] == cat]
        ax.scatter(sub["A"], sub["log2FoldChange"], c=clr, s=sz, alpha=al, edgecolors="none")
    ax.axhline(0, color=AC, lw=1, alpha=0.6)
    ax.set_xlabel("Average Expression", color=TX, fontsize=10)
    ax.set_ylabel(f"log2FC ({la}/{lb})", color=TX, fontsize=10)
    fig.tight_layout()
    return fig


def plot_heatmap(df, grp_a, grp_b, gc, n_top=40):
    sig = df[df["Category"] != "Not Significant"].copy()
    if len(sig) < 2:
        return None
    top = sig.nsmallest(min(n_top, len(sig)), "padj")
    cols = [c for c in grp_a + grp_b if c in top.columns]
    if not cols:
        return None
    mat = top[cols].copy()
    for c in cols:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")
    mat = mat.dropna()
    if mat.shape[0] < 2:
        return None
    mat = ((mat.T - mat.T.mean()) / (mat.T.std() + 1e-9)).T

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), min(16, n_top * 0.32)),
                            facecolor=BG)
    _ax(ax, "Top DEG Heatmap")
    if HAS_SEABORN:
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0,
                    yticklabels=(top[gc].astype(str).tolist()
                                 if gc in top.columns else False),
                    xticklabels=cols, linewidths=0.3,
                    cbar_kws={"shrink": 0.7, "label": "Z-score"})
        ax.tick_params(axis="x", colors=MT, labelsize=8, rotation=45)
        ax.tick_params(axis="y", colors=MT, labelsize=7)
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r")
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig


def plot_pca(df, grp_a, grp_b, la, lb):
    cols = [c for c in grp_a + grp_b if c in df.columns]
    if len(cols) < 3:
        return None
    mat = df[cols].select_dtypes(include=np.number).fillna(0).T
    if mat.shape[0] < 3:
        return None
    try:
        sc  = StandardScaler()
        X   = sc.fit_transform(mat)
        pca = PCA(n_components=2)
        Xp  = pca.fit_transform(X)
        labels = [la] * len(grp_a) + [lb] * len(grp_b)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
        _ax(ax, "PCA Plot")
        for lbl, clr in [(la, UP), (lb, DN)]:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            ax.scatter(Xp[idx, 0], Xp[idx, 1], c=clr, s=80,
                       alpha=0.85, label=lbl, edgecolors="white", linewidths=0.4)
        for i, col in enumerate(cols):
            ax.annotate(col, (Xp[i, 0], Xp[i, 1]),
                        fontsize=7, color=MT, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color=TX)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color=TX)
        ax.legend(framealpha=0.15, labelcolor="white", facecolor=AX, edgecolor=BD)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_dist(df, grp_a, grp_b, la, lb):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    _ax(ax, "Expression Distribution")
    for cols, lbl, clr in [(grp_a, la, UP), (grp_b, lb, DN)]:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            continue
        vals = df[valid].values.flatten()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        if HAS_SEABORN:
            sns.kdeplot(vals, ax=ax, color=clr, label=lbl, fill=True, alpha=0.3)
        else:
            ax.hist(vals, bins=60, color=clr, alpha=0.4, label=lbl, density=True)
    ax.set_xlabel("Expression Value", color=TX, fontsize=10)
    ax.set_ylabel("Density", color=TX, fontsize=10)
    ax.legend(framealpha=0.15, labelcolor="white", facecolor=AX, edgecolor=BD)
    fig.tight_layout()
    return fig


def plot_bar(up: int, down: int, la: str, lb: str):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    _ax(ax, "DEG Summary")
    bars = ax.bar([f"↑ Up\n({la})", f"↓ Down\n({lb})", "Total DEGs"],
                  [up, down, up + down],
                  color=[UP, DN, AC], edgecolor=BD, linewidth=0.8, width=0.55)
    for bar, val in zip(bars, [up, down, up + down]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:,}", ha="center", va="bottom", color="white",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("Gene Count", color=TX, fontsize=10)
    ax.set_ylim(0, max(up + down, 1) * 1.18)
    fig.tight_layout()
    return fig


def generate_all_plots(df: pd.DataFrame, gene_col: str,
                       grp_a: list, grp_b: list,
                       label_a: str, label_b: str,
                       lfc_thr: float, padj_thr: float,
                       n_top: int, tmp_dir: str) -> dict:
    """Generate all plots, save to tmp_dir, return {key: path} dict."""
    up   = (df["Category"] == "Upregulated").sum()
    down = (df["Category"] == "Downregulated").sum()
    saved = {}

    def sf(fig, k):
        if fig is None:
            return
        p = os.path.join(tmp_dir, f"{k}.png")
        _save(fig, p)
        saved[k] = p

    sf(plot_volcano(df, gene_col, label_a, label_b, lfc_thr, padj_thr), "volcano")
    if "mean_A" in df.columns:
        sf(plot_ma(df, label_a, label_b), "ma")
    if grp_a and grp_b:
        sf(plot_heatmap(df, grp_a, grp_b, gene_col, n_top), "heatmap")
        sf(plot_pca(df, grp_a, grp_b, label_a, label_b), "pca")
        sf(plot_dist(df, grp_a, grp_b, label_a, label_b), "dist")
    sf(plot_bar(up, down, label_a, label_b), "bar")
    return saved
