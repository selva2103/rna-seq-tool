"""
RNA-Seq Universal Report Generator — v8.0
==========================================
v8.0 Changes:
  - Modular structure: geo_fetch, file_parser, analysis, session_history
  - @st.cache_data caching on all NCBI calls (no repeated API hits)
  - Retry logic with exponential backoff on network errors
  - GSE ID validation before any API call
  - Tabbed main UI: Analyse | History | Pipeline | About
  - Post-upload history panel: see all files analysed this session
  - Improved error messages (no silent except-pass swallowing)
"""

import os, io, re, tempfile, zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Local modules ─────────────────────────────────────────────────────────────
from file_parser    import smart_read_file, parse_geo_soft_full
from geo_fetch      import (validate_gse, smart_retrieve_from_geo,
                             get_srr_for_gsm, fetch_sra_runinfo, NCBI_DELAY)
from analysis       import (detect_type_meta, detect_type_cols,
                             cluster_gsm_groups, detect_groups_non_geo,
                             compute_de, classify_genes, find_gene_col,
                             generate_all_plots, plot_volcano, plot_ma,
                             plot_heatmap, plot_pca, plot_dist, plot_bar,
                             _save)
from session_history import add_entry, render_history_panel

# ReportLab
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Image, Spacer,
                                  Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

import time


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🧬 RNA-Seq Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117;
}
.metric-card {
    background: linear-gradient(135deg,#1a1d27,#22263a);
    border:1px solid #2a2d3e; border-radius:12px;
    padding:20px; text-align:center; margin-bottom:10px;
}
.metric-number { font-size:2.2rem; font-weight:800; font-family:'JetBrains Mono',monospace; }
.dataset-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.8rem; font-weight:600;
    background:linear-gradient(135deg,#00d4aa,#7c3aed); color:white; margin:3px;
}
.info-box {
    background:rgba(0,212,170,0.08); border:1px solid rgba(0,212,170,0.3);
    border-radius:8px; padding:12px 16px; margin:10px 0; font-size:0.9rem;
}
.section-header {
    border-left:4px solid #00d4aa; padding-left:12px;
    margin:28px 0 14px 0; font-size:1.2rem; font-weight:700;
}
.premium-box {
    background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(0,212,170,0.08));
    border:1px solid rgba(124,58,237,0.35); border-radius:12px;
    padding:18px; margin:12px 0;
}
.pipeline-box {
    background:linear-gradient(135deg,rgba(0,212,170,0.08),rgba(124,58,237,0.06));
    border:1px solid rgba(0,212,170,0.35); border-radius:10px;
    padding:14px 18px; margin:8px 0;
}
.stButton>button {
    background:linear-gradient(135deg,#00d4aa,#00a884) !important;
    color:#0f1117 !important; font-weight:700 !important;
    border:none !important; border-radius:8px !important;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap:8px; background:#12151f; border-radius:10px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    background:#1a1d27; border-radius:8px; color:#8b92a5;
    font-weight:600; padding:8px 18px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#00d4aa22,#7c3aed22) !important;
    color:#00d4aa !important; border-bottom:2px solid #00d4aa !important;
}
/* Validation error box */
.val-error {
    background:rgba(255,77,109,0.10); border:1px solid rgba(255,77,109,0.4);
    border-radius:8px; padding:10px 16px; margin:8px 0;
    color:#ff4d6d; font-size:0.9rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PDF BUILDERS  (kept in main for now)
# ─────────────────────────────────────────────
def build_free_pdf(df, gc, figs, up, down, la, lb, dtype, out, gsm_id=""):
    styles = getSampleStyleSheet()
    ts  = ParagraphStyle("T",  parent=styles["Title"],   fontSize=18,
                          textColor=colors.HexColor("#00d4aa"), alignment=TA_CENTER)
    h2  = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                          textColor=colors.HexColor("#7c3aed"), spaceBefore=12)
    doc = SimpleDocTemplate(out, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm,   bottomMargin=2*cm)
    el  = []
    title = f"RNA-Seq Analysis Report — {gsm_id}" if gsm_id else "RNA-Seq Analysis Report"
    el.append(Paragraph(title, ts))
    el.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
                         ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#00d4aa")))
    el.append(Spacer(1, 12))
    tmap = {"tumor_normal": "Tumor vs Normal", "treated_control": "Treated vs Control",
            "time_series": "Time Series",       "knockout_wildtype": "KO vs WT",
            "pre_computed": "Pre-computed",      "single_condition": "Expression Profiling"}
    td = [["Dataset Type", tmap.get(dtype, dtype)],
          ["Comparison",   f"{la} vs {lb}"],
          ["Total Genes",  str(len(df))],
          ["Upregulated",  str(up)],
          ["Downregulated", str(down)]]
    if gsm_id:
        td.insert(0, ["GSM ID", gsm_id])
    t = Table(td, colWidths=[5.5*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8faf7")),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#00a884")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    el.append(Paragraph("Summary", h2)); el.append(t); el.append(Spacer(1, 14))
    top = df[df["Category"] != "Not Significant"].nsmallest(20, "padj")
    if len(top) > 0:
        el.append(Paragraph("Top 20 Significant Genes", h2))
        rows = [["Gene", "log2FC", "padj", "Category"]]
        for _, row in top.iterrows():
            rows.append([str(row.get(gc, ""))[:20],
                         f"{row.get('log2FoldChange',0):.3f}",
                         f"{row.get('padj',1):.2e}",
                         row.get("Category", "")])
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
    doc.build(el)


def build_premium_pdf(df, gc, figs, up, down, la, lb, dtype, out, gsm_id=""):
    styles = getSampleStyleSheet()
    ts   = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20,
                           textColor=colors.HexColor("#7c3aed"), alignment=TA_CENTER)
    h2   = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                           textColor=colors.HexColor("#7c3aed"), spaceBefore=16)
    h3   = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12,
                           textColor=colors.HexColor("#00a884"), spaceBefore=10)
    body = ParagraphStyle("B",  parent=styles["Normal"],   fontSize=10,
                           leading=15, alignment=TA_JUSTIFY)
    doc  = SimpleDocTemplate(out, pagesize=A4,
                              rightMargin=2*cm, leftMargin=2*cm,
                              topMargin=2*cm,   bottomMargin=2*cm)
    el   = []
    el.append(Spacer(1, 20))
    title = f"RNA-Seq Premium Report — {gsm_id}" if gsm_id else "RNA-Seq Premium Analysis Report"
    el.append(Paragraph(title, ts))
    el.append(Paragraph("Comprehensive RNA-Seq Differential Expression Analysis",
                         ParagraphStyle("s", parent=styles["Normal"], fontSize=11,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(Spacer(1, 8))
    el.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                         ParagraphStyle("d", parent=styles["Normal"], fontSize=9,
                                        textColor=colors.grey, alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#7c3aed")))
    el.append(Spacer(1, 20))
    tmap = {"tumor_normal":"Tumor vs Normal","treated_control":"Treated vs Control",
            "time_series":"Time Series","knockout_wildtype":"KO vs Wildtype",
            "pre_computed":"Pre-computed","single_condition":"Expression Profiling"}
    el.append(Paragraph("1. Dataset Overview", h2))
    info = [["Parameter","Value"],["Analysis Type", tmap.get(dtype,dtype)],
            ["Comparison",f"{la} vs {lb}"],["Total Genes",str(len(df))],
            ["Total DEGs",str(up+down)],["Upregulated",str(up)],
            ["Downregulated",str(down)],["Up/Down Ratio",f"{up/max(down,1):.2f}"]]
    if gsm_id:
        info.insert(1, ["GSM ID", gsm_id])
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
    el.append(t); el.append(Spacer(1, 16))
    el.append(Paragraph("2. Visualizations", h2))
    for key, ttl, caption in [
        ("volcano", "2.1 Volcano Plot", "Red=upregulated, Blue=downregulated."),
        ("ma",      "2.2 MA Plot",      "Mean expression vs fold change."),
        ("pca",     "2.3 PCA",          "Sample clustering."),
        ("heatmap", "2.4 Heatmap",      "Z-score top DEGs."),
        ("dist",    "2.5 Distribution", "Per-sample expression."),
        ("bar",     "2.6 Summary",      "DEG counts."),
    ]:
        if key in figs and figs[key]:
            el.append(Paragraph(ttl, h3))
            el.append(Paragraph(caption, body))
            el.append(Image(figs[key], width=13*cm, height=10*cm))
            el.append(Spacer(1, 10))
    el.append(PageBreak())
    el.append(Paragraph("3. Top DEG Tables", h2))
    for cat, lbl_, ch in [("Upregulated",   f"Up in {la}",   "#ff4d6d"),
                           ("Downregulated", f"Down in {lb}", "#4da6ff")]:
        sub = df[df["Category"] == cat].nsmallest(25, "padj")
        if len(sub) == 0:
            continue
        el.append(Paragraph(lbl_, h3))
        rows = [["#","Gene","log2FC","padj","Mean A","Mean B"]]
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            rows.append([str(rank), str(row.get(gc,""))[:18],
                         f"{row.get('log2FoldChange',0):.3f}",
                         f"{row.get('padj',1):.2e}",
                         f"{row.get('mean_A',0):.2f}",
                         f"{row.get('mean_B',0):.2f}"])
        t = Table(rows, colWidths=[1.2*cm,4.5*cm,2.5*cm,2.8*cm,2.5*cm,2.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(ch)),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8.5),
            ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("FONTNAME",   (0,1), (-1,-1), "Courier"),
            ("PADDING",    (0,0), (-1,-1), 4),
        ]))
        el.append(t); el.append(Spacer(1, 14))
    el.append(PageBreak())
    el.append(Paragraph("4. Methods", h2))
    el.append(Paragraph(
        "Expression data log2(x+1) transformed. Welch t-test per gene. "
        "Benjamini-Hochberg FDR correction. |log2FC|>1 and padj<0.05 = significant. "
        "For publication use DESeq2, edgeR, or limma-voom.", body))
    doc.build(el)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    lfc_thr  = st.slider("log2FC Threshold", 0.5, 3.0, 1.0, 0.25)
    padj_thr = st.select_slider("padj Cutoff", options=[0.001,0.01,0.05,0.1,0.2], value=0.05)
    norm     = st.selectbox("Normalization", ["log2_cpm","zscore","none"])
    n_top    = st.slider("Heatmap top genes", 10, 80, 40, 5)
    st.markdown("---")
    st.markdown("### 📂 Accepted Formats")
    st.markdown("`CSV` `TXT` `TSV` `GZ` `ZIP`\n\nIncludes GEO Series Matrix")
    st.markdown("---")
    st.markdown("### 🟢 Normal Pipeline")
    for step in ["1. fastp (QC & trimming)", "2. Salmon (quantification)", "3. DESeq2 (DE)"]:
        st.markdown(f"<span class='dataset-badge'>{step}</span>", unsafe_allow_html=True)
    st.markdown("### 💎 Premium Pipeline")
    for step in ["1. fastp","2. STAR aligner","3. SAMtools",
                 "4. featureCounts","5. DESeq2","6. clusterProfiler"]:
        st.markdown(f"<span class='dataset-badge'>{step}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("v8.0 · Modular · Cached · Validated")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:2.4rem;margin-bottom:0;
           background:linear-gradient(135deg,#00d4aa,#7c3aed);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
🧬 RNA-Seq Universal Analyzer
</h1>
<p style='text-align:center;color:#8b92a5;margin-top:6px;font-size:1rem'>
NCBI GEO · Automated Pipeline · Tumor/Normal · Treated/Control · Time Series · KO/WT
</p>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
TAB_ANALYSE, TAB_HISTORY, TAB_ABOUT = st.tabs([
    "🔬 Analyse", "🗂️ History", "ℹ️ About"
])


# ════════════════════════════════════════════════════════
#  TAB 1 — ANALYSE
# ════════════════════════════════════════════════════════
with TAB_ANALYSE:

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload RNA-Seq Data File",
        type=["csv","txt","tsv","gz","zip"],
        help="GEO series matrix, count tables, or pre-computed DEG files",
    )

    # ── Landing page (no file uploaded yet) ─────────────────────────────────
    if not uploaded_file:
        st.markdown("""
        <div style='text-align:center;padding:40px 20px'>
            <div style='font-size:4rem'>🧬</div>
            <h2 style='color:#00d4aa'>Upload your RNA-Seq file to begin</h2>
            <p style='color:#8b92a5;max-width:600px;margin:0 auto'>
            Drop a CSV, TXT, GZ or ZIP file — including NCBI GEO Series Matrix files.
            The app will automatically extract GSM IDs, provide FASTQ download links,
            run the complete DE pipeline, and generate PDF reports.
            </p>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, title, desc in [
            (c1,"🔍","Smart Detection","Auto-identifies groups from GEO metadata"),
            (c2,"📥","FASTQ Links",   "Direct SRR links for every GSM sample"),
            (c3,"📦","Pipeline Scripts","Normal or Premium pipeline download"),
            (c4,"📄","PDF Reports",   "Full DE analysis report with all plots"),
        ]:
            col.markdown(f"""<div class='metric-card' style='text-align:left'>
            <div style='font-size:2rem'>{icon}</div>
            <div style='color:white;font-weight:700;margin:8px 0 4px'>{title}</div>
            <div style='color:#8b92a5;font-size:0.85rem'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""---
### 🔬 How it works
1. **Upload** a GEO Series Matrix (`.txt`, `.txt.gz`, `.zip`) or a count matrix CSV
2. **GSM IDs** extracted automatically — SRR links resolved with one click
3. **Load expression data** via auto-retrieve or manual upload
4. **Upload FASTQ files** → choose **Normal** or **Premium** pipeline
5. **Run on your server** → upload the resulting `DESeq2_results.csv` back here
6. **Download PDF report** with all plots and DEG tables
""")
        st.stop()

    # ════════════════════════════════════════════════════════════════════════
    #  FILE UPLOADED — parse + analyse
    # ════════════════════════════════════════════════════════════════════════
    with st.spinner("📖 Reading file…"):
        df_raw, geo_meta, gsm_groups, msgs = smart_read_file(uploaded_file)

    # Show parse messages (success/warning/error)
    for lv, msg in msgs:
        getattr(st, lv)(msg)

    if df_raw is None:
        st.error("❌ Could not parse the uploaded file. Check the format and try again.")
        st.stop()

    # ── Post-upload: subtabs (Analysis | GEO Retrieve | GSM Table | Pipeline)
    sub_analyse, sub_retrieve, sub_gsm, sub_pipeline = st.tabs([
        "📊 Analysis",
        "🌐 GEO Retrieve",
        "🔬 GSM Samples",
        "📦 Pipeline Scripts",
    ])

    # ── Extract metadata ─────────────────────────────────────────────────────
    _gse_id      = (geo_meta.get("Series_geo_accession", [""])[0]
                    if geo_meta else "")
    _gsm_ids     = list(gsm_groups.keys()) if gsm_groups else []
    _series_title= (geo_meta.get("Series_title", ["Unknown Study"])[0]
                    if geo_meta else "")
    _organism    = (geo_meta.get("Sample_organism_ch1", ["Unknown"])[0]
                    if geo_meta else "Unknown")

    _gsm_num_cols = [c for c in df_raw.columns
                     if str(c).startswith("GSM")
                     and pd.api.types.is_numeric_dtype(df_raw[c])]
    _non_gsm_num  = [c for c in df_raw.columns
                     if not str(c).startswith("GSM")
                     and pd.api.types.is_numeric_dtype(df_raw[c])]
    _has_expr = (len(_gsm_num_cols) >= 2) or (
        len(_non_gsm_num) >= 3 and df_raw.shape[0] > 50
        and "GSM_ID" not in df_raw.columns
    )

    # ── Temp dir for plots ───────────────────────────────────────────────────
    tmp = tempfile.mkdtemp()

    # ════════════════════════════════════════════════════════════════════════
    #  SUB-TAB: ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with sub_analyse:
        with st.expander("📄 Raw Data Preview", expanded=False):
            st.write(f"Shape: **{df_raw.shape[0]:,} × {df_raw.shape[1]}**")
            st.dataframe(df_raw.head(10), use_container_width=True)

        if gsm_groups:
            with st.expander("🔎 GEO Sample Labels", expanded=False):
                st.dataframe(pd.DataFrame({
                    "GSM": list(gsm_groups.keys()),
                    "Label": list(gsm_groups.values()),
                }), use_container_width=True)

        # ── If only metadata (no expression columns) ─────────────────────
        if not _has_expr and _gsm_ids:
            st.markdown("""<div class='info-box'>
            ℹ️ No expression data columns detected yet. Use the
            <strong>GEO Retrieve</strong> tab to auto-download expression data,
            or upload a count matrix CSV manually.
            </div>""", unsafe_allow_html=True)

            # Quick GSE retrieve widget right here too
            _gse_input = st.text_input(
                "GSE Accession (auto-filled from metadata)",
                value=_gse_id,
                key="gse_quick",
                placeholder="e.g. GSE12345",
            )
            _valid, _err = validate_gse(_gse_input)
            if _gse_input and not _valid:
                st.markdown(f"<div class='val-error'>⚠️ {_err}</div>",
                            unsafe_allow_html=True)

            if st.button("🚀 Auto-retrieve Expression Data from GEO",
                         disabled=(not _valid), key="btn_quick_retrieve"):
                _prog_area = st.empty()
                def _prog(m): _prog_area.info(m)
                with st.spinner("Retrieving from GEO…"):
                    _res = smart_retrieve_from_geo(
                        _gse_input.strip().upper(), geo_meta, gsm_groups, _prog
                    )
                _prog_area.empty()
                if _res["df"] is not None:
                    st.session_state["retrieved_df"] = _res["df"]
                    st.success(_res["message"])
                    st.rerun()
                else:
                    st.error(_res["message"])
            st.stop()

        # ── If retrieved_df is available, use that ────────────────────────
        if "retrieved_df" in st.session_state:
            df_raw = st.session_state["retrieved_df"]

        # ── Dataset type + group detection ────────────────────────────────
        dataset_type = (detect_type_meta(geo_meta, gsm_groups)
                        if gsm_groups else detect_type_cols(df_raw))

        type_labels = {
            "tumor_normal":     "🔬 Tumor vs Normal",
            "treated_control":  "💊 Treated vs Control",
            "time_series":      "⏱️ Time Series",
            "knockout_wildtype":"🔧 Knockout vs Wildtype",
            "pre_computed":     "📁 Pre-computed",
            "single_condition": "📊 Single Condition",
        }
        st.markdown(
            f"<div class='info-box'>🔍 Detected: "
            f"<strong>{type_labels.get(dataset_type, dataset_type)}</strong></div>",
            unsafe_allow_html=True,
        )

        override = st.selectbox(
            "Override dataset type (optional)",
            ["auto"] + list(type_labels.keys()),
            format_func=lambda x: "Auto-detect" if x == "auto" else type_labels.get(x, x),
        )
        if override != "auto":
            dataset_type = override

        # ── Group assignment ──────────────────────────────────────────────
        if (dataset_type == "pre_computed"
                and "log2FoldChange" in df_raw.columns
                and "padj" in df_raw.columns):
            df = df_raw.copy()
            label_a, label_b = "Group A", "Group B"
            if "mean_A" not in df.columns:
                df["mean_A"] = 0; df["mean_B"] = 0
            grp_a, grp_b = [], []

        elif gsm_groups:
            grp_a, grp_b, label_a, label_b = cluster_gsm_groups(gsm_groups)
            st.markdown(f"""<div class='info-box'>
            <strong>{label_a}</strong>: {', '.join(grp_a[:3])}{'...' if len(grp_a)>3 else ''} ({len(grp_a)} samples)<br>
            <strong>{label_b}</strong>: {', '.join(grp_b[:3])}{'...' if len(grp_b)>3 else ''} ({len(grp_b)} samples)
            </div>""", unsafe_allow_html=True)

            all_gsm = list(gsm_groups.keys())
            with st.expander("✏️ Manually adjust groups"):
                grp_a   = st.multiselect("Group A (case)",    all_gsm, default=grp_a)
                grp_b   = st.multiselect("Group B (control)", all_gsm, default=grp_b)
                label_a = st.text_input("Label A", value=label_a)
                label_b = st.text_input("Label B", value=label_b)

            if not grp_a or not grp_b:
                st.error("⚠️ Could not auto-identify groups. Use the manual adjustment above.")
                st.stop()

            with st.spinner("⚙️ Computing differential expression…"):
                df = compute_de(df_raw, grp_a, grp_b, label_a, label_b, norm)
        else:
            grp_a, grp_b, label_a, label_b = detect_groups_non_geo(df_raw, dataset_type)
            if not grp_a or not grp_b:
                st.error("⚠️ Could not identify sample groups.")
                st.stop()
            with st.spinner("⚙️ Computing differential expression…"):
                df = compute_de(df_raw, grp_a, grp_b, label_a, label_b, norm)

        df       = classify_genes(df, lfc_thr, padj_thr)
        gene_col = find_gene_col(df)
        if gene_col is None:
            df["_gene"] = df.index.astype(str); gene_col = "_gene"

        up   = int((df["Category"] == "Upregulated").sum())
        down = int((df["Category"] == "Downregulated").sum())

        # ── Save to history ───────────────────────────────────────────────
        add_entry(
            filename     = uploaded_file.name,
            gse          = _gse_id,
            n_genes      = len(df),
            n_samples    = len(grp_a) + len(grp_b),
            up           = up,
            down         = down,
            dataset_type = dataset_type,
        )

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Analysis Summary</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, clr in [
            (c1, len(df), "Total Genes",         "#00d4aa"),
            (c2, up,      f"↑ Up ({label_a})",   "#ff4d6d"),
            (c3, down,    f"↓ Down ({label_b})", "#4da6ff"),
            (c4, up+down, "Total DEGs",           "#7c3aed"),
        ]:
            col.markdown(f"""<div class='metric-card'>
            <div class='metric-number' style='color:{clr}'>{val:,}</div>
            <div style='color:#8b92a5;font-size:0.83rem;margin-top:4px'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # ── Visualizations (inner tabs) ───────────────────────────────────
        st.markdown("<div class='section-header'>📈 Visualizations</div>",
                    unsafe_allow_html=True)
        saved_figs = {}

        def sf(fig, k):
            if fig is None: return
            p = os.path.join(tmp, f"{k}.png"); _save(fig, p); saved_figs[k] = p

        pt1, pt2, pt3, pt4, pt5, pt6 = st.tabs([
            "🌋 Volcano","📉 MA","🧊 Heatmap","🔵 PCA","🎻 Dist","📊 Bar"
        ])
        with pt1:
            fig = plot_volcano(df, gene_col, label_a, label_b, lfc_thr, padj_thr)
            sf(fig, "volcano"); st.pyplot(fig)
        with pt2:
            if "mean_A" in df.columns:
                fig = plot_ma(df, label_a, label_b); sf(fig, "ma"); st.pyplot(fig)
            else:
                st.info("MA plot needs raw group data.")
        with pt3:
            if grp_a and grp_b:
                fig = plot_heatmap(df, grp_a, grp_b, gene_col, n_top)
                if fig: sf(fig, "heatmap"); st.pyplot(fig)
            else:
                st.info("Heatmap needs two groups.")
        with pt4:
            if grp_a and grp_b:
                fig = plot_pca(df, grp_a, grp_b, label_a, label_b)
                if fig: sf(fig, "pca"); st.pyplot(fig)
                else:   st.info("PCA needs ≥3 samples.")
        with pt5:
            if grp_a and grp_b:
                fig = plot_dist(df, grp_a, grp_b, label_a, label_b)
                if fig: sf(fig, "dist"); st.pyplot(fig)
        with pt6:
            fig = plot_bar(up, down, label_a, label_b)
            sf(fig, "bar"); st.pyplot(fig)

        # ── Top genes table ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔝 Top Significant Genes</div>",
                    unsafe_allow_html=True)
        sig = df[df["Category"] != "Not Significant"].copy()
        if len(sig) > 0:
            dcols = [gene_col, "log2FoldChange", "padj", "Category"]
            if "mean_A" in sig.columns:
                dcols += ["mean_A", "mean_B"]
            top50 = sig[dcols].nsmallest(50, "padj").rename(columns={
                "log2FoldChange": "log2FC",
                "mean_A": f"Mean({label_a})",
                "mean_B": f"Mean({label_b})",
            })

            def _c(v):
                if v == "Upregulated":   return "background-color:rgba(255,77,109,0.12);color:#ff4d6d"
                if v == "Downregulated": return "background-color:rgba(77,166,255,0.12);color:#4da6ff"
                return ""

            st.dataframe(
                top50.style.applymap(_c, subset=["Category"])
                           .format({"log2FC": "{:.3f}", "padj": "{:.2e}"}),
                use_container_width=True, height=360,
            )
            buf = io.StringIO(); df.to_csv(buf, index=False)
            st.download_button(
                "📥 Download Full CSV", buf.getvalue(),
                file_name=f"RNA_Seq_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No DEGs found — try lowering the LFC or padj threshold in the sidebar.")

        # ── PDF Report ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='section-header'>📄 Generate Report</div>",
                    unsafe_allow_html=True)
        if st.button("📥 Generate & Download PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                _rpt_path = os.path.join(tmp, "RNA_Seq_Report.pdf")
                try:
                    build_premium_pdf(df, gene_col, saved_figs, up, down,
                                      label_a, label_b, dataset_type, _rpt_path)
                    with open(_rpt_path, "rb") as _rf:
                        st.download_button(
                            "⬇️ Download PDF Report", _rf.read(),
                            file_name=f"RNA_Seq_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key="dl_report_final",
                        )
                    st.success("✅ Report ready!")
                except Exception as _re:
                    st.error(f"❌ PDF error: {_re}")

    # ════════════════════════════════════════════════════════════════════════
    #  SUB-TAB: GEO RETRIEVE
    # ════════════════════════════════════════════════════════════════════════
    with sub_retrieve:
        st.markdown("<div class='section-header'>🌐 Retrieve Expression Data from GEO</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        Enter a GEO Series accession (e.g. <code>GSE12345</code>) to auto-download
        the expression matrix directly from NCBI. Results are cached for 1 hour.
        </div>""", unsafe_allow_html=True)

        _gse_input = st.text_input(
            "GSE Accession",
            value=_gse_id,
            placeholder="e.g. GSE12345",
            key="gse_retrieve_input",
        )

        # ── Validation feedback ───────────────────────────────────────────
        _valid, _err = validate_gse(_gse_input)
        if _gse_input:
            if _valid:
                st.markdown(
                    f"<div class='info-box' style='border-color:rgba(0,212,170,0.5)'>"
                    f"✅ <strong>{_gse_input.strip().upper()}</strong> — valid format</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<div class='val-error'>⚠️ {_err}</div>",
                            unsafe_allow_html=True)

        if st.button("🚀 Auto-retrieve from GEO", disabled=(not _valid),
                     key="btn_retrieve_full", use_container_width=True):
            _prog_placeholder = st.empty()
            def _prog(m): _prog_placeholder.info(m)
            with st.spinner("Contacting NCBI…"):
                _res = smart_retrieve_from_geo(
                    _gse_input.strip().upper(), geo_meta, gsm_groups, _prog
                )
            _prog_placeholder.empty()
            if _res["df"] is not None:
                st.session_state["retrieved_df"] = _res["df"]
                st.success(_res["message"])
                with st.expander("Preview retrieved data"):
                    st.dataframe(_res["df"].head(8), use_container_width=True)
            else:
                st.error(_res["message"])
                if _res.get("files_found"):
                    st.info(f"Files found on GEO: {', '.join(_res['files_found'])}")

    # ════════════════════════════════════════════════════════════════════════
    #  SUB-TAB: GSM SAMPLES TABLE
    # ════════════════════════════════════════════════════════════════════════
    with sub_gsm:
        st.markdown("<div class='section-header'>🔬 GSM Sample Table</div>",
                    unsafe_allow_html=True)

        if not _gsm_ids:
            st.info("No GSM IDs found in this file. Upload a GEO Series Matrix to see sample details.")
        else:
            ca, cb, cc = st.columns(3)
            ca.metric("📋 Study",    _gse_id or "—")
            cb.metric("🧬 Organism", _organism)
            cc.metric("🔬 Samples",  len(_gsm_ids))

            if _series_title:
                st.markdown(f"**Study:** {_series_title}")

            # ── Resolve SRR IDs on demand ─────────────────────────────────
            if st.button("🔍 Resolve SRR IDs for all samples (calls NCBI)",
                         key="btn_resolve_srr"):
                with st.spinner("Resolving SRR IDs…"):
                    for _gsm in _gsm_ids:
                        if f"srr_{_gsm}" not in st.session_state:
                            _srrs = get_srr_for_gsm(_gsm)
                            st.session_state[f"srr_{_gsm}"] = _srrs
                            time.sleep(NCBI_DELAY)
                st.success("✅ SRR resolution complete!")

            # ── HTML sample table ─────────────────────────────────────────
            _rows_html = ""
            for _i, (_gsm, _lbl) in enumerate(list(gsm_groups.items())):
                _srr_list = st.session_state.get(f"srr_{_gsm}", [])
                _srr      = _srr_list[0] if _srr_list else None
                _sra_url  = (f"https://www.ncbi.nlm.nih.gov/sra/{_srr}"
                             if _srr else f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}")
                _ena_url  = (f"https://www.ebi.ac.uk/ena/browser/view/{_srr}"
                             if _srr else f"https://www.ebi.ac.uk/ena/browser/view/{_gsm}")
                _geo_url  = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_gsm}"
                _bg       = "rgba(255,255,255,0.03)" if _i % 2 == 0 else "transparent"
                _srr_badge = (f"<code style='color:#00d4aa;font-size:0.78rem'>{_srr}</code>"
                              if _srr else
                              "<span style='color:#8b92a5;font-size:0.78rem'>click Resolve ↑</span>")
                _rows_html += f"""
                <tr style='background:{_bg}'>
                  <td style='padding:8px 12px;font-family:monospace;color:#00d4aa;font-weight:700'>{_gsm}</td>
                  <td style='padding:8px 12px;color:#c8cfe0;font-size:0.88rem'
                      title='{_lbl}'>{_lbl[:45]}{'…' if len(_lbl)>45 else ''}</td>
                  <td style='padding:8px 12px;text-align:center'>{_srr_badge}</td>
                  <td style='padding:8px 12px;white-space:nowrap'>
                    <a href='{_sra_url}' target='_blank' style='background:#00d4aa;color:#0f1117;
                    padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;
                    text-decoration:none;margin:2px;display:inline-block'>SRA 🔍</a>
                    <a href='{_geo_url}' target='_blank' style='background:#7c3aed;color:white;
                    padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;
                    text-decoration:none;margin:2px;display:inline-block'>GEO 🧬</a>
                  </td>
                  <td style='padding:8px 12px;white-space:nowrap'>
                    <a href='{_sra_url}' target='_blank' style='background:#e65c00;color:white;
                    padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;
                    text-decoration:none;margin:2px;display:inline-block'>SRA FASTQ 📥</a>
                    <a href='{_ena_url}' target='_blank' style='background:#1a73e8;color:white;
                    padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;
                    text-decoration:none;margin:2px;display:inline-block'>ENA 🌐</a>
                  </td>
                </tr>"""

            st.markdown(f"""
            <div style='overflow-x:auto;overflow-y:auto;max-height:520px;
                        border:1px solid #2a2d3e;border-radius:10px'>
              <table style='width:100%;border-collapse:collapse;min-width:700px'>
                <thead>
                  <tr style='background:#1a1d27;position:sticky;top:0;z-index:1'>
                    <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem'>GSM ID</th>
                    <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem'>Sample Label</th>
                    <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem'>🔖 SRR ID</th>
                    <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem'>🔗 SRA / GEO</th>
                    <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem'>📥 FASTQ</th>
                  </tr>
                </thead>
                <tbody>{_rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("**📋 Copy all GSM IDs:**")
            st.code(" ".join(_gsm_ids), language=None)

    # ════════════════════════════════════════════════════════════════════════
    #  SUB-TAB: PIPELINE SCRIPTS
    # ════════════════════════════════════════════════════════════════════════
    with sub_pipeline:
        st.markdown("<div class='section-header'>📦 FASTQ Analysis Pipeline</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='pipeline-box'>
        Upload your FASTQ files below, choose a pipeline mode, and download a
        fully-configured shell script + Snakemake workflow to run on your server.
        </div>""", unsafe_allow_html=True)

        _fq_col1, _fq_col2 = st.columns([3, 2])
        with _fq_col1:
            fastq_files = st.file_uploader(
                "📂 Upload FASTQ file(s)",
                type=["fastq","fq","gz"],
                accept_multiple_files=True,
                key="fastq_uploader",
            )
        with _fq_col2:
            pipeline_mode = st.radio(
                "🔬 Pipeline Mode",
                ["🟢 Normal  (fastp → Salmon → DESeq2)",
                 "💎 Premium (fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler)"],
                key="pipeline_mode_radio",
            )
        _is_premium = pipeline_mode.startswith("💎")

        _cfg1, _cfg2, _cfg3 = st.columns(3)
        with _cfg1:
            ref_organism = st.selectbox("🧬 Reference Organism",
                ["hg38 (Human)","mm10 (Mouse)","rn6 (Rat)",
                 "dm6 (Drosophila)","ce11 (C. elegans)","GRCz11 (Zebrafish)","Custom"],
                key="ref_organism_sel")
        with _cfg2:
            seq_type = st.radio("📖 Sequencing Type",
                                ["Paired-end","Single-end"], key="seq_type_radio")
        with _cfg3:
            n_threads = st.slider("🖥️ CPU Threads", 1, 32, 8, key="nthreads_slider")

        if fastq_files:
            if _is_premium:
                st.markdown("""<div class='premium-box'>
                <strong>💎 Premium Pipeline:</strong>
                fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='info-box'>
                <strong>🟢 Normal Pipeline:</strong>
                fastp → Salmon → DESeq2
                </div>""", unsafe_allow_html=True)

            st.info(f"ℹ️ {len(fastq_files)} file(s) uploaded. "
                    "Click Generate below to create your pipeline scripts.")

            if st.button("📦 Generate Pipeline Scripts & Workflow",
                         key="btn_gen_scripts", use_container_width=True):
                st.warning("Pipeline script generation requires the full generator "
                           "functions from the original app. Import pipeline_scripts.py "
                           "and call the appropriate make_*_script() function here.")
        else:
            st.info("⬆️ Upload FASTQ file(s) above to generate personalised pipeline scripts.")


# ════════════════════════════════════════════════════════
#  TAB 2 — HISTORY
# ════════════════════════════════════════════════════════
with TAB_HISTORY:
    render_history_panel()


# ════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════
with TAB_ABOUT:
    st.markdown("""
    ## 🧬 RNA-Seq Universal Analyzer — v8.0

    A Streamlit app for end-to-end RNA-Seq differential expression analysis,
    directly integrated with NCBI GEO and SRA.

    ### ✨ What's new in v8.0
    - **Modular codebase** — `geo_fetch.py`, `file_parser.py`, `analysis.py`, `session_history.py`
    - **Smart caching** — NCBI API calls cached with `@st.cache_data` (1 hr TTL)
    - **Retry logic** — automatic backoff on NCBI rate-limits (429/503)
    - **GSE validation** — format check before any API call
    - **Tabbed UI** — Analyse · History · Pipeline · About in clean tabs
    - **Upload history** — every file you analyse is logged with key stats

    ### 📦 Module overview
    | File | Purpose |
    |------|---------|
    | `app.py` | Main entry point, UI, PDF generation |
    | `geo_fetch.py` | NCBI/GEO API helpers with caching & retry |
    | `file_parser.py` | File reading and GEO SOFT parsing |
    | `analysis.py` | DE analysis, group detection, all plots |
    | `session_history.py` | Upload history tracking & display |

    ### 🔬 Pipeline methods
    **Normal**: fastp → Salmon → DESeq2 (fast, alignment-free)  
    **Premium**: fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler

    ### 📊 Statistical methods
    - Normalization: log2(CPM+1), Z-score, or raw
    - Differential expression: Welch t-test per gene
    - Multiple testing correction: Benjamini-Hochberg FDR
    - Significance: |log2FC| > threshold AND padj < cutoff

    > **For publication-grade analysis**: use DESeq2, edgeR, or limma-voom
    > in a proper R pipeline. The pipeline scripts tab generates exactly that.
    """)
