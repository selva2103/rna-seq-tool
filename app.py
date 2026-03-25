"""
RNA-Seq Universal Report Generator — v3.1 AI + Auto-Retrieval Edition
=======================================================================
NEW in v3.1:
  - Auto-retrieval of expression data directly from NCBI GEO / SRA
  - Reads GSM/GSE IDs from series matrix and fetches processed counts
  - Falls back to SRA RunInfo if no supplementary counts found
  - All v3.0 AI features retained (interpretation, chatbot, pathways, auto-settings)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import zipfile, gzip, io, os, re, json, tempfile, time
import urllib.request, urllib.error, urllib.parse
from datetime import datetime

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer,
    Table, TableStyle, PageBreak, HRFlowable
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
    page_title="🧬 RNA-Seq AI Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

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
.ai-box {
    background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(0,212,170,0.08));
    border:1px solid rgba(124,58,237,0.4);
    border-radius:12px; padding:18px; margin:12px 0;
}
.chat-user {
    background:rgba(0,212,170,0.1); border-radius:12px 12px 4px 12px;
    padding:10px 14px; margin:6px 0; text-align:right;
}
.chat-ai {
    background:rgba(124,58,237,0.1); border-radius:12px 12px 12px 4px;
    padding:10px 14px; margin:6px 0;
    border-left:3px solid #7c3aed;
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
.ai-badge {
    background:linear-gradient(135deg,#7c3aed,#00d4aa);
    color:white; padding:3px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:700; margin-left:8px;
}
.stButton>button {
    background:linear-gradient(135deg,#00d4aa,#00a884) !important;
    color:#0f1117 !important; font-weight:700 !important;
    border:none !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CLAUDE API HELPER
# ─────────────────────────────────────────────
def call_claude(system_prompt: str, user_message: str,
                max_tokens: int = 1500) -> str:
    """
    Call the Anthropic Claude API.
    Returns the text response or an error string.
    """
    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"


def call_claude_chat(messages: list, system_prompt: str,
                     max_tokens: int = 1000) -> str:
    """Multi-turn chat version of the API call."""
    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except Exception as e:
        return f"[AI Error: {str(e)}]"


# ─────────────────────────────────────────────
#  AI FEATURE 1 — AUTO-SELECT SETTINGS
# ─────────────────────────────────────────────
def ai_auto_settings(geo_meta: dict, gsm_groups: dict,
                     series_title: str) -> dict:
    """
    Ask Claude to recommend analysis settings based on dataset metadata.
    Returns dict with recommended lfc_threshold, padj_threshold,
    norm_method, dataset_type, label_a, label_b.
    """
    sample_labels = list(set(gsm_groups.values()))[:20]
    prompt = f"""You are an expert bioinformatician. Given this RNA-Seq dataset metadata,
recommend the best analysis settings. Respond ONLY with a JSON object, no other text.

Series title: {series_title}
Sample labels (first 20): {sample_labels}
Number of samples: {len(gsm_groups)}

Return JSON with exactly these keys:
{{
  "dataset_type": one of [tumor_normal, treated_control, time_series, knockout_wildtype, single_condition],
  "label_a": "name for the case/treatment group",
  "label_b": "name for the control/reference group",
  "lfc_threshold": number between 0.5 and 2.0,
  "padj_threshold": one of [0.001, 0.01, 0.05, 0.1],
  "norm_method": one of [log2_cpm, zscore, none],
  "reasoning": "one sentence explaining your choices"
}}"""

    response = call_claude(
        "You are a bioinformatics expert. Return only valid JSON.",
        prompt, max_tokens=400
    )
    try:
        # Strip any markdown code fences if present
        clean = re.sub(r"```json|```", "", response).strip()
        return json.loads(clean)
    except Exception:
        return {}


# ─────────────────────────────────────────────
#  AI FEATURE 2 — BIOLOGICAL INTERPRETATION
# ─────────────────────────────────────────────
def ai_interpret_results(df, gene_col, dataset_type,
                         label_a, label_b, up, down,
                         series_title: str = "") -> str:
    """Generate deep biological interpretation using Claude."""
    up_genes  = df[df["Category"] == "Upregulated"][gene_col].astype(str).tolist()[:30]
    dn_genes  = df[df["Category"] == "Downregulated"][gene_col].astype(str).tolist()[:30]

    # Include top stats
    top_up = df[df["Category"] == "Upregulated"].nsmallest(10, "padj")[
        [gene_col, "log2FoldChange", "padj"]].to_string(index=False)
    top_dn = df[df["Category"] == "Downregulated"].nsmallest(10, "padj")[
        [gene_col, "log2FoldChange", "padj"]].to_string(index=False)

    system = """You are a senior computational biologist and biomedical researcher.
Write clear, scientifically accurate biological interpretations of RNA-Seq results.
Your response should be structured with these sections:
1. Overview of findings
2. Upregulated pathway analysis
3. Downregulated pathway analysis
4. Key genes of interest
5. Biological significance and implications
6. Recommended follow-up experiments

Be specific, mention actual gene functions, known pathways, and disease relevance where applicable."""

    user = f"""Analyze these RNA-Seq differential expression results:

Study: {series_title if series_title else 'RNA-Seq Analysis'}
Comparison: {label_a} vs {label_b}
Dataset type: {dataset_type}
Total DEGs: {up + down} ({up} upregulated, {down} downregulated)

Top 10 UPREGULATED genes (in {label_a}):
{top_up}

Top 10 DOWNREGULATED genes (in {label_b}):
{top_dn}

All upregulated genes: {', '.join(up_genes)}
All downregulated genes: {', '.join(dn_genes)}

Write a comprehensive biological interpretation."""

    return call_claude(system, user, max_tokens=2000)


# ─────────────────────────────────────────────
#  AI FEATURE 3 — PATHWAY ENRICHMENT
# ─────────────────────────────────────────────
def ai_pathway_enrichment(up_genes: list, dn_genes: list,
                           label_a: str, label_b: str,
                           organism: str = "mouse/human") -> str:
    """AI-powered pathway and gene set enrichment analysis."""
    system = """You are an expert in pathway analysis and gene ontology.
Analyze gene lists and provide enrichment analysis including:
- KEGG pathways likely enriched
- GO Biological Processes
- Disease associations
- Transcription factor targets
- Known drug targets among the genes
Be specific and cite known biology."""

    user = f"""Perform pathway enrichment analysis on these RNA-Seq gene lists.
Organism: {organism}
Comparison: {label_a} vs {label_b}

UPREGULATED genes ({len(up_genes)} total, showing up to 40):
{', '.join(up_genes[:40])}

DOWNREGULATED genes ({len(dn_genes)} total, showing up to 40):
{', '.join(dn_genes[:40])}

Provide:
1. Top 5 enriched KEGG pathways for upregulated genes (with likely p-value estimates)
2. Top 5 enriched KEGG pathways for downregulated genes
3. Key GO Biological Processes (top 5 each direction)
4. Notable transcription factors likely regulating these changes
5. Any drug targets or clinically actionable genes
6. Disease relevance based on the gene signatures"""

    return call_claude(system, user, max_tokens=2000)


# ─────────────────────────────────────────────
#  AI FEATURE 4 — DATA CHATBOT
# ─────────────────────────────────────────────
def build_data_context(df, gene_col, label_a, label_b,
                        up, down, dataset_type,
                        series_title: str,
                        geo_meta: dict) -> str:
    """Build a context string about the dataset for the chatbot."""
    up_genes = df[df["Category"] == "Upregulated"][gene_col].astype(str).tolist()[:50]
    dn_genes = df[df["Category"] == "Downregulated"][gene_col].astype(str).tolist()[:50]

    top_df = df[df["Category"] != "Not Significant"].nsmallest(20, "padj")
    top_table = top_df[[gene_col, "log2FoldChange", "padj", "Category"]].to_string(index=False)

    organism = ""
    if "Sample_organism_ch1" in geo_meta:
        organism = geo_meta["Sample_organism_ch1"][0]

    return f"""You are an AI assistant helping a researcher understand their RNA-Seq data.

=== DATASET CONTEXT ===
Study title: {series_title}
Organism: {organism}
Dataset type: {dataset_type}
Comparison: {label_a} (case) vs {label_b} (control)
Total genes analyzed: {len(df):,}
Upregulated genes: {up}
Downregulated genes: {down}
Total DEGs: {up + down}

Top 20 significant genes:
{top_table}

All upregulated genes: {', '.join(up_genes)}
All downregulated genes: {', '.join(dn_genes)}

=== YOUR ROLE ===
Answer questions about this specific dataset. Be concise but accurate.
If asked about a specific gene, explain its biological function and relevance.
If asked about pathways, identify which DEGs are involved.
If asked for recommendations, suggest follow-up experiments or analyses.
Always refer to the actual genes and numbers from this dataset."""


# ─────────────────────────────────────────────
#  NCBI AUTO-RETRIEVAL MODULE  ← NEW in v3.1
# ─────────────────────────────────────────────

NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_FTP     = "https://ftp.ncbi.nlm.nih.gov/geo/series"
NCBI_DELAY  = 0.4          # seconds between API calls (NCBI rate limit = 3/sec)


def _ncbi_get(url: str, timeout: int = 30) -> bytes:
    """HTTP GET with NCBI-friendly headers. Returns raw bytes or raises."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RNASeqTool/3.1 (research; contact@example.com)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _geo_ftp_folder(gse: str) -> str:
    """Return the GEO FTP folder path for a GSE accession."""
    num    = gse.replace("GSE", "")
    prefix = num[:-3] + "nnn" if len(num) > 3 else "0nnn"
    return f"{GEO_FTP}/GSE{prefix}/{gse}"


def fetch_geo_supplementary_files(gse: str) -> list:
    """
    List supplementary files for a GSE from the GEO FTP.
    Returns list of dicts: [{"name": ..., "url": ...}]
    """
    folder  = _geo_ftp_folder(gse)
    suppl   = f"{folder}/suppl/"
    results = []
    try:
        raw  = _ncbi_get(suppl, timeout=20).decode("utf-8", errors="replace")
        # Parse FTP index HTML — filenames in href attributes
        names = re.findall(r'href="([^"]+\.(?:gz|txt|csv|tsv|xlsx)[^"]*)"', raw, re.I)
        for name in names:
            name = name.split("/")[-1]
            if name and not name.startswith("?"):
                results.append({"name": name, "url": f"{suppl}{name}"})
    except Exception:
        pass
    return results


def download_geo_supplementary(url: str, filename: str) -> pd.DataFrame | None:
    """
    Download a GEO supplementary count file and parse it into a DataFrame.
    Handles .gz, .txt, .csv automatically.
    """
    try:
        raw = _ncbi_get(url, timeout=60)
    except Exception as e:
        return None

    try:
        # Decompress if needed
        if filename.endswith(".gz"):
            import gzip as gz
            raw = gz.decompress(raw)

        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

        # Try tab then comma
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
                if df.shape[1] > 2:
                    return df
            except Exception:
                pass
    except Exception:
        pass
    return None


def fetch_geo_series_matrix(gse: str) -> tuple:
    """
    Download the series matrix file directly from NCBI GEO FTP.
    Returns (df, geo_meta, gsm_groups) or (None, {}, {}).
    """
    folder = _geo_ftp_folder(gse)
    # Try common filename patterns
    candidates = [
        f"{folder}/matrix/{gse}_series_matrix.txt.gz",
        f"{folder}/matrix/{gse}-GPL570_series_matrix.txt.gz",
        f"{folder}/matrix/{gse}-GPL13112_series_matrix.txt.gz",
    ]
    # Also try listing the matrix directory
    try:
        listing_url = f"{folder}/matrix/"
        raw_listing = _ncbi_get(listing_url, timeout=15).decode("utf-8", errors="replace")
        found = re.findall(r'href="([^"]+_series_matrix\.txt\.gz)"', raw_listing, re.I)
        for f in found:
            fname = f.split("/")[-1]
            candidates.insert(0, f"{listing_url}{fname}")
    except Exception:
        pass

    for url in candidates:
        try:
            raw = _ncbi_get(url, timeout=60)
            text = gzip.decompress(raw).decode("latin1", errors="replace")
            df, geo_meta, gsm_groups = parse_geo_soft_full(text)
            if df is not None and df.shape[1] > 1 and df.shape[0] > 10:
                return df, geo_meta, gsm_groups
        except Exception:
            continue
    return None, {}, {}


def get_srx_to_srr_mapping(srx_ids: list) -> dict:
    """
    Convert SRX experiment IDs to SRR run IDs using NCBI eutils.
    Returns {srx: [srr1, srr2, ...]}
    """
    mapping = {}
    for srx in srx_ids[:20]:   # limit to avoid rate-limiting
        try:
            time.sleep(NCBI_DELAY)
            # Search SRA for the SRX
            search_url = (f"{NCBI_BASE}/esearch.fcgi"
                          f"?db=sra&term={srx}&retmax=5&retmode=json")
            data = json.loads(_ncbi_get(search_url, timeout=15))
            ids  = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            # Fetch run info
            fetch_url = (f"{NCBI_BASE}/efetch.fcgi"
                         f"?db=sra&id={','.join(ids)}&rettype=runinfo&retmode=csv")
            csv_raw = _ncbi_get(fetch_url, timeout=20).decode("utf-8", errors="replace")
            runs = []
            for row in csv_raw.strip().split("\n")[1:]:
                parts = row.split(",")
                if parts and parts[0].startswith("SRR"):
                    runs.append(parts[0])
            if runs:
                mapping[srx] = runs
        except Exception:
            continue
    return mapping


def fetch_sra_runinfo(srp_or_srx: str) -> pd.DataFrame | None:
    """
    Fetch run info table for a SRP project or SRX experiment.
    Returns DataFrame with columns: Run, Experiment, Sample, BioSample, etc.
    """
    try:
        time.sleep(NCBI_DELAY)
        search_url = (f"{NCBI_BASE}/esearch.fcgi"
                      f"?db=sra&term={srp_or_srx}&retmax=200&retmode=json")
        data = json.loads(_ncbi_get(search_url, timeout=15))
        ids  = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        fetch_url = (f"{NCBI_BASE}/efetch.fcgi"
                     f"?db=sra&id={','.join(ids)}&rettype=runinfo&retmode=csv")
        csv_raw = _ncbi_get(fetch_url, timeout=30).decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(csv_raw))
        return df if len(df) > 0 else None
    except Exception:
        return None


def smart_retrieve_from_geo(gse: str, geo_meta: dict,
                              gsm_groups: dict,
                              progress_callback=None) -> dict:
    """
    Master retrieval function. Given a GSE accession:
    1. Try supplementary count files first (best option)
    2. Try downloading series matrix with actual data
    3. Fall back to SRA runinfo (metadata only, no counts)

    Returns dict:
      {
        "status":       "counts" | "matrix" | "runinfo" | "failed",
        "df":           DataFrame or None,
        "files_found":  list of filenames tried,
        "runinfo":      DataFrame or None,
        "message":      str,
      }
    """
    result = {
        "status": "failed", "df": None,
        "files_found": [], "runinfo": None, "message": ""
    }

    def _prog(msg):
        if progress_callback:
            progress_callback(msg)

    # ── STEP 1: Supplementary processed count files ──────────────────────
    _prog("🔍 Searching GEO supplementary files...")
    suppl_files = fetch_geo_supplementary_files(gse)
    result["files_found"] = [f["name"] for f in suppl_files]

    # Prioritise count/expression files
    COUNT_KEYWORDS = ["count", "expr", "fpkm", "rpkm", "tpm",
                      "normalized", "matrix", "raw", "read"]
    priority = [f for f in suppl_files
                if any(kw in f["name"].lower() for kw in COUNT_KEYWORDS)]
    others   = [f for f in suppl_files if f not in priority]
    ordered  = priority + others

    for finfo in ordered[:6]:   # try up to 6 files
        _prog(f"📥 Trying supplementary file: {finfo['name']}")
        df = download_geo_supplementary(finfo["url"], finfo["name"])
        if df is not None and df.shape[0] > 100:
            # Check it has numeric sample columns
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                result["status"]  = "counts"
                result["df"]      = df
                result["message"] = (f"✅ Retrieved processed count matrix: "
                                     f"{finfo['name']} "
                                     f"({df.shape[0]:,} genes × {df.shape[1]} columns)")
                return result

    # ── STEP 2: Series matrix with actual data ────────────────────────────
    _prog("🔍 Trying to fetch series matrix from GEO FTP...")
    df, new_meta, new_gsm = fetch_geo_series_matrix(gse)
    if df is not None and df.shape[0] > 10:
        # Merge metadata if useful
        if new_gsm:
            gsm_groups.update(new_gsm)
        result["status"]  = "matrix"
        result["df"]      = df
        result["message"] = (f"✅ Retrieved series matrix: "
                             f"{df.shape[0]:,} genes × {df.shape[1]} samples")
        return result

    # ── STEP 3: SRA RunInfo (metadata, no counts) ─────────────────────────
    srp = None
    for val in geo_meta.get("Series_relation", []):
        m = re.search(r"SRA:\s*.*?term=(\w+)", val)
        if m:
            srp = m.group(1); break

    if srp:
        _prog(f"📋 Fetching SRA run info for {srp}...")
        runinfo = fetch_sra_runinfo(srp)
        if runinfo is not None:
            result["status"]  = "runinfo"
            result["runinfo"] = runinfo
            result["message"] = (f"⚠️ No count matrix found. Retrieved SRA run info "
                                 f"({len(runinfo)} runs for {srp}). "
                                 f"Raw FASTQ files must be processed with a pipeline "
                                 f"(STAR + featureCounts) to get expression counts.")
            return result

    # ── STEP 4: Extract SRX IDs from meta and show them ───────────────────
    srx_ids = []
    for val in geo_meta.get("Sample_relation", []):
        m = re.search(r"term=(SRX\w+)", val)
        if m:
            srx_ids.append(m.group(1))
    srx_ids = list(set(srx_ids))

    if srx_ids:
        result["status"]  = "srx_only"
        result["message"] = (f"ℹ️ Found {len(srx_ids)} SRX experiment IDs. "
                             f"No processed counts available on GEO. "
                             f"Use the SRX IDs below to download raw data from SRA.")
        result["srx_ids"] = srx_ids[:20]
        return result

    result["message"] = ("❌ Could not retrieve any data for this GSE. "
                         "The dataset may be private or have no processed files.")
    return result


def build_retrieval_summary(retrieve_result: dict, gse: str) -> str:
    """Build a human-readable summary of what was retrieved."""
    status = retrieve_result.get("status", "failed")
    files  = retrieve_result.get("files_found", [])

    lines = [f"**GSE Accession:** {gse}"]

    if status == "counts":
        df = retrieve_result["df"]
        lines += [
            f"**Status:** ✅ Processed count matrix retrieved",
            f"**Genes:** {df.shape[0]:,}",
            f"**Samples:** {df.shape[1] - 1}",
            f"**Source:** GEO Supplementary Files",
        ]
    elif status == "matrix":
        df = retrieve_result["df"]
        lines += [
            f"**Status:** ✅ Series matrix retrieved",
            f"**Genes:** {df.shape[0]:,}",
            f"**Samples:** {df.shape[1] - 1}",
            f"**Source:** GEO FTP Series Matrix",
        ]
    elif status == "runinfo":
        ri = retrieve_result.get("runinfo")
        lines += [
            f"**Status:** ⚠️ SRA run metadata only (no expression counts)",
            f"**Runs found:** {len(ri) if ri is not None else 0}",
            f"**Next step:** Download FASTQ → align → count (STAR + featureCounts)",
        ]
    elif status == "srx_only":
        srx = retrieve_result.get("srx_ids", [])
        lines += [
            f"**Status:** ℹ️ SRX IDs found but no processed counts on GEO",
            f"**SRX IDs:** {', '.join(srx[:5])}{'...' if len(srx)>5 else ''}",
        ]
    else:
        lines.append("**Status:** ❌ Retrieval failed")

    if files:
        lines.append(f"**Files scanned:** {', '.join(files[:5])}")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────
#  GEO PARSER (unchanged from v2.2)
# ─────────────────────────────────────────────
def parse_geo_soft_full(content):
    lines = content.split("\n")
    geo_meta = {}
    data_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\r")
        if line.startswith("!") and not in_table:
            m = re.match(r'^!([\w]+)\s*=\s*(.+)$', line)
            if m:
                key = m.group(1)
                vals = [v.strip().strip('"') for v in m.group(2).split("\t")]
                geo_meta[key] = vals
            continue
        if "table_begin" in line.lower():
            in_table = True; continue
        if "table_end" in line.lower():
            in_table = False; continue
        if in_table and line.strip():
            data_lines.append(line)

    df = None
    if data_lines:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep="\t", index_col=None, low_memory=False)
        except Exception:
            pass

    gsm_groups = {}
    if df is not None:
        gsm_cols = [c for c in df.columns if str(c).startswith("GSM")]
        n = len(gsm_cols)
        label_source = None
        for key in ["Sample_title", "Sample_source_name_ch1",
                    "Sample_characteristics_ch1", "Sample_description"]:
            if key in geo_meta and len(geo_meta[key]) >= n:
                label_source = geo_meta[key][:n]
                break
        if label_source and len(label_source) == n:
            for gsm, lbl in zip(gsm_cols, label_source):
                gsm_groups[gsm] = lbl
        else:
            for i, gsm in enumerate(gsm_cols):
                gsm_groups[gsm] = f"sample_{i+1}"

    return df, geo_meta, gsm_groups


def cluster_gsm_groups(gsm_groups):
    if not gsm_groups:
        return [], [], "Group A", "Group B"

    gsm_list   = list(gsm_groups.keys())
    label_list = list(gsm_groups.values())

    POS = [r"tumor",r"cancer",r"malignant",r"treated?",r"treatment",
           r"drug",r"stimulat",r"infected",r"\bko\b",r"knockout",
           r"\bkd\b",r"knockdown",r"siRNA",r"mutant",r"crispr",
           r"stress",r"disease",r"case",r"patient",r"\bcus\b",r"mdd",
           r"late",r"high",r"depressed?"]
    NEG = [r"normal",r"benign",r"control",r"\bctrl\b",r"untreat",
           r"vehicle",r"dmso",r"mock",r"\bwt\b",r"wildtype",
           r"scramble",r"healthy",r"early",r"low",r"adjacent",
           r"naive",r"baseline",r"sham"]

    grp_a, grp_b = [], []
    for gsm, lbl in zip(gsm_list, label_list):
        ll = lbl.lower()
        pos = any(re.search(p, ll) for p in POS)
        neg = any(re.search(p, ll) for p in NEG)
        if pos and not neg:   grp_a.append(gsm)
        elif neg and not pos: grp_b.append(gsm)

    if not grp_a or not grp_b:
        unique = list(dict.fromkeys(label_list))
        if len(unique) >= 2:
            mid = len(unique) // 2
            set_a = set(unique[mid:]); set_b = set(unique[:mid])
            grp_a = [g for g,l in zip(gsm_list,label_list) if l in set_a]
            grp_b = [g for g,l in zip(gsm_list,label_list) if l in set_b]
        else:
            mid = len(gsm_list) // 2
            grp_a = gsm_list[mid:]; grp_b = gsm_list[:mid]

    def _label(subset):
        vals = [gsm_groups[g] for g in subset if g in gsm_groups]
        if not vals: return "Group"
        words = re.findall(r"[A-Za-z]{3,}", " ".join(vals))
        if words:
            from collections import Counter
            return Counter(words).most_common(1)[0][0].capitalize()
        return vals[0][:20]

    la = _label(grp_a); lb = _label(grp_b)
    if la == lb: la, lb = "Group A", "Group B"
    return grp_a, grp_b, la, lb


def smart_read_file(uploaded_file):
    fname = uploaded_file.name
    msgs  = []

    def _decode(raw):
        for enc in ["utf-8","latin1","cp1252"]:
            try: return raw.decode(enc)
            except: pass
        return raw.decode("latin1", errors="replace")

    def _parse(text):
        if "table_begin" in text.lower():
            df, gm, gg = parse_geo_soft_full(text)
            if df is not None and df.shape[1] > 1:
                return df, gm, gg
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep,
                                 encoding="latin1", comment="!", low_memory=False)
                if df.shape[1] > 1: return df, {}, {}
            except: pass
        return None, {}, {}

    try:
        if fname.endswith(".zip"):
            msgs.append(("info","📦 ZIP detected"))
            with zipfile.ZipFile(uploaded_file) as z:
                for name in z.namelist():
                    if name.endswith("/"): continue
                    with z.open(name) as f:
                        raw = _decode(f.read())
                    df, gm, gg = _parse(raw)
                    if df is not None:
                        msgs.append(("success",f"✅ Parsed `{name}`"))
                        return df, gm, gg, msgs
            msgs.append(("error","Could not parse ZIP contents."))
            return None,{},{},msgs

        elif fname.endswith(".gz"):
            msgs.append(("info","🗜️ GZ decompressing..."))
            with gzip.open(uploaded_file,"rt",encoding="latin1",errors="replace") as f:
                raw = f.read()
            df,gm,gg = _parse(raw)
            if df is not None:
                msgs.append(("success","✅ GZ parsed"))
                return df,gm,gg,msgs

        elif fname.endswith((".txt",".tsv")):
            raw = _decode(uploaded_file.read())
            df,gm,gg = _parse(raw)
            if df is not None:
                msgs.append(("success","✅ Text file parsed"))
                return df,gm,gg,msgs

        elif fname.endswith(".csv"):
            for enc in ["utf-8","latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    msgs.append(("success",f"✅ CSV parsed"))
                    return df,{},{},msgs
                except: pass

    except Exception as e:
        msgs.append(("error",f"Fatal: {e}"))
        return None,{},{},msgs

    msgs.append(("error","Unsupported format."))
    return None,{},{},msgs


# ─────────────────────────────────────────────
#  DATASET TYPE DETECTION
# ─────────────────────────────────────────────
DTYPE_PATS = {
    "tumor_normal":      [r"\btumor\b",r"\bnormal\b",r"\bcancer\b",r"\bmalignant\b"],
    "treated_control":   [r"\btreated?\b",r"\bcontrol\b",r"\bdrug\b",r"\bstress\b",r"\bcus\b"],
    "time_series":       [r"\b\d+h\b",r"\b\d+hr\b",r"\b\d+days?\b",r"\btime\b"],
    "knockout_wildtype": [r"\bko\b",r"\bwt\b",r"\bknockout\b",r"\bknockdown\b"],
}

def detect_type_meta(geo_meta, gsm_groups):
    combined = " ".join(gsm_groups.values()).lower() if gsm_groups else ""
    for dtype, pats in DTYPE_PATS.items():
        for p in pats:
            if re.search(p, combined, re.IGNORECASE): return dtype
    return "single_condition"

def detect_type_cols(df):
    cols = " ".join(df.columns).lower()
    for dtype, pats in DTYPE_PATS.items():
        for p in pats:
            if re.search(p, cols, re.IGNORECASE): return dtype
    if "log2FoldChange" in df.columns and "padj" in df.columns:
        return "pre_computed"
    return "single_condition"

def detect_groups_non_geo(df, dtype):
    gene_like = {"gene","geneid","gene_id","symbol","genename","probe_id",
                 "probeid","id_ref","id","ensembl","transcript","name"}
    scols = [c for c in df.columns
             if c.lower().replace("_","") not in gene_like
             and pd.api.types.is_numeric_dtype(df[c])]
    if len(scols) < 2: return scols[:1], scols[1:2], "Group A", "Group B"

    def _m(pats,cols): return [c for c in cols if any(re.search(p,c,re.IGNORECASE) for p in pats)]

    if dtype=="tumor_normal":
        a=_m([r"tumor",r"cancer",r"malignant"],scols); b=_m([r"normal",r"benign"],scols); la,lb="Tumor","Normal"
    elif dtype=="treated_control":
        a=_m([r"treat",r"drug",r"stimul"],scols); b=_m([r"control",r"untreat",r"dmso"],scols); la,lb="Treated","Control"
    elif dtype=="knockout_wildtype":
        a=_m([r"\bko\b",r"knockout",r"\bkd\b"],scols); b=_m([r"\bwt\b",r"wildtype"],scols); la,lb="KO","WT"
    elif dtype=="time_series":
        def _t(c):
            m=re.search(r"(\d+)",c); return int(m.group(1)) if m else 9999
        sc=sorted(scols,key=_t); mid=len(sc)//2; a,b,la,lb=sc[mid:],sc[:mid],"Late","Early"
    else:
        half=len(scols)//2; a,b,la,lb=scols[half:],scols[:half],"Group A","Group B"

    if not a or not b:
        half=len(scols)//2; b,a=scols[:half],scols[half:]
    return a,b,la,lb


# ─────────────────────────────────────────────
#  DIFFERENTIAL EXPRESSION
# ─────────────────────────────────────────────
def compute_de(df, grp_a, grp_b, label_a, label_b, norm="log2_cpm"):
    expr = df.copy()
    for c in grp_a+grp_b: expr[c] = pd.to_numeric(expr[c], errors="coerce")
    expr = expr.dropna(subset=grp_a+grp_b, how="all")

    av = expr[grp_a]; bv = expr[grp_b]
    if norm=="log2_cpm":
        an=np.log2(av.clip(lower=0)+1); bn=np.log2(bv.clip(lower=0)+1)
    elif norm=="zscore":
        an=(av-av.mean())/(av.std()+1e-9); bn=(bv-bv.mean())/(bv.std()+1e-9)
    else: an,bn=av,bv

    expr["mean_A"]=an.mean(axis=1); expr["mean_B"]=bn.mean(axis=1)
    expr["log2FoldChange"]=expr["mean_A"]-expr["mean_B"]

    pv=[]
    if len(grp_a)>=2 and len(grp_b)>=2:
        for i in range(len(expr)):
            ra=an.iloc[i].dropna().values; rb=bn.iloc[i].dropna().values
            if len(ra)>=2 and len(rb)>=2:
                _,p=ttest_ind(ra,rb,equal_var=False)
                pv.append(float(p) if np.isfinite(p) else 1.0)
            else: pv.append(1.0)
    else:
        pv=(1/(1+expr["log2FoldChange"].abs())).tolist()

    expr["pvalue"]=pv
    pvn=np.array(pv); n=len(pvn); o=np.argsort(pvn)
    pa=pvn.copy(); pa[o]=pvn[o]*n/(np.arange(n)+1)
    pa=np.minimum.accumulate(pa[::-1])[::-1]
    expr["padj"]=np.clip(pa,0,1)
    expr["label_A"]=label_a; expr["label_B"]=label_b
    return expr


def classify_genes(df, lfc=1.0, padj=0.05):
    df=df.copy(); df["Category"]="Not Significant"
    df.loc[(df["padj"]<padj)&(df["log2FoldChange"]> lfc),"Category"]="Upregulated"
    df.loc[(df["padj"]<padj)&(df["log2FoldChange"]<-lfc),"Category"]="Downregulated"
    return df


def find_gene_col(df):
    for c in ["gene","Gene","gene_id","GeneID","gene_name","symbol","Symbol",
              "ID_REF","probe_id","NAME","name","Ensembl","feature_id"]:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype==object: return c
    return None


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
BG="#0f1117"; AX="#12151f"; BD="#2a2d3e"
TX="#c8cfe0"; MT="#8b92a5"; UP="#ff4d6d"; DN="#4da6ff"; AC="#00d4aa"

def _ax(ax,t=""):
    ax.set_facecolor(AX); ax.tick_params(colors=MT,labelsize=9)
    for s in ax.spines.values(): s.set_color(BD)
    if t: ax.set_title(t,color="white",fontsize=13,fontweight="bold",pad=10)

def _save(fig,path):
    fig.savefig(path,dpi=110,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.close(fig)

def plot_volcano(df,gc,la,lb,lt,pt):
    df=df.dropna(subset=["log2FoldChange","padj"]).copy()
    df["lp"]=-np.log10(df["padj"].clip(lower=1e-300))
    fig,ax=plt.subplots(figsize=(9,7),facecolor=BG); _ax(ax,"Volcano Plot")
    cm={UP_cat:UP,DN_cat:DN,"Not Significant":"#3a3f55"}
    sm={UP_cat:25,DN_cat:25,"Not Significant":7}
    am={UP_cat:0.85,DN_cat:0.85,"Not Significant":0.3}
    UP_cat="Upregulated"; DN_cat="Downregulated"
    for cat in ["Not Significant","Downregulated","Upregulated"]:
        sub=df[df["Category"]==cat]
        ax.scatter(sub["log2FoldChange"],sub["lp"],
                   c=(UP if cat=="Upregulated" else DN if cat=="Downregulated" else "#3a3f55"),
                   s=(25 if cat!="Not Significant" else 7),
                   alpha=(0.85 if cat!="Not Significant" else 0.3),
                   label=f"{cat} (n={len(sub)})",edgecolors="none")
    ax.axvline(x=lt,color=BD,ls="--",lw=1,alpha=0.8)
    ax.axvline(x=-lt,color=BD,ls="--",lw=1,alpha=0.8)
    ax.axhline(y=-np.log10(pt),color=BD,ls="--",lw=1,alpha=0.8)
    for _,row in pd.concat([df[df["Category"]=="Upregulated"].nsmallest(8,"padj"),
                             df[df["Category"]=="Downregulated"].nsmallest(8,"padj")]).iterrows():
        clr=UP if row["Category"]=="Upregulated" else DN
        ax.annotate(str(row[gc])[:14],(row["log2FoldChange"],row["lp"]),
                    fontsize=7,color=clr,fontfamily="monospace",
                    xytext=(5,3),textcoords="offset points",
                    arrowprops=dict(arrowstyle="-",color=clr,alpha=0.4,lw=0.6))
    ax.set_xlabel(f"log2FC ({la}/{lb})",color=TX,fontsize=10)
    ax.set_ylabel("-log10(padj)",color=TX,fontsize=10)
    ax.legend(framealpha=0.15,labelcolor="white",fontsize=8,facecolor=AX,edgecolor=BD)
    fig.tight_layout(); return fig

def plot_ma(df,la,lb):
    df=df.dropna(subset=["log2FoldChange","mean_A","mean_B"]).copy()
    df["bm"]=(df["mean_A"]+df["mean_B"])/2
    fig,ax=plt.subplots(figsize=(8,6),facecolor=BG); _ax(ax,"MA Plot")
    for cat,clr,sz,al in [("Not Significant","#3a3f55",6,0.25),
                           ("Downregulated",DN,18,0.8),("Upregulated",UP,18,0.8)]:
        sub=df[df["Category"]==cat]
        ax.scatter(sub["bm"],sub["log2FoldChange"],c=clr,s=sz,alpha=al,edgecolors="none",label=cat)
    ax.axhline(y=0,color=AC,lw=1,alpha=0.6)
    ax.set_xlabel("Mean Expression",color=TX,fontsize=10)
    ax.set_ylabel(f"log2FC",color=TX,fontsize=10)
    ax.legend(framealpha=0.15,labelcolor="white",fontsize=8,facecolor=AX,edgecolor=BD)
    fig.tight_layout(); return fig

def plot_heatmap(df,ga,gb,gc,n=40):
    sig=df[df["Category"]!="Not Significant"].copy()
    if len(sig)==0: sig=df.copy()
    sig=sig.nsmallest(min(n,len(sig)),"padj")
    cu=[c for c in (gb+ga) if c in df.columns]
    if not cu: return None
    mat=sig[cu].apply(pd.to_numeric,errors="coerce").fillna(0)
    rs=mat.std(axis=1).replace(0,1)
    mz=mat.sub(mat.mean(axis=1),axis=0).div(rs,axis=0)
    gl=sig[gc].astype(str).values[:len(mz)]
    sc=[c[:10]+"…" if len(c)>12 else c for c in cu]
    fig,ax=plt.subplots(figsize=(max(8,len(cu)*0.9+3),max(7,len(mz)*0.27)),facecolor=BG)
    ax.set_facecolor(AX)
    im=ax.imshow(mz.values,cmap=plt.cm.RdBu_r,aspect="auto",vmin=-2.5,vmax=2.5,interpolation="nearest")
    ax.set_xticks(range(len(sc))); ax.set_xticklabels(sc,rotation=45,ha="right",fontsize=8,color=TX)
    ax.set_yticks(range(len(mz))); ax.set_yticklabels(gl,fontsize=7.5,color=TX,fontfamily="monospace")
    ax.set_title("Top DEG Heatmap (Z-score)",color="white",fontsize=13,fontweight="bold")
    cb=plt.colorbar(im,ax=ax,fraction=0.02,pad=0.04)
    cb.ax.tick_params(colors=MT,labelsize=8); cb.set_label("Z-score",color=MT,fontsize=9)
    for s in ax.spines.values(): s.set_color(BD)
    fig.tight_layout(); return fig

def plot_pca(df,ga,gb,la,lb):
    cols=[c for c in ga+gb if c in df.columns]
    if len(cols)<3: return None
    mat=df[cols].apply(pd.to_numeric,errors="coerce").fillna(0).T.values
    ms=StandardScaler().fit_transform(mat)
    nc=min(2,ms.shape[0],ms.shape[1])
    if nc<2: return None
    pca=PCA(n_components=nc); pca.fit(ms); pcs=pca.transform(ms); var=pca.explained_variance_ratio_*100
    fig,ax=plt.subplots(figsize=(8,6),facecolor=BG); _ax(ax,"PCA — Sample Clustering")
    lbls=[lb]*len(gb)+[la]*len(ga); cm2={la:UP,lb:DN}
    for lbl in set(lbls):
        idx=[i for i,l in enumerate(lbls) if l==lbl]
        ax.scatter(pcs[idx,0],pcs[idx,1],c=cm2.get(lbl,AC),s=90,label=lbl,
                   alpha=0.9,edgecolors="white",linewidths=0.5)
        for i in idx:
            ax.annotate(cols[i][:10],(pcs[i,0],pcs[i,1]),fontsize=7,color=MT,
                        xytext=(4,4),textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)",color=TX,fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)",color=TX,fontsize=10)
    ax.legend(framealpha=0.15,labelcolor="white",fontsize=9,facecolor=AX,edgecolor=BD)
    fig.tight_layout(); return fig

def plot_dist(df,ga,gb,la,lb):
    all_d,all_l=[],[]
    for c in ga:
        if c in df.columns:
            all_d.append(pd.to_numeric(df[c],errors="coerce").dropna().clip(-20,20).values)
            all_l.append(f"{la[:8]}\n{c[:8]}")
    for c in gb:
        if c in df.columns:
            all_d.append(pd.to_numeric(df[c],errors="coerce").dropna().clip(-20,20).values)
            all_l.append(f"{lb[:8]}\n{c[:8]}")
    if not all_d: return None
    fig,ax=plt.subplots(figsize=(max(8,len(all_d)*1.1),5),facecolor=BG)
    _ax(ax,"Expression Distribution")
    vp=ax.violinplot(all_d,positions=range(len(all_d)),showmedians=True)
    for i,body in enumerate(vp["bodies"]):
        clr=UP if i<len(ga) else DN
        body.set_facecolor(clr); body.set_alpha(0.55); body.set_edgecolor(clr)
    for part in ["cmedians","cmins","cmaxes","cbars"]:
        vp[part].set_edgecolor(TX); vp[part].set_linewidth(1)
    ax.set_xticks(range(len(all_l))); ax.set_xticklabels(all_l,color=TX,fontsize=7.5)
    ax.set_ylabel("Expression",color=TX,fontsize=10)
    fig.tight_layout(); return fig

def plot_bar(up,down,la,lb):
    fig,ax=plt.subplots(figsize=(6,4),facecolor=BG); _ax(ax,"DEG Summary")
    bars=ax.bar([f"Up ({la})",f"Down ({lb})"],[up,down],
                color=[UP,DN],width=0.45,edgecolor=BG,alpha=0.9)
    for bar,val in zip(bars,[up,down]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                str(val),ha="center",va="bottom",color="white",fontsize=13,fontweight="bold")
    ax.set_ylabel("Genes",color=TX,fontsize=10); ax.set_ylim(0,max(up,down,1)*1.3)
    fig.tight_layout(); return fig


# ─────────────────────────────────────────────
#  PDF BUILDERS
# ─────────────────────────────────────────────
def build_free_pdf(df,gc,figs,up,down,la,lb,dtype,out):
    styles=getSampleStyleSheet()
    ts=ParagraphStyle("T",parent=styles["Title"],fontSize=18,
                       textColor=colors.HexColor("#00d4aa"),alignment=TA_CENTER)
    h2=ParagraphStyle("H2",parent=styles["Heading2"],fontSize=13,
                       textColor=colors.HexColor("#7c3aed"),spaceBefore=12)
    doc=SimpleDocTemplate(out,pagesize=A4,
                           rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    el=[]
    el.append(Paragraph("RNA-Seq Analysis Report (Free)",ts))
    el.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
                         ParagraphStyle("s",parent=styles["Normal"],fontSize=9,
                                        textColor=colors.grey,alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%",thickness=2,color=colors.HexColor("#00d4aa")))
    el.append(Spacer(1,12))
    tmap={"tumor_normal":"Tumor vs Normal","treated_control":"Treated vs Control",
          "time_series":"Time Series","knockout_wildtype":"KO vs WT",
          "pre_computed":"Pre-computed","single_condition":"Expression Profiling"}
    td=[["Dataset Type",tmap.get(dtype,dtype)],["Comparison",f"{la} vs {lb}"],
        ["Total Genes",str(len(df))],["Upregulated",str(up)],["Downregulated",str(down)]]
    t=Table(td,colWidths=[5.5*cm,9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#e8faf7")),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#00a884")),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),10),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
        ("PADDING",(0,0),(-1,-1),6),
    ]))
    el.append(Paragraph("Summary",h2)); el.append(t); el.append(Spacer(1,14))
    top=df[df["Category"]!="Not Significant"].nsmallest(20,"padj")
    if len(top)>0:
        el.append(Paragraph("Top 20 Significant Genes",h2))
        rows=[["Gene","log2FC","padj","Category"]]
        for _,row in top.iterrows():
            rows.append([str(row.get(gc,""))[:20],
                         f"{row.get('log2FoldChange',0):.3f}",
                         f"{row.get('padj',1):.2e}",row.get("Category","")])
        t2=Table(rows,colWidths=[5*cm,3*cm,3*cm,4.5*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#00d4aa")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS",(1,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
            ("PADDING",(0,0),(-1,-1),5),
        ]))
        el.append(t2)
    el.append(Spacer(1,14))
    if "volcano" in figs and figs["volcano"]:
        el.append(Paragraph("Volcano Plot",h2))
        el.append(Image(figs["volcano"],width=13*cm,height=10*cm))
    el.append(Spacer(1,16))
    el.append(Paragraph("Upgrade to Premium for AI interpretation, all plots, and full gene tables.",
                         ParagraphStyle("n",parent=styles["Normal"],fontSize=8,
                                        textColor=colors.grey,alignment=TA_CENTER)))
    doc.build(el)


def build_premium_pdf(df,gc,figs,up,down,la,lb,dtype,ai_interp,ai_pathways,out):
    styles=getSampleStyleSheet()
    ts=ParagraphStyle("T",parent=styles["Title"],fontSize=20,
                       textColor=colors.HexColor("#7c3aed"),alignment=TA_CENTER)
    h2=ParagraphStyle("H2",parent=styles["Heading2"],fontSize=14,
                       textColor=colors.HexColor("#7c3aed"),spaceBefore=16)
    h3=ParagraphStyle("H3",parent=styles["Heading3"],fontSize=12,
                       textColor=colors.HexColor("#00a884"),spaceBefore=10)
    body=ParagraphStyle("B",parent=styles["Normal"],fontSize=10,leading=15,alignment=TA_JUSTIFY)
    doc=SimpleDocTemplate(out,pagesize=A4,
                           rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    el=[]
    el.append(Spacer(1,20))
    el.append(Paragraph("RNA-Seq Premium AI Analysis Report",ts))
    el.append(Paragraph("Powered by Claude AI — Comprehensive Biological Interpretation",
                         ParagraphStyle("s",parent=styles["Normal"],fontSize=11,
                                        textColor=colors.grey,alignment=TA_CENTER)))
    el.append(Spacer(1,8))
    el.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                         ParagraphStyle("d",parent=styles["Normal"],fontSize=9,
                                        textColor=colors.grey,alignment=TA_CENTER)))
    el.append(HRFlowable(width="100%",thickness=2,color=colors.HexColor("#7c3aed")))
    el.append(Spacer(1,20))

    tmap={"tumor_normal":"Tumor vs Normal","treated_control":"Treated vs Control",
          "time_series":"Time Series","knockout_wildtype":"KO vs Wildtype",
          "pre_computed":"Pre-computed","single_condition":"Expression Profiling"}
    el.append(Paragraph("1. Dataset Overview",h2))
    info=[["Parameter","Value"],["Analysis Type",tmap.get(dtype,dtype)],
          ["Comparison",f"{la} vs {lb}"],["Total Genes",str(len(df))],
          ["Total DEGs",str(up+down)],["Upregulated",str(up)],["Downregulated",str(down)],
          ["Up/Down Ratio",f"{up/max(down,1):.2f}"],["AI Powered","Yes — Claude Sonnet"]]
    t=Table(info,colWidths=[7*cm,8.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#7c3aed")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BACKGROUND",(0,1),(0,-1),colors.HexColor("#f5f0ff")),("FONTSIZE",(0,0),(-1,-1),10),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#faf8ff")]),
        ("PADDING",(0,0),(-1,-1),6),
    ]))
    el.append(t); el.append(Spacer(1,16))

    el.append(Paragraph("2. AI Biological Interpretation",h2))
    el.append(Paragraph("(Generated by Claude AI — Anthropic)",
                         ParagraphStyle("ai",parent=styles["Normal"],fontSize=8,
                                        textColor=colors.HexColor("#7c3aed"))))
    el.append(Spacer(1,6))
    for line in ai_interp.split("\n"):
        if line.strip(): el.append(Paragraph(line,body))
    el.append(Spacer(1,16))

    if ai_pathways:
        el.append(Paragraph("3. AI Pathway Enrichment Analysis",h2))
        el.append(Paragraph("(Generated by Claude AI)",
                             ParagraphStyle("ai",parent=styles["Normal"],fontSize=8,
                                            textColor=colors.HexColor("#7c3aed"))))
        el.append(Spacer(1,6))
        for line in ai_pathways.split("\n"):
            if line.strip(): el.append(Paragraph(line,body))
        el.append(Spacer(1,16))

    el.append(Paragraph("4. Visualizations",h2))
    for key,title,caption in [
        ("volcano","4.1 Volcano Plot","Red=upregulated, Blue=downregulated."),
        ("ma","4.2 MA Plot","Mean expression vs fold change."),
        ("pca","4.3 PCA","Sample clustering."),
        ("heatmap","4.4 Heatmap","Z-score top DEGs."),
        ("dist","4.5 Distribution","Per-sample expression."),
        ("bar","4.6 Summary","DEG counts."),
    ]:
        if key in figs and figs[key]:
            el.append(Paragraph(title,h3))
            el.append(Paragraph(caption,body))
            el.append(Image(figs[key],width=13*cm,height=10*cm))
            el.append(Spacer(1,10))

    el.append(PageBreak())
    el.append(Paragraph("5. Top DEG Tables",h2))
    for cat,lbl,ch in [("Upregulated",f"Up in {la}","#ff4d6d"),
                        ("Downregulated",f"Down in {lb}","#4da6ff")]:
        sub=df[df["Category"]==cat].nsmallest(25,"padj")
        if len(sub)==0: continue
        el.append(Paragraph(lbl,h3))
        rows=[["#","Gene","log2FC","padj","Mean A","Mean B"]]
        for rank,(_,row) in enumerate(sub.iterrows(),1):
            rows.append([str(rank),str(row.get(gc,""))[:18],
                         f"{row.get('log2FoldChange',0):.3f}",
                         f"{row.get('padj',1):.2e}",
                         f"{row.get('mean_A',0):.2f}",
                         f"{row.get('mean_B',0):.2f}"])
        t=Table(rows,colWidths=[1.2*cm,4.5*cm,2.5*cm,2.8*cm,2.5*cm,2.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(ch)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
            ("FONTNAME",(0,1),(-1,-1),"Courier"),("PADDING",(0,0),(-1,-1),4),
        ]))
        el.append(t); el.append(Spacer(1,14))

    el.append(PageBreak())
    el.append(Paragraph("6. Methods",h2))
    el.append(Paragraph(
        "Expression data log2(x+1) transformed. Welch t-test per gene. "
        "Benjamini-Hochberg FDR correction. |log2FC|>1 and padj<0.05 = significant. "
        "Biological interpretation generated by Claude AI (Anthropic). "
        "For publication use DESeq2, edgeR, or limma-voom.",body))
    doc.build(el)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    lfc_thr  = st.slider("log2FC Threshold",0.5,3.0,1.0,0.25)
    padj_thr = st.select_slider("padj Cutoff",options=[0.001,0.01,0.05,0.1,0.2],value=0.05)
    norm     = st.selectbox("Normalization",["log2_cpm","zscore","none"])
    n_top    = st.slider("Heatmap top genes",10,80,40,5)
    st.markdown("---")
    st.markdown("### 🤖 AI Features")
    for f in ["Auto-select settings","Biological interpretation",
               "Pathway enrichment","Data chatbot"]:
        st.markdown(f"<span class='dataset-badge'>✨ {f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📂 Accepted Formats")
    st.markdown("`CSV` `TXT` `TSV` `GZ` `ZIP`\n\nIncludes GEO Series Matrix")


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:2.2rem;margin-bottom:0'>
🧬 RNA-Seq Universal Analyzer
<span class='ai-badge'>AI Powered</span>
</h1>
<p style='text-align:center;color:#8b92a5;margin-top:6px'>
Claude AI · NCBI GEO · Tumor/Normal · Treated/Control · Time Series · KO/WT
</p>
""", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload RNA-Seq Data File",
    type=["csv","txt","tsv","gz","zip"],
    help="GEO series matrix, count tables, or pre-computed DEG files"
)

if uploaded_file:
    # ── Read
    with st.spinner("Reading file..."):
        df_raw,geo_meta,gsm_groups,msgs = smart_read_file(uploaded_file)
    for lv,msg in msgs: getattr(st,lv)(msg)
    if df_raw is None: st.stop()

    # ── Preview
    with st.expander("📄 Raw Data Preview",expanded=False):
        st.write(f"Shape: **{df_raw.shape[0]:,} × {df_raw.shape[1]}**")
        st.dataframe(df_raw.head(10),use_container_width=True)

    # ── GEO metadata
    series_title = geo_meta.get("Series_title",[""])[0] if geo_meta else ""
    if gsm_groups:
        with st.expander("🔎 GEO Sample Labels",expanded=False):
            st.dataframe(pd.DataFrame({"GSM":list(gsm_groups.keys()),
                                        "Label":list(gsm_groups.values())}),
                          use_container_width=True)

    # ─────────────────────────────────────────
    #  AUTO-RETRIEVAL FROM NCBI  ← NEW v3.1
    # ─────────────────────────────────────────
    gse_id = geo_meta.get("Series_geo_accession", [""])[0] if geo_meta else ""

    if gse_id:
        st.markdown("""<div class='section-header'>
        📡 Auto-Retrieve Data from NCBI
        <span class='ai-badge'>NEW</span>
        </div>""", unsafe_allow_html=True)

        # Check if uploaded file has actual expression data
        gsm_cols = [c for c in df_raw.columns if str(c).startswith("GSM")]
        has_data = len(df_raw) > 10

        if not has_data:
            st.markdown(f"""
            <div class='ai-box'>
            ⚠️ <strong>Your uploaded file ({gse_id}) has no expression values</strong>
            — it is a metadata-only series matrix.<br><br>
            Click <strong>Auto-Retrieve</strong> to fetch the actual count data
            directly from NCBI GEO and SRA databases automatically.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='info-box'>
            ✅ Your file has expression data. You can still click
            <strong>Auto-Retrieve</strong> to check if a newer or more complete
            version exists on NCBI GEO for <strong>{gse_id}</strong>.
            </div>""", unsafe_allow_html=True)

        # Show what we already know from the metadata
        srp_ids = []
        srx_ids_meta = []
        for val in geo_meta.get("Series_relation", []):
            m = re.search(r"SRA:.*?term=(\w+)", val)
            if m: srp_ids.append(m.group(1))
        for val in geo_meta.get("Sample_relation", []):
            m = re.search(r"term=(SRX\w+)", val)
            if m: srx_ids_meta.append(m.group(1))
        srp_ids     = list(set(srp_ids))
        srx_ids_meta= list(set(srx_ids_meta))

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("GSE Accession",  gse_id)
        col_info2.metric("SRP Projects",   len(srp_ids))
        col_info3.metric("SRX Experiments",len(srx_ids_meta))

        if srp_ids:
            st.markdown(f"**SRA Project IDs:** `{'`, `'.join(srp_ids)}`")
        if srx_ids_meta:
            with st.expander(f"🔗 View all {len(srx_ids_meta)} SRX Experiment IDs"):
                st.write(", ".join(srx_ids_meta))

        # Retrieval button
        if "retrieve_result" not in st.session_state:
            st.session_state.retrieve_result = None

        if st.button("📡 Auto-Retrieve Expression Data from NCBI"):
            progress_box = st.empty()
            log_msgs = []

            def _prog(msg):
                log_msgs.append(msg)
                progress_box.markdown(
                    "\n\n".join([f"<div class='info-box'>{m}</div>"
                                 for m in log_msgs[-4:]]),
                    unsafe_allow_html=True
                )

            with st.spinner(f"Connecting to NCBI for {gse_id}..."):
                st.session_state.retrieve_result = smart_retrieve_from_geo(
                    gse_id, geo_meta, gsm_groups, _prog
                )
            progress_box.empty()

        # Show results if retrieved
        if st.session_state.retrieve_result:
            rr = st.session_state.retrieve_result
            status = rr.get("status", "failed")

            # Status banner
            if status in ("counts", "matrix"):
                st.success(rr["message"])
                retrieved_df = rr["df"]

                with st.expander("📄 Retrieved Data Preview", expanded=True):
                    st.write(f"Shape: **{retrieved_df.shape[0]:,} genes × "
                             f"{retrieved_df.shape[1]} columns**")
                    st.dataframe(retrieved_df.head(10), use_container_width=True)

                # REPLACE df_raw with retrieved data
                st.markdown("""
                <div class='ai-box'>
                🎯 <strong>Retrieved data is ready!</strong>
                Click below to use this data for your analysis instead of the uploaded file.
                </div>""", unsafe_allow_html=True)

                if st.button("✅ Use Retrieved Data for Analysis"):
                    df_raw = retrieved_df
                    # Re-detect groups from new data
                    gsm_cols_new = [c for c in df_raw.columns if str(c).startswith("GSM")]
                    if gsm_cols_new and gsm_groups:
                        grp_a, grp_b, label_a, label_b = cluster_gsm_groups(gsm_groups)
                    else:
                        dataset_type_new = detect_type_cols(df_raw)
                        grp_a, grp_b, label_a, label_b = detect_groups_non_geo(
                            df_raw, dataset_type_new
                        )
                    st.success(f"✅ Now using retrieved data: "
                               f"{label_a} ({len(grp_a)} samples) vs "
                               f"{label_b} ({len(grp_b)} samples)")
                    st.rerun()

            elif status == "runinfo":
                st.warning(rr["message"])
                runinfo_df = rr.get("runinfo")
                if runinfo_df is not None:
                    st.markdown("### 📋 SRA Run Info")
                    st.dataframe(runinfo_df[["Run","Experiment","Sample",
                                             "BioSample","Organism",
                                             "Instrument","spots","bases"]
                                            ].head(20) if all(
                                    c in runinfo_df.columns for c in
                                    ["Run","Experiment","Sample","BioSample"]
                                ) else runinfo_df.head(20),
                                 use_container_width=True)

                    buf = io.StringIO()
                    runinfo_df.to_csv(buf, index=False)
                    st.download_button(
                        "📥 Download SRA Run Info CSV",
                        buf.getvalue(),
                        file_name=f"{gse_id}_SRA_runinfo.csv",
                        mime="text/csv"
                    )

                    st.markdown(f"""
                    <div class='ai-box'>
                    📌 <strong>Next steps to get expression counts:</strong><br><br>
                    1️⃣ Download FASTQ files using the SRR IDs above:<br>
                    <code>fasterq-dump SRR_ID --outdir ./fastq/</code><br><br>
                    2️⃣ Align with STAR:<br>
                    <code>STAR --genomeDir ./genome --readFilesIn sample.fastq --outSAMtype BAM SortedByCoordinate</code><br><br>
                    3️⃣ Count with featureCounts:<br>
                    <code>featureCounts -a annotation.gtf -o counts.txt *.bam</code><br><br>
                    4️⃣ Upload the resulting <code>counts.txt</code> to this tool.
                    </div>""", unsafe_allow_html=True)

            elif status == "srx_only":
                st.info(rr["message"])
                srx_show = rr.get("srx_ids", [])
                if srx_show:
                    st.markdown("**SRX Experiment IDs:**")
                    st.code("\n".join(srx_show))
                    st.markdown(f"""
                    <div class='ai-box'>
                    💡 <strong>How to get expression data from these SRX IDs:</strong><br>
                    Visit <a href="https://www.ncbi.nlm.nih.gov/sra" target="_blank">NCBI SRA</a>
                    and search each SRX ID, or use:<br>
                    <code>prefetch {srx_show[0] if srx_show else 'SRX...'}</code><br>
                    Then process FASTQ files through your alignment pipeline.
                    </div>""", unsafe_allow_html=True)

            else:
                st.error(rr.get("message", "Retrieval failed."))
                st.markdown(f"""
                **Manual download steps for {gse_id}:**
                1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}
                2. Scroll to **Supplementary files** at the bottom
                3. Download files ending in `_counts.txt.gz` or `_matrix.txt.gz`
                4. Upload that file here
                """)

        # Always show direct GEO link
        st.markdown(f"🔗 [Open {gse_id} on NCBI GEO]"
                    f"(https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id})",
                    unsafe_allow_html=False)
        st.markdown("---")

    # ─────────────────────────────────────────
    #  AI FEATURE 1 — AUTO-SELECT SETTINGS
    # ─────────────────────────────────────────
    ai_settings = {}
    if gsm_groups and series_title:
        st.markdown("<div class='section-header'>🤖 AI Analysis Settings</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='ai-box'>
        <strong>✨ AI Auto-Settings</strong> — Claude will read your dataset metadata
        and recommend the optimal analysis configuration.
        </div>""", unsafe_allow_html=True)

        if st.button("🤖 Let AI Choose Settings"):
            with st.spinner("🧠 Claude is reading your dataset metadata..."):
                ai_settings = ai_auto_settings(geo_meta, gsm_groups, series_title)

            if ai_settings:
                st.success("✅ AI has recommended settings!")
                col1,col2,col3 = st.columns(3)
                col1.metric("Dataset Type",   ai_settings.get("dataset_type","—"))
                col2.metric("Recommended LFC", ai_settings.get("lfc_threshold","—"))
                col3.metric("Group A Label",   ai_settings.get("label_a","—"))

                if "reasoning" in ai_settings:
                    st.markdown(f"""
                    <div class='info-box'>
                    🧠 <strong>AI Reasoning:</strong> {ai_settings['reasoning']}
                    </div>""", unsafe_allow_html=True)

                # Apply AI settings to sliders
                if st.checkbox("Apply AI-recommended settings?", value=True):
                    lfc_thr  = float(ai_settings.get("lfc_threshold", lfc_thr))
                    padj_thr = float(ai_settings.get("padj_threshold", padj_thr))
                    norm     = ai_settings.get("norm_method", norm)
            else:
                st.warning("AI settings unavailable — using manual settings.")

    # ── Detect dataset type
    if ai_settings.get("dataset_type"):
        dataset_type = ai_settings["dataset_type"]
    elif gsm_groups:
        dataset_type = detect_type_meta(geo_meta, gsm_groups)
    else:
        dataset_type = detect_type_cols(df_raw)

    type_labels = {
        "tumor_normal":"🔬 Tumor vs Normal","treated_control":"💊 Treated vs Control",
        "time_series":"⏱️ Time Series","knockout_wildtype":"🔧 Knockout vs Wildtype",
        "pre_computed":"📁 Pre-computed","single_condition":"📊 Single Condition",
    }
    st.markdown(f"<div class='info-box'>🔍 Dataset type: "
                f"<strong>{type_labels.get(dataset_type,dataset_type)}</strong>"
                f"{'  <span class=\"ai-badge\">AI</span>' if ai_settings.get('dataset_type') else ''}"
                f"</div>", unsafe_allow_html=True)

    override = st.selectbox("Override dataset type (optional)",
                             ["auto"]+list(type_labels.keys()),
                             format_func=lambda x: "Auto-detect" if x=="auto" else type_labels.get(x,x))
    if override != "auto": dataset_type = override

    # ── Compute groups + DE
    if dataset_type=="pre_computed" and \
       "log2FoldChange" in df_raw.columns and "padj" in df_raw.columns:
        df=df_raw.copy(); label_a,label_b="Group A","Group B"
        if "mean_A" not in df.columns: df["mean_A"]=0; df["mean_B"]=0
        grp_a,grp_b=[],[]

    elif gsm_groups:
        if ai_settings.get("label_a") and ai_settings.get("label_b"):
            grp_a,grp_b,la_auto,lb_auto = cluster_gsm_groups(gsm_groups)
            label_a = ai_settings["label_a"]
            label_b = ai_settings["label_b"]
        else:
            grp_a,grp_b,label_a,label_b = cluster_gsm_groups(gsm_groups)

        st.markdown(f"""<div class='info-box'>
        <strong>{label_a}</strong>: {', '.join(grp_a[:3])}{'...' if len(grp_a)>3 else ''} ({len(grp_a)} samples)<br>
        <strong>{label_b}</strong>: {', '.join(grp_b[:3])}{'...' if len(grp_b)>3 else ''} ({len(grp_b)} samples)
        </div>""", unsafe_allow_html=True)

        all_gsm=list(gsm_groups.keys())
        with st.expander("✏️ Manually adjust groups (if AI got it wrong)"):
            grp_a=st.multiselect("Group A (case)",all_gsm,default=grp_a,key="ga")
            grp_b=st.multiselect("Group B (control)",all_gsm,default=grp_b,key="gb")
            label_a=st.text_input("Label A",value=label_a)
            label_b=st.text_input("Label B",value=label_b)

        if not grp_a or not grp_b:
            st.error("Could not identify groups. Use the manual adjustment above.")
            st.stop()

        with st.spinner("Computing differential expression..."):
            df=compute_de(df_raw,grp_a,grp_b,label_a,label_b,norm)
    else:
        grp_a,grp_b,label_a,label_b=detect_groups_non_geo(df_raw,dataset_type)
        if not grp_a or not grp_b:
            st.error("Could not identify sample groups."); st.stop()
        with st.spinner("Computing differential expression..."):
            df=compute_de(df_raw,grp_a,grp_b,label_a,label_b,norm)

    df=classify_genes(df,lfc_thr,padj_thr)
    gene_col=find_gene_col(df)
    if gene_col is None: df["_gene"]=df.index.astype(str); gene_col="_gene"

    up  =(df["Category"]=="Upregulated").sum()
    down=(df["Category"]=="Downregulated").sum()

    # ── Metrics
    st.markdown("<div class='section-header'>📊 Analysis Summary</div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for col,val,lbl,clr in [
        (c1,len(df),"Total Genes","#00d4aa"),
        (c2,up,f"↑ Up ({label_a})","#ff4d6d"),
        (c3,down,f"↓ Down ({label_b})","#4da6ff"),
        (c4,up+down,"Total DEGs","#7c3aed"),
    ]:
        col.markdown(f"""<div class='metric-card'>
        <div class='metric-number' style='color:{clr}'>{val:,}</div>
        <div style='color:#8b92a5;font-size:0.83rem;margin-top:4px'>{lbl}</div>
        </div>""",unsafe_allow_html=True)

    # ── Plots
    st.markdown("<div class='section-header'>📈 Visualizations</div>",unsafe_allow_html=True)
    saved_figs={}; tmp=tempfile.mkdtemp()
    def sf(fig,k):
        if fig is None: return
        p=os.path.join(tmp,f"{k}.png"); _save(fig,p); saved_figs[k]=p

    t1,t2,t3,t4,t5,t6=st.tabs(["🌋 Volcano","📉 MA","🧊 Heatmap","🔵 PCA","🎻 Dist","📊 Bar"])
    with t1:
        fig=plot_volcano(df,gene_col,label_a,label_b,lfc_thr,padj_thr)
        sf(fig,"volcano"); st.pyplot(fig)
    with t2:
        if "mean_A" in df.columns:
            fig=plot_ma(df,label_a,label_b); sf(fig,"ma"); st.pyplot(fig)
        else: st.info("MA needs raw group data.")
    with t3:
        if grp_a and grp_b:
            fig=plot_heatmap(df,grp_a,grp_b,gene_col,n_top)
            if fig: sf(fig,"heatmap"); st.pyplot(fig)
    with t4:
        if grp_a and grp_b:
            fig=plot_pca(df,grp_a,grp_b,label_a,label_b)
            if fig: sf(fig,"pca"); st.pyplot(fig)
            else: st.info("PCA needs ≥3 samples.")
    with t5:
        if grp_a and grp_b:
            fig=plot_dist(df,grp_a,grp_b,label_a,label_b)
            if fig: sf(fig,"dist"); st.pyplot(fig)
    with t6:
        fig=plot_bar(up,down,label_a,label_b); sf(fig,"bar"); st.pyplot(fig)

    # ── Top genes table
    st.markdown("<div class='section-header'>🔝 Top Significant Genes</div>",unsafe_allow_html=True)
    sig=df[df["Category"]!="Not Significant"].copy()
    if len(sig)>0:
        dcols=[gene_col,"log2FoldChange","padj","Category"]
        if "mean_A" in sig.columns: dcols+=["mean_A","mean_B"]
        top50=sig[dcols].nsmallest(50,"padj").rename(columns={
            "log2FoldChange":"log2FC","mean_A":f"Mean({label_a})","mean_B":f"Mean({label_b})"})
        def _c(v):
            if v=="Upregulated": return "background-color:rgba(255,77,109,0.12);color:#ff4d6d"
            if v=="Downregulated": return "background-color:rgba(77,166,255,0.12);color:#4da6ff"
            return ""
        st.dataframe(top50.style.applymap(_c,subset=["Category"])
                               .format({"log2FC":"{:.3f}","padj":"{:.2e}"}),
                     use_container_width=True,height=360)
        buf=io.StringIO(); df.to_csv(buf,index=False)
        st.download_button("📥 Download Full CSV",buf.getvalue(),
                            file_name=f"RNA_Seq_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv")
    else:
        st.warning("No DEGs found — try lowering the LFC or padj threshold in the sidebar.")

    # ─────────────────────────────────────────
    #  AI FEATURE 2 — BIOLOGICAL INTERPRETATION
    # ─────────────────────────────────────────
    st.markdown("""<div class='section-header'>
    🤖 AI Biological Interpretation
    <span class='ai-badge'>Claude AI</span>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div class='ai-box'>
    Claude AI will analyze your DEG results and write a comprehensive biological
    interpretation — covering pathway activation, key gene functions,
    disease relevance, and suggested follow-up experiments.
    </div>""",unsafe_allow_html=True)

    if "ai_interp" not in st.session_state:
        st.session_state.ai_interp = ""

    if up+down > 0:
        if st.button("🧠 Generate AI Interpretation"):
            with st.spinner("🤖 Claude is analyzing your results... (15-30 seconds)"):
                st.session_state.ai_interp = ai_interpret_results(
                    df,gene_col,dataset_type,label_a,label_b,
                    up,down,series_title
                )
        if st.session_state.ai_interp:
            st.markdown(f"""<div class='ai-box'>
            <strong>🧬 AI Biological Interpretation</strong><br><br>
            {st.session_state.ai_interp.replace(chr(10),'<br>')}
            </div>""",unsafe_allow_html=True)
    else:
        st.info("Run analysis first to generate AI interpretation.")

    # ─────────────────────────────────────────
    #  AI FEATURE 3 — PATHWAY ENRICHMENT
    # ─────────────────────────────────────────
    st.markdown("""<div class='section-header'>
    🔬 AI Pathway Enrichment
    <span class='ai-badge'>Claude AI</span>
    </div>""",unsafe_allow_html=True)

    if "ai_pathways" not in st.session_state:
        st.session_state.ai_pathways = ""

    organism = geo_meta.get("Sample_organism_ch1",["human/mouse"])[0] if geo_meta else "human/mouse"
    up_genes = df[df["Category"]=="Upregulated"][gene_col].astype(str).tolist()
    dn_genes = df[df["Category"]=="Downregulated"][gene_col].astype(str).tolist()

    if up+down > 0:
        if st.button("🔬 Run AI Pathway Analysis"):
            with st.spinner("🤖 Claude is performing pathway enrichment... (15-30 seconds)"):
                st.session_state.ai_pathways = ai_pathway_enrichment(
                    up_genes,dn_genes,label_a,label_b,organism
                )
        if st.session_state.ai_pathways:
            st.markdown(f"""<div class='ai-box'>
            <strong>🗺️ AI Pathway Enrichment Analysis</strong><br><br>
            {st.session_state.ai_pathways.replace(chr(10),'<br>')}
            </div>""",unsafe_allow_html=True)
    else:
        st.info("Run analysis first to perform pathway enrichment.")

    # ─────────────────────────────────────────
    #  AI FEATURE 4 — DATA CHATBOT
    # ─────────────────────────────────────────
    st.markdown("""<div class='section-header'>
    💬 Ask AI About Your Data
    <span class='ai-badge'>Claude AI</span>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div class='ai-box'>
    Chat directly with Claude about your results. Ask anything:<br>
    <em>"What does DUSP6 downregulation mean?"</em> &nbsp;·&nbsp;
    <em>"Which genes are related to apoptosis?"</em> &nbsp;·&nbsp;
    <em>"What experiments should I do next?"</em>
    </div>""",unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "data_context" not in st.session_state:
        st.session_state.data_context = build_data_context(
            df,gene_col,label_a,label_b,up,down,
            dataset_type,series_title,geo_meta
        )

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>🤖 {msg['content']}</div>",
                        unsafe_allow_html=True)

    # Quick question buttons
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "What are the top 5 most significant genes and their functions?",
        "Which apoptosis-related genes changed significantly?",
        "What follow-up experiments do you recommend?",
    ]
    for i,(qcol,q) in enumerate(zip(qcols,quick_qs)):
        if qcol.button(f"💡 {q[:35]}...", key=f"quick_{i}"):
            with st.spinner("🤖 Claude is thinking..."):
                st.session_state.chat_history.append({"role":"user","content":q})
                reply = call_claude_chat(
                    st.session_state.chat_history,
                    st.session_state.data_context,
                    max_tokens=800
                )
                st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.rerun()

    # Chat input
    user_q = st.text_input("Ask a question about your data:",
                            placeholder="e.g. What pathways are enriched in the upregulated genes?",
                            key="chat_input")
    col_send, col_clear = st.columns([3,1])
    with col_send:
        if st.button("Send 📨") and user_q.strip():
            with st.spinner("🤖 Claude is thinking..."):
                st.session_state.chat_history.append({"role":"user","content":user_q})
                reply = call_claude_chat(
                    st.session_state.chat_history,
                    st.session_state.data_context,
                    max_tokens=800
                )
                st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.rerun()
    with col_clear:
        if st.button("Clear Chat 🗑️"):
            st.session_state.chat_history=[]
            st.rerun()

    # ── PDF REPORTS
    st.markdown("---")
    st.markdown("<div class='section-header'>📄 Download Reports</div>",unsafe_allow_html=True)
    cf,cp=st.columns(2)

    with cf:
        st.markdown("### 🆓 Free Report")
        st.markdown("Summary + Top Genes + Volcano Plot")
        if st.button("📥 Generate Free PDF"):
            with st.spinner("Building PDF..."):
                path=os.path.join(tmp,"Free.pdf")
                try:
                    build_free_pdf(df,gene_col,saved_figs,up,down,label_a,label_b,dataset_type,path)
                    with open(path,"rb") as f:
                        st.download_button("⬇️ Download Free PDF",f.read(),
                                            file_name=f"RNA_Free_{datetime.now().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf")
                    st.success("✅ Ready!")
                except Exception as e: st.error(f"PDF error: {e}")

    with cp:
        st.markdown("""<div class='premium-box'>
        <h3 style='color:#7c3aed;margin:0 0 6px'>💎 Premium AI Report</h3>
        <p style='font-size:0.88rem;color:#888;margin:0'>
        All plots · AI biological interpretation · AI pathway enrichment ·
        Full gene tables · Methods section
        </p></div>""",unsafe_allow_html=True)
        code=st.text_input("🔑 Access Code",type="password",placeholder="Enter after payment")
        VALID={"BIO100","RNASEQ2025","PREMIUM99"}
        if code in VALID:
            st.success("✅ Access granted!")
            if st.button("💎 Generate Premium AI PDF"):
                with st.spinner("Building premium AI PDF..."):
                    interp = st.session_state.get("ai_interp","")
                    pathways = st.session_state.get("ai_pathways","")
                    if not interp:
                        with st.spinner("Generating AI interpretation..."):
                            interp = ai_interpret_results(
                                df,gene_col,dataset_type,label_a,label_b,
                                up,down,series_title
                            )
                    path=os.path.join(tmp,"Premium.pdf")
                    try:
                        build_premium_pdf(df,gene_col,saved_figs,up,down,
                                          label_a,label_b,dataset_type,
                                          interp,pathways,path)
                        with open(path,"rb") as f:
                            st.download_button("⬇️ Download Premium PDF",f.read(),
                                                file_name=f"RNA_Premium_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                mime="application/pdf")
                        st.success("✅ Premium AI Report ready!")
                        st.balloons()
                    except Exception as e: st.error(f"PDF error: {e}")
        elif code: st.error("❌ Invalid code.")
        else: st.info("Enter access code to unlock.")

else:
    # Landing page
    st.markdown("""
    <div style='text-align:center;padding:40px 20px'>
        <div style='font-size:4rem'>🧬</div>
        <h2 style='color:#00d4aa'>Upload your RNA-Seq file to begin</h2>
        <p style='color:#8b92a5;max-width:600px;margin:0 auto'>
        Drop CSV, TXT, GZ or ZIP — including NCBI GEO Series Matrix.
        Powered by Claude AI for intelligent analysis and interpretation.
        </p>
    </div>""",unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,icon,title,desc in [
        (c1,"🔍","Smart Detection","Auto-identifies groups from GEO metadata"),
        (c2,"🤖","Claude AI","Biological interpretation & pathway analysis"),
        (c3,"💬","AI Chatbot","Ask questions about your specific results"),
        (c4,"📄","AI Reports","Free summary + Premium full AI analysis"),
    ]:
        col.markdown(f"""<div class='metric-card' style='text-align:left'>
        <div style='font-size:2rem'>{icon}</div>
        <div style='color:white;font-weight:700;margin:8px 0 4px'>{title}</div>
        <div style='color:#8b92a5;font-size:0.85rem'>{desc}</div>
        </div>""",unsafe_allow_html=True)
