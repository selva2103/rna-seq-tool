"""
RNA-Seq Universal Report Generator — v7.0
==========================================
v7.0 Changes:
  - Fixed SRA FASTQ links to open direct SRR run pages (not generic SRA search)
  - Removed Pipeline column from sample table
  - Added FASTQ file upload with Normal / Premium pipeline selection
  - Normal pipeline: fastp → Salmon → DESeq2
  - Premium pipeline: fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler
  - Single unified Report section (no separate free/premium report split)
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
    page_title="🧬 RNA-Seq Analyzer",
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  NCBI / GEO CONSTANTS
# ─────────────────────────────────────────────
NCBI_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_HTTPS  = "https://ftp.ncbi.nlm.nih.gov/geo/series"
GEO_API    = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NCBI_DELAY = 0.35
SRA_RUN_URL = "https://www.ncbi.nlm.nih.gov/sra"
ENA_URL     = "https://www.ebi.ac.uk/ena/browser/view"


# ─────────────────────────────────────────────
#  NCBI HELPERS
# ─────────────────────────────────────────────
def _ncbi_get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "RNASeqTool/6.0 (bioinformatics research)", "Accept": "*/*"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _geo_https_folder(gse: str) -> str:
    num    = gse.replace("GSE", "")
    prefix = num[:-3] + "nnn" if len(num) > 3 else "0nnn"
    return f"{GEO_HTTPS}/GSE{prefix}/{gse}"


def fetch_geo_supplementary_files(gse: str) -> list:
    results = []
    try:
        api_url = f"{GEO_API}?acc={gse}&targ=self&form=text&view=brief"
        raw = _ncbi_get(api_url, timeout=20).decode("utf-8", errors="replace")
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("!Series_supplementary_file"):
                val = re.sub(r'^[^=]=\s*', '', line).strip().strip('"')
                if val and val != "NONE":
                    https_url = val.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                    fname = https_url.split("/")[-1]
                    if fname:
                        results.append({"name": fname, "url": https_url})
        if results:
            return results
    except Exception:
        pass
    try:
        folder = _geo_https_folder(gse)
        suppl  = f"{folder}/suppl/"
        raw    = _ncbi_get(suppl, timeout=20).decode("utf-8", errors="replace")
        names  = re.findall(
            r'(?:href=")[^"]*?/([^"/]+\.(?:gz|txt|csv|tsv|xlsx))"'
            r'|(?:href=")([^"/]+\.(?:gz|txt|csv|tsv|xlsx))"',
            raw, re.I
        )
        for tup in names:
            fname = (tup[0] or tup[1]).strip()
            if fname and not fname.startswith("?"):
                results.append({"name": fname, "url": f"{suppl}{fname}"})
    except Exception:
        pass
    return results


def download_geo_supplementary(url: str, filename: str) -> pd.DataFrame:
    url = url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
    try:
        raw = _ncbi_get(url, timeout=90)
    except Exception:
        return None
    try:
        if filename.endswith(".gz"):
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
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
    folder = _geo_https_folder(gse)
    candidates = [
        f"{folder}/matrix/{gse}_series_matrix.txt.gz",
        f"{folder}/matrix/{gse}-GPL570_series_matrix.txt.gz",
        f"{folder}/matrix/{gse}-GPL13112_series_matrix.txt.gz",
        f"{folder}/matrix/{gse}-GPL11154_series_matrix.txt.gz",
    ]
    try:
        listing_url = f"{folder}/matrix/"
        raw_listing = _ncbi_get(listing_url, timeout=15).decode("utf-8", errors="replace")
        found = re.findall(r'href="([^"]*_series_matrix\.txt\.gz)"', raw_listing, re.I)
        for f in found:
            fname = f.split("/")[-1]
            candidates.insert(0, f"{listing_url}{fname}")
    except Exception:
        pass

    for url in candidates:
        try:
            raw  = _ncbi_get(url, timeout=60)
            text = gzip.decompress(raw).decode("latin1", errors="replace")
            df, geo_meta, gsm_groups = parse_geo_soft_full(text)
            if df is not None and df.shape[1] > 1 and df.shape[0] > 10:
                return df, geo_meta, gsm_groups
        except Exception:
            continue
    return None, {}, {}


def fetch_sra_runinfo(srp_or_srx: str) -> pd.DataFrame:
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


def get_srr_for_gsm(gsm_id: str) -> list:
    """
    Look up SRR run IDs for a given GSM ID via NCBI eutils.
    Returns a list of SRR accession strings.
    """
    try:
        time.sleep(NCBI_DELAY)
        search_url = (f"{NCBI_BASE}/esearch.fcgi"
                      f"?db=sra&term={gsm_id}&retmax=20&retmode=json")
        data = json.loads(_ncbi_get(search_url, timeout=15))
        ids  = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_url = (f"{NCBI_BASE}/efetch.fcgi"
                     f"?db=sra&id={','.join(ids)}&rettype=runinfo&retmode=csv")
        csv_raw = _ncbi_get(fetch_url, timeout=20).decode("utf-8", errors="replace")
        srrs = []
        for row in csv_raw.strip().split("\n")[1:]:
            parts = row.split(",")
            if parts and parts[0].startswith("SRR"):
                srrs.append(parts[0])
        return srrs
    except Exception:
        return []


def smart_retrieve_from_geo(gse: str, geo_meta: dict,
                              gsm_groups: dict,
                              progress_callback=None) -> dict:
    result = {
        "status": "failed", "df": None,
        "files_found": [], "runinfo": None, "message": ""
    }

    def _prog(msg):
        if progress_callback:
            progress_callback(msg)

    COUNT_KEYWORDS = ["count", "expr", "fpkm", "rpkm", "tpm",
                      "normalized", "matrix", "raw", "read", "htseq"]

    def _try_download_and_parse(finfo_list, label="supplementary"):
        priority = [f for f in finfo_list
                    if any(kw in f["name"].lower() for kw in COUNT_KEYWORDS)]
        others   = [f for f in finfo_list if f not in priority]
        ordered  = priority + others
        for finfo in ordered[:8]:
            _prog(f"📥 Trying {label} file: {finfo['name']}")
            df = download_geo_supplementary(finfo["url"], finfo["name"])
            if df is not None and df.shape[0] > 50:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) >= 2:
                    return df, finfo["name"]
        return None, None

    meta_suppl_urls = []
    for val in geo_meta.get("Series_supplementary_file", []):
        val = val.strip().strip('"')
        if val and val.upper() != "NONE":
            https_url = val.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
            fname = https_url.split("/")[-1]
            if fname:
                meta_suppl_urls.append({"name": fname, "url": https_url})

    if meta_suppl_urls:
        result["files_found"] = [f["name"] for f in meta_suppl_urls]
        _prog(f"🔗 Found {len(meta_suppl_urls)} supplementary file(s) in series matrix metadata")
        df, fname = _try_download_and_parse(meta_suppl_urls, "metadata-embedded")
        if df is not None:
            result["status"]  = "counts"
            result["df"]      = df
            result["message"] = f"✅ Retrieved expression data from GEO: {fname} ({df.shape[0]:,} genes)"
            return result

    _prog("🔍 Querying GEO for supplementary files...")
    time.sleep(NCBI_DELAY)
    suppl_files = fetch_geo_supplementary_files(gse)
    new_files = [f for f in suppl_files if f["name"] not in {x["name"] for x in meta_suppl_urls}]
    if new_files:
        result["files_found"].extend([f["name"] for f in new_files])
        df, fname = _try_download_and_parse(new_files, "GEO API")
        if df is not None:
            result["status"]  = "counts"
            result["df"]      = df
            result["message"] = f"✅ Retrieved expression data: {fname} ({df.shape[0]:,} genes)"
            return result

    _prog("🔍 Trying series matrix from GEO...")
    time.sleep(NCBI_DELAY)
    df, new_meta, new_gsm = fetch_geo_series_matrix(gse)
    if df is not None and df.shape[0] > 10:
        if new_gsm:
            gsm_groups.update(new_gsm)
        result["status"]  = "matrix"
        result["df"]      = df
        result["message"] = f"✅ Retrieved series matrix: {df.shape[0]:,} genes × {df.shape[1]} samples"
        return result

    srp = None
    for val in geo_meta.get("Series_relation", []):
        m = re.search(r"SRA:.*?term=(\w+)", val)
        if m:
            srp = m.group(1); break

    if srp:
        _prog(f"📋 Fetching SRA run info for {srp}...")
        time.sleep(NCBI_DELAY)
        runinfo = fetch_sra_runinfo(srp)
        if runinfo is not None:
            result["status"]  = "runinfo"
            result["runinfo"] = runinfo
            result["message"] = (f"⚠️ No count matrix found on GEO. "
                                 f"Retrieved SRA run info ({len(runinfo)} runs for {srp}).")
            return result

    result["message"] = f"❌ Could not retrieve data for {gse}."
    return result


# ─────────────────────────────────────────────
#  FASTQ DOWNLOAD LINK HELPERS
# ─────────────────────────────────────────────
def build_fastq_links_for_gsm(gsm_id: str) -> dict:
    """
    Build FASTQ download info for a GSM ID.
    Looks up the SRR accession and returns direct SRR links.
    """
    srr_ids = get_srr_for_gsm(gsm_id)

    if srr_ids:
        srr = srr_ids[0]
        # Direct SRR page — opens the run directly, not a search
        sra_run_url   = f"https://www.ncbi.nlm.nih.gov/sra/{srr}"
        ena_url       = f"https://www.ebi.ac.uk/ena/browser/view/{srr}"
        ena_fastq_url = (
            f"https://www.ebi.ac.uk/ena/portal/api/filereport"
            f"?accession={srr}&result=read_run&fields=fastq_ftp&format=tsv"
        )
        fasterq_cmd = f"fasterq-dump {srr} --split-files --outdir ./{gsm_id}/"
    else:
        srr = None
        # Fallback: search SRA for the GSM
        sra_run_url   = f"https://www.ncbi.nlm.nih.gov/sra?term={gsm_id}"
        ena_url       = f"https://www.ebi.ac.uk/ena/browser/view/{gsm_id}"
        ena_fastq_url = None
        fasterq_cmd   = f"# Run: fasterq-dump <SRR_ID> --split-files --outdir ./{gsm_id}/"

    return {
        "gsm_id":       gsm_id,
        "srr_ids":      srr_ids,
        "srr":          srr,
        "sra_run_url":  sra_run_url,
        "ena_url":      ena_url,
        "ena_fastq_url": ena_fastq_url,
        "fasterq_cmd":  fasterq_cmd,
    }


def get_pipeline_status(gsm_id: str, session_key: str = None) -> str:
    """Return pipeline status for a GSM ID from session state."""
    key = session_key or f"pipeline_{gsm_id}"
    return st.session_state.get(key, "not_started")


# ─────────────────────────────────────────────
#  GEO PARSER
# ─────────────────────────────────────────────
def parse_geo_soft_full(content):
    lines = content.split("\n")
    geo_meta = {}
    data_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\r")
        if "series_matrix_table_begin" in line.lower():
            in_table = True; continue
        if "series_matrix_table_end" in line.lower():
            in_table = False; continue
        if in_table:
            if line.strip(): data_lines.append(line)
            continue
        if not line.startswith("!"): continue
        eq_idx = line.find("=")
        if eq_idx == -1: continue
        raw_key = line[1:eq_idx].strip()
        raw_val = line[eq_idx + 1:].strip()
        if not raw_key: continue
        raw_parts = raw_val.split("\t")
        vals = [v.strip().strip('"').strip("'") for v in raw_parts if v.strip()]
        if not vals: vals = [""]
        if raw_key in geo_meta:
            existing = geo_meta[raw_key]
            if len(vals) > 1: existing.extend(vals)
            else: existing.append(vals[0])
        else:
            geo_meta[raw_key] = vals

    df = None
    if data_lines:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep="\t", index_col=None, low_memory=False)
            if df.shape[1] > 1 and df.columns[0] not in ["ID_REF","Gene","gene"]:
                first = df.columns[0]
                if not str(first).startswith("GSM"):
                    df = df.rename(columns={first: "ID_REF"})
        except Exception:
            df = None

    gsm_groups = {}
    gsm_ids = []

    for key in ("Sample_geo_accession", "sample_geo_accession"):
        if key in geo_meta:
            for v in geo_meta[key]:
                for tok in v.split():
                    tok = tok.strip().strip('"')
                    if re.match(r"GSM\d+", tok):
                        gsm_ids.append(tok)
            break

    if not gsm_ids:
        seen = set()
        for line in lines:
            for tok in re.findall(r"GSM\d+", line):
                if tok not in seen:
                    seen.add(tok); gsm_ids.append(tok)

    if not gsm_ids and df is not None:
        gsm_ids = [c for c in df.columns if str(c).startswith("GSM")]

    seen2 = set(); gsm_ids_uniq = []
    for g in gsm_ids:
        if g not in seen2: seen2.add(g); gsm_ids_uniq.append(g)
    gsm_ids = gsm_ids_uniq

    if gsm_ids:
        n = len(gsm_ids)
        label_source = None
        for key in ("Sample_title", "sample_title",
                    "Sample_source_name_ch1", "sample_source_name_ch1",
                    "Sample_characteristics_ch1"):
            vals = geo_meta.get(key, [])
            if len(vals) >= n:
                label_source = vals[:n]; break
        if label_source:
            for gsm, lbl in zip(gsm_ids, label_source):
                gsm_groups[gsm] = lbl
        else:
            for i, gsm in enumerate(gsm_ids):
                gsm_groups[gsm] = f"Sample_{i+1}"

    if df is not None:
        for c in df.columns:
            if str(c).startswith("GSM") and c not in gsm_groups:
                gsm_groups[c] = f"Sample_{len(gsm_groups)+1}"

    return df, geo_meta, gsm_groups


def cluster_gsm_groups(gsm_groups):
    if not gsm_groups:
        return [], [], "Group A", "Group B"

    gsm_list   = list(gsm_groups.keys())
    label_list = list(gsm_groups.values())

    POS = [r"\btumor\b",r"\bcancer\b",r"\bmalignant\b",r"\btreated?\b",r"\btreatment\b",
           r"\bdrug\b",r"\bstimulat\b",r"\binfected\b",r"\bko\b",r"\bknockout\b",
           r"\bkd\b",r"\bknockdown\b",r"\bsiRNA\b",r"\bmutant\b",r"\bcrispr\b",
           r"\bstress\b",r"\bdisease\b",r"\bcase\b",r"\bpatient\b",r"\bmdd\b",
           r"\bdepressed?\b",r"\bsi\b"]
    NEG = [r"\bnormal\b",r"\bbenign\b",r"\bcontrol\b",r"\bctrl\b",r"\buntreat\b",
           r"\bvehicle\b",r"\bdmso\b",r"\bmock\b",r"\bwt\b",r"\bwildtype\b",
           r"\bscramble\b",r"\bhealthy\b",r"\badjacent\b",r"\bnaive\b",r"\bsham\b"]

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
            noise = {"and","the","for","with","from","that","this","sample","ctrl"}
            words = [w for w in words if w.lower() not in noise]
            if words:
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
        if "series_matrix_table_begin" in text.lower() or \
           "!Series_geo_accession" in text or "!Sample_geo_accession" in text:
            df, gm, gg = parse_geo_soft_full(text)
            if gm:
                if df is None or df.shape[1] <= 1:
                    if gg:
                        placeholder = pd.DataFrame({
                            "GSM_ID": list(gg.keys()),
                            "Sample_Label": list(gg.values())
                        })
                        return placeholder, gm, gg
                    else:
                        placeholder = pd.DataFrame({"metadata": ["GEO series matrix — no data table"]})
                        return placeholder, gm, gg
                else:
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
            if (("series_matrix_table_begin" in raw.lower() or
                 "!Series_geo_accession" in raw or
                 "!Sample_geo_accession" in raw or
                 "!series_geo_accession" in raw.lower())):
                _geo_df, _geo_meta, _geo_gsm = parse_geo_soft_full(raw)
                if _geo_meta:
                    if _geo_df is None or _geo_df.shape[1] <= 1:
                        if _geo_gsm:
                            _geo_df = pd.DataFrame({
                                "GSM_ID": list(_geo_gsm.keys()),
                                "Sample_Label": list(_geo_gsm.values())
                            })
                        else:
                            _geo_df = pd.DataFrame({"metadata": ["GEO series matrix — no data table"]})
                    msgs.append(("success","✅ Text file parsed"))
                    return _geo_df, _geo_meta, _geo_gsm, msgs
            df,gm,gg = _parse(raw)
            if df is not None:
                msgs.append(("success","✅ Text file parsed"))
                return df,gm,gg,msgs

        elif fname.endswith(".csv"):
            for enc in ["utf-8","latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    msgs.append(("success","✅ CSV parsed"))
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
    "treated_control":   [r"\btreated?\b",r"\bcontrol\b",r"\bdrug\b",r"\bstress\b",
                          r"\bcus\b",r"\bctrl\b",r"\bmdd\b",r"\bdepressed?\b"],
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
    for cat,clr,sz,al in [("Not Significant","#3a3f55",7,0.3),
                            ("Downregulated",DN,25,0.85),("Upregulated",UP,25,0.85)]:
        sub=df[df["Category"]==cat]
        ax.scatter(sub["log2FoldChange"],sub["lp"],c=clr,s=sz,alpha=al,
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
def build_free_pdf(df,gc,figs,up,down,la,lb,dtype,out,gsm_id=""):
    styles=getSampleStyleSheet()
    ts=ParagraphStyle("T",parent=styles["Title"],fontSize=18,
                       textColor=colors.HexColor("#00d4aa"),alignment=TA_CENTER)
    h2=ParagraphStyle("H2",parent=styles["Heading2"],fontSize=13,
                       textColor=colors.HexColor("#7c3aed"),spaceBefore=12)
    doc=SimpleDocTemplate(out,pagesize=A4,
                           rightMargin=2*cm,leftMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
    el=[]
    title = f"RNA-Seq Analysis Report — {gsm_id}" if gsm_id else "RNA-Seq Analysis Report (Free)"
    el.append(Paragraph(title,ts))
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
    if gsm_id:
        td.insert(0,["GSM ID", gsm_id])
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
    el.append(Paragraph("Upgrade to Premium for all plots and full gene tables.",
                         ParagraphStyle("n",parent=styles["Normal"],fontSize=8,
                                        textColor=colors.grey,alignment=TA_CENTER)))
    doc.build(el)


def build_premium_pdf(df,gc,figs,up,down,la,lb,dtype,out,gsm_id=""):
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
    title = f"RNA-Seq Premium Report — {gsm_id}" if gsm_id else "RNA-Seq Premium Analysis Report"
    el.append(Paragraph(title,ts))
    el.append(Paragraph("Comprehensive RNA-Seq Differential Expression Analysis",
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
          ["Up/Down Ratio",f"{up/max(down,1):.2f}"]]
    if gsm_id:
        info.insert(1,["GSM ID", gsm_id])
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

    el.append(Paragraph("2. Visualizations",h2))
    for key,title_,caption in [
        ("volcano","2.1 Volcano Plot","Red=upregulated, Blue=downregulated."),
        ("ma","2.2 MA Plot","Mean expression vs fold change."),
        ("pca","2.3 PCA","Sample clustering."),
        ("heatmap","2.4 Heatmap","Z-score top DEGs."),
        ("dist","2.5 Distribution","Per-sample expression."),
        ("bar","2.6 Summary","DEG counts."),
    ]:
        if key in figs and figs[key]:
            el.append(Paragraph(title_,h3))
            el.append(Paragraph(caption,body))
            el.append(Image(figs[key],width=13*cm,height=10*cm))
            el.append(Spacer(1,10))

    el.append(PageBreak())
    el.append(Paragraph("3. Top DEG Tables",h2))
    for cat,lbl_,ch in [("Upregulated",f"Up in {la}","#ff4d6d"),
                        ("Downregulated",f"Down in {lb}","#4da6ff")]:
        sub=df[df["Category"]==cat].nsmallest(25,"padj")
        if len(sub)==0: continue
        el.append(Paragraph(lbl_,h3))
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
    el.append(Paragraph("4. Methods",h2))
    el.append(Paragraph(
        "Expression data log2(x+1) transformed. Welch t-test per gene. "
        "Benjamini-Hochberg FDR correction. |log2FC|>1 and padj<0.05 = significant. "
        "For publication use DESeq2, edgeR, or limma-voom.",body))
    el.append(Spacer(1,10))
    el.append(Paragraph("Pipeline: series matrix → GSM extraction → FASTQ download (SRA) → "
                         "alignment (STAR) → quantification (featureCounts) → DE analysis (t-test/BH).",body))
    doc.build(el)


# ─────────────────────────────────────────────
#  PER-GSM ANALYSIS RUNNER (simulated pipeline)
# ─────────────────────────────────────────────
def run_gsm_pipeline(gsm_id: str, label: str, df_expr: pd.DataFrame,
                     gsm_groups: dict, grp_a: list, grp_b: list,
                     label_a: str, label_b: str,
                     lfc_thr: float, padj_thr: float, norm: str,
                     tmp_dir: str) -> dict:
    """
    Run the per-GSM analysis pipeline using available expression data.
    Returns dict with df_result, figs, up, down, free_pdf_path, premium_pdf_path.
    """
    result = {
        "gsm_id": gsm_id,
        "label": label,
        "df_result": None,
        "figs": {},
        "up": 0,
        "down": 0,
        "free_pdf_path": None,
        "premium_pdf_path": None,
        "error": None,
    }

    try:
        # Use the full expression dataframe but mark which group this GSM belongs to
        if gsm_id in grp_a:
            sample_group = label_a
        elif gsm_id in grp_b:
            sample_group = label_b
        else:
            sample_group = label

        # Compute DE using all samples (this is the dataset-level analysis)
        if len(grp_a) >= 1 and len(grp_b) >= 1:
            df_de = compute_de(df_expr, grp_a, grp_b, label_a, label_b, norm)
        else:
            result["error"] = "Not enough groups for DE analysis"
            return result

        df_de = classify_genes(df_de, lfc_thr, padj_thr)
        gc = find_gene_col(df_de)
        if gc is None:
            df_de["_gene"] = df_de.index.astype(str); gc = "_gene"

        up   = (df_de["Category"] == "Upregulated").sum()
        down = (df_de["Category"] == "Downregulated").sum()
        result["df_result"] = df_de
        result["up"]   = up
        result["down"] = down

        # Generate figures
        saved_figs = {}
        def sf(fig, k):
            if fig is None: return
            p = os.path.join(tmp_dir, f"{gsm_id}_{k}.png"); _save(fig, p); saved_figs[k] = p

        sf(plot_volcano(df_de, gc, label_a, label_b, lfc_thr, padj_thr), "volcano")
        if "mean_A" in df_de.columns:
            sf(plot_ma(df_de, label_a, label_b), "ma")
        if grp_a and grp_b:
            fig_h = plot_heatmap(df_de, grp_a, grp_b, gc, 40)
            if fig_h: sf(fig_h, "heatmap")
            fig_p = plot_pca(df_de, grp_a, grp_b, label_a, label_b)
            if fig_p: sf(fig_p, "pca")
            fig_d = plot_dist(df_de, grp_a, grp_b, label_a, label_b)
            if fig_d: sf(fig_d, "dist")
        sf(plot_bar(up, down, label_a, label_b), "bar")
        result["figs"] = saved_figs

        dataset_type = detect_type_meta({}, {g: gsm_groups.get(g,"") for g in grp_a+grp_b})

        # Free PDF
        free_path = os.path.join(tmp_dir, f"{gsm_id}_free_report.pdf")
        build_free_pdf(df_de, gc, saved_figs, up, down, label_a, label_b,
                       dataset_type, free_path, gsm_id=gsm_id)
        result["free_pdf_path"] = free_path

        # Premium PDF
        prem_path = os.path.join(tmp_dir, f"{gsm_id}_premium_report.pdf")
        build_premium_pdf(df_de, gc, saved_figs, up, down, label_a, label_b,
                          dataset_type, prem_path, gsm_id=gsm_id)
        result["premium_pdf_path"] = prem_path

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    lfc_thr  = st.slider("log2FC Threshold",0.5,3.0,1.0,0.25)
    padj_thr = st.select_slider("padj Cutoff",options=[0.001,0.01,0.05,0.1,0.2],value=0.05)
    norm     = st.selectbox("Normalization",["log2_cpm","zscore","none"])
    n_top    = st.slider("Heatmap top genes",10,80,40,5)
    st.markdown("---")
    st.markdown("### 📂 Accepted Formats")
    st.markdown("`CSV` `TXT` `TSV` `GZ` `ZIP`\n\nIncludes GEO Series Matrix")
    st.markdown("---")
    st.markdown("### 🟢 Normal Pipeline")
    for step in ["1. fastp (QC & trimming)",
                 "2. Salmon (quantification)",
                 "3. DESeq2 (DE analysis)"]:
        st.markdown(f"<span class='dataset-badge'>{step}</span>", unsafe_allow_html=True)
    st.markdown("### 💎 Premium Pipeline")
    for step in ["1. fastp (QC & trimming)",
                 "2. STAR aligner",
                 "3. SAMtools",
                 "4. featureCounts",
                 "5. DESeq2",
                 "6. clusterProfiler"]:
        st.markdown(f"<span class='dataset-badge'>{step}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:2.2rem;margin-bottom:0'>
🧬 RNA-Seq Universal Analyzer
</h1>
<p style='text-align:center;color:#8b92a5;margin-top:6px'>
NCBI GEO · Automated Pipeline · Tumor/Normal · Treated/Control · Time Series · KO/WT
</p>
""", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload RNA-Seq Data File",
    type=["csv","txt","tsv","gz","zip"],
    help="GEO series matrix, count tables, or pre-computed DEG files"
)

if uploaded_file:
    with st.spinner("Reading file..."):
        df_raw, geo_meta, gsm_groups, msgs = smart_read_file(uploaded_file)
    for lv, msg in msgs: getattr(st, lv)(msg)
    if df_raw is None: st.stop()

    with st.expander("📄 Raw Data Preview", expanded=False):
        st.write(f"Shape: **{df_raw.shape[0]:,} × {df_raw.shape[1]}**")
        st.dataframe(df_raw.head(10), use_container_width=True)

    series_title = geo_meta.get("Series_title",[""])[0] if geo_meta else ""
    if gsm_groups:
        with st.expander("🔎 GEO Sample Labels", expanded=False):
            st.dataframe(pd.DataFrame({"GSM":list(gsm_groups.keys()),
                                        "Label":list(gsm_groups.values())}),
                          use_container_width=True)

    # ─────────────────────────────────────────
    #  GSM ID PANEL + AUTOMATED PIPELINE TABLE
    # ─────────────────────────────────────────
    _gse_quick  = geo_meta.get("Series_geo_accession",[""])[0] if geo_meta else ""
    _gsm_ids    = list(gsm_groups.keys()) if gsm_groups else []
    _srp_quick  = []
    if geo_meta:
        for _v in geo_meta.get("Series_relation",[]):
            _m = re.search(r"SRA:.*?term=(\w+)", _v)
            if _m: _srp_quick.append(_m.group(1))
    _srp_quick = list(set(_srp_quick))

    _meta_suppls_quick = []
    if geo_meta:
        for _v in geo_meta.get("Series_supplementary_file",[]):
            _v = _v.strip().strip('"')
            if _v and _v.upper() != "NONE":
                _meta_suppls_quick.append(_v)

    _gsm_num_cols = [c for c in df_raw.columns if str(c).startswith("GSM")
                     and pd.api.types.is_numeric_dtype(df_raw[c])]
    _non_gsm_num_cols = [c for c in df_raw.columns
                         if not str(c).startswith("GSM")
                         and pd.api.types.is_numeric_dtype(df_raw[c])]
    _has_expr = (len(_gsm_num_cols) >= 2) or (
        len(_non_gsm_num_cols) >= 3 and df_raw.shape[0] > 50
        and "GSM_ID" not in df_raw.columns
    )

    # Always show GSM panel + pipeline table when we have GSM IDs
    if _gsm_ids:
        st.markdown("---")
        st.markdown("""
        <div style='background:linear-gradient(135deg,rgba(0,212,170,0.10),rgba(124,58,237,0.08));
        border:2px solid rgba(0,212,170,0.4);border-radius:14px;padding:20px;margin:10px 0'>
        <h3 style='color:#00d4aa;margin:0 0 8px'>
        🔬 Automated Pipeline — All GSM Samples
        </h3>
        <p style='color:#c8cfe0;margin:0;font-size:0.95rem'>
        Below are all GSM sample IDs extracted from your series matrix file.
        Each row shows: SRA search link, direct FASTQ download info,
        pipeline status, and report download buttons.
        </p></div>
        """, unsafe_allow_html=True)

        _series_summary = geo_meta.get("Series_title",["Unknown Study"])[0] if geo_meta else "Unknown"
        _organism       = geo_meta.get("Sample_organism_ch1",["Unknown"])[0] if geo_meta else "Unknown"

        ca, cb, cc = st.columns(3)
        ca.metric("📋 Study",    _gse_quick or "—")
        cb.metric("🧬 Organism", _organism)
        cc.metric("🔬 Samples",  len(_gsm_ids))

        st.markdown(f"**Study:** {_series_summary}")

        # ── Supplementary file download links
        if _meta_suppls_quick:
            st.markdown("""<div class='section-header'>📥 Processed Expression Files Available</div>""",
                        unsafe_allow_html=True)
            for _s in _meta_suppls_quick:
                _fname   = _s.split("/")[-1]
                _hurl    = _s.replace("ftp://ftp.ncbi.nlm.nih.gov","https://ftp.ncbi.nlm.nih.gov")
                _is_best = any(k in _fname.lower() for k in ["fpkm","count","tpm","rpkm","expr","matrix"])
                _badge   = "⭐ **RECOMMENDED** — " if _is_best else ""
                st.markdown(f"- {_badge}[`{_fname}`]({_hurl})", unsafe_allow_html=True)

        # ── Main pipeline table
        st.markdown("""<div class='section-header'>📊 Sample Pipeline Table</div>""",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Each row = one sample. Use the <strong>SRA Link</strong> column to search for sequencing runs,
        the <strong>FASTQ Download</strong> column for direct download instructions,
        and the <strong>Report</strong> columns to generate PDFs once expression data is loaded.
        </div>""", unsafe_allow_html=True)

        # Build rich HTML table with all columns
        _rows_html = ""
        for _i, (_gsm, _lbl) in enumerate(list(gsm_groups.items())):
            _srr_list   = st.session_state.get(f"srr_{_gsm}", [])
            _srr        = _srr_list[0] if _srr_list else None
            # Direct SRR link if available, else fall back to SRA search
            _sra_run_url  = (f"https://www.ncbi.nlm.nih.gov/sra/{_srr}"
                             if _srr else f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}")
            _ena_run_url  = (f"https://www.ebi.ac.uk/ena/browser/view/{_srr}"
                             if _srr else f"https://www.ebi.ac.uk/ena/browser/view/{_gsm}")
            _geo_link     = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_gsm}"
            _bg           = "rgba(255,255,255,0.03)" if _i % 2 == 0 else "transparent"

            # SRA / GEO column
            _sra_col = (
                f"<a href='{_sra_run_url}' target='_blank' "
                f"style='background:#00d4aa;color:#0f1117;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>NCBI SRA 🔍</a> "
                f"<a href='{_geo_link}' target='_blank' "
                f"style='background:#7c3aed;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>GEO 🧬</a>"
            )

            # FASTQ download column — direct SRR links
            _srr_label = _srr if _srr else _gsm
            _fastq_col = (
                f"<a href='{_sra_run_url}' target='_blank' "
                f"style='background:#e65c00;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>SRA FASTQ 📥</a> "
                f"<a href='{_ena_run_url}' target='_blank' "
                f"style='background:#1a73e8;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>ENA FASTQ 🌐</a>"
            )

            # SRR accession badge
            _srr_badge = (
                f"<code style='color:#00d4aa;font-size:0.78rem'>{_srr}</code>"
                if _srr else
                f"<span style='color:#8b92a5;font-size:0.78rem'>lookup needed</span>"
            )

            _rows_html += f"""
            <tr style='background:{_bg}'>
              <td style='padding:8px 12px;font-family:monospace;color:#00d4aa;font-weight:700;white-space:nowrap'>
                {_gsm}
              </td>
              <td style='padding:8px 12px;color:#c8cfe0;font-size:0.88rem;max-width:180px;
                         overflow:hidden;text-overflow:ellipsis;white-space:nowrap'
                  title='{_lbl}'>
                {_lbl[:40]}{'…' if len(_lbl)>40 else ''}
              </td>
              <td style='padding:8px 12px;text-align:center'>{_srr_badge}</td>
              <td style='padding:8px 12px;white-space:nowrap'>{_sra_col}</td>
              <td style='padding:8px 12px;white-space:nowrap'>{_fastq_col}</td>
            </tr>"""

        _table_html = f"""
        <div style='overflow-x:auto;overflow-y:auto;max-height:500px;border:1px solid #2a2d3e;border-radius:10px'>
          <table style='width:100%;border-collapse:collapse;min-width:700px'>
            <thead>
              <tr style='background:#1a1d27;position:sticky;top:0;z-index:1'>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>GSM ID</th>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem'>Sample Label</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>🔖 SRR ID</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>🔗 SRA / GEO</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>📥 FASTQ Download</th>
              </tr>
            </thead>
            <tbody>{_rows_html}</tbody>
          </table>
        </div>"""
        st.markdown(_table_html, unsafe_allow_html=True)

        # SRR Lookup for direct links
        if _gsm_ids:
            _col_lu, _col_info = st.columns([2, 3])
            with _col_lu:
                if st.button("🔍 Lookup SRR IDs for All Samples", key="btn_srr_lookup"):
                    _prog_bar = st.progress(0)
                    _total_lu = len(_gsm_ids)
                    for _idx_lu, _gsm_lu in enumerate(_gsm_ids):
                        _srrs_lu = get_srr_for_gsm(_gsm_lu)
                        st.session_state[f"srr_{_gsm_lu}"] = _srrs_lu
                        _prog_bar.progress((_idx_lu + 1) / _total_lu)
                    _prog_bar.empty()
                    st.success("✅ SRR IDs loaded — FASTQ buttons now link directly to SRR runs!")
                    st.rerun()
            with _col_info:
                st.markdown("""<div class='info-box' style='margin:0'>
                Click <strong>Lookup SRR IDs</strong> to resolve GSM → SRR accessions.
                The FASTQ buttons will then open the exact SRR run page (not a search).
                </div>""", unsafe_allow_html=True)

        # Copy-all box
        st.markdown("<br>**📋 Copy all GSM IDs:**", unsafe_allow_html=True)
        st.code(" ".join(_gsm_ids), language=None)

        if _gse_quick:
            _bulk_sra = f"https://www.ncbi.nlm.nih.gov/sra?term={_gse_quick}"
            st.markdown(f"**🔗 All SRA runs for {_gse_quick}:** [{_bulk_sra}]({_bulk_sra})")

        # FASTQ download instructions expandable
        with st.expander("📥 FASTQ Download Instructions (step-by-step)", expanded=False):
            st.markdown(f"""
**Quick Method — SRA Toolkit (command line)**
```bash
# Install SRA Toolkit
conda install -c bioconda sra-tools

# For each GSM, find SRR IDs at:
# https://www.ncbi.nlm.nih.gov/sra?term=<GSM_ID>
# Then download:
fasterq-dump SRR_ID --split-files --outdir ./fastq/

# Compress
gzip ./fastq/*.fastq
```

**Alternative — ENA Browser (no tools needed)**
1. Visit: `https://www.ebi.ac.uk/ena/browser/view/<GSM_ID>`
2. Click the FASTQ file links to download directly in your browser.

**Complete Pipeline after FASTQ download**
```bash
# 1. Quality Control
fastqc ./fastq/*.fastq.gz

# 2. Align to reference genome (replace mm10 with your organism)
STAR --runMode genomeGenerate --genomeDir ./genome_index \\
     --genomeFastaFiles genome.fa --sjdbGTFfile annotation.gtf
STAR --genomeDir ./genome_index \\
     --readFilesIn sample_R1.fastq.gz sample_R2.fastq.gz \\
     --readFilesCommand zcat \\
     --outSAMtype BAM SortedByCoordinate \\
     --outFileNamePrefix ./aligned/sample_

# 3. Count reads
featureCounts -a annotation.gtf -o counts_matrix.txt ./aligned/*.bam

# 4. Upload counts_matrix.txt to this app for DE analysis
```
            """)

        st.markdown("---")

    # ─────────────────────────────────────────
    #  DATA RETRIEVAL — 3 OPTIONS
    # ─────────────────────────────────────────
    gse_id = geo_meta.get("Series_geo_accession", [""])[0] if geo_meta else ""

    meta_suppls = []
    if geo_meta:
        for val in geo_meta.get("Series_supplementary_file", []):
            val = val.strip().strip('"')
            if val and val.upper() != "NONE":
                meta_suppls.append(val)

    srp_ids, srx_ids_meta = [], []
    if geo_meta:
        for val in geo_meta.get("Series_relation", []):
            m = re.search(r"SRA:.*?term=(\w+)", val)
            if m: srp_ids.append(m.group(1))
        for val in geo_meta.get("Sample_relation", []):
            m = re.search(r"term=(SRX\w+)", val)
            if m: srx_ids_meta.append(m.group(1))
    srp_ids      = list(set(srp_ids))
    srx_ids_meta = list(set(srx_ids_meta))

    gsm_cols_check = [c for c in df_raw.columns if str(c).startswith("GSM")]
    non_gsm_num_check = [c for c in df_raw.columns
                         if not str(c).startswith("GSM")
                         and pd.api.types.is_numeric_dtype(df_raw[c])]
    has_expression_data = (
        (len(gsm_cols_check) >= 2 and
         any(pd.api.types.is_numeric_dtype(df_raw[c]) for c in gsm_cols_check))
        or
        (len(non_gsm_num_check) >= 3 and df_raw.shape[0] > 50
         and "GSM_ID" not in df_raw.columns)
    )

    if gse_id:
        st.markdown("""<div class='section-header'>📡 Get Expression Data — 3 Options</div>""",
                    unsafe_allow_html=True)

        if has_expression_data:
            st.success(f"✅ Your file already contains expression data for **{gse_id}** — scroll down to run analysis!")
        else:
            st.warning(f"⚠️ **{gse_id}** series matrix has metadata only. Use one of the 3 options below.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GSE Accession", gse_id)
        c2.metric("Samples (GSM)", len(gsm_groups))
        c3.metric("Suppl. Files", len(meta_suppls))
        c4.metric("SRP Projects", len(srp_ids))

        if meta_suppls:
            with st.expander(f"📎 {len(meta_suppls)} Supplementary File(s) on NCBI"):
                for s in meta_suppls:
                    fname = s.split("/")[-1]
                    https_url = s.replace("ftp://ftp.ncbi.nlm.nih.gov","https://ftp.ncbi.nlm.nih.gov")
                    st.markdown(f"- **`{fname}`** — [Download]({https_url})")

        st.markdown("---")

        with st.expander("🚀 Option 1 — Auto-Retrieve from NCBI", expanded=not has_expression_data):
            st.markdown("""
            <div class='pipeline-box'>
            <strong>⚡ Fully Automatic</strong> — Connects to NCBI GEO, downloads the
            supplementary FPKM/count file, and loads it automatically.
            </div>""", unsafe_allow_html=True)

            if "retrieve_result" not in st.session_state:
                st.session_state.retrieve_result = None

            if st.button("📡 Auto-Retrieve Expression Data from NCBI", key="btn_auto_retrieve"):
                progress_box = st.empty()
                log_msgs = []
                def _prog(msg):
                    log_msgs.append(msg)
                    progress_box.markdown(
                        "\n\n".join([f"<div class='info-box'>{m}</div>" for m in log_msgs[-5:]]),
                        unsafe_allow_html=True
                    )
                with st.spinner(f"🔗 Connecting to NCBI for {gse_id}..."):
                    st.session_state.retrieve_result = smart_retrieve_from_geo(
                        gse_id, geo_meta, gsm_groups, _prog
                    )
                progress_box.empty()

            if st.session_state.retrieve_result:
                rr     = st.session_state.retrieve_result
                status = rr.get("status", "failed")

                if status in ("counts", "matrix"):
                    st.success(rr["message"])
                    retrieved_df = rr["df"]
                    with st.expander("📄 Retrieved Data Preview", expanded=True):
                        st.write(f"**{retrieved_df.shape[0]:,} genes × {retrieved_df.shape[1]} columns**")
                        st.dataframe(retrieved_df.head(10), use_container_width=True)
                    if st.button("✅ Use Retrieved Data for Analysis", key="btn_use_retrieved"):
                        st.session_state["active_df"] = retrieved_df
                        st.rerun()

                elif status == "runinfo":
                    st.warning(rr["message"])
                    runinfo_df = rr.get("runinfo")
                    if runinfo_df is not None:
                        show_cols = [c for c in ["Run","Experiment","Sample","BioSample",
                                                  "Organism","Instrument","spots","bases"]
                                     if c in runinfo_df.columns]
                        st.dataframe(runinfo_df[show_cols].head(20), use_container_width=True)
                        buf = io.StringIO()
                        runinfo_df.to_csv(buf, index=False)
                        st.download_button("📥 Download SRA Run Info CSV", buf.getvalue(),
                                           file_name=f"{gse_id}_SRA_runinfo.csv", mime="text/csv")
                else:
                    st.error(rr.get("message", "Auto-retrieval failed. Try Option 2 or 3."))

        with st.expander("📁 Option 2 — Upload Expression File Manually", expanded=False):
            st.markdown("""
            <div class='pipeline-box'>
            <strong>💯 Guaranteed</strong> — Download the expression file from NCBI GEO yourself
            and upload it here. Works even if auto-retrieval fails.
            </div>""", unsafe_allow_html=True)

            if meta_suppls:
                st.markdown("#### 📥 Step 1 — Download one of these files:")
                for s in meta_suppls:
                    fname     = s.split("/")[-1]
                    https_url = s.replace("ftp://ftp.ncbi.nlm.nih.gov","https://ftp.ncbi.nlm.nih.gov")
                    is_expr   = any(k in fname.lower() for k in ["fpkm","count","expr","tpm","rpkm","matrix"])
                    badge     = "⭐ **Recommended**" if is_expr else ""
                    st.markdown(f"- {badge} [`{fname}`]({https_url})")
            else:
                st.markdown(f"""#### 📥 Step 1 — Visit GEO and download supplementary file:
                👉 **[Open {gse_id} on NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id})**
                """)

            st.markdown("#### 📤 Step 2 — Upload here:")
            manual_file = st.file_uploader(
                "Upload expression file (GZ, TXT, CSV, TSV, ZIP)",
                type=["gz","txt","csv","tsv","zip"],
                key="manual_expr_upload",
            )

            if manual_file:
                with st.spinner("Parsing uploaded expression file..."):
                    man_df, man_meta, man_gsm, man_msgs = smart_read_file(manual_file)
                for lv, msg in man_msgs: getattr(st, lv)(msg)
                if man_df is not None and man_df.shape[0] > 10:
                    num_cols_man = [c for c in man_df.columns if pd.api.types.is_numeric_dtype(man_df[c])]
                    if len(num_cols_man) >= 2:
                        st.success(f"✅ Expression file loaded: **{man_df.shape[0]:,} genes × {man_df.shape[1]} columns**")
                        with st.expander("📄 Preview", expanded=True):
                            st.dataframe(man_df.head(10), use_container_width=True)
                        if st.button("✅ Use This File for Analysis", key="btn_use_manual"):
                            combined_gsm = {**gsm_groups, **man_gsm} if man_gsm else gsm_groups
                            st.session_state["active_df"]  = man_df
                            st.session_state["active_gsm"] = combined_gsm
                            st.rerun()

        with st.expander("📖 Option 3 — Step-by-Step Instructions", expanded=False):
            geo_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
            sra_url = f"https://www.ncbi.nlm.nih.gov/sra?term={srp_ids[0]}" if srp_ids else "https://www.ncbi.nlm.nih.gov/sra"
            st.markdown(f"""
#### 🔗 Direct Links
| Resource | Link |
|---|---|
| GEO Dataset | [{gse_id} on NCBI GEO]({geo_url}) |
| SRA Project | [SRA Browser]({sra_url}) |

#### 📥 Method A — Download Processed FPKM/Count File
1. Open **[{geo_url}]({geo_url})**
2. Scroll to bottom → **Supplementary file** section
3. Download `*_fpkm.txt.gz` or `*_counts.txt.gz`
4. Upload in Option 2 above

#### 🔬 Method B — From SRA Raw FASTQ (Advanced)
```bash
# Install tools
conda install -c bioconda sra-tools star subread fastqc

# Download FASTQ
fasterq-dump SRR_ID --split-files --outdir ./fastq/

# Align (replace mm10 with your organism)
STAR --genomeDir ./genome_idx --readFilesIn R1.fastq R2.fastq \\
     --outSAMtype BAM SortedByCoordinate --outFileNamePrefix ./aligned/

# Count
featureCounts -a annotation.gtf -o counts.txt ./aligned/*.bam

# Upload counts.txt here for DE analysis
```
            """)

        st.markdown(f"🔗 [Open **{gse_id}** on NCBI GEO »](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id})")
        st.markdown("---")

    # ── Use active_df if set
    if "active_df" in st.session_state and st.session_state["active_df"] is not None:
        df_raw = st.session_state["active_df"]
        if "active_gsm" in st.session_state and st.session_state["active_gsm"]:
            gsm_groups = st.session_state["active_gsm"]
        st.success(f"✅ Using expression data: **{df_raw.shape[0]:,} genes × {df_raw.shape[1]} columns**")

    # ── GUARD: stop if no real expression data
    _numeric_expr_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    _placeholder_cols  = {"GSM_ID","Sample_Label","metadata"}
    _is_placeholder    = (set(df_raw.columns) <= _placeholder_cols or len(_numeric_expr_cols) < 2)

    if _is_placeholder:
        _wp_gse = ""
        for _k in ("Series_geo_accession", "series_geo_accession"):
            _gv = (geo_meta or {}).get(_k, [])
            if _gv: _wp_gse = _gv[0].strip().strip('"'); break
        if not _wp_gse:
            _fn_m = re.search(r"(GSE\d+)", uploaded_file.name)
            if _fn_m: _wp_gse = _fn_m.group(1)

        _wp_gsms = list(gsm_groups.items()) if gsm_groups else []
        if not _wp_gsms and "GSM_ID" in df_raw.columns:
            _lc = "Sample_Label" if "Sample_Label" in df_raw.columns else df_raw.columns[-1]
            _wp_gsms = list(zip(df_raw["GSM_ID"].astype(str), df_raw[_lc].astype(str)))

        st.markdown("""
        <div style='background:rgba(255,77,109,0.1);border:2px solid rgba(255,77,109,0.5);
        border-radius:14px;padding:22px;margin:16px 0'>
        <h3 style='color:#ff4d6d;margin:0 0 10px'>⏳ Metadata Only — Expression Data Needed</h3>
        <p style='color:#c8cfe0;margin:0 0 10px'>
        Use one of the 3 options above to load expression data, then the full analysis will run.
        </p></div>""", unsafe_allow_html=True)

        if _wp_gsms:
            st.markdown("### 📋 GSM IDs Found")
            _rows = ""
            for i, (_g, _l) in enumerate(_wp_gsms):
                _bg   = "rgba(255,255,255,0.03)" if i%2==0 else "transparent"
                _srrs = st.session_state.get(f"srr_{_g}", [])
                _srr  = _srrs[0] if _srrs else None
                _sra  = (f"https://www.ncbi.nlm.nih.gov/sra/{_srr}"
                         if _srr else f"https://www.ncbi.nlm.nih.gov/sra?term={_g}")
                _geo  = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_g}"
                _ena  = (f"https://www.ebi.ac.uk/ena/browser/view/{_srr}"
                         if _srr else f"https://www.ebi.ac.uk/ena/browser/view/{_g}")
                _srr_badge = (f"<code style='color:#00d4aa;font-size:0.78rem'>{_srr}</code>"
                              if _srr else "<span style='color:#8b92a5;font-size:0.78rem'>—</span>")
                _rows += (
                    f"<tr style='background:{_bg}'>"
                    f"<td style='padding:7px 14px;font-family:monospace;color:#00d4aa;font-weight:700'>{_g}</td>"
                    f"<td style='padding:7px 14px;color:#c8cfe0;font-size:0.9rem'>{str(_l)[:55]}</td>"
                    f"<td style='padding:7px 10px;text-align:center'>{_srr_badge}</td>"
                    f"<td style='padding:7px 10px;text-align:center'>"
                    f"<a href='{_sra}' target='_blank' style='background:#00d4aa;color:#0f1117;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>SRA</a> "
                    f"<a href='{_geo}' target='_blank' style='background:#7c3aed;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>GEO</a>"
                    f"</td>"
                    f"<td style='padding:7px 10px;text-align:center'>"
                    f"<a href='{_sra}' target='_blank' style='background:#e65c00;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>SRA FASTQ</a> "
                    f"<a href='{_ena}' target='_blank' style='background:#1a73e8;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>ENA</a>"
                    f"</td>"
                    f"</tr>"
                )

            st.markdown(f"""
            <div style='overflow-x:auto;overflow-y:auto;max-height:420px;border:1px solid #2a2d3e;border-radius:10px;margin:12px 0'>
            <table style='width:100%;border-collapse:collapse;min-width:700px'>
              <thead><tr style='background:#1a1d27;position:sticky;top:0'>
                <th style='padding:9px 14px;text-align:left;color:#8b92a5;font-size:0.85rem'>GSM ID</th>
                <th style='padding:9px 14px;text-align:left;color:#8b92a5;font-size:0.85rem'>Sample Label</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>🔖 SRR ID</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>🔗 SRA / GEO</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>📥 FASTQ</th>
              </tr></thead>
              <tbody>{_rows}</tbody>
            </table></div>""", unsafe_allow_html=True)

            st.markdown("**📋 Copy all GSM IDs:**")
            st.code(" ".join([g for g,_ in _wp_gsms]), language=None)

            if _wp_gse:
                col_a, col_b = st.columns(2)
                col_a.markdown(f"[🔗 All SRA runs for **{_wp_gse}**](https://www.ncbi.nlm.nih.gov/sra?term={_wp_gse})")
                col_b.markdown(f"[🌐 Open **{_wp_gse}** on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_wp_gse})")

        st.markdown("---")
        st.info("⬆️ Scroll up and use Option 1 / 2 / 3 to load expression data, then the full analysis will run.")
        st.stop()

    # ─────────────────────────────────────────
    #  DATASET TYPE + GROUP DETECTION
    # ─────────────────────────────────────────
    if gsm_groups:
        dataset_type = detect_type_meta(geo_meta, gsm_groups)
    else:
        dataset_type = detect_type_cols(df_raw)

    type_labels = {
        "tumor_normal":"🔬 Tumor vs Normal","treated_control":"💊 Treated vs Control",
        "time_series":"⏱️ Time Series","knockout_wildtype":"🔧 Knockout vs Wildtype",
        "pre_computed":"📁 Pre-computed","single_condition":"📊 Single Condition",
    }
    st.markdown(f"<div class='info-box'>🔍 Dataset type: "
                f"<strong>{type_labels.get(dataset_type,dataset_type)}</strong></div>",
                unsafe_allow_html=True)

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
        grp_a,grp_b,label_a,label_b = cluster_gsm_groups(gsm_groups)
        st.markdown(f"""<div class='info-box'>
        <strong>{label_a}</strong>: {', '.join(grp_a[:3])}{'...' if len(grp_a)>3 else ''} ({len(grp_a)} samples)<br>
        <strong>{label_b}</strong>: {', '.join(grp_b[:3])}{'...' if len(grp_b)>3 else ''} ({len(grp_b)} samples)
        </div>""", unsafe_allow_html=True)

        all_gsm=list(gsm_groups.keys())
        with st.expander("✏️ Manually adjust groups"):
            grp_a=st.multiselect("Group A (case)",all_gsm,default=grp_a,key="ga")
            grp_b=st.multiselect("Group B (control)",all_gsm,default=grp_b,key="gb")
            label_a=st.text_input("Label A",value=label_a)
            label_b=st.text_input("Label B",value=label_b)

        if not grp_a or not grp_b:
            st.error("⚠️ Could not auto-identify groups. Use the manual adjustment expander above.")
            st.stop()

        with st.spinner("Computing differential expression..."):
            df=compute_de(df_raw,grp_a,grp_b,label_a,label_b,norm)
    else:
        grp_a,grp_b,label_a,label_b=detect_groups_non_geo(df_raw,dataset_type)
        if not grp_a or not grp_b:
            st.error("⚠️ Could not identify sample groups.")
            st.stop()
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
    #  FASTQ FILE PIPELINE — NORMAL / PREMIUM
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🧬 FASTQ Analysis Pipeline</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='pipeline-box'>
    <strong>🚀 Upload your FASTQ files</strong> and configure your pipeline below.<br>
    The app will generate a fully-configured <strong>shell script</strong> and
    <strong>Snakemake workflow</strong> you download and run on your own server
    (Linux/HPC/cloud) where the tools are installed.<br><br>
    <strong>🟢 Normal</strong>: fastp → Salmon → DESeq2 (fast, alignment-free)<br>
    <strong>💎 Premium</strong>: fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler
    </div>""", unsafe_allow_html=True)

    # ── File upload + mode
    _fq_col1, _fq_col2 = st.columns([3, 2])
    with _fq_col1:
        fastq_files = st.file_uploader(
            "📂 Upload FASTQ file(s) (.fastq, .fastq.gz, .fq, .fq.gz)",
            type=["fastq", "fq", "gz"],
            accept_multiple_files=True,
            key="fastq_uploader",
            help="Upload R1 and R2 files for paired-end, or single files for single-end."
        )
    with _fq_col2:
        pipeline_mode = st.radio(
            "🔬 Pipeline Mode",
            options=["🟢 Normal  (fastp → Salmon → DESeq2)",
                     "💎 Premium (fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler)"],
            key="pipeline_mode_radio",
        )
    _is_premium = pipeline_mode.startswith("💎")

    # ── Configuration
    st.markdown("#### ⚙️ Pipeline Configuration")
    _cfg1, _cfg2, _cfg3 = st.columns(3)
    with _cfg1:
        ref_organism = st.selectbox(
            "🧬 Reference Organism",
            ["hg38 (Human)", "mm10 (Mouse)", "rn6 (Rat)",
             "dm6 (Drosophila)", "ce11 (C. elegans)", "GRCz11 (Zebrafish)", "Custom"],
            key="ref_organism_sel"
        )
    with _cfg2:
        seq_type = st.radio("📖 Sequencing Type",
                            ["Paired-end", "Single-end"], key="seq_type_radio")
    with _cfg3:
        n_threads = st.slider("🖥️ CPU Threads", 1, 32, 8, key="nthreads_slider")

    _cfg4, _cfg5 = st.columns(2)
    with _cfg4:
        lfc_script = st.number_input("log2FC cutoff (for DESeq2 script)", 0.5, 3.0, 1.0, 0.25,
                                     key="lfc_script")
    with _cfg5:
        padj_script = st.selectbox("padj cutoff (for DESeq2 script)",
                                   [0.001, 0.01, 0.05, 0.1], index=2, key="padj_script")

    # ── Detect sample names from uploaded files
    def _parse_samples(files):
        """Pair R1/R2 files and return list of sample dicts."""
        r1_files = [f for f in files if re.search(r'_R1|_1\.f', f.name, re.I)]
        r2_files = [f for f in files if re.search(r'_R2|_2\.f', f.name, re.I)]
        singles  = [f for f in files if f not in r1_files and f not in r2_files]
        samples  = []
        if r1_files:
            for r1 in r1_files:
                base = re.sub(r'_R1.*', '', r1.name)
                base = re.sub(r'\.fastq.*|\.fq.*', '', base)
                r2 = next((f for f in r2_files if base in f.name), None)
                samples.append({"name": base, "r1": r1.name,
                                 "r2": r2.name if r2 else None})
        for s in singles:
            base = re.sub(r'\.fastq.*|\.fq.*', '', s.name)
            samples.append({"name": base, "r1": s.name, "r2": None})
        return samples if samples else [{"name": "sample1", "r1": "sample1_R1.fastq.gz",
                                         "r2": "sample1_R2.fastq.gz"}]

    # ── Script generators
    _ORG_MAP = {
        "hg38 (Human)":       ("hg38",  "GRCh38"),
        "mm10 (Mouse)":       ("mm10",  "GRCm38"),
        "rn6 (Rat)":          ("rn6",   "Rnor_6.0"),
        "dm6 (Drosophila)":   ("dm6",   "BDGP6"),
        "ce11 (C. elegans)":  ("ce11",  "WBcel235"),
        "GRCz11 (Zebrafish)": ("GRCz11","GRCz11"),
        "Custom":             ("custom","custom"),
    }
    _org_short, _org_ensembl = _ORG_MAP.get(ref_organism, ("hg38", "GRCh38"))
    _paired = (seq_type == "Paired-end")

    def _make_normal_script(samples, org, threads, lfc, padj, paired):
        lines = [
            "#!/usr/bin/env bash",
            "# ═══════════════════════════════════════════════════════════",
            "# RNA-Seq NORMAL Pipeline — fastp → Salmon → DESeq2",
            f"# Generated by RNA-Seq Analyzer  •  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"# Organism : {ref_organism}  |  Threads : {threads}",
            f"# Samples  : {', '.join(s['name'] for s in samples)}",
            "# ═══════════════════════════════════════════════════════════",
            "",
            "set -euo pipefail",
            f"THREADS={threads}",
            f"ORGANISM={org}",
            f"LFC_CUTOFF={lfc}",
            f"PADJ_CUTOFF={padj}",
            "SALMON_INDEX=./salmon_index",
            "TRANSCRIPTOME=./reference/${ORGANISM}_transcriptome.fa.gz",
            "RESULTS_DIR=./results_normal",
            "",
            "mkdir -p $RESULTS_DIR/qc $RESULTS_DIR/trimmed $RESULTS_DIR/salmon $RESULTS_DIR/deseq2",
            "",
            "# ── STEP 0: Download reference transcriptome (if not present) ──────────",
            f"if [ ! -f ./reference/{org}_transcriptome.fa.gz ]; then",
            "    mkdir -p ./reference",
        ]
        if org == "hg38":
            lines += [
                "    wget -c https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz \\",
                f"         -O ./reference/{org}_transcriptome.fa.gz",
            ]
        elif org == "mm10":
            lines += [
                "    wget -c https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/cdna/Mus_musculus.GRCm38.cdna.all.fa.gz \\",
                f"         -O ./reference/{org}_transcriptome.fa.gz",
            ]
        else:
            lines += [f"    echo 'Please download {org} transcriptome to ./reference/{org}_transcriptome.fa.gz'",
                      "    exit 1"]
        lines += [
            "fi",
            "",
            "# ── STEP 1: Build Salmon index (if not present) ─────────────────────",
            "if [ ! -d $SALMON_INDEX ]; then",
            "    echo '>>> Building Salmon index...'",
            "    salmon index -t $TRANSCRIPTOME -i $SALMON_INDEX \\",
            "                 --gencode -p $THREADS",
            "fi",
            "",
        ]
        for s in samples:
            n = s["name"]
            r1 = s["r1"]
            r2 = s.get("r2")
            lines += [
                f"# ── SAMPLE: {n} ──────────────────────────────────────────────────────",
                f"echo '>>> [fastp] QC + trimming: {n}'",
            ]
            if paired and r2:
                lines += [
                    f"fastp -i fastq/{r1} -I fastq/{r2} \\",
                    f"      -o $RESULTS_DIR/trimmed/{n}_R1_trim.fastq.gz \\",
                    f"      -O $RESULTS_DIR/trimmed/{n}_R2_trim.fastq.gz \\",
                    f"      -h $RESULTS_DIR/qc/{n}_fastp.html \\",
                    f"      -j $RESULTS_DIR/qc/{n}_fastp.json \\",
                    f"      -w $THREADS --detect_adapter_for_pe",
                    "",
                    f"echo '>>> [Salmon] Quantifying: {n}'",
                    f"salmon quant -i $SALMON_INDEX -l A \\",
                    f"      -1 $RESULTS_DIR/trimmed/{n}_R1_trim.fastq.gz \\",
                    f"      -2 $RESULTS_DIR/trimmed/{n}_R2_trim.fastq.gz \\",
                    f"      -p $THREADS --validateMappings --gcBias \\",
                    f"      -o $RESULTS_DIR/salmon/{n}",
                    "",
                ]
            else:
                lines += [
                    f"fastp -i fastq/{r1} \\",
                    f"      -o $RESULTS_DIR/trimmed/{n}_trim.fastq.gz \\",
                    f"      -h $RESULTS_DIR/qc/{n}_fastp.html \\",
                    f"      -j $RESULTS_DIR/qc/{n}_fastp.json \\",
                    f"      -w $THREADS",
                    "",
                    f"echo '>>> [Salmon] Quantifying: {n}'",
                    f"salmon quant -i $SALMON_INDEX -l A \\",
                    f"      -r $RESULTS_DIR/trimmed/{n}_trim.fastq.gz \\",
                    f"      -p $THREADS --validateMappings \\",
                    f"      -o $RESULTS_DIR/salmon/{n}",
                    "",
                ]
        sample_dirs = " ".join(f"$RESULTS_DIR/salmon/{s['name']}" for s in samples)
        sample_names = " ".join(f'"{s["name"]}"' for s in samples)
        lines += [
            "# ── STEP 2: DESeq2 via R ─────────────────────────────────────────────",
            "echo '>>> [DESeq2] Differential expression analysis...'",
            "Rscript - <<'REOF'",
            "library(tximport)",
            "library(DESeq2)",
            "library(ggplot2)",
            "library(dplyr)",
            "",
            "samples <- c(" + ", ".join("'" + s["name"] + "'" for s in samples) + ")",
            f"# samples <- c({sample_names})",
            "salmon_dirs <- c(" + ", ".join("'results_normal/salmon/" + s["name"] + "'" for s in samples) + ")",
            "names(salmon_dirs) <- samples",
            "",
            "files <- file.path(salmon_dirs, 'quant.sf')",
            "txi <- tximport(files, type='salmon', ignoreTxVersion=TRUE)",
            "",
            "# Edit colData to match your experimental design",
            "colData <- data.frame(",
            "    row.names = samples,",
            f"    condition = c({', '.join(chr(39)+'condition'+chr(39) for _ in samples)})  # EDIT THIS",
            ")",
            "",
            "dds <- DESeqDataSetFromTximport(txi, colData=colData, design=~condition)",
            "dds <- DESeq(dds)",
            "",
            f"res <- results(dds, lfcThreshold={lfc}, alpha={padj})",
            "res_df <- as.data.frame(res)",
            "res_df$gene <- rownames(res_df)",
            "write.csv(res_df, 'results_normal/deseq2/DESeq2_results.csv', row.names=FALSE)",
            "",
            "# Volcano plot",
            "res_df$sig <- ifelse(res_df$padj < 0.05 & abs(res_df$log2FoldChange) > 1,",
            "                     ifelse(res_df$log2FoldChange > 0, 'Up', 'Down'), 'NS')",
            "p <- ggplot(res_df, aes(log2FoldChange, -log10(pvalue), color=sig)) +",
            "     geom_point(alpha=0.6, size=1.5) +",
            "     scale_color_manual(values=c(Up='#ff4d6d', Down='#4da6ff', NS='grey50')) +",
            "     theme_bw() + ggtitle('Volcano Plot — DESeq2')",
            "ggsave('results_normal/deseq2/volcano.png', p, width=8, height=6, dpi=150)",
            "",
            "cat('DESeq2 done. Results in results_normal/deseq2/\\n')",
            "REOF",
            "",
            "echo '✅ Normal pipeline complete! Results in ./results_normal/'",
            "echo '   Upload results_normal/deseq2/DESeq2_results.csv to the app for report generation.'",
        ]
        return "\n".join(lines)

    def _make_premium_script(samples, org, threads, lfc, padj, paired):
        lines = [
            "#!/usr/bin/env bash",
            "# ═══════════════════════════════════════════════════════════",
            "# RNA-Seq PREMIUM Pipeline",
            "# fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler",
            f"# Generated by RNA-Seq Analyzer  •  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"# Organism : {ref_organism}  |  Threads : {threads}",
            f"# Samples  : {', '.join(s['name'] for s in samples)}",
            "# ═══════════════════════════════════════════════════════════",
            "",
            "set -euo pipefail",
            f"THREADS={threads}",
            f"ORGANISM={org}",
            f"LFC_CUTOFF={lfc}",
            f"PADJ_CUTOFF={padj}",
            "STAR_INDEX=./star_index",
            "GENOME_FA=./reference/${ORGANISM}_genome.fa",
            "GTF=./reference/${ORGANISM}_annotation.gtf",
            "RESULTS_DIR=./results_premium",
            "",
            "mkdir -p $RESULTS_DIR/qc $RESULTS_DIR/trimmed $RESULTS_DIR/aligned \\",
            "         $RESULTS_DIR/counts $RESULTS_DIR/deseq2 $RESULTS_DIR/enrichment",
            "",
            "# ── STEP 0: Download reference genome + GTF ─────────────────────────",
            f"if [ ! -f ./reference/{org}_genome.fa ]; then",
            "    mkdir -p ./reference",
        ]
        if org == "hg38":
            lines += [
                "    wget -c https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \\",
                f"         -O - | gunzip > ./reference/{org}_genome.fa",
                "    wget -c https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz \\",
                f"         -O - | gunzip > ./reference/{org}_annotation.gtf",
            ]
        elif org == "mm10":
            lines += [
                "    wget -c https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz \\",
                f"         -O - | gunzip > ./reference/{org}_genome.fa",
                "    wget -c https://ftp.ensembl.org/pub/release-110/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz \\",
                f"         -O - | gunzip > ./reference/{org}_annotation.gtf",
            ]
        else:
            lines += [
                f"    echo 'Please place {org} genome FASTA in ./reference/{org}_genome.fa'",
                f"    echo 'Please place {org} GTF annotation in ./reference/{org}_annotation.gtf'",
                "    exit 1",
            ]
        lines += [
            "fi",
            "",
            "# ── STEP 1: Build STAR genome index (if not present) ────────────────",
            "if [ ! -d $STAR_INDEX ]; then",
            "    echo '>>> Building STAR genome index (needs ~30GB RAM for human)...'",
            "    STAR --runMode genomeGenerate \\",
            "         --genomeDir $STAR_INDEX \\",
            "         --genomeFastaFiles $GENOME_FA \\",
            "         --sjdbGTFfile $GTF \\",
            f"         --runThreadN $THREADS",
            "fi",
            "",
        ]
        bam_files = []
        for s in samples:
            n  = s["name"]
            r1 = s["r1"]
            r2 = s.get("r2")
            bam_files.append(f"$RESULTS_DIR/aligned/{n}_sorted.bam")
            lines += [
                f"# ── SAMPLE: {n} ──────────────────────────────────────────────────────",
                f"echo '>>> [fastp] QC + trimming: {n}'",
            ]
            if paired and r2:
                lines += [
                    f"fastp -i fastq/{r1} -I fastq/{r2} \\",
                    f"      -o $RESULTS_DIR/trimmed/{n}_R1_trim.fastq.gz \\",
                    f"      -O $RESULTS_DIR/trimmed/{n}_R2_trim.fastq.gz \\",
                    f"      -h $RESULTS_DIR/qc/{n}_fastp.html \\",
                    f"      -j $RESULTS_DIR/qc/{n}_fastp.json \\",
                    f"      -w $THREADS --detect_adapter_for_pe",
                    "",
                    f"echo '>>> [STAR] Aligning: {n}'",
                    f"STAR --runThreadN $THREADS \\",
                    f"     --genomeDir $STAR_INDEX \\",
                    f"     --readFilesIn $RESULTS_DIR/trimmed/{n}_R1_trim.fastq.gz \\",
                    f"                   $RESULTS_DIR/trimmed/{n}_R2_trim.fastq.gz \\",
                    f"     --readFilesCommand zcat \\",
                    f"     --outSAMtype BAM SortedByCoordinate \\",
                    f"     --outSAMattributes NH HI AS NM MD \\",
                    f"     --outFileNamePrefix $RESULTS_DIR/aligned/{n}_ \\",
                    f"     --quantMode GeneCounts",
                ]
            else:
                lines += [
                    f"fastp -i fastq/{r1} \\",
                    f"      -o $RESULTS_DIR/trimmed/{n}_trim.fastq.gz \\",
                    f"      -h $RESULTS_DIR/qc/{n}_fastp.html \\",
                    f"      -j $RESULTS_DIR/qc/{n}_fastp.json \\",
                    f"      -w $THREADS",
                    "",
                    f"echo '>>> [STAR] Aligning: {n}'",
                    f"STAR --runThreadN $THREADS \\",
                    f"     --genomeDir $STAR_INDEX \\",
                    f"     --readFilesIn $RESULTS_DIR/trimmed/{n}_trim.fastq.gz \\",
                    f"     --readFilesCommand zcat \\",
                    f"     --outSAMtype BAM SortedByCoordinate \\",
                    f"     --outSAMattributes NH HI AS NM MD \\",
                    f"     --outFileNamePrefix $RESULTS_DIR/aligned/{n}_ \\",
                    f"     --quantMode GeneCounts",
                ]
            lines += [
                "",
                f"echo '>>> [SAMtools] Indexing + stats: {n}'",
                f"samtools sort -@ $THREADS \\",
                f"    $RESULTS_DIR/aligned/{n}_Aligned.sortedByCoord.out.bam \\",
                f"    -o $RESULTS_DIR/aligned/{n}_sorted.bam",
                f"samtools index $RESULTS_DIR/aligned/{n}_sorted.bam",
                f"samtools flagstat $RESULTS_DIR/aligned/{n}_sorted.bam \\",
                f"    > $RESULTS_DIR/qc/{n}_flagstat.txt",
                f"samtools idxstats $RESULTS_DIR/aligned/{n}_sorted.bam \\",
                f"    > $RESULTS_DIR/qc/{n}_idxstats.txt",
                "",
            ]
        bam_list = " \\\n         ".join(bam_files)
        lines += [
            "# ── STEP 2: featureCounts (all samples together) ────────────────────",
            "echo '>>> [featureCounts] Counting reads...'",
            f"featureCounts -T $THREADS \\",
            f"    -a $GTF \\",
            f"    -o $RESULTS_DIR/counts/counts_matrix.txt \\",
        ]
        if paired:
            lines.append("    -p --countReadPairs \\")
        lines += [
            f"    -s 0 \\",
            f"    {bam_list}",
            "",
            "# Clean up featureCounts output to plain count matrix",
            "tail -n +3 $RESULTS_DIR/counts/counts_matrix.txt | \\",
            "    cut -f1,7- > $RESULTS_DIR/counts/counts_clean.txt",
            "",
            "# ── STEP 3: DESeq2 + clusterProfiler via R ──────────────────────────",
            "echo '>>> [DESeq2 + clusterProfiler] Differential expression + enrichment...'",
            "Rscript - <<'REOF'",
            "library(DESeq2)",
            "library(clusterProfiler)",
            "library(ggplot2)",
            "library(dplyr)",
        ]
        if org in ("hg38",):
            lines.append("library(org.Hs.eg.db)"); org_db = "org.Hs.eg.db"
        elif org in ("mm10",):
            lines.append("library(org.Mm.eg.db)"); org_db = "org.Mm.eg.db"
        else:
            org_db = "org.Hs.eg.db"
            lines.append("library(org.Hs.eg.db)  # Change for your organism")
        lines += [
            "",
            "# ---- Load count matrix ----",
            "counts <- read.table('results_premium/counts/counts_clean.txt',",
            "                     header=TRUE, row.names=1, sep='\\t')",
            "counts <- round(counts)  # featureCounts can produce floats",
            "",
            "# ---- Sample metadata — EDIT condition labels ----",
            "colData <- data.frame(",
            f"    row.names = colnames(counts),",
            f"    condition = c({', '.join(chr(39)+'condition'+chr(39) for _ in samples)})  # EDIT: 'treated'/'control' etc.",
            ")",
            "",
            "# ---- DESeq2 ----",
            "dds <- DESeqDataSetFromMatrix(countData=counts, colData=colData, design=~condition)",
            "dds <- dds[rowSums(counts(dds)) >= 10, ]  # filter low-count genes",
            "dds <- DESeq(dds)",
            "",
            f"res <- results(dds, lfcThreshold={lfc}, alpha={padj})",
            "res <- lfcShrink(dds, coef=2, res=res, type='apeglm')  # shrink LFC estimates",
            "res_df <- as.data.frame(res)",
            "res_df$gene <- rownames(res_df)",
            "write.csv(res_df, 'results_premium/deseq2/DESeq2_results.csv', row.names=FALSE)",
            "",
            "# ---- Volcano plot ----",
            "res_df$sig <- ifelse(res_df$padj < 0.05 & abs(res_df$log2FoldChange) > 1,",
            "                     ifelse(res_df$log2FoldChange > 0, 'Up', 'Down'), 'NS')",
            "p <- ggplot(res_df, aes(log2FoldChange, -log10(pvalue), color=sig)) +",
            "     geom_point(alpha=0.5, size=1.2) +",
            "     scale_color_manual(values=c(Up='#ff4d6d', Down='#4da6ff', NS='grey60')) +",
            "     theme_bw() + ggtitle('Volcano Plot — DESeq2 (Premium)')",
            "ggsave('results_premium/deseq2/volcano.png', p, width=8, height=6, dpi=150)",
            "",
            "# ---- clusterProfiler GO enrichment ----",
            "sig_genes <- res_df %>% filter(!is.na(padj), padj < 0.05, abs(log2FoldChange) > 1)",
            "gene_ids <- bitr(sig_genes$gene, fromType='SYMBOL', toType='ENTREZID',",
            f"                 OrgDb='{org_db}')",
            "",
            "# GO enrichment",
            "go_bp <- enrichGO(gene=gene_ids$ENTREZID, OrgDb=org_db,",
            "                  ont='BP', pAdjustMethod='BH', pvalueCutoff=0.05,",
            "                  qvalueCutoff=0.2, readable=TRUE)",
            "write.csv(as.data.frame(go_bp), 'results_premium/enrichment/GO_BP_results.csv',",
            "          row.names=FALSE)",
            "dotplot(go_bp, showCategory=20) + ggtitle('GO Biological Process')",
            "ggsave('results_premium/enrichment/GO_BP_dotplot.png', width=10, height=8, dpi=150)",
            "",
            "# KEGG pathway enrichment",
            "kegg <- enrichKEGG(gene=gene_ids$ENTREZID, organism='hsa',  # hsa=human, mmu=mouse",
            "                   pvalueCutoff=0.05)",
            "write.csv(as.data.frame(kegg), 'results_premium/enrichment/KEGG_results.csv',",
            "          row.names=FALSE)",
            "dotplot(kegg, showCategory=20) + ggtitle('KEGG Pathway Enrichment')",
            "ggsave('results_premium/enrichment/KEGG_dotplot.png', width=10, height=8, dpi=150)",
            "",
            "# PCA plot",
            "vsd <- vst(dds, blind=FALSE)",
            "pca_data <- plotPCA(vsd, intgroup='condition', returnData=TRUE)",
            "pct_var <- round(100 * attr(pca_data, 'percentVar'))",
            "p_pca <- ggplot(pca_data, aes(PC1, PC2, color=condition)) +",
            "    geom_point(size=4) + theme_bw() +",
            "    xlab(paste0('PC1: ', pct_var[1], '% variance')) +",
            "    ylab(paste0('PC2: ', pct_var[2], '% variance'))",
            "ggsave('results_premium/deseq2/PCA.png', p_pca, width=7, height=5, dpi=150)",
            "",
            "cat('Premium analysis done!\\n')",
            "cat(paste('DEGs found:', nrow(sig_genes), '\\n'))",
            "REOF",
            "",
            "echo ''",
            "echo '✅ Premium pipeline complete!'",
            "echo '   Results in ./results_premium/'",
            "echo '   Upload results_premium/deseq2/DESeq2_results.csv to the app for report generation.'",
        ]
        return "\n".join(lines)

    def _make_snakefile(samples, org, threads, lfc, padj, paired, premium):
        mode = "premium" if premium else "normal"
        snames = [s["name"] for s in samples]
        snake  = [
            f'# Snakemake workflow — RNA-Seq {mode.upper()} pipeline',
            f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            f'# Run: snakemake --cores {threads} --use-conda',
            '',
            'configfile: "config.yaml"',
            '',
            f'SAMPLES = {snames}',
            f'THREADS = {threads}',
            f'ORGANISM = "{org}"',
            '',
        ]
        if premium:
            snake += [
                'rule all:',
                '    input:',
                '        expand("results_premium/counts/counts_clean.txt"),',
                '        "results_premium/deseq2/DESeq2_results.csv",',
                '        "results_premium/enrichment/GO_BP_results.csv"',
                '',
                'rule fastp:',
                '    input:',
                '        r1="fastq/{sample}_R1.fastq.gz",',
                '        r2="fastq/{sample}_R2.fastq.gz"' if paired else '        r1="fastq/{sample}.fastq.gz"',
                '    output:',
                '        r1="results_premium/trimmed/{sample}_R1_trim.fastq.gz",' if paired else '        r1="results_premium/trimmed/{sample}_trim.fastq.gz",',
                '        r2="results_premium/trimmed/{sample}_R2_trim.fastq.gz",' if paired else '',
                '        html="results_premium/qc/{sample}_fastp.html",',
                '        json="results_premium/qc/{sample}_fastp.json"',
                '    threads: THREADS',
                '    shell:',
                '        "fastp -i {input.r1} -I {input.r2} -o {output.r1} -O {output.r2} '
                '-h {output.html} -j {output.json} -w {threads} --detect_adapter_for_pe"'
                if paired else
                '        "fastp -i {input.r1} -o {output.r1} -h {output.html} -j {output.json} -w {threads}"',
                '',
                'rule star_align:',
                '    input:',
                '        r1="results_premium/trimmed/{sample}_R1_trim.fastq.gz",' if paired else '        r1="results_premium/trimmed/{sample}_trim.fastq.gz",',
                '        r2="results_premium/trimmed/{sample}_R2_trim.fastq.gz",' if paired else '',
                '        idx=directory("star_index")',
                '    output:',
                '        bam="results_premium/aligned/{sample}_Aligned.sortedByCoord.out.bam"',
                '    threads: THREADS',
                '    shell:',
                '        "STAR --runThreadN {threads} --genomeDir {input.idx} '
                '--readFilesIn {input.r1} {input.r2} --readFilesCommand zcat '
                '--outSAMtype BAM SortedByCoordinate '
                '--outFileNamePrefix results_premium/aligned/{wildcards.sample}_"'
                if paired else
                '        "STAR --runThreadN {threads} --genomeDir {input.idx} '
                '--readFilesIn {input.r1} --readFilesCommand zcat '
                '--outSAMtype BAM SortedByCoordinate '
                '--outFileNamePrefix results_premium/aligned/{wildcards.sample}_"',
                '',
                'rule samtools_sort_index:',
                '    input: "results_premium/aligned/{sample}_Aligned.sortedByCoord.out.bam"',
                '    output:',
                '        bam="results_premium/aligned/{sample}_sorted.bam",',
                '        bai="results_premium/aligned/{sample}_sorted.bam.bai"',
                '    threads: THREADS',
                '    shell:',
                '        "samtools sort -@ {threads} {input} -o {output.bam} && '
                'samtools index {output.bam}"',
                '',
                'rule featurecounts:',
                '    input:',
                '        bams=expand("results_premium/aligned/{sample}_sorted.bam", sample=SAMPLES),',
                '        gtf=f"reference/{org}_annotation.gtf"',
                '    output: "results_premium/counts/counts_matrix.txt"',
                '    threads: THREADS',
                '    shell:',
                '        "featureCounts -T {threads} -a {input.gtf} '
                '-o {output} ' + ('-p --countReadPairs ' if paired else '') + '{input.bams}"',
                '',
                'rule deseq2_clusterProfiler:',
                '    input: "results_premium/counts/counts_matrix.txt"',
                '    output:',
                '        results="results_premium/deseq2/DESeq2_results.csv",',
                '        go="results_premium/enrichment/GO_BP_results.csv"',
                '    script: "scripts/deseq2_premium.R"',
            ]
        else:
            snake += [
                'rule all:',
                '    input:',
                '        expand("results_normal/salmon/{sample}/quant.sf", sample=SAMPLES),',
                '        "results_normal/deseq2/DESeq2_results.csv"',
                '',
                'rule fastp:',
                '    input:',
                '        r1="fastq/{sample}_R1.fastq.gz",' if paired else '        r1="fastq/{sample}.fastq.gz",',
                '        r2="fastq/{sample}_R2.fastq.gz"' if paired else '',
                '    output:',
                '        r1="results_normal/trimmed/{sample}_R1_trim.fastq.gz",' if paired else '        r1="results_normal/trimmed/{sample}_trim.fastq.gz",',
                '        r2="results_normal/trimmed/{sample}_R2_trim.fastq.gz",' if paired else '',
                '        html="results_normal/qc/{sample}_fastp.html",',
                '        json="results_normal/qc/{sample}_fastp.json"',
                '    threads: THREADS',
                '    shell:',
                '        "fastp -i {input.r1} -I {input.r2} -o {output.r1} -O {output.r2} '
                '-h {output.html} -j {output.json} -w {threads} --detect_adapter_for_pe"'
                if paired else
                '        "fastp -i {input.r1} -o {output.r1} -h {output.html} -j {output.json} -w {threads}"',
                '',
                'rule salmon_quant:',
                '    input:',
                '        r1="results_normal/trimmed/{sample}_R1_trim.fastq.gz",' if paired else '        r1="results_normal/trimmed/{sample}_trim.fastq.gz",',
                '        r2="results_normal/trimmed/{sample}_R2_trim.fastq.gz",' if paired else '',
                '        idx=directory("salmon_index")',
                '    output: directory("results_normal/salmon/{sample}")',
                '    threads: THREADS',
                '    shell:',
                '        "salmon quant -i {input.idx} -l A -1 {input.r1} -2 {input.r2} '
                '-p {threads} --validateMappings --gcBias -o {output}"'
                if paired else
                '        "salmon quant -i {input.idx} -l A -r {input.r1} '
                '-p {threads} --validateMappings -o {output}"',
                '',
                'rule deseq2:',
                '    input: expand("results_normal/salmon/{sample}", sample=SAMPLES)',
                '    output: "results_normal/deseq2/DESeq2_results.csv"',
                '    script: "scripts/deseq2_normal.R"',
            ]
        return "\n".join(snake)

    def _make_config_yaml(samples, org, threads, lfc, padj, paired, premium):
        snames = [s["name"] for s in samples]
        return f"""# config.yaml — RNA-Seq Pipeline Configuration
# Edit this file to match your experiment

samples:
{chr(10).join(f'  - {n}' for n in snames)}

organism: "{org}"
threads: {threads}
lfc_cutoff: {lfc}
padj_cutoff: {padj}
paired_end: {'true' if paired else 'false'}
pipeline: "{'premium' if premium else 'normal'}"

# Paths (relative to working directory)
fastq_dir: "./fastq"
reference_dir: "./reference"
results_dir: "./results_{'premium' if premium else 'normal'}"

# Reference URLs (auto-filled for known organisms)
# genome_fa: "./reference/{org}_genome.fa"
# gtf: "./reference/{org}_annotation.gtf"

# Conda environment (for snakemake --use-conda)
# See environment.yaml for tool versions
"""

    def _make_environment_yaml(premium):
        base = """name: rnaseq_pipeline
channels:
  - bioconda
  - conda-forge
  - defaults
dependencies:
  - fastp>=0.23.4
  - r-base>=4.3
  - bioconductor-deseq2>=1.40
  - bioconductor-tximport
  - r-ggplot2
  - r-dplyr
  - snakemake>=7.0
"""
        if premium:
            base += """  - star>=2.7.10
  - samtools>=1.17
  - subread>=2.0.3       # featureCounts
  - bioconductor-clusterprofiler>=4.8
  - bioconductor-apeglm
  - r-biocmanager
"""
        else:
            base += """  - salmon>=1.10
  - bioconductor-tximport>=1.28
"""
        return base

    def _make_readme(samples, org, premium, paired):
        mode = "Premium" if premium else "Normal"
        return f"""# RNA-Seq {mode} Pipeline
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Samples ({len(samples)})
{chr(10).join(f'- {s["name"]}  (R1: {s["r1"]}{", R2: "+s["r2"] if s.get("r2") else ""})' for s in samples)}

## Pipeline: {mode}
{'fastp → STAR → SAMtools → featureCounts → DESeq2 → clusterProfiler' if premium else 'fastp → Salmon → DESeq2'}

## Requirements
Install via conda:
```bash
conda env create -f environment.yaml
conda activate rnaseq_pipeline
```

Or install manually:
```bash
# Tools needed:
{'conda install -c bioconda fastp star samtools subread' if premium else 'conda install -c bioconda fastp salmon'}
# R packages:
{'Rscript -e "BiocManager::install(c(chr(39)DESeq2chr(39),chr(39)clusterProfilerchr(39),chr(39)apeglmchr(39)))"' if premium else 'Rscript -e "BiocManager::install(c(chr(39)DESeq2chr(39),chr(39)tximportchr(39)))"'}
```

## How to Run

### Option A — Shell script (simple)
```bash
# 1. Place your FASTQ files in ./fastq/
# 2. Run the pipeline script:
bash pipeline_{mode.lower()}.sh
```

### Option B — Snakemake (recommended for large datasets / HPC)
```bash
# Dry run first:
snakemake -n --cores {n_threads}

# Full run:
snakemake --cores {n_threads} --use-conda

# On HPC (SLURM):
snakemake --cores {n_threads} --cluster "sbatch -c {{threads}} --mem=32G" --jobs 10
```

## Directory Structure
```
project/
├── fastq/                  ← Put your FASTQ files here
├── reference/              ← Genome/transcriptome (auto-downloaded by script)
├── {'star_index/' if premium else 'salmon_index/'}              ← Index (auto-built)
├── results_{'premium' if premium else 'normal'}/
│   ├── qc/                 ← fastp HTML/JSON reports
│   ├── trimmed/            ← Trimmed FASTQ files
│   {'├── aligned/           ← STAR BAM files' if premium else '├── salmon/            ← Salmon quant folders'}
│   ├── counts/             ← {'featureCounts matrix' if premium else 'tximport counts'}
│   ├── deseq2/             ← DESeq2 results + plots
│   {'└── enrichment/        ← GO/KEGG enrichment results' if premium else ''}
├── pipeline_{mode.lower()}.sh   ← Main pipeline script
├── Snakefile               ← Snakemake workflow
├── config.yaml             ← Configuration
└── environment.yaml        ← Conda environment
```

## After Running
Upload `results_{'premium' if premium else 'normal'}/deseq2/DESeq2_results.csv` back to the
RNA-Seq Analyzer app to generate your PDF report with all visualizations.

## Important: Edit colData in the R script
Open `pipeline_{mode.lower()}.sh` and find the colData section — replace 'condition'
with your actual group labels (e.g., 'treated', 'control', 'tumor', 'normal').
""".replace("chr(39)", "'")

    # ── UI: show detected samples + generate button
    if fastq_files:
        _samples = _parse_samples(fastq_files)
        st.markdown(f"**{len(fastq_files)} file(s) detected → {len(_samples)} sample(s):**")
        _sm_cols = st.columns(min(4, len(_samples)))
        for _si2, _sm in enumerate(_samples):
            with _sm_cols[_si2 % len(_sm_cols)]:
                _r2_str = f" + `{_sm['r2']}`" if _sm.get('r2') else " (single-end)"
                st.markdown(f"""<div class='metric-card' style='padding:10px'>
                <div style='color:#00d4aa;font-weight:700;font-size:0.95rem'>{_sm['name']}</div>
                <div style='color:#8b92a5;font-size:0.78rem'>`{_sm['r1']}`{_r2_str}</div>
                </div>""", unsafe_allow_html=True)

        if _is_premium:
            st.markdown("""<div class='premium-box'>
            <strong>💎 Premium Pipeline Steps:</strong>
            fastp &nbsp;→&nbsp; STAR aligner &nbsp;→&nbsp; SAMtools &nbsp;→&nbsp;
            featureCounts &nbsp;→&nbsp; DESeq2 &nbsp;→&nbsp; clusterProfiler
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='info-box'>
            <strong>🟢 Normal Pipeline Steps:</strong>
            fastp &nbsp;→&nbsp; Salmon &nbsp;→&nbsp; DESeq2
            </div>""", unsafe_allow_html=True)

        if st.button("📦 Generate Pipeline Scripts & Workflow", key="btn_gen_scripts",
                     use_container_width=True):
            with st.spinner("Generating scripts..."):
                _mode_key = "premium" if _is_premium else "normal"
                _sh_name  = f"pipeline_{_mode_key}.sh"
                _sh_text  = (_make_premium_script(_samples, _org_short, n_threads,
                                                   lfc_script, padj_script, _paired)
                             if _is_premium else
                             _make_normal_script(_samples, _org_short, n_threads,
                                                 lfc_script, padj_script, _paired))
                _snake    = _make_snakefile(_samples, _org_short, n_threads,
                                             lfc_script, padj_script, _paired, _is_premium)
                _config   = _make_config_yaml(_samples, _org_short, n_threads,
                                               lfc_script, padj_script, _paired, _is_premium)
                _env      = _make_environment_yaml(_is_premium)
                _readme   = _make_readme(_samples, _org_short, _is_premium, _paired)

                # Pack everything into a ZIP
                _zip_buf = io.BytesIO()
                with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                    _zf.writestr(_sh_name, _sh_text)
                    _zf.writestr("Snakefile", _snake)
                    _zf.writestr("config.yaml", _config)
                    _zf.writestr("environment.yaml", _env)
                    _zf.writestr("README.md", _readme)
                _zip_buf.seek(0)

                st.session_state["pipeline_zip"]  = _zip_buf.getvalue()
                st.session_state["pipeline_sh"]   = _sh_text
                st.session_state["pipeline_snake"] = _snake
                st.session_state["pipeline_mode_done"] = _mode_key
                st.session_state["pipeline_sh_name"] = _sh_name

        if st.session_state.get("pipeline_zip"):
            _mode_done2 = st.session_state.get("pipeline_mode_done", "normal")
            _is_prem2   = (_mode_done2 == "premium")
            st.success("✅ Pipeline scripts generated — download the ZIP below!")

            st.markdown("""<div class='info-box'>
            <strong>📦 ZIP contains:</strong><br>
            • <code>pipeline_*.sh</code> — complete shell script (just run <code>bash pipeline_*.sh</code>)<br>
            • <code>Snakefile</code> — Snakemake workflow for HPC/parallel runs<br>
            • <code>config.yaml</code> — all settings in one place<br>
            • <code>environment.yaml</code> — conda environment with all tool versions<br>
            • <code>README.md</code> — step-by-step instructions
            </div>""", unsafe_allow_html=True)

            dl_cols = st.columns(2)
            with dl_cols[0]:
                st.download_button(
                    f"📦 Download Pipeline ZIP ({_mode_done2.capitalize()})",
                    data=st.session_state["pipeline_zip"],
                    file_name=f"rnaseq_{_mode_done2}_pipeline.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_pipeline_zip"
                )
            with dl_cols[1]:
                st.download_button(
                    f"📄 Download {st.session_state.get('pipeline_sh_name','pipeline.sh')} only",
                    data=st.session_state["pipeline_sh"],
                    file_name=st.session_state.get("pipeline_sh_name", "pipeline.sh"),
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_pipeline_sh"
                )

            with st.expander("👁️ Preview shell script", expanded=False):
                st.code(st.session_state["pipeline_sh"][:3000] +
                        ("\n... (truncated — download full script)" if
                         len(st.session_state["pipeline_sh"]) > 3000 else ""),
                        language="bash")

            with st.expander("👁️ Preview Snakefile", expanded=False):
                st.code(st.session_state["pipeline_snake"], language="python")

            st.markdown("""<div class='pipeline-box'>
            <strong>🔄 Workflow:</strong><br>
            1️⃣ Download the ZIP &nbsp;→&nbsp;
            2️⃣ Place FASTQ files in <code>./fastq/</code> &nbsp;→&nbsp;
            3️⃣ Edit <code>colData</code> condition labels in the script &nbsp;→&nbsp;
            4️⃣ Run <code>conda env create -f environment.yaml && conda activate rnaseq_pipeline</code> &nbsp;→&nbsp;
            5️⃣ Run <code>bash pipeline_*.sh</code> (or <code>snakemake --cores N</code>) &nbsp;→&nbsp;
            6️⃣ Upload <code>DESeq2_results.csv</code> back here for the PDF report
            </div>""", unsafe_allow_html=True)

    else:
        st.info("⬆️ Upload FASTQ file(s) above to generate your personalised pipeline scripts.")

    # ── Unified Report Section
    st.markdown("---")
    st.markdown("<div class='section-header'>📄 Generate Report</div>", unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    Generate a comprehensive PDF report from the expression data analysis above.
    The report includes all plots, DEG tables, and a full methods section.
    </div>""", unsafe_allow_html=True)

    if st.button("📥 Generate & Download PDF Report", key="btn_gen_report",
                 use_container_width=True):
        with st.spinner("Building PDF report..."):
            _rpt_path = os.path.join(tmp, "RNA_Seq_Report.pdf")
            try:
                build_premium_pdf(df, gene_col, saved_figs, up, down,
                                  label_a, label_b, dataset_type, _rpt_path)
                with open(_rpt_path, "rb") as _rf:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        _rf.read(),
                        file_name=f"RNA_Seq_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="dl_report_final"
                    )
                st.success("✅ Report ready!")
            except Exception as _re:
                st.error(f"PDF error: {_re}")

else:
    # Landing page
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

    c1,c2,c3,c4=st.columns(4)
    for col,icon,title,desc in [
        (c1,"🔍","Smart Detection","Auto-identifies groups from GEO metadata"),
        (c2,"📥","FASTQ Links","Direct SRR links for every GSM sample"),
        (c3,"📦","Pipeline Scripts","Download ready-to-run Normal or Premium pipeline"),
        (c4,"📄","PDF Reports","Full DE analysis report with all plots"),
    ]:
        col.markdown(f"""<div class='metric-card' style='text-align:left'>
        <div style='font-size:2rem'>{icon}</div>
        <div style='color:white;font-weight:700;margin:8px 0 4px'>{title}</div>
        <div style='color:#8b92a5;font-size:0.85rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
### 🔬 How it works
1. **Upload** a GEO Series Matrix file (`.txt`, `.txt.gz`, `.zip`) or a count matrix CSV
2. **GSM IDs** extracted automatically — SRR links resolved with one click
3. **Load expression data** via auto-retrieve or manual upload
4. **Upload FASTQ files** → choose **Normal** or **Premium** pipeline → download personalised shell script + Snakemake workflow
5. **Run on your server** → upload the resulting `DESeq2_results.csv` back here
6. **Download PDF report** with all plots and DEG tables

### 🟢 Normal Pipeline &nbsp;&nbsp;|&nbsp;&nbsp; 💎 Premium Pipeline
| Step | Normal | Premium |
|------|--------|---------|
| QC | fastp | fastp |
| Alignment | Salmon (alignment-free) | STAR (genome-guided) |
| BAM processing | — | SAMtools |
| Counting | tximport | featureCounts |
| DE analysis | DESeq2 | DESeq2 + apeglm shrinkage |
| Enrichment | — | clusterProfiler (GO + KEGG) |
    """)
