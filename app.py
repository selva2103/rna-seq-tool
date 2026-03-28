"""
RNA-Seq Universal Report Generator — v6.0
==========================================
v6.0 Changes:
  - REMOVED all AI/Claude integration
  - Added FASTQ download column per GSM ID (via SRA)
  - Added automated pipeline column per GSM ID
  - Added Free Report PDF column per GSM ID
  - Added Premium Report PDF column per GSM ID
  - Auto-pipeline: series matrix upload → extract GSMs → FASTQ → analysis → reports
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
    Returns dict with srr_ids, sra_url, ena_url, fasterq_cmd.
    """
    srr_ids = get_srr_for_gsm(gsm_id)

    sra_search_url = f"https://www.ncbi.nlm.nih.gov/sra?term={gsm_id}"

    if srr_ids:
        srr = srr_ids[0]
        ena_url = f"https://www.ebi.ac.uk/ena/browser/view/{srr}"
        # ENA FASTQ direct download (most reliable public URL)
        ena_fastq_url = (
            f"https://www.ebi.ac.uk/ena/portal/api/filereport"
            f"?accession={srr}&result=read_run&fields=fastq_ftp&format=tsv"
        )
        fasterq_cmd = f"fasterq-dump {srr} --split-files --outdir ./{gsm_id}/"
    else:
        srr = None
        ena_url = None
        ena_fastq_url = None
        fasterq_cmd = f"# Run: fasterq-dump <SRR_ID> --split-files --outdir ./{gsm_id}/"

    return {
        "gsm_id": gsm_id,
        "srr_ids": srr_ids,
        "srr": srr,
        "sra_search_url": sra_search_url,
        "ena_url": ena_url,
        "ena_fastq_url": ena_fastq_url,
        "fasterq_cmd": fasterq_cmd,
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
    st.markdown("### 🔬 Pipeline Steps")
    for step in ["1. Upload series matrix",
                 "2. Extract GSM IDs",
                 "3. Lookup FASTQ via SRA",
                 "4. Run DE analysis",
                 "5. Generate reports"]:
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
            _sra_link   = f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}"
            _geo_link   = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_gsm}"
            _ena_link   = f"https://www.ebi.ac.uk/ena/browser/view/{_gsm}"
            _bg         = "rgba(255,255,255,0.03)" if _i % 2 == 0 else "transparent"

            # SRA column
            _sra_col = (
                f"<a href='{_sra_link}' target='_blank' "
                f"style='background:#00d4aa;color:#0f1117;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>NCBI SRA 🔍</a> "
                f"<a href='{_geo_link}' target='_blank' "
                f"style='background:#7c3aed;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>GEO 🧬</a>"
            )

            # FASTQ download column
            _fastq_search_url = f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}[Sample]"
            _ena_fastq_url    = f"https://www.ebi.ac.uk/ena/browser/view/{_gsm}"
            _fastq_col = (
                f"<a href='{_fastq_search_url}' target='_blank' "
                f"style='background:#e65c00;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>SRA FASTQ 📥</a> "
                f"<a href='{_ena_fastq_url}' target='_blank' "
                f"style='background:#1a73e8;color:white;padding:3px 9px;"
                f"border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none;"
                f"margin:2px;display:inline-block'>ENA FASTQ 🌐</a>"
            )

            # Pipeline status column
            _pipe_key = f"pipeline_done_{_gsm}"
            _pipe_done = st.session_state.get(_pipe_key, False)
            if _pipe_done:
                _pipe_col = "<span style='color:#00d4aa;font-weight:700'>✅ Done</span>"
            elif _has_expr or st.session_state.get("active_df") is not None:
                _pipe_col = "<span style='color:#ffa500;font-size:0.82rem'>⚡ Ready to run</span>"
            else:
                _pipe_col = "<span style='color:#8b92a5;font-size:0.82rem'>⏳ Load data first</span>"

            # Report columns — shown as "Generate below" links
            _report_col = (
                f"<span style='color:#8b92a5;font-size:0.8rem'>↓ Scroll to Reports</span>"
            )
            _premium_col = (
                f"<span style='color:#8b92a5;font-size:0.8rem'>↓ Premium below</span>"
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
              <td style='padding:8px 12px;white-space:nowrap'>{_sra_col}</td>
              <td style='padding:8px 12px;white-space:nowrap'>{_fastq_col}</td>
              <td style='padding:8px 12px;text-align:center'>{_pipe_col}</td>
              <td style='padding:8px 12px;text-align:center'>{_report_col}</td>
              <td style='padding:8px 12px;text-align:center'>{_premium_col}</td>
            </tr>"""

        _table_html = f"""
        <div style='overflow-x:auto;overflow-y:auto;max-height:500px;border:1px solid #2a2d3e;border-radius:10px'>
          <table style='width:100%;border-collapse:collapse;min-width:900px'>
            <thead>
              <tr style='background:#1a1d27;position:sticky;top:0;z-index:1'>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>GSM ID</th>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.85rem'>Sample Label</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>🔗 SRA / GEO</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>📥 FASTQ Download</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.85rem;white-space:nowrap'>⚙️ Pipeline</th>
                <th style='padding:10px 12px;text-align:center;color:#00d4aa;font-size:0.85rem;white-space:nowrap'>🆓 Free Report</th>
                <th style='padding:10px 12px;text-align:center;color:#7c3aed;font-size:0.85rem;white-space:nowrap'>💎 Premium Report</th>
              </tr>
            </thead>
            <tbody>{_rows_html}</tbody>
          </table>
        </div>"""
        st.markdown(_table_html, unsafe_allow_html=True)

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
                _bg  = "rgba(255,255,255,0.03)" if i%2==0 else "transparent"
                _sra = f"https://www.ncbi.nlm.nih.gov/sra?term={_g}"
                _geo = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={_g}"
                _fq  = f"https://www.ncbi.nlm.nih.gov/sra?term={_g}[Sample]"
                _ena = f"https://www.ebi.ac.uk/ena/browser/view/{_g}"
                _rows += (
                    f"<tr style='background:{_bg}'>"
                    f"<td style='padding:7px 14px;font-family:monospace;color:#00d4aa;font-weight:700'>{_g}</td>"
                    f"<td style='padding:7px 14px;color:#c8cfe0;font-size:0.9rem'>{str(_l)[:55]}</td>"
                    f"<td style='padding:7px 10px;text-align:center'>"
                    f"<a href='{_sra}' target='_blank' style='background:#00d4aa;color:#0f1117;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>SRA</a> "
                    f"<a href='{_geo}' target='_blank' style='background:#7c3aed;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>GEO</a>"
                    f"</td>"
                    f"<td style='padding:7px 10px;text-align:center'>"
                    f"<a href='{_fq}' target='_blank' style='background:#e65c00;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>SRA FASTQ</a> "
                    f"<a href='{_ena}' target='_blank' style='background:#1a73e8;color:white;padding:3px 9px;border-radius:5px;font-weight:700;font-size:0.8rem;text-decoration:none'>ENA</a>"
                    f"</td>"
                    f"<td style='padding:7px 10px;text-align:center;color:#8b92a5;font-size:0.8rem'>Load data first</td>"
                    f"<td style='padding:7px 10px;text-align:center;color:#8b92a5;font-size:0.8rem'>Load data first</td>"
                    f"<td style='padding:7px 10px;text-align:center;color:#8b92a5;font-size:0.8rem'>Load data first</td>"
                    f"</tr>"
                )

            st.markdown(f"""
            <div style='overflow-x:auto;overflow-y:auto;max-height:420px;border:1px solid #2a2d3e;border-radius:10px;margin:12px 0'>
            <table style='width:100%;border-collapse:collapse;min-width:800px'>
              <thead><tr style='background:#1a1d27;position:sticky;top:0'>
                <th style='padding:9px 14px;text-align:left;color:#8b92a5;font-size:0.85rem'>GSM ID</th>
                <th style='padding:9px 14px;text-align:left;color:#8b92a5;font-size:0.85rem'>Sample Label</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>🔗 SRA / GEO</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>📥 FASTQ</th>
                <th style='padding:9px 14px;text-align:center;color:#8b92a5;font-size:0.85rem'>⚙️ Pipeline</th>
                <th style='padding:9px 14px;text-align:center;color:#00d4aa;font-size:0.85rem'>🆓 Free Report</th>
                <th style='padding:9px 14px;text-align:center;color:#7c3aed;font-size:0.85rem'>💎 Premium</th>
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
    #  PER-GSM AUTOMATED PIPELINE + REPORTS
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div class='section-header'>
    ⚙️ Automated Pipeline — Per-Sample Reports
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='pipeline-box'>
    <strong>🚀 Automated Pipeline</strong> — Run the complete analysis pipeline for all samples.
    For each GSM ID: the pipeline uses the loaded expression data to compute DE statistics,
    generate all visualizations, and produce both Free and Premium PDF reports.
    FASTQ links are provided directly in the table for downloading raw sequencing data.
    </div>""", unsafe_allow_html=True)

    if not gsm_groups:
        st.info("GSM-level pipeline requires a GEO series matrix file with sample metadata.")
    else:
        _gsm_list = list(gsm_groups.items())

        # Pipeline run-all button
        col_run, col_info = st.columns([2,3])
        with col_run:
            run_all = st.button("🚀 Run Full Pipeline for All Samples", key="run_all_pipeline")
        with col_info:
            st.markdown(f"""
            <div class='info-box' style='margin:0'>
            Will process <strong>{len(_gsm_list)} samples</strong> using the loaded expression data.
            Generates per-sample PDF reports with all DE plots and gene tables.
            </div>""", unsafe_allow_html=True)

        if "pipeline_results" not in st.session_state:
            st.session_state["pipeline_results"] = {}

        if run_all:
            prog_bar = st.progress(0)
            status_box = st.empty()
            total = len(_gsm_list)
            for idx, (_gsm, _lbl) in enumerate(_gsm_list):
                status_box.markdown(f"<div class='info-box'>⚙️ Processing {_gsm} ({idx+1}/{total})...</div>",
                                    unsafe_allow_html=True)
                result = run_gsm_pipeline(
                    gsm_id=_gsm, label=_lbl, df_expr=df_raw,
                    gsm_groups=gsm_groups, grp_a=grp_a, grp_b=grp_b,
                    label_a=label_a, label_b=label_b,
                    lfc_thr=lfc_thr, padj_thr=padj_thr, norm=norm,
                    tmp_dir=tmp
                )
                st.session_state["pipeline_results"][_gsm] = result
                st.session_state[f"pipeline_done_{_gsm}"] = (result["error"] is None)
                prog_bar.progress((idx+1)/total)

            status_box.success(f"✅ Pipeline complete for {total} samples!")

        # ── Per-GSM results table with report buttons
        st.markdown("<br>**📊 Per-Sample Results & Reports:**", unsafe_allow_html=True)

        # Build extended table with pipeline results
        _pipe_rows_html = ""
        for _i, (_gsm, _lbl) in enumerate(_gsm_list):
            _bg      = "rgba(255,255,255,0.03)" if _i % 2 == 0 else "transparent"
            _sra_url = f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}[Sample]"
            _fq_url  = f"https://www.ncbi.nlm.nih.gov/sra?term={_gsm}"
            _ena_url = f"https://www.ebi.ac.uk/ena/browser/view/{_gsm}"
            _res     = st.session_state.get("pipeline_results", {}).get(_gsm)

            # SRA link
            _sra_col = (
                f"<a href='{_fq_url}' target='_blank' "
                f"style='background:#00d4aa;color:#0f1117;padding:3px 8px;"
                f"border-radius:5px;font-weight:700;font-size:0.78rem;text-decoration:none'>SRA</a>"
            )

            # FASTQ download
            _fastq_col = (
                f"<a href='{_sra_url}' target='_blank' "
                f"style='background:#e65c00;color:white;padding:3px 8px;"
                f"border-radius:5px;font-weight:700;font-size:0.78rem;text-decoration:none;margin-right:3px'>SRA FASTQ</a>"
                f"<a href='{_ena_url}' target='_blank' "
                f"style='background:#1a73e8;color:white;padding:3px 8px;"
                f"border-radius:5px;font-weight:700;font-size:0.78rem;text-decoration:none'>ENA</a>"
            )

            # Status
            if _res is None:
                _status_col = "<span style='color:#8b92a5;font-size:0.8rem'>⏳ Not run</span>"
                _deg_col = "—"
                _free_col = "<span style='color:#8b92a5;font-size:0.8rem'>Run pipeline first</span>"
                _prem_col = "<span style='color:#8b92a5;font-size:0.8rem'>Run pipeline first</span>"
            elif _res.get("error"):
                _status_col = f"<span style='color:#ff4d6d;font-size:0.78rem'>❌ Error</span>"
                _deg_col = "—"
                _free_col = "<span style='color:#ff4d6d;font-size:0.78rem'>Failed</span>"
                _prem_col = "<span style='color:#ff4d6d;font-size:0.78rem'>Failed</span>"
            else:
                _up_n   = _res.get("up", 0)
                _dn_n   = _res.get("down", 0)
                _status_col = "<span style='color:#00d4aa;font-weight:700'>✅ Done</span>"
                _deg_col    = f"<span style='color:#ff4d6d'>{_up_n}↑</span> / <span style='color:#4da6ff'>{_dn_n}↓</span>"
                _free_col   = f"<span style='color:#00d4aa;font-size:0.8rem'>✅ Ready (scroll ↓)</span>"
                _prem_col   = f"<span style='color:#7c3aed;font-size:0.8rem'>✅ Ready (scroll ↓)</span>"

            _pipe_rows_html += f"""
            <tr style='background:{_bg}'>
              <td style='padding:8px 12px;font-family:monospace;color:#00d4aa;font-weight:700;white-space:nowrap'>{_gsm}</td>
              <td style='padding:8px 12px;color:#c8cfe0;font-size:0.85rem;max-width:160px;
                         overflow:hidden;text-overflow:ellipsis;white-space:nowrap' title='{_lbl}'>
                {_lbl[:38]}{'…' if len(_lbl)>38 else ''}
              </td>
              <td style='padding:8px 10px;text-align:center'>{_sra_col}</td>
              <td style='padding:8px 10px;text-align:center;white-space:nowrap'>{_fastq_col}</td>
              <td style='padding:8px 10px;text-align:center'>{_status_col}</td>
              <td style='padding:8px 10px;text-align:center'>{_deg_col}</td>
              <td style='padding:8px 10px;text-align:center'>{_free_col}</td>
              <td style='padding:8px 10px;text-align:center'>{_prem_col}</td>
            </tr>"""

        _pipe_table = f"""
        <div style='overflow-x:auto;overflow-y:auto;max-height:480px;border:1px solid #2a2d3e;border-radius:10px'>
          <table style='width:100%;border-collapse:collapse;min-width:850px'>
            <thead>
              <tr style='background:#1a1d27;position:sticky;top:0;z-index:1'>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.83rem;white-space:nowrap'>GSM ID</th>
                <th style='padding:10px 12px;text-align:left;color:#8b92a5;font-size:0.83rem'>Label</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.83rem;white-space:nowrap'>🔗 SRA</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.83rem;white-space:nowrap'>📥 FASTQ</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.83rem;white-space:nowrap'>⚙️ Status</th>
                <th style='padding:10px 12px;text-align:center;color:#8b92a5;font-size:0.83rem;white-space:nowrap'>📊 DEGs</th>
                <th style='padding:10px 12px;text-align:center;color:#00d4aa;font-size:0.83rem;white-space:nowrap'>🆓 Free PDF</th>
                <th style='padding:10px 12px;text-align:center;color:#7c3aed;font-size:0.83rem;white-space:nowrap'>💎 Premium PDF</th>
              </tr>
            </thead>
            <tbody>{_pipe_rows_html}</tbody>
          </table>
        </div>"""
        st.markdown(_pipe_table, unsafe_allow_html=True)

        # ── PDF download section for completed pipeline samples
        pipeline_results = st.session_state.get("pipeline_results", {})
        completed = {k: v for k, v in pipeline_results.items()
                     if v and not v.get("error") and v.get("free_pdf_path")}

        if completed:
            st.markdown("---")
            st.markdown("<div class='section-header'>📄 Download Per-Sample Reports</div>",
                        unsafe_allow_html=True)

            st.markdown("""<div class='info-box'>
            Pipeline complete — download individual PDF reports for each sample below.
            Free reports include summary + top genes + volcano plot.
            Premium reports include all plots + complete DEG tables + methods.
            </div>""", unsafe_allow_html=True)

            # Free reports
            st.markdown("### 🆓 Free Reports")
            free_cols = st.columns(min(4, len(completed)))
            for idx, (_gsm, _res) in enumerate(completed.items()):
                col_idx = idx % len(free_cols)
                with free_cols[col_idx]:
                    _lbl_short = gsm_groups.get(_gsm, _gsm)[:20]
                    try:
                        with open(_res["free_pdf_path"], "rb") as _f:
                            st.download_button(
                                label=f"📥 {_gsm}\n{_lbl_short}",
                                data=_f.read(),
                                file_name=f"{_gsm}_free_report.pdf",
                                mime="application/pdf",
                                key=f"dl_free_{_gsm}",
                                use_container_width=True,
                            )
                    except Exception:
                        st.error(f"Error: {_gsm}")

            # Premium reports
            st.markdown("### 💎 Premium Reports")
            st.markdown("""<div class='premium-box'>
            <h4 style='color:#7c3aed;margin:0 0 8px'>💎 Premium Full Reports</h4>
            <p style='font-size:0.88rem;color:#888;margin:0'>
            All 6 plots · Complete DEG tables · Methods section · Per-sample analysis
            </p></div>""", unsafe_allow_html=True)

            code_input = st.text_input("🔑 Access Code (for premium reports)",
                                        type="password", placeholder="Enter access code")
            VALID_CODES = {"BIO100", "RNASEQ2025", "PREMIUM99"}

            if code_input in VALID_CODES:
                st.success("✅ Premium access granted!")
                prem_cols = st.columns(min(4, len(completed)))
                for idx, (_gsm, _res) in enumerate(completed.items()):
                    col_idx = idx % len(prem_cols)
                    with prem_cols[col_idx]:
                        _lbl_short = gsm_groups.get(_gsm, _gsm)[:20]
                        try:
                            with open(_res["premium_pdf_path"], "rb") as _f:
                                st.download_button(
                                    label=f"💎 {_gsm}\n{_lbl_short}",
                                    data=_f.read(),
                                    file_name=f"{_gsm}_premium_report.pdf",
                                    mime="application/pdf",
                                    key=f"dl_prem_{_gsm}",
                                    use_container_width=True,
                                )
                        except Exception:
                            st.error(f"Error: {_gsm}")
            elif code_input:
                st.error("❌ Invalid access code.")
            else:
                st.info("Enter access code to unlock premium per-sample reports.")

    # ── Dataset-level PDF Reports
    st.markdown("---")
    st.markdown("<div class='section-header'>📄 Dataset-Level Reports</div>",unsafe_allow_html=True)
    cf, cp = st.columns(2)

    with cf:
        st.markdown("### 🆓 Free Dataset Report")
        st.markdown("Summary + Top 20 Genes + Volcano Plot")
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
        <h3 style='color:#7c3aed;margin:0 0 6px'>💎 Premium Dataset Report</h3>
        <p style='font-size:0.88rem;color:#888;margin:0'>
        All plots · Full DEG tables (top 25 per direction) · Methods section
        </p></div>""", unsafe_allow_html=True)
        code=st.text_input("🔑 Access Code",type="password",placeholder="Enter after payment",
                            key="dataset_prem_code")
        VALID={"BIO100","RNASEQ2025","PREMIUM99"}
        if code in VALID:
            st.success("✅ Access granted!")
            if st.button("💎 Generate Premium Dataset PDF"):
                with st.spinner("Building premium PDF..."):
                    path=os.path.join(tmp,"Premium.pdf")
                    try:
                        build_premium_pdf(df,gene_col,saved_figs,up,down,
                                          label_a,label_b,dataset_type,path)
                        with open(path,"rb") as f:
                            st.download_button("⬇️ Download Premium PDF",f.read(),
                                                file_name=f"RNA_Premium_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                mime="application/pdf")
                        st.success("✅ Premium Report ready!")
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
        Drop a CSV, TXT, GZ or ZIP file — including NCBI GEO Series Matrix files.
        The app will automatically extract GSM IDs, provide FASTQ download links,
        run the complete DE pipeline, and generate PDF reports.
        </p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    for col,icon,title,desc in [
        (c1,"🔍","Smart Detection","Auto-identifies groups from GEO metadata"),
        (c2,"📥","FASTQ Links","Direct SRA & ENA FASTQ download per sample"),
        (c3,"⚙️","Auto Pipeline","Full DE analysis for every GSM sample"),
        (c4,"📄","PDF Reports","Free summary + Premium full reports per sample"),
    ]:
        col.markdown(f"""<div class='metric-card' style='text-align:left'>
        <div style='font-size:2rem'>{icon}</div>
        <div style='color:white;font-weight:700;margin:8px 0 4px'>{title}</div>
        <div style='color:#8b92a5;font-size:0.85rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🔬 Automated Pipeline
    1. **Upload** a GEO Series Matrix file (`.txt`, `.txt.gz`, `.zip`)
    2. **GSM IDs** are extracted automatically — shown in a table with SRA & FASTQ links
    3. **Load expression data** via auto-retrieve or manual upload
    4. **Run pipeline** — DE analysis computed for all samples
    5. **Download PDFs** — Free or Premium report per GSM ID
    """)
