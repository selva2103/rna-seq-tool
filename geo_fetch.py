"""
geo_fetch.py — NCBI / GEO data fetching helpers
Extracted from app.py v7.0 for modular structure.
Adds: @st.cache_data caching, retry logic, detailed error logging.
"""

import re, json, gzip, io, time
import urllib.request, urllib.error
import pandas as pd
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────
NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_HTTPS   = "https://ftp.ncbi.nlm.nih.gov/geo/series"
GEO_API     = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NCBI_DELAY  = 0.35
MAX_RETRIES = 3


# ── Validation ───────────────────────────────────────────────────────────────
def validate_gse(gse: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty error means valid."""
    gse = gse.strip().upper()
    if not gse:
        return False, "Please enter a GSE accession number."
    if not re.match(r"^GSE\d{1,9}$", gse):
        return False, f"'{gse}' is not a valid GSE ID. Format must be GSE followed by digits (e.g. GSE12345)."
    return True, ""


# ── Low-level HTTP ───────────────────────────────────────────────────────────
def _ncbi_get(url: str, timeout: int = 45, retries: int = MAX_RETRIES) -> bytes:
    """GET with retry + backoff. Raises urllib.error.URLError on final failure."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RNASeqTool/7.0 (bioinformatics research)",
            "Accept": "*/*",
        },
    )
    last_exc = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (429, 503):          # rate-limit → back off longer
                time.sleep(2 ** attempt)
            else:
                raise                          # 404, 403 etc. → no retry
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            time.sleep(0.5 * (attempt + 1))
    raise last_exc


def _geo_https_folder(gse: str) -> str:
    num    = gse.replace("GSE", "")
    prefix = num[:-3] + "nnn" if len(num) > 3 else "0nnn"
    return f"{GEO_HTTPS}/GSE{prefix}/{gse}"


# ── Supplementary files ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geo_supplementary_files(gse: str) -> list:
    """Return list of {name, url} dicts for GEO supplementary files."""
    results = []
    # Try the GEO API first
    try:
        api_url = f"{GEO_API}?acc={gse}&targ=self&form=text&view=brief"
        raw = _ncbi_get(api_url, timeout=20).decode("utf-8", errors="replace")
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("!Series_supplementary_file"):
                val = re.sub(r'^[^=]=\s*', '', line).strip().strip('"')
                if val and val != "NONE":
                    https_url = val.replace("ftp://ftp.ncbi.nlm.nih.gov",
                                            "https://ftp.ncbi.nlm.nih.gov")
                    fname = https_url.split("/")[-1]
                    if fname:
                        results.append({"name": fname, "url": https_url})
        if results:
            return results
    except Exception as e:
        st.warning(f"⚠️ GEO API lookup failed for {gse}: {e}. Trying FTP directory…")

    # Fallback: scrape the HTTPS FTP folder
    try:
        folder = _geo_https_folder(gse)
        suppl  = f"{folder}/suppl/"
        raw    = _ncbi_get(suppl, timeout=20).decode("utf-8", errors="replace")
        names  = re.findall(
            r'(?:href=")[^"]*?/([^"/]+\.(?:gz|txt|csv|tsv|xlsx))"'
            r'|(?:href=")([^"/]+\.(?:gz|txt|csv|tsv|xlsx))"',
            raw, re.I,
        )
        for tup in names:
            fname = (tup[0] or tup[1]).strip()
            if fname and not fname.startswith("?"):
                results.append({"name": fname, "url": f"{suppl}{fname}"})
    except Exception as e:
        st.warning(f"⚠️ FTP directory lookup also failed: {e}")

    return results


@st.cache_data(ttl=3600, show_spinner=False)
def download_geo_supplementary(url: str, filename: str) -> pd.DataFrame | None:
    """Download and parse a GEO supplementary file into a DataFrame."""
    url = url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
    try:
        raw = _ncbi_get(url, timeout=90)
    except Exception as e:
        st.warning(f"⚠️ Could not download {filename}: {e}")
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
    except Exception as e:
        st.warning(f"⚠️ Could not parse {filename}: {e}")
    return None


# ── Series matrix ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geo_series_matrix(gse: str) -> tuple:
    """Fetch and parse the GEO series matrix. Returns (df, geo_meta, gsm_groups)."""
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
            from file_parser import parse_geo_soft_full
            df, geo_meta, gsm_groups = parse_geo_soft_full(text)
            if df is not None and df.shape[1] > 1 and df.shape[0] > 10:
                return df, geo_meta, gsm_groups
        except Exception:
            continue
    return None, {}, {}


# ── SRA helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sra_runinfo(srp_or_srx: str) -> pd.DataFrame | None:
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
    except Exception as e:
        st.warning(f"⚠️ SRA run info fetch failed for {srp_or_srx}: {e}")
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_srr_for_gsm(gsm_id: str) -> list:
    """Look up SRR run IDs for a given GSM ID. Returns list of SRR strings."""
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
    except Exception as e:
        st.warning(f"⚠️ SRR lookup failed for {gsm_id}: {e}")
        return []


# ── Smart retrieval orchestrator ─────────────────────────────────────────────
def smart_retrieve_from_geo(gse: str, geo_meta: dict,
                             gsm_groups: dict,
                             progress_callback=None) -> dict:
    result = {
        "status": "failed", "df": None,
        "files_found": [], "runinfo": None, "message": "",
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
        for finfo in (priority + others)[:8]:
            _prog(f"📥 Trying {label} file: {finfo['name']}")
            df = download_geo_supplementary(finfo["url"], finfo["name"])
            if df is not None and df.shape[0] > 50:
                num_cols = [c for c in df.columns
                            if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) >= 2:
                    return df, finfo["name"]
        return None, None

    # 1. Metadata-embedded supplementary URLs
    meta_suppl_urls = []
    for val in geo_meta.get("Series_supplementary_file", []):
        val = val.strip().strip('"')
        if val and val.upper() != "NONE":
            https_url = val.replace("ftp://ftp.ncbi.nlm.nih.gov",
                                    "https://ftp.ncbi.nlm.nih.gov")
            fname = https_url.split("/")[-1]
            if fname:
                meta_suppl_urls.append({"name": fname, "url": https_url})

    if meta_suppl_urls:
        result["files_found"] = [f["name"] for f in meta_suppl_urls]
        _prog(f"🔗 Found {len(meta_suppl_urls)} supplementary file(s) in metadata")
        df, fname = _try_download_and_parse(meta_suppl_urls, "metadata-embedded")
        if df is not None:
            result.update(status="counts", df=df,
                          message=f"✅ Retrieved expression data: {fname} ({df.shape[0]:,} genes)")
            return result

    # 2. GEO API supplementary files
    _prog("🔍 Querying GEO for supplementary files…")
    time.sleep(NCBI_DELAY)
    suppl_files = fetch_geo_supplementary_files(gse)
    new_files   = [f for f in suppl_files
                   if f["name"] not in {x["name"] for x in meta_suppl_urls}]
    if new_files:
        result["files_found"].extend([f["name"] for f in new_files])
        df, fname = _try_download_and_parse(new_files, "GEO API")
        if df is not None:
            result.update(status="counts", df=df,
                          message=f"✅ Retrieved expression data: {fname} ({df.shape[0]:,} genes)")
            return result

    # 3. Series matrix
    _prog("🔍 Trying series matrix from GEO…")
    time.sleep(NCBI_DELAY)
    df, new_meta, new_gsm = fetch_geo_series_matrix(gse)
    if df is not None and df.shape[0] > 10:
        if new_gsm:
            gsm_groups.update(new_gsm)
        result.update(status="matrix", df=df,
                      message=f"✅ Retrieved series matrix: {df.shape[0]:,} genes × {df.shape[1]} samples")
        return result

    # 4. SRA runinfo
    srp = None
    for val in geo_meta.get("Series_relation", []):
        m = re.search(r"SRA:.*?term=(\w+)", val)
        if m:
            srp = m.group(1); break

    if srp:
        _prog(f"📋 Fetching SRA run info for {srp}…")
        time.sleep(NCBI_DELAY)
        runinfo = fetch_sra_runinfo(srp)
        if runinfo is not None:
            result.update(status="runinfo", runinfo=runinfo,
                          message=(f"⚠️ No count matrix found on GEO. "
                                   f"Retrieved SRA run info ({len(runinfo)} runs for {srp})."))
            return result

    result["message"] = f"❌ Could not retrieve data for {gse}."
    return result
