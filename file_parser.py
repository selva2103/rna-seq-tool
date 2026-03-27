"""
file_parser.py — File reading, GEO SOFT parsing, group detection
Extracted from app.py v7.0 for modular structure.
"""

import re, gzip, io, zipfile
import pandas as pd
import streamlit as st


# ── GEO SOFT parser ──────────────────────────────────────────────────────────
def parse_geo_soft_full(content: str) -> tuple:
    """Parse a GEO series matrix text into (df, geo_meta dict, gsm_groups dict)."""
    lines      = content.split("\n")
    geo_meta   = {}
    data_lines = []
    in_table   = False

    for line in lines:
        line = line.rstrip("\r")
        if "series_matrix_table_begin" in line.lower():
            in_table = True; continue
        if "series_matrix_table_end" in line.lower():
            in_table = False; continue
        if in_table:
            if line.strip():
                data_lines.append(line)
            continue
        if not line.startswith("!"):
            continue
        eq_idx = line.find("=")
        if eq_idx == -1:
            continue
        raw_key = line[1:eq_idx].strip()
        raw_val = line[eq_idx + 1:].strip()
        if not raw_key:
            continue
        raw_parts = raw_val.split("\t")
        vals = [v.strip().strip('"').strip("'") for v in raw_parts if v.strip()]
        if not vals:
            vals = [""]
        if raw_key in geo_meta:
            existing = geo_meta[raw_key]
            if len(vals) > 1:
                existing.extend(vals)
            else:
                existing.append(vals[0])
        else:
            geo_meta[raw_key] = vals

    df = None
    if data_lines:
        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)),
                             sep="\t", index_col=None, low_memory=False)
            if df.shape[1] > 1 and df.columns[0] not in ["ID_REF", "Gene", "gene"]:
                first = df.columns[0]
                if not str(first).startswith("GSM"):
                    df = df.rename(columns={first: "ID_REF"})
        except Exception:
            df = None

    # ── Extract GSM IDs ──────────────────────────────────────────────────────
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
                    seen.add(tok)
                    gsm_ids.append(tok)

    if not gsm_ids and df is not None:
        gsm_ids = [c for c in df.columns if str(c).startswith("GSM")]

    # Deduplicate
    seen2 = set(); gsm_ids_uniq = []
    for g in gsm_ids:
        if g not in seen2:
            seen2.add(g)
            gsm_ids_uniq.append(g)
    gsm_ids = gsm_ids_uniq

    # ── Build gsm_groups ─────────────────────────────────────────────────────
    gsm_groups = {}
    if gsm_ids:
        n = len(gsm_ids)
        label_source = None
        for key in ("Sample_title", "sample_title",
                    "Sample_source_name_ch1", "sample_source_name_ch1",
                    "Sample_characteristics_ch1"):
            vals = geo_meta.get(key, [])
            if len(vals) >= n:
                label_source = vals[:n]
                break
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


# ── Smart file reader ────────────────────────────────────────────────────────
def smart_read_file(uploaded_file) -> tuple:
    """
    Read any supported file format.
    Returns (df, geo_meta, gsm_groups, messages).
    messages = list of (level, text) tuples, level in {info, success, error, warning}.
    """
    fname = uploaded_file.name
    msgs  = []

    def _decode(raw):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                return raw.decode(enc)
            except Exception:
                pass
        return raw.decode("latin1", errors="replace")

    def _parse(text):
        is_geo = (
            "series_matrix_table_begin" in text.lower()
            or "!Series_geo_accession" in text
            or "!Sample_geo_accession" in text
        )
        if is_geo:
            df, gm, gg = parse_geo_soft_full(text)
            if gm:
                if df is None or df.shape[1] <= 1:
                    if gg:
                        placeholder = pd.DataFrame({
                            "GSM_ID": list(gg.keys()),
                            "Sample_Label": list(gg.values()),
                        })
                        return placeholder, gm, gg
                    else:
                        placeholder = pd.DataFrame(
                            {"metadata": ["GEO series matrix — no data table"]}
                        )
                        return placeholder, gm, gg
                return df, gm, gg
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep,
                                 encoding="latin1", comment="!", low_memory=False)
                if df.shape[1] > 1:
                    return df, {}, {}
            except Exception:
                pass
        return None, {}, {}

    try:
        if fname.endswith(".zip"):
            msgs.append(("info", "📦 ZIP detected"))
            with zipfile.ZipFile(uploaded_file) as z:
                for name in z.namelist():
                    if name.endswith("/"):
                        continue
                    with z.open(name) as f:
                        raw = _decode(f.read())
                    df, gm, gg = _parse(raw)
                    if df is not None:
                        msgs.append(("success", f"✅ Parsed `{name}`"))
                        return df, gm, gg, msgs
            msgs.append(("error", "Could not parse any file inside the ZIP."))
            return None, {}, {}, msgs

        elif fname.endswith(".gz"):
            msgs.append(("info", "🗜️ GZ decompressing…"))
            with gzip.open(uploaded_file, "rt", encoding="latin1", errors="replace") as f:
                raw = f.read()
            df, gm, gg = _parse(raw)
            if df is not None:
                msgs.append(("success", "✅ GZ parsed"))
                return df, gm, gg, msgs
            msgs.append(("error", "Could not parse contents of GZ file."))
            return None, {}, {}, msgs

        elif fname.endswith((".txt", ".tsv")):
            raw = _decode(uploaded_file.read())
            df, gm, gg = _parse(raw)
            if df is not None:
                msgs.append(("success", "✅ Text file parsed"))
                return df, gm, gg, msgs
            msgs.append(("error", "Could not parse TXT/TSV file."))
            return None, {}, {}, msgs

        elif fname.endswith(".csv"):
            for enc in ["utf-8", "latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    msgs.append(("success", "✅ CSV parsed"))
                    return df, {}, {}, msgs
                except Exception:
                    pass
            msgs.append(("error", "Could not read CSV file."))
            return None, {}, {}, msgs

    except Exception as e:
        msgs.append(("error", f"Fatal read error: {e}"))
        return None, {}, {}, msgs

    msgs.append(("error", f"Unsupported file format: {fname}"))
    return None, {}, {}, msgs
