"""
session_history.py — Upload history & session log management
Persists file upload history across reruns using st.session_state.
"""

import json
from datetime import datetime
import streamlit as st


HISTORY_KEY = "upload_history"


def _init():
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []


def add_entry(filename: str, gse: str, n_genes: int, n_samples: int,
              up: int, down: int, dataset_type: str):
    """Add a new analysis entry to the session history."""
    _init()
    entry = {
        "id":           len(st.session_state[HISTORY_KEY]) + 1,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename":     filename,
        "gse":          gse or "—",
        "n_genes":      n_genes,
        "n_samples":    n_samples,
        "up":           up,
        "down":         down,
        "dataset_type": dataset_type,
    }
    # Avoid duplicate consecutive entries for the same file
    hist = st.session_state[HISTORY_KEY]
    if hist and hist[-1]["filename"] == filename and hist[-1]["gse"] == (gse or "—"):
        hist[-1] = entry           # overwrite with fresh stats
    else:
        hist.append(entry)


def get_history() -> list:
    _init()
    return list(reversed(st.session_state[HISTORY_KEY]))  # newest first


def clear_history():
    st.session_state[HISTORY_KEY] = []


def history_as_csv() -> str:
    """Return history as a CSV string for download."""
    hist = get_history()
    if not hist:
        return "No history yet."
    cols = ["id", "timestamp", "filename", "gse", "n_genes",
            "n_samples", "up", "down", "dataset_type"]
    lines = [",".join(cols)]
    for e in hist:
        lines.append(",".join(str(e.get(c, "")) for c in cols))
    return "\n".join(lines)


# ── UI renderer ──────────────────────────────────────────────────────────────
def render_history_panel():
    """Render the full history panel inside the current Streamlit context."""
    _init()
    hist = get_history()

    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(0,212,170,0.08),rgba(124,58,237,0.06));
    border:1px solid rgba(0,212,170,0.3);border-radius:14px;padding:20px;margin-bottom:18px'>
    <h3 style='color:#00d4aa;margin:0 0 6px'>🗂️ Upload & Analysis History</h3>
    <p style='color:#8b92a5;margin:0;font-size:0.9rem'>
    All files you've analysed this session are listed below.
    History is preserved while the browser tab stays open.
    </p></div>
    """, unsafe_allow_html=True)

    if not hist:
        st.info("No files analysed yet. Upload a file above to get started.")
        return

    # ── Summary metrics
    total_degs = sum(e["up"] + e["down"] for e in hist)
    c1, c2, c3 = st.columns(3)
    c1.metric("📁 Files Analysed", len(hist))
    c2.metric("🧬 Total DEGs Found", f"{total_degs:,}")
    c3.metric("🕐 Session Started",
              st.session_state[HISTORY_KEY][0]["timestamp"] if st.session_state[HISTORY_KEY] else "—")

    st.markdown("---")

    # ── History cards
    TYPE_ICONS = {
        "tumor_normal":      "🔬",
        "treated_control":   "💊",
        "time_series":       "⏱️",
        "knockout_wildtype": "🔧",
        "pre_computed":      "📁",
        "single_condition":  "📊",
    }
    TYPE_LABELS = {
        "tumor_normal":      "Tumor vs Normal",
        "treated_control":   "Treated vs Control",
        "time_series":       "Time Series",
        "knockout_wildtype": "Knockout vs WT",
        "pre_computed":      "Pre-computed",
        "single_condition":  "Single Condition",
    }

    for e in hist:
        icon  = TYPE_ICONS.get(e["dataset_type"], "🧬")
        label = TYPE_LABELS.get(e["dataset_type"], e["dataset_type"])
        up    = e["up"]
        down  = e["down"]
        gse   = e["gse"]

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid #2a2d3e;
        border-radius:10px;padding:14px 18px;margin:8px 0;
        border-left:4px solid #00d4aa'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px'>
          <div>
            <div style='color:white;font-weight:700;font-size:1rem'>
              {icon} {e["filename"]}
            </div>
            <div style='color:#8b92a5;font-size:0.82rem;margin-top:3px'>
              {label} &nbsp;·&nbsp; {e["timestamp"]}
              {"&nbsp;·&nbsp;<code style='color:#00d4aa'>" + gse + "</code>" if gse != "—" else ""}
            </div>
          </div>
          <div style='display:flex;gap:18px;flex-wrap:wrap'>
            <div style='text-align:center'>
              <div style='color:#00d4aa;font-weight:800;font-size:1.3rem;font-family:monospace'>
                {e["n_genes"]:,}
              </div>
              <div style='color:#8b92a5;font-size:0.76rem'>genes</div>
            </div>
            <div style='text-align:center'>
              <div style='color:#ff4d6d;font-weight:800;font-size:1.3rem;font-family:monospace'>
                {up:,}
              </div>
              <div style='color:#8b92a5;font-size:0.76rem'>↑ up</div>
            </div>
            <div style='text-align:center'>
              <div style='color:#4da6ff;font-weight:800;font-size:1.3rem;font-family:monospace'>
                {down:,}
              </div>
              <div style='color:#8b92a5;font-size:0.76rem'>↓ down</div>
            </div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_dl, col_cl = st.columns([2, 1])
    with col_dl:
        st.download_button(
            "📥 Export History CSV",
            data=history_as_csv(),
            file_name=f"rnaseq_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_cl:
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.rerun()
