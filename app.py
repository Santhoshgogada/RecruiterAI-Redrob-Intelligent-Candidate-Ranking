"""
app.py — RecruiterAI Streamlit Demo
=====================================
Live sandbox for judges to verify the system end-to-end.

Supports three data sources:
  1. Built-in toy demo (6 hand-written candidates, fast, illustrative)
  2. A real-data sample (N candidates streamed from the actual
     candidates.jsonl without loading the full ~487MB file into memory)
  3. User-uploaded file

Run: streamlit run app.py

To enable mode (2), place the real candidates.jsonl in the same directory
as this file (it is gitignored / not checked in due to size).
"""

import json
import time
import csv
import io
import os
import random
import streamlit as st
from scorer import score_candidate, ScoreBreakdown

st.set_page_config(
    page_title="RecruiterAI — Redrob Candidate Ranking",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
# Page styles
# ─────────────────────────────────────────────
st.markdown("""
<style>
.score-high  { color: #16a34a; font-weight: 700; font-size: 28px; }
.score-mid   { color: #d97706; font-weight: 700; font-size: 28px; }
.score-low   { color: #dc2626; font-weight: 700; font-size: 28px; }
.honeypot    { background: #fef2f2; border-left: 4px solid #dc2626;
               padding: 8px 12px; border-radius: 4px; font-size: 13px; }
.data-note   { background: #eff6ff; border: 1px solid #bfdbfe;
               border-radius: 8px; padding: 10px 14px; font-size: 13px;
               color: #1e40af; margin-bottom: 12px; }

/* Custom global overrides to target button elements securely without breaking text flow */
.stDownloadButton, .stDownloadButton button {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🎯 RecruiterAI — Intelligent Candidate Ranking")
st.caption("Redrob AI Hackathon 2026 · Track 1: Data & AI Challenge · Hybrid scoring engine · Zero API calls · 14.9s for 100K candidates on CLI")

st.divider()

# ─────────────────────────────────────────────
# Streaming loaders — never hold the full 487MB file in memory at once
# ─────────────────────────────────────────────

REAL_DATA_PATH = "candidates.jsonl"


def real_data_available() -> bool:
    return os.path.exists(REAL_DATA_PATH)


def count_lines_fast(path: str) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def stream_first_n(path: str, n: int) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def stream_random_n(path: str, n: int, total_lines: int, seed: int = 42) -> list[dict]:
    """Single sequential pass, picking lines whose index is in a random target set."""
    random.seed(seed)
    if n >= total_lines:
        return stream_first_n(path, total_lines)

    target_idx = set(random.sample(range(total_lines), n))
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in target_idx:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def load_from_upload(file) -> list[dict]:
    content = file.read().decode("utf-8")
    if content.strip().startswith("["):
        return json.loads(content)
    candidates = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return candidates


def load_demo() -> list[dict]:
    try:
        with open("sample_candidates.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ─────────────────────────────────────────────
# Data source selector
# ─────────────────────────────────────────────

st.subheader("📂 Choose a data source")

source_options = []
if real_data_available():
    source_options.append("Real dataset sample (from actual candidates.jsonl)")
source_options.append("Built-in toy demo (6 candidates)")
source_options.append("Upload my own file")

source = st.radio("Data source", source_options, label_visibility="collapsed")

uploaded = None
sample_size = None
sample_mode = None

if source.startswith("Real dataset"):
    if "_total_lines" not in st.session_state:
        with st.spinner("Indexing real dataset (one-time)..."):
            st.session_state["_total_lines"] = count_lines_fast(REAL_DATA_PATH)
    total = st.session_state["_total_lines"]

    st.markdown(
        f'<div class="data-note">📊 Real file detected: <b>{total:,}</b> candidates available. '
        f'Sampling avoids loading the full file into memory in this browser session — '
        f'the actual top-100 submission was generated by running the full 100K pipeline via the '
        f'CLI (<code>python rank.py</code>, 14.9s), validated against <code>validate_submission.py</code>.</div>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Sample size", min_value=50, max_value=min(5000, total), value=500, step=50)
    with col2:
        sample_mode = st.selectbox("Sampling method", ["Random sample", "First N rows"])

elif source.startswith("Built-in toy"):
    st.caption("6 hand-written illustrative candidates including 1 honeypot — useful for explaining the system, not for evaluating real ranking quality.")

else:
    uploaded = st.file_uploader("Upload candidates.jsonl or any JSON/JSONL matching the schema", type=["json", "jsonl"])

st.divider()

# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Show top N candidates", min_value=5, max_value=100, value=10)
with col2:
    show_honeypots = st.checkbox("Show honeypot analysis", value=True)
    show_breakdown = st.checkbox("Show score breakdown per candidate", value=True)

st.divider()

# ─────────────────────────────────────────────
# Run ranking
# ─────────────────────────────────────────────

if st.button("🚀 Rank Candidates", type="primary", width="stretch"):

    candidates: list[dict] = []

    if source.startswith("Real dataset"):
        with st.spinner(f"Streaming {sample_size} candidates from the real dataset..."):
            t_load = time.time()
            if sample_mode == "Random sample":
                candidates = stream_random_n(REAL_DATA_PATH, sample_size, st.session_state["_total_lines"])
            else:
                candidates = stream_first_n(REAL_DATA_PATH, sample_size)
            load_time = time.time() - t_load
        st.caption(f"Loaded {len(candidates):,} real candidates in {load_time:.2f}s without reading the full file into memory.")

    elif source.startswith("Built-in toy"):
        candidates = load_demo()

    else:
        if uploaded is not None:
            candidates = load_from_upload(uploaded)
        else:
            st.warning("Upload a file first.")
            st.stop()

    if not candidates:
        st.error("No candidates loaded.")
        st.stop()

    st.info(f"Scoring **{len(candidates):,}** candidates...")
    progress = st.progress(0)
    t0 = time.time()

    results: list[ScoreBreakdown] = []
    honeypots: list[ScoreBreakdown] = []
    skipped = 0

    for i, c in enumerate(candidates):
        try:
            r = score_candidate(c)
        except Exception:
            skipped += 1
            continue
        if r.is_honeypot:
            honeypots.append(r)
        else:
            results.append(r)
        step = max(1, len(candidates) // 100)
        if (i + 1) % step == 0 or i == len(candidates) - 1:
            progress.progress((i + 1) / len(candidates))

    results.sort(key=lambda r: -r.final_score)
    for i, r in enumerate(results, 1):
        r.rank = i

    elapsed = time.time() - t0
    progress.empty()

    if skipped:
        st.warning(f"Skipped {skipped} candidates due to schema errors.")

    st.success(f"✅ Ranked {len(results):,} candidates + filtered {len(honeypots):,} honeypots in **{elapsed:.2f}s**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total candidates", f"{len(candidates):,}")
    c2.metric("Viable candidates", f"{len(results):,}")
    c3.metric("Honeypots filtered", f"{len(honeypots):,}")
    if results:
        c4.metric("Top score", f"{results[0].final_score:.1f}/100")

    if source.startswith("Real dataset") and len(candidates) < 5000:
        st.caption(
            "ℹ️ This is a sample, not the full 100K dataset — the live ranking and honeypot "
            "rate here are illustrative of real-data behavior. The official submission.csv "
            "was generated against the complete candidates.jsonl via the CLI."
        )

    st.divider()

    # ── Top candidates ──
    st.subheader(f"🏆 Top {min(top_n, len(results))} Candidates")

    for r in results[:top_n]:
        color_class = "score-high" if r.final_score >= 75 else ("score-mid" if r.final_score >= 50 else "score-low")

        with st.expander(f"#{r.rank}  {r.name}  —  {r.current_title}  [{r.final_score:.1f}/100]", expanded=(r.rank <= 3)):
            col_a, col_b = st.columns([3, 2])

            with col_a:
                st.markdown(f"**Reasoning:** {r.reasoning}")
                if show_breakdown:
                    st.markdown("**Score breakdown:**")
                    breakdown_data = {
                        "Dimension": ["Title fit", "Skills coverage", "Work depth", "Experience", "Education"],
                        "Score": [r.title_score, r.skills_score, r.work_depth_score, r.experience_score, r.education_score],
                        "Weight": ["20%", "25%", "30%", "15%", "10%"],
                        "Note": [r.title_note[:60], r.skills_note[:60], r.work_depth_note[:60], r.experience_note[:60], r.education_note[:60]],
                    }
                    st.dataframe(breakdown_data, width="stretch", hide_index=True)

            with col_b:
                st.markdown(f"<div class='{color_class}'>{r.final_score:.1f}</div>", unsafe_allow_html=True)
                st.caption("Overall score / 100")
                st.metric("Signals multiplier", f"×{r.signals_multiplier:.2f}")
                st.caption(r.signals_note[:80])

    # ── Honeypot section ──
    if show_honeypots and honeypots:
        st.divider()
        st.subheader(f"🪤 Honeypots Detected ({len(honeypots):,})")
        st.caption("Disqualifier title + AI skills listed + insufficient supporting work history — exactly the trap the JD warns about.")
        for h in honeypots[:30]:
            st.markdown(
                f'<div class="honeypot">❌ <b>{h.name}</b> ({h.current_title}) — {h.honeypot_reason}</div>',
                unsafe_allow_html=True
            )
            st.write("")
        if len(honeypots) > 30:
            st.caption(f"... and {len(honeypots) - 30:,} more.")

    # ── CSV download ──
    st.divider()
    st.subheader("📥 Download Ranked CSV")

    if len(results) >= 100:
        top100 = results[:100]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top100:
            writer.writerow([r.candidate_id, r.rank, f"{r.final_score:.2f}", r.reasoning.replace("\n", " ")])

        st.download_button(
            "⬇️ Download submission.csv (top 100, validator-ready)",
            data=buf.getvalue().encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )
        st.caption("This file is formatted to pass validate_submission.py")
    else:
        st.warning(
            f"Only {len(results)} viable candidates in this sample — need at least 100 for a "
            f"valid submission.csv. Increase the sample size, or run the full dataset via "
            f"`python rank.py --candidates candidates.jsonl --out submission.csv` for the real submission."
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([r.candidate_id, r.rank, f"{r.final_score:.2f}", r.reasoning.replace("\n", " ")])
        st.download_button(
            f"⬇️ Download this sample's ranking ({len(results)} rows, not validator-ready)",
            data=buf.getvalue().encode("utf-8"),
            file_name="sample_ranking.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Choose a data source and click **Rank Candidates** to start.")
    with st.expander("How this system works", expanded=True):
        st.markdown("""
**5-component hybrid scoring engine — no LLM calls needed:**

| Component | Weight | Method |
|---|---|---|
| Title fit | 20% | Exhaustive match against the dataset's real title vocabulary |
| Skill coverage | 25% | Must-have vs nice-to-have skills, weighted by proficiency, **backed by work history** (not just listed) |
| Work depth | 30% | AI engineering signals in job descriptions — shipped, deployed, real users |
| Experience | 15% | Years calibrated to JD target (3–8 yrs), seniority arc, job-hopping check |
| Education | 10% | Institution tier (from the dataset's own tiering) + field relevance |

**Signals multiplier (×0.50–1.20):** all 23 real Redrob platform signal fields combined —
availability, response rate, GitHub activity, interview readiness, skill assessment scores, etc.

**Honeypot detection:** calibrated against the real dataset. Legitimate ML-titled candidates
average ~4 AI-engineering signal hits in their work history; candidates with a non-AI title who
also list AI skills but show <2 such signals are flagged as keyword-stuffing — exactly the trap
the JD explicitly warns about.

**Why "Real dataset sample" instead of the full 100K in this browser tab:** the full file is
~487MB, and Streamlit Cloud's free tier has limited memory and a slow cold-start file read
(~11s just to iterate the raw lines). The sampler above streams a chosen number of real rows
without holding the rest of the file in memory, so judges can verify real-data behavior live.
The actual top-100 submission was generated by running the full pipeline via the CLI
(`python rank.py`), which scores all 100,000 candidates in 14.9 seconds and was validated
against the official `validate_submission.py`.
        """)