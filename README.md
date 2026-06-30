# 🎯 RecruiterAI — Redrob Intelligent Candidate Ranking

**India Runs by Redrob AI × Hack2skill — Track 1: Data & AI Challenge**

> Calibrated and validated against the **real 100,000-candidate dataset**.
> Ranks 100K candidates in **14.9 seconds** on CPU. Zero API calls. Zero model downloads.
> Output passes the official `validate_submission.py` unmodified.

---

## Quick Start

```bash
pip install -r requirements.txt

# Rank the full 100K candidate pool (~15 seconds)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate output format (must pass before submission)
python validate_submission.py submission.csv
# → ✅ VALID — submission.csv passes all checks.

# 3. Spin up the modern interactive live demo safely inside your environment
python -m streamlit run app.py
```

### Streamlit demo: verifying against the real dataset, live

Drop the real `candidates.jsonl` next to `app.py` (it's git-ignored — too
large to commit, ~487MB) and the demo unlocks a **"Real dataset sample"**
mode: pick a sample size (50–5,000) and a random or first-N sampling
method, and it streams that many real candidates without ever loading the
full file into memory. This lets judges click through real-data behavior
— honeypot detection, score distribution, top candidates — in the browser,
rather than only trusting CLI output.

The full 100K run is still done via `python rank.py` (14.9s) for the
actual `submission.csv`; the in-browser sampler is for live verification,
not for generating the competition submission itself (Streamlit Cloud's
free tier has tighter memory/time limits than a local CLI run).

If `candidates.jsonl` isn't present, the app falls back cleanly to the
built-in 6-candidate toy demo or a file upload — no errors, no broken UI.

---

## This version is calibrated on real data, not guesses

The previous version of this system was built from inference about what the
dataset might look like. This version was rebuilt after directly inspecting
the real `candidates.jsonl` (100,000 rows, 487MB):

- **Schema**: discovered the real nested structure (`profile`, `career_history`,
  `education`, `skills`, `redrob_signals`) — completely different from the
  flat structure assumed earlier.
- **Title vocabulary**: enumerated distinct `current_title` values directly
  from the data and classified all of them into 4 relevance tiers by direct
  inspection, rather than fuzzy keyword guessing.
- **Skill vocabulary**: extracted the exact AI-related `skill.name` values
  that appear in the dataset (RAG, Fine-tuning LLMs, QLoRA, Sentence
  Transformers, etc.) rather than a generic ML skill list.
- **Honeypot calibration**: measured that legitimate ML-titled candidates
  average ~4 AI-engineering signal hits in their `career_history`
  descriptions, while keyword-stuffed honeypots average ~0–1. The
  honeypot threshold (2 signals) was set from this real distribution.
- **23 signals**: confirmed exact field names and types directly from the
  data (`open_to_work_flag`, `github_activity_score`,
  `recruiter_response_rate`, etc.) — not invented field names.

---

## Results on the real dataset

| Metric | Value |
|---|---|
| Total candidates | 100,000 |
| Scoring time | 14.9 seconds |
| True honeypots filtered | 5,517 (5.5%) |
| Viable candidates | 94,483 |
| Top-100 title purity | 100% genuinely AI/ML-titled |

**Top-100 title distribution** (every single one is AI/ML-relevant):

| Title | Count |
|---|---|
| Senior Software Engineer (ML) | 30 |
| ML Engineer | 19 |
| AI Research Engineer | 12 |
| Applied ML Engineer | 12 |
| AI Specialist | 9 |
| Junior ML Engineer | 5 |
| Senior Data Scientist | 4 |
| Senior AI Engineer | 2 |
| Senior NLP Engineer | 2 |
| Senior Machine Learning Engineer | 2 |
| Data Scientist, Staff ML Engineer, Lead AI Engineer | 1 each |

No Marketing Manager, Accountant, Sales Executive, or other irrelevant title
appears anywhere in the top 100 — and none of the keyword-stuffed honeypot
profiles (disqualifier title + AI skills listed + no work backing) made it
through either.

---

## Architecture

```
candidates.jsonl (100K profiles, nested schema)
        │
        ▼
┌─────────────────────────────────────────┐
│  1. HONEYPOT FILTER                     │
│     Disqualifier title (e.g. Marketing) │
│     + AI skills listed                  │
│     + <2 AI signals in career_history   │
│     → filtered (5,517 caught)           │
└─────────────────┬───────────────────────┘
                  │ viable candidates
                  ▼
┌─────────────────────────────────────────┐
│  2. HYBRID SCORER (5 dimensions)        │
│                                         │
│  Title fit        ×  20%   (exhaustive │
│  Skill coverage   ×  25%    closed     │
│  Work depth       ×  30%    vocab,     │
│  Experience       ×  15%    exact      │
│  Education        ×  10%    matched)   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. SIGNALS MULTIPLIER (×0.50–1.20)    │
│     All 23 real redrob_signals fields   │
│     github_activity_score is a strong   │
│     signal for an engineering role      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
        final_score = base × multiplier × 10
                  │
                  ▼
        submission.csv (top-100 ranked)
```

---

## Scoring Methodology

### Title classification (exhaustive, not fuzzy)

Titles in this dataset were directly enumerated and classified into 4
tiers by inspection, rather than fuzzy keyword matching:

| Tier | Examples | Title score |
|---|---|---|
| A — Core AI/ML | ML Engineer, AI Specialist, Senior SE (ML), AI Research Engineer | 7.0–9.5 |
| B — Adjacent | Data Scientist, Data Engineer, Backend Engineer, Software Engineer | 3.0–4.5 |
| C — Weak technical | Java Developer, Frontend Engineer, QA Engineer, Mobile Developer | 1.5 |
| D — Non-technical | Marketing Manager, Accountant, HR Manager, Sales Executive | 0.5–1.0 |

### Honeypot detection (calibrated, not assumed)

The JD warns: *"A candidate who has all the AI keywords listed as skills
but whose title is 'Marketing Manager' is not a fit."*

Measured on real data: legitimate ML-titled candidates average ~4 AI
engineering signal hits (trained, deployed, fine-tuned, RAG, etc.) in their
`career_history` descriptions. Disqualifier-titled candidates who also list
AI skills average ~0–1. The honeypot filter uses this measured gap — a
candidate is only flagged if they have BOTH a disqualifier title AND list
AI skills AND show fewer than 2 supporting signals in their actual work
history. Plain irrelevant candidates (no AI skills listed at all) are not
"honeypots" — they're scored normally and naturally rank low.

### Skill scoring — proficiency and endorsement weighted

Each skill in `skills[]` has a `proficiency` (beginner/intermediate/
advanced/expert) and `endorsements` count. Must-have AI skills are weighted
by proficiency level, with a small endorsement boost, and cross-checked
against `career_history` text — skills with zero supporting work-history
mentions are penalized as likely list-padding.

### The 23 Redrob signals — real field names

| Signal | Effect |
|---|---|
| `open_to_work_flag` | +0.08 |
| `last_active_date` ≤7 days | +0.06 / >180 days: −0.12 |
| `recruiter_response_rate` ≥0.6 | +0.06 / <0.15: −0.08 |
| `github_activity_score` ≥70 | +0.07 (strong signal for engineering role) |
| `skill_assessment_scores` avg ≥70 on AI skills | +0.06 |
| `saved_by_recruiters_30d` ≥5 | +0.03 |
| `verified_email` + `verified_phone` | +0.02 |
| `notice_period_days` ≤15 | +0.03 |
| Clamp range | ×0.50 – ×1.20 |

---

## File Structure

```
├── rank.py                    ★ Main entry point — run this
├── scorer.py                  ★ All scoring logic (calibrated on real data)
├── jd.py                      ★ JD constants, exhaustive title/skill vocab
├── app.py                     Streamlit demo
├── sample_candidates.json     50 real candidates (sampled from actual data)
├── submission.csv             ★ Top-100 output from the real 100K dataset
└── requirements.txt
```

---

## Performance

| Metric | Value |
|---|---|
| 100K candidates | 14.9 seconds on CPU |
| API calls | 0 |
| Model downloads | 0 |
| Dependencies | numpy, pandas, streamlit, python-dateutil |

---

## Known limitations / honest caveats

- The skill-backing text match is exact-substring, so minor wording
  variants (e.g. "sentence-transformer" vs "sentence transformers") can
  miss the bonus — this only affects a secondary scoring nudge, not the
  primary ranking signal, and was spot-checked to not change top-100
  membership.
- Title classification covers the titles observed during calibration;
  any title outside this set falls through to a generic low-relevance
  default. If the leaderboard dataset has a meaningfully different title
  vocabulary, this should be re-verified.
- There is no ground-truth "good hire" label to validate against — scoring
  soundness was checked via distribution analysis and manual spot-checks
  of borderline cases, not against an oracle.
