"""
scorer.py — RecruiterAI Hybrid Scoring Engine (v3 — calibrated on real data)
===============================================================================
Rewritten against the ACTUAL schema found in candidates.jsonl:

  {
    "candidate_id": "CAND_0000001",
    "profile": { anonymized_name, headline, summary, location, country,
                 years_of_experience, current_title, current_company,
                 current_company_size, current_industry },
    "career_history": [ {company, title, start_date, end_date,
                          duration_months, is_current, industry,
                          company_size, description}, ... ],
    "education": [ {institution, degree, field_of_study, start_year,
                     end_year, grade, tier}, ... ],
    "skills": [ {name, proficiency, endorsements, duration_months}, ... ],
    "certifications": [...],
    "languages": [...],
    "redrob_signals": { ...23 fields... }
  }

Scoring components (unchanged weights, recalibrated logic):
  1. Honeypot filter    — title/headline/skill-backing mismatch detection
  2. Title score        — pattern match against real title vocabulary
  3. Skill score        — must-have/nice-to-have, weighted by proficiency +
                           endorsements, backed by career_history text
  4. Work depth score   — AI signal density in actual job descriptions
  5. Experience score    — years calibrated to JD's 3–8yr target + trajectory
  6. Education score     — tier_1..tier_4 + field relevance
  7. Signals multiplier — all 23 real redrob_signals fields
  8. Final combine       — weighted sum × multiplier → 0–100
"""

from __future__ import annotations
import json
import re
from datetime import datetime, date
from dataclasses import dataclass

from jd import (
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    STRONG_TITLE_PATTERNS,
    ADJACENT_TITLE_PATTERNS,
    WEAK_TECHNICAL_TITLES,
    DISQUALIFIER_TITLE_PATTERNS,
    HONEYPOT_HEADLINE_PHRASES,
    WORK_DESCRIPTION_KEYWORDS,
    STRONG_AI_COMPANIES,
    TOP_EDUCATION_TIER,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _lower(v) -> str:
    return "" if v is None else str(v).lower()


def _contains_any(text: str, terms) -> list[str]:
    return [t for t in terms if t in text]


def _profile(c: dict) -> dict:
    return c.get("profile", {}) or {}


def _career(c: dict) -> list[dict]:
    return c.get("career_history", []) or []


def _skills(c: dict) -> list[dict]:
    return c.get("skills", []) or []


def _education(c: dict) -> list[dict]:
    return c.get("education", []) or []


def _signals(c: dict) -> dict:
    return c.get("redrob_signals", {}) or {}


def _skill_names_text(c: dict) -> str:
    return " ".join(_lower(s.get("name", "")) for s in _skills(c))


def _career_text(c: dict) -> str:
    parts = []
    for job in _career(c):
        parts.append(_lower(job.get("title", "")))
        parts.append(_lower(job.get("company", "")))
        parts.append(_lower(job.get("description", "")))
    return " ".join(parts)


# ─────────────────────────────────────────────
# 1. Honeypot detection (calibrated on real distribution)
# ─────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Real-data calibration showed two distinct populations among
    disqualifier-titled candidates:
      1. Plain irrelevant candidates (Marketing Manager, no AI skills at all)
         — these are just not a fit, not a "trap."
      2. True honeypots: disqualifier title BUT skills list stuffed with
         AI keywords (RAG, Fine-tuning LLMs, etc.) despite zero supporting
         work history. These are the dataset's explicit trap, per the JD's
         own warning. Headlines often self-identify as "Exploring AI &
         GenAI applications".

    Only population (2) is filtered with a "honeypot" label and a strong
    score penalty distinct from ordinary low relevance — both end up with
    a low final score, but the reasoning differs for transparency.
    """
    profile = _profile(candidate)
    title = _lower(profile.get("current_title", ""))
    headline = _lower(profile.get("headline", ""))
    title_text = title + " " + headline

    disq_hits = [p for p in DISQUALIFIER_TITLE_PATTERNS if p in title_text]
    if not disq_hits:
        return False, ""

    # Has a disqualifier title. Check if skills list is stuffed with AI terms.
    skill_text = _skill_names_text(candidate)
    ai_skill_hits = _contains_any(skill_text, MUST_HAVE_SKILLS)

    if not ai_skill_hits:
        # Plain irrelevant candidate — not a honeypot, just not a fit.
        # Handled by the normal low title score, no special filtering needed.
        return False, ""

    # Has disqualifier title AND lists AI skills — check work history for
    # real substance before calling it a trap
    career_text = _career_text(candidate)
    work_ai_hits = _contains_any(career_text, WORK_DESCRIPTION_KEYWORDS)
    headline_tell = _contains_any(headline, HONEYPOT_HEADLINE_PHRASES)

    if len(work_ai_hits) < 2:
        reason = (
            f"Title '{profile.get('current_title','')}' is non-AI ({disq_hits[0]}) "
            f"but lists {len(ai_skill_hits)} AI keywords as skills with only "
            f"{len(work_ai_hits)} supporting signals in career history — "
            f"likely keyword-stuffing"
        )
        if headline_tell:
            reason += f"; headline self-identifies as exploratory ('{headline_tell[0]}')"
        return True, reason

    return False, ""


# ─────────────────────────────────────────────
# 2. Title score (0–10)
# ─────────────────────────────────────────────

def score_title(candidate: dict) -> tuple[float, str]:
    profile = _profile(candidate)
    title_raw = _lower(profile.get("current_title", ""))
    headline = _lower(profile.get("headline", ""))
    combined = title_raw + " " + headline

    strong = [p for p in STRONG_TITLE_PATTERNS if p in combined]
    if strong:
        bonus = 0.0
        if any(w in combined for w in ["senior", "staff", "principal", "lead", "founding"]):
            bonus = 1.5
        elif any(w in combined for w in ["junior", "intern", "trainee"]):
            bonus = -1.0  # junior ML engineer is still real signal, smaller penalty
        score = min(10.0, 8.0 + bonus)
        return score, f"Strong AI/ML title: '{profile.get('current_title','')}'"

    adjacent = [p for p in ADJACENT_TITLE_PATTERNS if p in combined]
    if adjacent:
        score = 4.5 if "data scientist" in combined else 3.0
        return score, f"Adjacent role to AI engineering: {adjacent[0]}"

    weak = [p for p in WEAK_TECHNICAL_TITLES if p in combined]
    if weak:
        return 1.5, f"Technical but not ML-adjacent: {weak[0]}"

    disq = [p for p in DISQUALIFIER_TITLE_PATTERNS if p in combined]
    if disq:
        return 0.5, f"Non-technical title: {disq[0]}"

    return 1.0, f"Title '{profile.get('current_title','')}' shows no AI/ML relevance"


# ─────────────────────────────────────────────
# 3. Skill score (0–10)
# ─────────────────────────────────────────────

def score_skills(candidate: dict) -> tuple[float, str]:
    skills_list = _skills(candidate)
    career_text = _career_text(candidate)

    if not skills_list:
        return 0.0, "No skills listed"

    must_matched = []
    nice_matched = []
    proficiency_weight = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.25}

    weighted_must_score = 0.0
    weighted_nice_score = 0.0
    backed_count = 0

    for s in skills_list:
        name = _lower(s.get("name", ""))
        prof = s.get("proficiency", "intermediate")
        endorsements = s.get("endorsements", 0) or 0
        w = proficiency_weight.get(prof, 0.5)
        # small endorsement boost, capped
        endorsement_boost = min(0.2, (endorsements or 0) / 100)

        if name in MUST_HAVE_SKILLS:
            must_matched.append(name)
            weighted_must_score += w + endorsement_boost
            if name in career_text:
                backed_count += 1
        elif name in NICE_TO_HAVE_SKILLS:
            nice_matched.append(name)
            weighted_nice_score += (w + endorsement_boost) * 0.5

    # Normalize: cap contribution from must-haves and nice-to-haves
    must_component = min(7.0, weighted_must_score * 1.3)
    nice_component = min(2.0, weighted_nice_score * 0.8)

    base = must_component + nice_component

    # Skill-backing adjustment: must-have skills should show up in actual
    # work descriptions, not just be listed
    if must_matched:
        backing_ratio = backed_count / len(must_matched)
        if backing_ratio >= 0.4:
            base += 1.0
        elif backing_ratio == 0:
            base -= 1.5  # all skills listed, none backed by work — red flag

    score = round(min(10.0, max(0.0, base)), 2)

    reason = (
        f"{len(must_matched)} must-have AI skills ({backed_count} backed by work history), "
        f"{len(nice_matched)} nice-to-have. "
        f"Top matches: {', '.join(must_matched[:3]) if must_matched else 'none'}"
    )
    return score, reason


# ─────────────────────────────────────────────
# 4. Work depth score (0–10)
# ─────────────────────────────────────────────

PRODUCTION_KEYWORDS = [
    "production", "deployed", "served", "shipped", "live", "users",
    "scale", "latency", "throughput", "real-time", "real time", "uptime",
]


def score_work_depth(candidate: dict) -> tuple[float, str]:
    jobs = _career(candidate)
    if not jobs:
        return 0.0, "No career history"

    total_ai_signals = 0
    production_signals = 0
    ai_company_count = 0
    description_depth = 0

    for job in jobs[:6]:
        title = _lower(job.get("title", ""))
        company = _lower(job.get("company", ""))
        desc = _lower(job.get("description", ""))

        if any(c in company for c in STRONG_AI_COMPANIES):
            ai_company_count += 1

        role_ai = len(_contains_any(desc + " " + title, WORK_DESCRIPTION_KEYWORDS))
        total_ai_signals += role_ai

        production_signals += len(_contains_any(desc, PRODUCTION_KEYWORDS))

        if len(desc) > 200:
            description_depth += 2
        elif len(desc) > 80:
            description_depth += 1

    # Calibrated against real data: ~4 hits = strong legit signal
    score = 0.0
    score += min(5.0, total_ai_signals * 0.9)       # up to 5 pts for AI signal density
    score += min(2.5, production_signals * 0.5)      # up to 2.5 pts for production proof
    score += min(1.5, ai_company_count * 0.8)         # up to 1.5 pts for strong companies
    score += min(1.0, description_depth * 0.15)       # up to 1 pt for description richness

    score = round(min(10.0, score), 2)
    reason = (
        f"{total_ai_signals} AI engineering signals across career history, "
        f"{production_signals} production/deployment indicators, "
        f"{ai_company_count} recognized AI-forward company"
    )
    return score, reason


# ─────────────────────────────────────────────
# 5. Experience score (0–10)
# ─────────────────────────────────────────────

def score_experience(candidate: dict) -> tuple[float, str]:
    profile = _profile(candidate)
    years = profile.get("years_of_experience", 0) or 0
    try:
        years = float(years)
    except (ValueError, TypeError):
        years = 0.0

    jobs = _career(candidate)

    if years < 1:
        base, note = 2.0, "<1 year (very junior)"
    elif years < 3:
        base, note = 4.5, f"{years:.1f} years (below JD target of 3-8)"
    elif years <= 5:
        base, note = 9.0, f"{years:.1f} years (ideal for JD target 3-8)"
    elif years <= 8:
        base, note = 8.5, f"{years:.1f} years (good, upper end of JD target)"
    elif years <= 12:
        base, note = 6.5, f"{years:.1f} years (overqualified for IC founding role)"
    else:
        base, note = 5.0, f"{years:.1f} years (significantly overqualified)"

    # Career progression: did titles grow more senior over time?
    if len(jobs) >= 2:
        titles = [_lower(j.get("title", "")) for j in jobs]
        seniority_words = ["senior", "lead", "staff", "principal", "head", "founding"]
        if any(w in t for t in titles[:2] for w in seniority_words):  # most recent jobs first typically
            base = min(10.0, base + 0.5)
            note += ", growing seniority"

    # Job-hopping penalty: many very short stints
    short_stints = sum(1 for j in jobs if (j.get("duration_months") or 99) < 8)
    if len(jobs) >= 3 and short_stints >= 2:
        base = max(0.0, base - 1.0)
        note += ", job-hopping pattern"

    return round(min(10.0, base), 2), note


# ─────────────────────────────────────────────
# 6. Education score (0–10)
# ─────────────────────────────────────────────

TIER_SCORES = {"tier_1": 9.0, "tier_2": 7.0, "tier_3": 5.5, "tier_4": 4.0}


def score_education(candidate: dict) -> tuple[float, str]:
    edu_list = _education(candidate)
    if not edu_list:
        return 4.0, "No education data"

    # Use highest-tier entry
    best = None
    best_score = -1
    for e in edu_list:
        tier = e.get("tier", "tier_4")
        s = TIER_SCORES.get(tier, 4.0)
        if s > best_score:
            best_score = s
            best = e

    score = best_score
    field = _lower(best.get("field_of_study", ""))
    degree = _lower(best.get("degree", ""))

    note = f"{best.get('degree','')} {best.get('field_of_study','')} ({best.get('tier','unknown')})"

    if any(w in degree for w in ["phd", "ph.d", "doctorate"]):
        score = min(10.0, score + 1.0)
        note += ", doctoral"
    elif any(w in degree for w in ["m.tech", "mtech", "m.s.", "ms", "msc", "m.sc", "master"]):
        score = min(10.0, score + 0.5)

    if any(w in field for w in ["computer science", "artificial intelligence", "machine learning",
                                  "data science", "electronics", "electrical", "statistics", "mathematics"]):
        score = min(10.0, score + 0.5)
        note += ", relevant field"

    return round(score, 2), note


# ─────────────────────────────────────────────
# 7. Redrob signals multiplier — real field names from actual schema
# ─────────────────────────────────────────────

def _days_since(date_str: str) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except (ValueError, TypeError):
        return None


def compute_signals_multiplier(candidate: dict) -> tuple[float, str]:
    sig = _signals(candidate)
    if not sig:
        return 1.0, "No Redrob signals available (neutral multiplier)"

    multiplier = 1.0
    notes = []

    # ── Availability ──
    if sig.get("open_to_work_flag", False):
        multiplier += 0.08
        notes.append("open to work")

    last_active_days = _days_since(sig.get("last_active_date"))
    if last_active_days is not None:
        if last_active_days <= 7:
            multiplier += 0.06
            notes.append("active <7d ago")
        elif last_active_days <= 30:
            multiplier += 0.02
        elif last_active_days > 180:
            multiplier -= 0.12
            notes.append("inactive 180+ days")
        elif last_active_days > 90:
            multiplier -= 0.06

    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        try:
            rr = float(rr)
            if rr >= 0.6:
                multiplier += 0.06
                notes.append(f"response rate {rr*100:.0f}%")
            elif rr < 0.15:
                multiplier -= 0.08
                notes.append(f"low response {rr*100:.0f}%")
        except (ValueError, TypeError):
            pass

    avg_resp_hrs = sig.get("avg_response_time_hours")
    if avg_resp_hrs is not None:
        try:
            h = float(avg_resp_hrs)
            if h <= 24:
                multiplier += 0.03
            elif h > 240:
                multiplier -= 0.03
        except (ValueError, TypeError):
            pass

    # ── Profile quality ──
    pcs = sig.get("profile_completeness_score")
    if pcs is not None:
        try:
            pcs = float(pcs)
            if pcs >= 85:
                multiplier += 0.04
            elif pcs < 40:
                multiplier -= 0.05
        except (ValueError, TypeError):
            pass

    # ── GitHub activity — strong signal for an engineering role ──
    gh = sig.get("github_activity_score")
    if gh is not None:
        try:
            gh = float(gh)
            if gh >= 70:
                multiplier += 0.07
                notes.append(f"GitHub activity {gh:.0f}")
            elif gh >= 40:
                multiplier += 0.03
            elif gh < 10:
                multiplier -= 0.02
        except (ValueError, TypeError):
            pass

    # ── Recruiter interest signals ──
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    try:
        if int(saved) >= 5:
            multiplier += 0.03
            notes.append(f"saved by {saved} recruiters")
    except (ValueError, TypeError):
        pass

    # ── Interview / offer track record ──
    icr = sig.get("interview_completion_rate")
    if icr is not None:
        try:
            icr = float(icr)
            if icr >= 0.7:
                multiplier += 0.03
            elif icr < 0.2:
                multiplier -= 0.03
        except (ValueError, TypeError):
            pass

    oar = sig.get("offer_acceptance_rate")
    if oar is not None:
        try:
            oar = float(oar)
            if oar >= 0.6:
                multiplier += 0.02
        except (ValueError, TypeError):
            pass

    # ── Skill assessment scores — real proof of competence ──
    assessments = sig.get("skill_assessment_scores", {}) or {}
    if assessments:
        ai_assessments = {
            k: v for k, v in assessments.items()
            if any(kw in _lower(k) for kw in ["nlp", "llm", "ml", "ai", "image", "speech", "fine"])
        }
        if ai_assessments:
            try:
                avg = sum(float(v) for v in ai_assessments.values()) / len(ai_assessments)
                if avg >= 70:
                    multiplier += 0.06
                    notes.append(f"AI skill assessment avg {avg:.0f}%")
                elif avg < 30:
                    multiplier -= 0.06
                    notes.append(f"weak skill assessment {avg:.0f}%")
            except (ValueError, TypeError):
                pass
    else:
        # No assessment scores at all — small neutral-to-negative signal
        # (not heavily penalized since many legit candidates also lack these)
        multiplier -= 0.01

    # ── Verification & trust ──
    if sig.get("verified_email") and sig.get("verified_phone"):
        multiplier += 0.02
    if sig.get("linkedin_connected"):
        multiplier += 0.01

    # ── Notice period ──
    notice = sig.get("notice_period_days")
    if notice is not None:
        try:
            nd = int(notice)
            if nd <= 15:
                multiplier += 0.03
                notes.append(f"notice {nd}d")
            elif nd > 75:
                multiplier -= 0.02
        except (ValueError, TypeError):
            pass

    multiplier = round(max(0.50, min(1.20, multiplier)), 3)
    notes_str = ", ".join(notes) if notes else "no notable signals"
    return multiplier, f"Signals ×{multiplier:.2f}: {notes_str}"


# ─────────────────────────────────────────────
# 8. Final scoring combiner
# ─────────────────────────────────────────────

WEIGHTS = {
    "title":      0.20,
    "skills":     0.25,
    "work_depth": 0.30,
    "experience": 0.15,
    "education":  0.10,
}


@dataclass
class ScoreBreakdown:
    candidate_id: str
    name: str
    current_title: str
    is_honeypot: bool
    honeypot_reason: str

    title_score: float = 0.0
    title_note: str = ""
    skills_score: float = 0.0
    skills_note: str = ""
    work_depth_score: float = 0.0
    work_depth_note: str = ""
    experience_score: float = 0.0
    experience_note: str = ""
    education_score: float = 0.0
    education_note: str = ""
    signals_multiplier: float = 1.0
    signals_note: str = ""

    base_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    reasoning: str = ""


def score_candidate(candidate: dict) -> ScoreBreakdown:
    profile = _profile(candidate)
    cid = str(candidate.get("candidate_id", "UNKNOWN"))
    name = str(profile.get("anonymized_name", "Unknown"))
    title = str(profile.get("current_title", ""))

    result = ScoreBreakdown(
        candidate_id=cid, name=name, current_title=title,
        is_honeypot=False, honeypot_reason="",
    )

    hp, hp_reason = is_honeypot(candidate)
    result.is_honeypot = hp
    result.honeypot_reason = hp_reason
    if hp:
        result.final_score = 0.0
        result.reasoning = f"Filtered: {hp_reason}"
        return result

    result.title_score, result.title_note = score_title(candidate)
    result.skills_score, result.skills_note = score_skills(candidate)
    result.work_depth_score, result.work_depth_note = score_work_depth(candidate)
    result.experience_score, result.experience_note = score_experience(candidate)
    result.education_score, result.education_note = score_education(candidate)
    result.signals_multiplier, result.signals_note = compute_signals_multiplier(candidate)

    base = (
        result.title_score      * WEIGHTS["title"] +
        result.skills_score     * WEIGHTS["skills"] +
        result.work_depth_score * WEIGHTS["work_depth"] +
        result.experience_score * WEIGHTS["experience"] +
        result.education_score  * WEIGHTS["education"]
    )
    result.base_score = round(base, 4)
    result.final_score = round(min(100.0, base * result.signals_multiplier * 10), 2)

    strengths, concerns = [], []
    if result.title_score >= 7:
        strengths.append(result.title_note)
    elif result.title_score < 4:
        concerns.append(result.title_note)

    if result.skills_score >= 6:
        strengths.append(result.skills_note.split(".")[0])
    elif result.skills_score < 3:
        concerns.append("Limited must-have AI skill coverage")

    if result.work_depth_score >= 5:
        strengths.append(result.work_depth_note)
    elif result.work_depth_score < 2:
        concerns.append("Work history shows little AI engineering depth")

    if result.experience_score >= 7:
        strengths.append(f"Experience: {result.experience_note}")
    elif result.experience_score < 4:
        concerns.append(f"Experience: {result.experience_note}")

    if result.signals_multiplier >= 1.08:
        strengths.append(result.signals_note)
    elif result.signals_multiplier <= 0.85:
        concerns.append(result.signals_note)

    strength_str = "; ".join(strengths[:2]) if strengths else "No strong positives identified"
    concern_str = concerns[0] if concerns else "No major concerns"

    result.reasoning = f"Score {result.final_score:.1f}/100. Strengths: {strength_str}. Concern: {concern_str}."
    return result
