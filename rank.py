#!/usr/bin/env python3
"""
rank.py — RecruiterAI Main Entry Point
========================================
Produces submission.csv in the EXACT format required by validate_submission.py:
  - Row 1: header = candidate_id,rank,score,reasoning
  - Rows 2–101: exactly 100 data rows
  - candidate_id format: CAND_XXXXXXX (7 digits)
  - rank: 1–100 (integer, no duplicates)
  - score: float 0.00–100.00
  - reasoning: plain-English string, no commas inside (CSV-safe)

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py --candidates ./sample_candidates.json --out ./test_out.csv
  python rank.py --help
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from scorer import score_candidate, ScoreBreakdown


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    """Loads from .jsonl (one JSON object per line) or .json (array)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    candidates = []
    suffix = p.suffix.lower()

    with open(p, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Line {line_num} skipped: {e}", file=sys.stderr)
        else:
            # .json — may be array or newline-delimited
            content = f.read().strip()
            if content.startswith("["):
                candidates = json.loads(content)
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    print(f"  Loaded {len(candidates):,} candidates from {p.name}")
    return candidates


def sanitize_reasoning(text: str) -> str:
    """
    Ensures reasoning is CSV-safe (no unescaped commas breaking columns).
    The csv.writer handles quoting automatically, but we also strip newlines.
    """
    return text.replace("\n", " ").replace("\r", " ").strip()


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────

def rank_candidates(candidates: list[dict], verbose: bool = False) -> list[ScoreBreakdown]:
    """
    Scores all candidates and returns them sorted by final_score descending.
    """
    results: list[ScoreBreakdown] = []
    honeypot_count = 0
    t0 = time.time()

    for i, candidate in enumerate(candidates):
        result = score_candidate(candidate)
        results.append(result)
        if result.is_honeypot:
            honeypot_count += 1
        if verbose and (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Scored {i+1:,}/{len(candidates):,} ({elapsed:.1f}s) — {honeypot_count} honeypots filtered")

    # Sort: non-honeypots first by score descending, then honeypots
    real = sorted([r for r in results if not r.is_honeypot], key=lambda r: -r.final_score)
    traps = [r for r in results if r.is_honeypot]

    elapsed = time.time() - t0
    print(f"  Scored {len(candidates):,} candidates in {elapsed:.1f}s")
    print(f"  Honeypots filtered: {honeypot_count}")
    print(f"  Viable candidates: {len(real)}")

    # Assign ranks
    for i, r in enumerate(real, 1):
        r.rank = i

    return real + traps


def write_submission(results: list[ScoreBreakdown], out_path: str) -> None:
    """
    Writes exactly 100 rows (top-100 by score) in the required CSV format.
    Header: candidate_id,rank,score,reasoning
    """
    top100 = [r for r in results if not r.is_honeypot][:100]

    if len(top100) < 100:
        print(f"  [WARN] Only {len(top100)} viable candidates — padding with lowest-scored honeypots", file=sys.stderr)
        # Fill with lowest-scoring honeypots to reach 100 rows
        traps = [r for r in results if r.is_honeypot]
        for i, trap in enumerate(traps[:100 - len(top100)], len(top100) + 1):
            trap.rank = i
            top100.append(trap)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top100[:100]:
            writer.writerow([
                r.candidate_id,
                r.rank,
                f"{r.final_score:.2f}",
                sanitize_reasoning(r.reasoning),
            ])

    print(f"  Written: {out_path}")


def print_top10(results: list[ScoreBreakdown]) -> None:
    top = [r for r in results if not r.is_honeypot][:10]
    print("\n  ── Top 10 Candidates ──────────────────────────────────")
    for r in top:
        bar = "█" * int(r.final_score / 5) + "░" * (20 - int(r.final_score / 5))
        print(f"  #{r.rank:3d} {bar} {r.final_score:5.1f}  {r.name[:30]:<30}  {r.current_title[:35]}")
    print()


def print_score_distribution(results: list[ScoreBreakdown]) -> None:
    real = [r for r in results if not r.is_honeypot]
    if not real:
        return

    scores = [r.final_score for r in real]
    bands = {
        "90–100": sum(1 for s in scores if s >= 90),
        "75–89":  sum(1 for s in scores if 75 <= s < 90),
        "50–74":  sum(1 for s in scores if 50 <= s < 75),
        "25–49":  sum(1 for s in scores if 25 <= s < 50),
        "0–24":   sum(1 for s in scores if s < 25),
    }
    print("  ── Score distribution ─────────────────────────────────")
    for band, count in bands.items():
        bar = "█" * min(40, count // max(1, len(scores) // 40))
        print(f"  {band:8s}  {bar:<40}  {count:,}")
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RecruiterAI — Rank candidates for the Redrob AI Engineer role"
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl or sample_candidates.json"
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress every 5000 candidates"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top candidates to print in summary (default: 10)"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  RecruiterAI — Redrob Intelligent Candidate Ranking  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Load
    print("[1/3] Loading candidates...")
    candidates = load_candidates(args.candidates)

    # Rank
    print(f"\n[2/3] Scoring {len(candidates):,} candidates...")
    results = rank_candidates(candidates, verbose=args.verbose)

    # Print summary
    print_top10(results)
    print_score_distribution(results)

    # Write output
    print(f"[3/3] Writing submission CSV...")
    write_submission(results, args.out)

    print("\n✓ Done. Run the validator:")
    print(f"  python validate_submission.py {args.out}\n")


if __name__ == "__main__":
    main()
