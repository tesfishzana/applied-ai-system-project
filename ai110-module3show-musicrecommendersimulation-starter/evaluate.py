#!/usr/bin/env python3
"""
VibeFinder AI — Evaluation & Test Harness  (Stretch Feature: +2 pts)

Runs the system on predefined inputs and prints a pass/fail summary with
confidence ratings. Two test layers:

  Layer 1 — Deterministic engine tests (no API key needed)
    Checks the scoring engine against known-good expected outputs.
    Confidence score = (score_1 - score_2) / 5.5  (margin of victory).

  Layer 2 — LLM quality checks  (--with-llm flag, requires ANTHROPIC_API_KEY)
    Sends natural-language queries to the full agent and evaluates:
      -Whether the expected song title appears in the reply
      -Whether a numeric score is cited  (pattern: digits.digits)
      -Whether off-topic queries are correctly deflected

Usage:
    python evaluate.py               # deterministic layer only
    python evaluate.py --with-llm    # both layers (costs API tokens)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender import load_songs, recommend_songs

_DATA = os.path.join(os.path.dirname(__file__), "data", "songs.csv")
_KB = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.md")
_MAX_SCORE = 5.5  # theoretical ceiling with all advanced prefs


# ──────────────────────────────────────────────────────────────────────────────
# Test case definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineCase:
    name: str
    prefs: dict
    expected_top: str               # exact title of the expected #1 song
    min_top_score: float            # score must be ≥ this
    min_confidence: float = 0.0     # (score_1 − score_2) / 5.5 must be ≥ this


ENGINE_CASES: list[EngineCase] = [
    EngineCase(
        name="Chill Lofi Study",
        prefs={"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
        expected_top="Library Rain",
        min_top_score=4.0,
        min_confidence=0.00,  # LoRoom nearly tied — low margin is expected
    ),
    EngineCase(
        name="Weekend Pop Vibes",
        prefs={"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False},
        expected_top="Sunrise City",
        min_top_score=3.5,
        min_confidence=0.15,
    ),
    EngineCase(
        name="Deep Rock Intensity",
        prefs={"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
        expected_top="Storm Runner",
        min_top_score=3.5,
        min_confidence=0.35,
    ),
    EngineCase(
        name="Classical Serenity",
        prefs={"genre": "classical", "mood": "peaceful", "energy": 0.22, "likes_acoustic": True},
        expected_top="Moonlit Sonata",
        min_top_score=4.0,
        min_confidence=0.50,
    ),
    EngineCase(
        name="Hip-Hop Gym",
        prefs={"genre": "hip-hop", "mood": "confident", "energy": 0.85, "likes_acoustic": False},
        expected_top="Corner Store Flex",
        min_top_score=3.0,
        min_confidence=0.10,
    ),
    EngineCase(
        name="Relaxed Jazz Sunday",
        prefs={"genre": "jazz", "mood": "relaxed", "energy": 0.37, "likes_acoustic": True},
        expected_top="Coffee Shop Stories",
        min_top_score=3.5,
        min_confidence=0.50,
    ),
    EngineCase(
        name="Metal Aggression",
        prefs={"genre": "metal", "mood": "angry", "energy": 0.97, "likes_acoustic": False},
        expected_top="Iron Collapse",
        min_top_score=3.5,
        min_confidence=0.50,
    ),
    EngineCase(
        name="Late-Night Study (Advanced)",
        prefs={
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.38,
            "likes_acoustic": True,
            "preferred_decade": "2020s",
            "desired_mood_tags": ["calm", "focused", "dreamy"],
            "popularity_target": 52,
            "wants_instrumental": True,
        },
        expected_top="Midnight Coding",
        min_top_score=5.0,
        min_confidence=0.01,
    ),
    EngineCase(
        name="Unknown genre (guardrail)",
        prefs={"genre": "polka", "mood": "happy", "energy": 0.5, "likes_acoustic": False},
        # No genre bonus: top result is the happy-mood song with energy closest to 0.5
        # Rooftop Lights: mood(+1.0) + energy(+0.74, gap=0.26) = 1.74
        expected_top="Rooftop Lights",
        min_top_score=0.5,
        min_confidence=0.00,
    ),
]


@dataclass
class LLMCase:
    query: str
    expected_mentions: list[str] = field(default_factory=list)
    expect_score_cited: bool = True
    expect_refusal: bool = False


LLM_CASES: list[LLMCase] = [
    LLMCase(
        query="Something chill for studying late at night",
        expected_mentions=["Library Rain", "Midnight Coding"],
        expect_score_cited=True,
    ),
    LLMCase(
        query="I need maximum energy for the gym",
        expected_mentions=["Iron Collapse", "Storm Runner"],
        expect_score_cited=True,
    ),
    LLMCase(
        query="Explain why the first song scored so high",
        expected_mentions=[],    # can't predict context; just check score is cited
        expect_score_cited=True,
    ),
    LLMCase(
        query="What's the weather like today?",
        expected_mentions=[],
        expect_score_cited=False,
        expect_refusal=True,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 8) -> str:
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


def _pad(s: str, w: int) -> str:
    s = str(s)
    return s[:w].ljust(w)


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1 — Deterministic engine tests
# ──────────────────────────────────────────────────────────────────────────────

def run_engine_layer(songs: list[dict]) -> list[dict]:
    results = []
    for tc in ENGINE_CASES:
        recs = recommend_songs(tc.prefs, songs, k=len(songs))
        if not recs:
            results.append({
                "name": tc.name, "passed": False,
                "top_ok": False, "score": 0.0, "confidence": 0.0,
                "got": "(no results)", "note": "empty",
            })
            continue

        top_song, top_score, _ = recs[0]
        score_2 = recs[1][1] if len(recs) > 1 else 0.0
        confidence = (top_score - score_2) / _MAX_SCORE

        top_ok = top_song["title"] == tc.expected_top
        score_ok = top_score >= tc.min_top_score
        conf_ok = confidence >= tc.min_confidence
        passed = top_ok and score_ok and conf_ok

        results.append({
            "name": tc.name,
            "passed": passed,
            "top_ok": top_ok,
            "score_ok": score_ok,
            "conf_ok": conf_ok,
            "score": top_score,
            "confidence": confidence,
            "got": top_song["title"],
            "expected": tc.expected_top,
        })
    return results


def print_engine_report(results: list[dict]) -> None:
    print()
    print("=" * 74)
    print("  VIBEFINDER AI — DETERMINISTIC ENGINE EVALUATION")
    print("=" * 74)
    print(f"  {'Test Case':<30}  {'Status':<6}  {'Top Song':<22}  {'Score':>5}  {'Confidence':>10}")
    print("  " + "-" * 70)
    passed = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        conf_display = f"{r['confidence']:.2f} {_bar(r['confidence'], 5)}"
        print(
            f"  {_pad(r['name'], 30)}  {status:<6}  "
            f"{_pad(r['got'], 22)}  {r['score']:>5.2f}  {conf_display:>10}"
        )
        if r["passed"]:
            passed += 1
    print("  " + "-" * 70)

    n = len(results)
    avg_conf = sum(r["confidence"] for r in results) / n
    print(f"\n  RESULT  : {passed}/{n} passed")
    print(f"  Avg confidence margin : {avg_conf:.2f}  ({avg_conf * 100:.0f}% of max score gap)")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\n  Failed cases:")
        for r in fails:
            reasons = []
            if not r.get("top_ok"):
                reasons.append(f"expected '{r['expected']}', got '{r['got']}'")
            if not r.get("score_ok"):
                reasons.append(f"score {r['score']:.2f} too low")
            if not r.get("conf_ok"):
                reasons.append(f"confidence {r['confidence']:.2f} too low")
            print(f"    -{r['name']}: {'; '.join(reasons)}")
    else:
        print("\n  OK All deterministic engine tests passed.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2 — LLM quality checks
# ──────────────────────────────────────────────────────────────────────────────

def run_llm_layer() -> None:
    from src.logger import setup_logging
    from src.llm_agent import MusicAgent

    setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    kb = _KB if os.path.exists(_KB) else None
    agent = MusicAgent(_DATA, knowledge_path=kb)

    print("=" * 74)
    print("  VIBEFINDER AI — LLM QUALITY CHECKS")
    print("=" * 74)

    llm_passed = 0
    for i, case in enumerate(LLM_CASES, 1):
        print(f"\n  [{i}/{len(LLM_CASES)}] {case.query}")
        try:
            reply = agent.chat(case.query)
            steps = agent.reasoning_steps
            agent.reset()

            mentions_ok = all(m.lower() in reply.lower() for m in case.expected_mentions)
            score_cited = bool(re.search(r"\d\.\d{2}", reply))
            score_ok = not case.expect_score_cited or score_cited
            refusal_ok = not case.expect_refusal or any(
                w in reply.lower()
                for w in ["music", "recommend", "only", "can't", "cannot", "outside", "vibe"]
            )
            ok = mentions_ok and score_ok and refusal_ok

            status = "PASS" if ok else "FAIL"
            print(f"  Status         : {status}")
            print(f"  Tool steps     : {len(steps)} ({', '.join(s['tool'] for s in steps)})")
            print(f"  Mentions check : {'OK' if mentions_ok else 'XX'}  {case.expected_mentions}")
            print(f"  Score cited    : {'OK' if score_cited else 'XX'}")
            print(f"  Refusal check  : {'OK' if refusal_ok else 'XX'}")
            print(f"  Reply preview  : {reply[:150].strip()}{'…' if len(reply) > 150 else ''}")

            if ok:
                llm_passed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n  LLM RESULT: {llm_passed}/{len(LLM_CASES)} passed")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeFinder AI evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py              # deterministic engine only\n"
            "  python evaluate.py --with-llm   # full pipeline (needs API key)\n"
        ),
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run LLM quality checks (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    songs = load_songs(_DATA)
    results = run_engine_layer(songs)
    print_engine_report(results)

    if args.with_llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set. Cannot run LLM layer.\n")
            sys.exit(1)
        run_llm_layer()

    # Exit with non-zero if any engine test failed
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
