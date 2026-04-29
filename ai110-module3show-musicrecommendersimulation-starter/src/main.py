"""
Command line runner for the Music Recommender Simulation.

Sections:
  1. Standard user profiles  (Weekend Vibes, Chill Lofi, Deep Intense Rock)
  2. Adversarial edge-case profiles
  3. [Challenge 1] Advanced-features profiles (decade, mood tags, popularity, instrumental)
  4. [Challenge 2] Scoring-mode comparison across four strategies
  5. [Challenge 3] Diversity-penalty before/after demo
  6. [Challenge 4] All output displayed in formatted ASCII tables via tabulate
  7. Original weight-shift experiment (preserved from baseline)
"""

import sys

# Ensure Unicode box-drawing characters (used by tabulate) survive Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

from .recommender import (
    load_songs,
    recommend_songs,
    apply_diversity_penalty,
    DEFAULT_WEIGHTS,
    SCORING_MODES,
    SCORING_MODE_DESCRIPTIONS,
)

# Experimental weights from the original weight-shift experiment
_ENERGY_DOUBLED = {"genre": 1.0, "mood": 1.0, "energy": 2.0, "acoustic": 0.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_score(weights: dict, user_prefs: dict = None) -> float:
    """Theoretical maximum score for the given weights plus any active advanced prefs."""
    base = weights["genre"] + weights["mood"] + weights["energy"] + weights["acoustic"]
    if user_prefs:
        if user_prefs.get("preferred_decade"):
            base += 0.25
        n_tags = min(len(user_prefs.get("desired_mood_tags") or []), 3)
        base += n_tags * 0.10
        if (user_prefs.get("popularity_target", -1) or -1) >= 0:
            base += 0.20
        if user_prefs.get("wants_instrumental"):
            base += 0.25
    return base


def _render_table(rows: list, headers: list) -> str:
    """Render a table with tabulate when available; fall back to plain text."""
    if _HAS_TABULATE:
        return _tabulate(rows, headers=headers, tablefmt="rounded_outline",
                         maxcolwidths=[None, 22, 16, 12, None, 52])
    # Fallback plain-text table
    lines = ["  ".join(f"{h:<{w}}" for h, w in zip(headers, [5, 22, 16, 12, 14, 52]))]
    lines.append("-" * 100)
    for row in rows:
        lines.append("  ".join(f"{str(c):<{w}}" for c, w in zip(row, [5, 22, 16, 12, 14, 52])))
    return "\n".join(lines)


def _print_table(
    label: str,
    user_prefs: dict,
    songs: list,
    weights: dict = None,
    use_diversity: bool = False,
    k: int = 5,
) -> None:
    """Score, optionally re-rank for diversity, and print results as a table."""
    w = weights if weights is not None else DEFAULT_WEIGHTS
    max_s = _max_score(w, user_prefs)

    # Fetch all songs when diversity is on so the greedy selector has a full candidate pool
    fetch_k = len(songs) if use_diversity else k
    results = recommend_songs(user_prefs, songs, k=fetch_k, weights=w)
    if use_diversity:
        results = apply_diversity_penalty(results, k=k)
        label = f"{label}  [diversity ON]"

    rows = []
    for rank, (song, score, explanation) in enumerate(results, start=1):
        rows.append([
            f"#{rank}",
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.2f} / {max_s:.2f}",
            explanation,
        ])

    print(f"\nProfile : {label}")
    pref_line = (
        f"Prefs   : genre={user_prefs['genre']} | mood={user_prefs['mood']} "
        f"| energy={user_prefs['energy']} | acoustic={user_prefs.get('likes_acoustic', False)}"
    )
    if user_prefs.get("preferred_decade"):
        pref_line += (
            f"\n          decade={user_prefs['preferred_decade']}"
            f" | tags={user_prefs.get('desired_mood_tags', [])}"
            f" | pop_target={user_prefs.get('popularity_target', -1)}"
            f" | instrumental={user_prefs.get('wants_instrumental', False)}"
        )
    print(pref_line)
    print(_render_table(rows, ["Rank", "Title", "Artist", "Genre", "Score", "Why"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from catalog.")
    print(f"[Challenge 4] Output formatted with {'tabulate' if _HAS_TABULATE else 'plain-text'} tables.\n")

    # ── Standard Profiles ────────────────────────────────────────────────────

    _print_table(
        "Weekend Vibes (High-Energy Pop)",
        {"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False},
        songs,
    )

    _print_table(
        "Chill Lofi",
        {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True},
        songs,
    )

    _print_table(
        "Deep Intense Rock",
        {"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
        songs,
    )

    # ── Adversarial / Edge-Case Profiles ─────────────────────────────────────

    _print_table(
        "Conflicting Prefs — High Energy + Sad Mood  [mood not in catalog]",
        {"genre": "pop", "mood": "sad", "energy": 0.90, "likes_acoustic": False},
        songs,
    )

    _print_table(
        "Acoustic Metal Head  [acoustic + metal contradiction]",
        {"genre": "metal", "mood": "angry", "energy": 0.97, "likes_acoustic": True},
        songs,
    )

    _print_table(
        "Classical Serenity  [only 1 classical song in catalog]",
        {"genre": "classical", "mood": "peaceful", "energy": 0.22, "likes_acoustic": True},
        songs,
    )

    # ── Challenge 1: Advanced Features Demo ──────────────────────────────────
    print("\n" + "#" * 70)
    print("  CHALLENGE 1 — ADVANCED FEATURES DEMO")
    print("  New scored fields: popularity, release_decade, mood_tags, instrumentalness")
    print("#" * 70)

    # Late-night study: lofi listener who wants 2020s atmospheric instrumentals
    _print_table(
        "Late-Night Study (lofi / chill / 2020s / calm+focused / instrumental)",
        {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.38,
            "likes_acoustic": True,
            "preferred_decade": "2020s",
            "desired_mood_tags": ["calm", "focused", "dreamy"],
            "popularity_target": 52,
            "wants_instrumental": True,
        },
        songs,
    )

    # Throwback aggressor: 2010s rock fan hunting for powerful, driving energy
    _print_table(
        "Throwback Aggressor (rock / intense / 2010s / powerful+driving)",
        {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.90,
            "likes_acoustic": False,
            "preferred_decade": "2010s",
            "desired_mood_tags": ["powerful", "driving", "aggressive"],
            "popularity_target": 62,
            "wants_instrumental": False,
        },
        songs,
    )

    # ── Challenge 2: Scoring Mode Comparison ─────────────────────────────────
    print("\n" + "#" * 70)
    print("  CHALLENGE 2 — SCORING MODE COMPARISON")
    print("  Same profile run through four different ranking strategies.")
    print("  All modes sum to max 4.5.  Watch how #2 and #3 shift.")
    print("#" * 70)

    weekend_prefs = {"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False}

    mode_summary_rows = []
    for mode_name, mode_weights in SCORING_MODES.items():
        top5 = recommend_songs(weekend_prefs, songs, k=5, weights=mode_weights)
        _, top_score, _ = top5[0]
        mode_max = mode_weights["genre"] + mode_weights["mood"] + mode_weights["energy"] + mode_weights["acoustic"]
        ranking = " → ".join(s["title"] for s, _, _ in top5[:3])
        mode_summary_rows.append([
            mode_name,
            f"{top_score:.2f}/{mode_max:.2f}",
            ranking,
            SCORING_MODE_DESCRIPTIONS[mode_name],
        ])

    print("\n  Summary — top-3 song order per mode:")
    print(_render_table(
        mode_summary_rows,
        ["Mode", "#1 Score", "Top-3 Ranking", "Strategy"],
    ))

    # Full top-5 tables per mode
    for mode_name, mode_weights in SCORING_MODES.items():
        _print_table(
            f"Weekend Vibes [{mode_name}]",
            weekend_prefs,
            songs,
            weights=mode_weights,
        )

    # ── Challenge 3: Diversity Penalty Demo ──────────────────────────────────
    print("\n" + "#" * 70)
    print("  CHALLENGE 3 — DIVERSITY PENALTY DEMO")
    print("  Lofi profile: LoRoom appears twice in the top 5 (#2 and #3).")
    print("  Penalty rules: 2nd+ same artist → score × 0.75")
    print("                 3rd+ same genre  → score − 0.50")
    print("  Expected change: Spacewalk Thoughts (ambient) moves from #4 to #3.")
    print("#" * 70)

    lofi_prefs = {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True}

    _print_table("Chill Lofi  [NO diversity penalty]", lofi_prefs, songs)
    _print_table("Chill Lofi  [WITH diversity penalty]", lofi_prefs, songs, use_diversity=True)

    # ── Original Weight-Shift Experiment (preserved) ─────────────────────────
    print("\n" + "#" * 70)
    print("  WEIGHT EXPERIMENT")
    print("  Base profile: Weekend Vibes (pop / happy / energy=0.80)")
    print("  Change: genre 2.0→1.0  |  energy 1.0→2.0  (max stays 4.5)")
    print("#" * 70)

    _print_table(
        "Weekend Vibes [standard weights: genre=2.0, energy=1.0]",
        {"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False},
        songs,
        weights=DEFAULT_WEIGHTS,
    )

    _print_table(
        "Weekend Vibes [energy-doubled: genre=1.0, energy=2.0]",
        {"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False},
        songs,
        weights=_ENERGY_DOUBLED,
    )


if __name__ == "__main__":
    main()
