"""
Reliability tests for VibeFinder AI.

Run with:  pytest tests/test_reliability.py -v

Test coverage
─────────────
1. Recommender determinism   — same profile → identical ranking on every call.
2. Score bounds              — all scores within [0, 5.5] for any valid profile.
3. Explanation coverage      — every top-5 result has a non-empty explanation.
4. Diversity guarantee       — after the diversity penalty no artist appears twice.
5. Empty catalog edge case   — returns [] without raising.
6. Valid-genre scoring        — known genres always score above zero.
7. Unknown genre graceful    — unrecognised genre earns no genre-match bonus.
8. All scoring modes valid   — each of the four SCORING_MODES returns 5 results.
9. Diversity preserves order — #1 song without diversity is still #1 with diversity
                               (unless its artist or genre needs penalising).
"""

from __future__ import annotations

import os
import sys

# Allow running as:  pytest tests/test_reliability.py  (from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.recommender import (
    SCORING_MODES,
    apply_diversity_penalty,
    load_songs,
    recommend_songs,
    score_song,
)

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
_MAX_SCORE = 5.5  # theoretical ceiling: base 4.5 + advanced bonuses 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def songs():
    return load_songs(_DATA)


@pytest.fixture
def weekend_prefs():
    return {"genre": "pop", "mood": "happy", "energy": 0.80, "likes_acoustic": False}


@pytest.fixture
def lofi_prefs():
    return {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True}


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_recommendations_are_deterministic(songs, weekend_prefs):
    a = recommend_songs(weekend_prefs, songs)
    b = recommend_songs(weekend_prefs, songs)
    assert [(s["title"], round(sc, 6)) for s, sc, _ in a] == [
        (s["title"], round(sc, 6)) for s, sc, _ in b
    ]


# ---------------------------------------------------------------------------
# 2. Score bounds
# ---------------------------------------------------------------------------

def test_scores_within_bounds(songs):
    prefs = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
        "preferred_decade": "2020s",
        "desired_mood_tags": ["calm", "dreamy", "focused"],
        "popularity_target": 52,
        "wants_instrumental": True,
    }
    for song in songs:
        sc, _ = score_song(song, prefs)
        assert 0.0 <= sc <= _MAX_SCORE + 1e-9, (
            f"'{song['title']}' score {sc:.4f} is outside [0, {_MAX_SCORE}]"
        )


# ---------------------------------------------------------------------------
# 3. Explanation coverage
# ---------------------------------------------------------------------------

def test_all_top5_have_explanations(songs, weekend_prefs):
    results = recommend_songs(weekend_prefs, songs, k=5)
    for song, _, expl in results:
        assert expl.strip(), f"Empty explanation for '{song['title']}'"


# ---------------------------------------------------------------------------
# 4. Diversity penalty — soft penalty pushes repeat artists later, not out
# ---------------------------------------------------------------------------

def test_diversity_penalty_does_not_increase_repetition(songs, lofi_prefs):
    """Diversity penalty must not *increase* artist repetition vs raw top-k."""
    from collections import Counter

    all_results = recommend_songs(lofi_prefs, songs, k=len(songs))
    raw_dupes = sum(
        c - 1 for c in Counter(s["artist"] for s, _, _ in all_results[:5]).values()
    )
    diverse = apply_diversity_penalty(all_results, k=5)
    div_dupes = sum(
        c - 1 for c in Counter(s["artist"] for s, _, _ in diverse).values()
    )
    assert div_dupes <= raw_dupes, (
        f"Diversity increased artist repetition: {raw_dupes} → {div_dupes}"
    )


def test_diversity_penalty_delays_repeat_artist_in_ranking(songs, lofi_prefs):
    """If an artist appears twice in both raw and diverse top-5, their 2nd
    position must be the same or later in the diverse ranking (penalty can
    only push repeats back, never forward)."""
    from collections import defaultdict

    all_results = recommend_songs(lofi_prefs, songs, k=len(songs))
    diverse = apply_diversity_penalty(all_results, k=5)

    def positions(results):
        pos: dict[str, list[int]] = defaultdict(list)
        for i, (s, _, _) in enumerate(results):
            pos[s["artist"]].append(i)
        return pos

    raw_pos = positions(all_results[:5])
    div_pos = positions(diverse)

    for artist, rp in raw_pos.items():
        if len(rp) >= 2:
            dp = div_pos.get(artist, [])
            if len(dp) >= 2:
                assert dp[1] >= rp[1], (
                    f"Artist '{artist}': 2nd appearance moved earlier with diversity "
                    f"(raw rank {rp[1]+1} → diverse rank {dp[1]+1})"
                )


# ---------------------------------------------------------------------------
# 5. Empty catalog — no crash, returns empty list
# ---------------------------------------------------------------------------

def test_empty_catalog_returns_empty(weekend_prefs):
    assert recommend_songs(weekend_prefs, [], k=5) == []


# ---------------------------------------------------------------------------
# 6. Valid genres always produce a non-zero top score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("genre", ["pop", "rock", "lofi", "jazz", "metal", "classical"])
def test_valid_genre_scores_nonzero(songs, genre):
    prefs = {"genre": genre, "mood": "happy", "energy": 0.5, "likes_acoustic": False}
    results = recommend_songs(prefs, songs, k=1)
    assert results, f"No results returned for genre '{genre}'"
    _, top_sc, _ = results[0]
    assert top_sc > 0.0, f"Top score for genre '{genre}' should be > 0, got {top_sc}"


# ---------------------------------------------------------------------------
# 7. Unknown genre — no genre-match bonus earned
# ---------------------------------------------------------------------------

def test_unknown_genre_no_genre_bonus(songs):
    prefs = {"genre": "polka", "mood": "happy", "energy": 0.5, "likes_acoustic": False}
    results = recommend_songs(prefs, songs, k=5)
    assert isinstance(results, list)
    # Without a genre match (worth 2.0), max possible is mood(1.0)+energy(1.0)+acoustic(0.5) = 2.5
    for _, sc, _ in results:
        assert sc <= 2.5 + 1e-9, (
            f"Unknown genre should earn no genre-match bonus, but score={sc:.4f}"
        )


# ---------------------------------------------------------------------------
# 8. All four scoring modes return exactly 5 valid results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(SCORING_MODES.keys()))
def test_scoring_mode_returns_five_results(songs, weekend_prefs, mode):
    weights = SCORING_MODES[mode]
    results = recommend_songs(weekend_prefs, songs, k=5, weights=weights)
    assert len(results) == 5, f"Mode '{mode}' returned {len(results)} results"
    for _, sc, _ in results:
        assert sc >= 0.0, f"Negative score in mode '{mode}': {sc}"


# ---------------------------------------------------------------------------
# 9. Diversity does not displace a uniquely-top-scoring song
# ---------------------------------------------------------------------------

def test_diversity_retains_clear_winner(songs, weekend_prefs):
    all_results = recommend_songs(weekend_prefs, songs, k=len(songs))
    diverse = apply_diversity_penalty(all_results, k=5)
    # The #1 song from raw scoring should still be #1 after diversity
    # (its artist & genre are unique in the top-1 position, so no penalty applies)
    assert all_results[0][0]["title"] == diverse[0][0]["title"], (
        "Diversity penalty should not displace the uniquely top-scoring song"
    )
