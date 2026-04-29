import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Challenge 1: advanced features — defaults preserve backward compatibility with tests
    popularity: int = 0
    release_decade: str = ""
    mood_tags: list = field(default_factory=list)
    liveness: float = 0.0
    instrumentalness: float = 0.0


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Challenge 1: advanced preference fields — all optional with defaults
    preferred_decade: str = ""
    desired_mood_tags: list = field(default_factory=list)
    popularity_target: int = -1   # -1 means no preference
    wants_instrumental: bool = False


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = [(song, self._score(user, song)) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def _score(self, user: UserProfile, song: Song) -> float:
        score = 0.0
        # Base scoring (original four signals)
        if song.genre.lower() == user.favorite_genre.lower():
            score += 2.0
        if song.mood.lower() == user.favorite_mood.lower():
            score += 1.0
        energy_diff = abs(song.energy - user.target_energy)
        score += max(0.0, 1.0 - energy_diff)
        if user.likes_acoustic and song.acousticness >= 0.6:
            score += 0.5
        # Challenge 1: advanced feature bonuses
        if user.preferred_decade and song.release_decade == user.preferred_decade:
            score += 0.25
        if user.desired_mood_tags and song.mood_tags:
            for tag in user.desired_mood_tags:
                if tag in song.mood_tags:
                    score += 0.10
        if user.popularity_target >= 0:
            diff = abs(song.popularity - user.popularity_target)
            score += max(0.0, 0.20 * (1.0 - diff / 50.0))
        if user.wants_instrumental and song.instrumentalness >= 0.7:
            score += 0.25
        return score

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        reasons = []
        if song.genre.lower() == user.favorite_genre.lower():
            reasons.append(f"genre match ({song.genre})")
        if song.mood.lower() == user.favorite_mood.lower():
            reasons.append(f"mood match ({song.mood})")
        energy_diff = abs(song.energy - user.target_energy)
        energy_score = max(0.0, 1.0 - energy_diff)
        if energy_score >= 0.8:
            reasons.append(f"energy is a great fit ({song.energy:.2f})")
        elif energy_score >= 0.5:
            reasons.append(f"energy is a decent fit ({song.energy:.2f})")
        else:
            reasons.append(f"energy mismatch ({song.energy:.2f})")
        if user.likes_acoustic and song.acousticness >= 0.6:
            reasons.append(f"acoustic warmth ({song.acousticness:.2f})")
        if user.preferred_decade and song.release_decade == user.preferred_decade:
            reasons.append(f"era match ({song.release_decade})")
        if user.desired_mood_tags and song.mood_tags:
            hits = [t for t in user.desired_mood_tags if t in song.mood_tags]
            if hits:
                reasons.append(f"mood tags: {', '.join(hits)}")
        if user.popularity_target >= 0:
            diff = abs(song.popularity - user.popularity_target)
            if diff <= 10:
                reasons.append(f"popularity fit ({song.popularity})")
        if user.wants_instrumental and song.instrumentalness >= 0.7:
            reasons.append(f"instrumental ({song.instrumentalness:.2f})")
        return ", ".join(reasons) if reasons else "general recommendation"


# ---------------------------------------------------------------------------
# Scoring recipe — base signals
# ---------------------------------------------------------------------------
# Default max possible score (base): 4.5
#
#   +2.0  genre match       — most deliberate preference
#   +1.0  mood match        — contextual; half the genre weight
#   +1.0  energy similarity — linear decay: 1.0 - |song.energy - target|
#   +0.5  acoustic bonus    — small reward for acoustic preference match
#
# Challenge 1 — advanced feature bonuses (additive, max +1.00):
#   +0.25  exact decade match
#   +0.10  per matching mood tag (max 3 tags = +0.30)
#   +0.20  popularity proximity (linear decay over 50-point window)
#   +0.25  instrumentalness bonus (song.instrumentalness >= 0.70)
#
# Overall theoretical max: 4.5 + 1.0 = 5.5 (all advanced prefs satisfied)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict = {"genre": 2.0, "mood": 1.0, "energy": 1.0, "acoustic": 0.5}

# Challenge 2: Multiple Scoring Modes — each set of weights sums to 4.5
SCORING_MODES: Dict[str, Dict] = {
    "balanced":       {"genre": 2.0,  "mood": 1.0,  "energy": 1.0,  "acoustic": 0.5},
    "genre_first":    {"genre": 3.0,  "mood": 0.75, "energy": 0.5,  "acoustic": 0.25},
    "mood_first":     {"genre": 1.5,  "mood": 2.0,  "energy": 0.75, "acoustic": 0.25},
    "energy_focused": {"genre": 1.0,  "mood": 0.75, "energy": 2.5,  "acoustic": 0.25},
}

SCORING_MODE_DESCRIPTIONS: Dict[str, str] = {
    "balanced":       "Genre is primary; energy and mood give context (default).",
    "genre_first":    "Genre identity dominates; all other signals are tiebreakers.",
    "mood_first":     "Emotional context matters most; genre is secondary.",
    "energy_focused": "Match intensity first; cross-genre picks surface naturally.",
}


def score_song(song: Dict, user_prefs: Dict, weights: Dict = None) -> Tuple[float, str]:
    """Return (numeric_score, explanation) for one song.

    Base weights default to genre=2.0, mood=1.0, energy=1.0, acoustic=0.5 (max 4.5).
    Pass a custom ``weights`` dict to switch modes — e.g. SCORING_MODES["genre_first"].

    Advanced preference keys in user_prefs (preferred_decade, desired_mood_tags,
    popularity_target, wants_instrumental) award up to +1.0 in bonus points.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    score = 0.0
    reasons = []

    # --- Base scoring (original four signals) ---
    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += w["genre"]
        reasons.append(f"genre match ({song['genre']})")

    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += w["mood"]
        reasons.append(f"mood match ({song['mood']})")

    song_energy = float(song.get("energy", 0.5))
    target_energy = float(user_prefs.get("energy", 0.5))
    energy_diff = abs(song_energy - target_energy)
    raw_energy_fit = max(0.0, 1.0 - energy_diff)
    score += raw_energy_fit * w["energy"]
    if raw_energy_fit >= 0.8:
        reasons.append(f"energy is a great fit ({song_energy:.2f})")
    elif raw_energy_fit >= 0.5:
        reasons.append(f"energy is a decent fit ({song_energy:.2f})")
    else:
        reasons.append(f"energy mismatch ({song_energy:.2f})")

    if user_prefs.get("likes_acoustic") and float(song.get("acousticness", 0)) >= 0.6:
        score += w["acoustic"]
        reasons.append(f"acoustic warmth ({float(song['acousticness']):.2f})")

    # --- Challenge 1: Advanced feature bonuses (fixed weights, mode-independent) ---

    # Decade match (+0.25)
    preferred_decade = user_prefs.get("preferred_decade", "")
    if preferred_decade and song.get("release_decade", "") == preferred_decade:
        score += 0.25
        reasons.append(f"era match ({preferred_decade})")

    # Mood tag overlap (+0.10 per tag, capped at 3 tags = +0.30 max)
    desired_tags = user_prefs.get("desired_mood_tags") or []
    song_tags = song.get("mood_tags") or []
    if isinstance(song_tags, str):
        song_tags = [t.strip() for t in song_tags.split("|") if t.strip()]
    if desired_tags and song_tags:
        hits = [tag for tag in desired_tags if tag in song_tags]
        if hits:
            score += min(len(hits), 3) * 0.10
            reasons.append(f"mood tags: {', '.join(hits)}")

    # Popularity proximity (+0.20 max, linear decay over 50-point window)
    pop_target = user_prefs.get("popularity_target", -1)
    if pop_target is not None and pop_target >= 0:
        diff = abs(float(song.get("popularity", 0)) - pop_target)
        pop_bonus = max(0.0, 0.20 * (1.0 - diff / 50.0))
        score += pop_bonus
        if pop_bonus >= 0.16:
            reasons.append(f"popularity fit ({song.get('popularity', 0)})")

    # Instrumentalness preference (+0.25 when song is >= 70% instrumental)
    if user_prefs.get("wants_instrumental") and float(song.get("instrumentalness", 0)) >= 0.7:
        score += 0.25
        reasons.append(f"instrumental ({float(song['instrumentalness']):.2f})")

    explanation = ", ".join(reasons) if reasons else "no strong match"
    return score, explanation


def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with numeric fields cast to float/int."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mood_tags_raw = row.get("mood_tags", "")
            mood_tags = [t.strip() for t in mood_tags_raw.split("|") if t.strip()] if mood_tags_raw else []
            songs.append({
                "id":             int(row["id"]),
                "title":          row["title"],
                "artist":         row["artist"],
                "genre":          row["genre"],
                "mood":           row["mood"],
                "energy":         float(row["energy"]),
                "tempo_bpm":      float(row["tempo_bpm"]),
                "valence":        float(row["valence"]),
                "danceability":   float(row["danceability"]),
                "acousticness":   float(row["acousticness"]),
                # Challenge 1: advanced features (gracefully absent in older CSVs)
                "popularity":       int(row["popularity"]) if row.get("popularity") else 0,
                "release_decade":   row.get("release_decade", ""),
                "mood_tags":        mood_tags,
                "liveness":         float(row["liveness"]) if row.get("liveness") else 0.0,
                "instrumentalness": float(row["instrumentalness"]) if row.get("instrumentalness") else 0.0,
            })
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Dict = None) -> List[Tuple[Dict, float, str]]:
    """Score every song with score_song, sort highest-first, return top k as (song, score, explanation) tuples."""
    scored = []
    for song in songs:
        score, explanation = score_song(song, user_prefs, weights)
        scored.append((song, score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def apply_diversity_penalty(
    scored: List[Tuple[Dict, float, str]],
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """Greedy top-k selection that penalizes repeated artists and over-represented genres.

    Challenge 3: Diversity and Fairness Logic.

    At each round, every remaining candidate receives a temporary penalty based on
    what has already been selected, then the highest adjusted-score candidate wins.
    Original scores are preserved in the returned tuples so the display stays honest.

    Penalty rules:
        - 2nd+ song from the same artist: comparison score × 0.75
        - 3rd+ song from the same genre:  comparison score − 0.50
    """
    from collections import defaultdict
    remaining = list(scored)
    selected: List[Tuple[Dict, float, str]] = []
    artist_count: Dict[str, int] = defaultdict(int)
    genre_count: Dict[str, int] = defaultdict(int)

    while len(selected) < k and remaining:
        best_idx = 0
        best_penalized = float("-inf")
        for i, (song, base_score, _) in enumerate(remaining):
            penalized = base_score
            if artist_count[song["artist"]] >= 1:
                penalized *= 0.75
            if genre_count[song["genre"]] >= 2:
                penalized -= 0.50
            if penalized > best_penalized:
                best_penalized = penalized
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        artist_count[chosen[0]["artist"]] += 1
        genre_count[chosen[0]["genre"]] += 1

    return selected
