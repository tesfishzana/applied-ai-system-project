# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Goal / Task

VibeFinder's goal is to suggest five songs from a small catalog that best match a user's stated musical preferences. Given four inputs — favorite genre, favorite mood, preferred energy level, and whether the listener enjoys acoustic music — it ranks every song by how well it fits those preferences and returns the top results with a plain-English explanation for each pick.

This is a simulation and teaching tool. It is not trying to predict what a user will click on or learn from behavior history. It is trying to show how explicit preference-matching works before introducing machine-learning techniques.

---

## 3. Data Used

The catalog contains **18 songs** across **15 different genres**. Each song has the following attributes:

| Attribute | Description |
|-----------|-------------|
| title, artist | Song identity |
| genre | One of 15 genre labels |
| mood | One of 14 mood labels |
| energy | Float 0–1 (quiet to intense) |
| tempo_bpm | Beats per minute |
| valence | Musical positivity (0–1) |
| danceability | How suitable for dancing (0–1) |
| acousticness | How acoustic the song sounds (0–1) |

**Genres represented:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b, metal, country, reggae, blues, funk, classical.

**Moods represented:** happy, chill, intense, relaxed, focused, moody, peaceful, confident, romantic, angry, nostalgic, uplifting, melancholic, groovy.

**Limits:** The catalog is intentionally small for classroom use. Most genres have only 1–3 songs. Moods like "sad" and "excited" are missing entirely. Only two attributes — energy and acousticness — are used by the scoring algorithm; tempo, valence, and danceability are stored but ignored.

---

## 4. Algorithm Summary

Imagine you walk into a record store and tell the clerk: "I want pop music, something happy and high-energy, and I don't really care about acoustic guitars." The clerk checks every record in the store and gives each one a score based on how well it matches your request. VibeFinder does the same thing, just with numbers.

Each song earns points in four areas:

1. **Genre match (up to 2.0 points):** Exact match only — if a song's genre equals the one you asked for, it gets 2 points; otherwise zero. This is the single most important signal.
2. **Mood match (up to 1.0 point):** Same all-or-nothing rule. An exact mood label match earns 1 point; anything else earns nothing.
3. **Energy similarity (up to 1.0 point):** Energy is measured 0 to 1. A perfect energy match gives 1.0; the further away a song's energy is from your target, the fewer points it earns.
4. **Acoustic bonus (0.5 points):** If you said you enjoy acoustic music AND a song's acousticness is 0.6 or higher, it earns a small bonus.

The maximum possible score is **4.5 points**. Songs are ranked by total score and the top five are returned, each with a reason explaining which signals fired.

---

## 5. Observed Behavior / Biases

**Genre dominance creates a filter bubble.** Genre is worth 2.0 of 4.5 possible points (44%). A genre match alone pushes a song ahead of non-genre songs on almost every other dimension. Users whose preferred genre is rare in the catalog receive very little variety after their first result.

**All-or-nothing matching misses near-misses.** Genre and mood are binary checks — exact string match or zero. There is no recognition that "rock" and "metal" share high-energy character, or that "chill" and "relaxed" describe similar listener states. A rock fan gets zero credit from metal songs even if they would genuinely enjoy them.

**Missing moods silently fail.** The catalog has no songs labeled "sad," "excited," or several other common moods. A user who sets `mood: sad` will never receive the +1.0 mood bonus for any song. The system does not warn the user — it simply returns the best energy+genre matches as if the mood preference didn't exist.

**Contradictory preferences are handled silently.** An "Acoustic Metal Head" — someone who wants metal with acoustic warmth — gets the metal song correctly at #1, but positions #2 and #3 are slow country and blues songs because they happen to have high acousticness. The system cannot flag the impossibility; it substitutes what it has without explanation.

**Catalog sparsity degrades diversity.** With only 1–3 songs per genre, positions #2–#5 for niche profiles are filled by songs from completely unrelated genres that share an energy level or acoustic quality. A classical listener's second recommendation is an ambient electronic song — a very different sound — purely because both are quiet and acoustic.

---

## 6. Evaluation Process

Six user profiles were tested. Three were standard use-cases; three were adversarial edge cases designed to stress-test the scoring logic.

**Weekend Vibes (pop / happy / energy=0.80):** Results felt intuitive. Sunrise City ranked first (3.98/4.50) with genre, mood, and near-perfect energy matches. The large gap between #1 (3.98) and #3 (1.96) showed how dominant the genre+mood combination is.

**Chill Lofi (lofi / chill / energy=0.38 / acoustic):** The top three results were all lofi songs with high acousticness — accurate and useful. Library Rain and Midnight Coding were nearly tied (4.47 vs. 4.46) because both match genre, mood, and acousticness and differ only by 0.07 in energy proximity.

**Deep Intense Rock (rock / intense / energy=0.92):** Storm Runner won decisively (3.99/4.50). Gym Hero appeared at #2 — a pop song, not rock, but it shares the "intense" mood and near-perfect energy. This revealed mood as a meaningful secondary filter across genre boundaries.

**Conflicting — High Energy + Sad Mood [EDGE CASE]:** The mood "sad" does not exist in the catalog. No song ever earned a mood bonus. The system recommended Gym Hero ("intense" pop, energy=0.93) as the top result for someone who explicitly asked for sad music. This is the starkest failure mode — the system is confidently wrong.

**Acoustic Metal Head [EDGE CASE]:** Iron Collapse scored 4.00/4.50 correctly (losing only the acoustic bonus). Positions #2 and #3 were Dusty Back Roads (country) and Rainy Tuesday Blues (blues) — wrong genre but high acousticness. A user whose primary identity is "metal fan" receives country and blues in their top 3 because of acoustic score inflation.

**Classical Serenity [EDGE CASE]:** Moonlit Sonata achieved a perfect 4.50/4.50 — the only time any profile reached the maximum score. After that, the next four results were ambient, lofi, jazz, and lofi songs (scores 1.44–1.32). The catalog cliff is stark: one perfect match, then generic "acoustic and quiet" fallback.

**Weight Experiment — doubling energy, halving genre (Weekend Vibes):** With genre=1.0 and energy=2.0, Rooftop Lights (indie pop, energy=0.76) jumped from #3 to #2, displacing Gym Hero (pop, energy=0.93). Rooftop Lights is only 0.04 away from the target energy of 0.80; Gym Hero is 0.13 away. When energy weight doubled, that difference mattered more than Gym Hero's genre match. Neither ranking is objectively correct — they encode different theories about what music preference means.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**
- Educational tool for learning how preference-matching algorithms work
- Classroom demonstrations of scoring, weighting, and bias in recommender systems
- A starting point for students to experiment with changing weights and observing downstream effects
- A reference implementation showing how simple rules can produce recommendation-like behavior

**Non-intended use:**
- Not suitable for production music apps or commercial deployment — the 18-song catalog is far too small
- Not a substitute for behavior-based recommenders (like Spotify) that learn from listening history
- Should not be used to make decisions for users without showing them the explanation and the score — the system can be confidently wrong (see Conflicting Prefs edge case)
- Not designed to handle users with more than one preferred genre, mood, or energy state at a time

---

## 8. Ideas for Improvement

- **Expand the catalog significantly.** Even doubling to 36 songs per genre cluster would give positions #2–#5 real musical relevance instead of energy-proximity fallbacks. With 18 songs and 15 genres, many genres have only one representative.
- **Introduce soft / partial genre matching.** Group genres into families (e.g., rock / metal / punk as "high-energy guitar"; lofi / ambient / classical as "atmospheric") so that near-miss genres still earn partial credit rather than zero.
- **Warn when user preferences are impossible to satisfy.** If a mood label doesn't appear in the catalog, print a message like "Note: no songs labeled 'sad' — showing closest alternatives." This prevents the silent failure seen in the Conflicting profile.
- **Add tempo as a scored feature.** BPM is already in the dataset but unused. A workout profile asking for 140+ BPM should rank differently than a study session at 80 BPM, even when both want the same genre and energy level.
- **Support multi-genre or blended preferences.** Real listeners often enjoy more than one genre. Allowing `genres: ["lofi", "ambient"]` with shared weight would reduce the filter bubble effect.

---

## 9. Personal Reflection

**Biggest learning moment:** The most surprising discovery was how much of the recommender's behavior is determined by weight choices rather than the algorithm itself. The scoring formula is simple arithmetic, but small changes to the genre vs. energy balance produced genuinely different rankings. The weight experiment — doubling energy and halving genre — caused positions #2 and #3 to swap for the Weekend Vibes profile, and the new ranking arguably fit the user's stated target energy better. That showed me that "correct" is a design decision, not a math answer. Every recommender system encodes a theory about what user preference actually means.

**How AI tools helped — and where I double-checked:** AI tools were useful for generating the edge case profiles and thinking through what a "stress test" should look like for a scoring system. The Acoustic Metal Head profile came out of that kind of brainstorming. However, I had to verify every output manually by running the system and reading the scores myself. The AI suggestions for what the output "should" be were sometimes wrong because the tool didn't account for the exact acousticness threshold (0.6) in the code. The lesson: AI is good for generating ideas and structuring problems, but you have to run the actual code to know what the system truly does.

**What surprised me about simple algorithms feeling like recommendations:** The Classical Serenity profile achieving a perfect 4.50/4.50 felt genuinely satisfying — like the system "understood" exactly what the user wanted. But that happened only because one song in the catalog matched every preference by chance. The recommendation felt intelligent, but it was just arithmetic landing on a lucky exact match. This showed me why simple rule-based systems can feel impressive when the catalog aligns with the user, and feel frustratingly wrong when it doesn't. The algorithm never changed; only the data coverage did. Real-world systems feel smart primarily because they have huge catalogs — not because their algorithms are fundamentally different.

**What I would try next:** If I extended this project, I would first add the mood-missing warning so users know when their preference has no catalog coverage. Second, I would add tempo as a scored feature — BPM is already in the data and unused. Third, I would try soft genre matching using genre families, and measure whether it reduces the "country and blues for a metal fan" problem. Longer term, I would be curious to compare this rule-based approach side-by-side with a simple collaborative filtering approach (users who liked X also liked Y) on the same catalog, to see how differently they handle sparse data.

---
