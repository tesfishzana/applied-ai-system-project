# Model Card: VibeFinder AI (Module 5 — Applied AI System)

> **Base project:** VibeFinder 1.0 — Music Recommender Simulation (Module 3)
> **Extended into:** Full applied AI system with RAG pipeline, agentic workflow,
> structured logging, reliability testing, evaluation harness, and Streamlit UI.

---

## 1. Model / System Name

**VibeFinder AI** — conversational music recommendation system powered by Claude (Anthropic) with a deterministic retrieval backend.

---

## 2. Goal / Task

VibeFinder AI lets a user describe what music they want to hear in plain English ("something chill for late-night coding") and returns ranked song recommendations with transparent, score-grounded explanations. The system handles multi-turn refinement: follow-up messages like "make it more energetic" or "explain that first result" re-use conversation context without restarting.

**What this is NOT:** It is not a behavior-based system that learns from listening history. It is a teaching and demonstration system that shows how LLM intent extraction, rule-based retrieval, and structured explanation combine into an end-to-end recommender pipeline.

---

## 3. Architecture Summary

| Layer | Component | Role |
|---|---|---|
| Interface | `app.py` (Streamlit) | Chat UI, session state, reasoning steps panel |
| Orchestration | `src/llm_agent.py` (MusicAgent) | Agentic loop, tool routing, prompt caching |
| Language model | Claude Sonnet 4.6 (Anthropic API) | Intent extraction, planning, explanation generation |
| Retrieval | `src/recommender.py` (VibeFinder engine) | Deterministic score-based ranking |
| RAG context | `data/knowledge_base.md` | Genre profiles, mood psychology, activity → music map |
| Catalog | `data/songs.csv` | 18 songs × 15 features |
| Logging | `src/logger.py` | JSON-structured rotating log (`logs/vibefinder.log`) |

**RAG pattern:** The scoring engine acts as the retriever. Claude uses retrieved songs (with actual scores) as context for generation — it cannot hallucinate songs or invent scores that the engine did not produce.

**Agentic loop:** Claude may call three tools (`plan_search`, `get_recommendations`, `explain_song`) across up to 8 iterations before producing a final reply. The `plan_search` tool is called first on activity-based queries to make the intent-extraction step observable.

**Prompt caching:** The static system prompt (core instructions + catalog snapshot + knowledge base + few-shot examples, ~2,000 tokens) is marked `cache_control: ephemeral`. Turn 2+ in a conversation reuse the cache, saving ~90% of input token costs on multi-turn sessions.

Full diagram: [assets/system-diagram.md](assets/system-diagram.md)

---

## 4. Data Used

The catalog contains **18 songs** across **15 different genres**. Each song has:

| Attribute | Description |
|---|---|
| title, artist | Song identity |
| genre | One of 15 genre labels |
| mood | One of 14 mood labels |
| energy | Float 0–1 (quiet → intense) |
| acousticness | How acoustic the song sounds (0–1) |
| instrumentalness | Fraction of track that is instrumental (0–1) |
| release_decade | Era (e.g. "2020s") |
| popularity | 0–100 popularity score |
| mood_tags | Comma-separated tags (e.g. "calm, focused, dreamy") |

**Genres:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b, metal, country, reggae, blues, funk, classical.

**Catalog limits:** Most genres have only 1–3 songs. This is intentional for classroom scale but creates the sparsity biases described in Section 6.

---

## 5. Algorithm Summary

### Scoring Engine (src/recommender.py)

Each song earns points based on how well it matches the user's preference profile:

```
score = 2.00 × genre_match
      + 1.00 × mood_match
      + max(0, 1.0 − |song.energy − user.energy|)
      + 0.50 × acoustic_bonus
      + 0.25 × era_match              [advanced pref]
      + 0.10 × each_mood_tag_hit      [advanced pref, max 3]
      + 0.20 × popularity_fit         [advanced pref]
      + 0.25 × instrumental_bonus     [advanced pref]

maximum possible score = 5.50
```

A diversity penalty (`score × 0.75`) is applied to repeat artists after their first appearance in the ranked list.

### LLM Layer (src/llm_agent.py)

Claude does **not** generate scores — it calls the scoring engine as a tool and uses the returned scores to write explanations. Three tools:

- `plan_search(query, reasoning)` — articulates how the query was interpreted before searching
- `get_recommendations(prefs)` — calls the scoring engine; returns ranked songs with scores and reasons
- `explain_song(song_title, prefs)` — returns the detailed score breakdown for a single song

---

## 6. Observed Biases and Limitations

### Biases in the scoring engine

**Genre dominance (filter bubble):** Genre is worth 2.00 of 5.50 possible points (36% of max, but ~44% of base 4.5). A genre match alone puts a song far ahead of non-genre songs regardless of mood, energy, or acoustic fit. Users with a rare genre preference receive very little variety.

**All-or-nothing label matching:** Genre and mood are binary checks — exact string match or zero. "Rock" and "metal" share high-energy guitar character, but a rock fan gets zero genre credit from metal songs. A user who says "sad" will never receive a mood bonus because no catalog song uses that label.

**Energy is the only gradient:** Energy is the only feature that produces a smooth score (via distance). Tempo, valence, and danceability are in the dataset but unused by the scoring formula.

**Catalog sparsity degrades diversity:** With 1–3 songs per genre, positions #2–#5 for niche profiles fill with unrelated genres that share an energy level or acoustic quality. A classical listener's second recommendation may be an ambient electronic song solely because both are quiet.

### Biases introduced by the LLM layer

**Vocabulary normalization:** Claude maps colloquial terms to catalog values ("lo-fi vibes" → `genre: lofi`, "something sad" → `mood: melancholic`). This mapping is learned from Claude's training data, not from the catalog. If Claude's training data associates a term differently from the catalog labels, the wrong profile will be extracted.

**Confidence calibration:** Claude does not communicate how certain it is about extracted preferences. A vague query ("something nice") produces a profile with the same API structure as a precise query ("jazz, relaxed, energy around 0.4"). The system cannot flag low-confidence extractions.

**Guardrail scope:** Off-topic queries ("what's the weather?") are deflected by prompt instructions, but these instructions can be overridden by sufficiently creative prompting. The guardrails are not enforced at a system level — they rely on Claude following instructions.

**Hallucination risk at the boundary:** Claude is explicitly instructed to only cite songs and scores returned by the engine. If this instruction is ignored (model drift, adversarial prompt), Claude could invent plausible-sounding songs. The tool-use architecture makes this detectable but not impossible.

---

## 7. Reliability Testing Results

### pytest suite — 20 / 20 passing

```
tests/test_recommender.py        2 / 2   (original unit tests)
tests/test_reliability.py       18 / 18  (reliability suite)
```

| Test | What it checks | Result |
|---|---|---|
| `test_recommend_returns_songs_sorted_by_score` | Output is highest-first sorted | Pass |
| `test_explain_recommendation_returns_non_empty_string` | Each result has a non-empty explanation | Pass |
| `test_recommendations_are_deterministic` | Same profile → identical ranking every call | Pass |
| `test_scores_within_bounds` | All scores in [0, 5.5] | Pass |
| `test_all_top5_have_explanations` | Every top-5 result has an explanation | Pass |
| `test_diversity_penalty_does_not_increase_repetition` | Diversity never makes artist variety worse | Pass |
| `test_diversity_penalty_delays_repeat_artist_in_ranking` | Penalty pushes repeat artists later | Pass |
| `test_empty_catalog_returns_empty` | Empty catalog → [] without crashing | Pass |
| `test_valid_genre_scores_nonzero` (×6 genres) | Known genres produce non-zero top scores | Pass |
| `test_unknown_genre_no_genre_bonus` | Unrecognised genre earns no genre bonus | Pass |
| `test_scoring_mode_returns_five_results` (×4 modes) | All scoring modes return 5 results ≥ 0 | Pass |
| `test_diversity_retains_clear_winner` | Diversity doesn't displace the uniquely top song | Pass |

### evaluate.py — 9 / 9 deterministic cases passing

```
python evaluate.py

RESULT  : 9/9 passed
Avg confidence margin : 0.31  (31% of max score gap)
OK All deterministic engine tests passed.
```

Each case verifies: (1) expected song is ranked #1, (2) top score meets a minimum threshold, (3) margin-of-victory confidence score meets a minimum. The confidence score `(score_1 − score_2) / 5.5` measures how decisively the top song won.

| Test case | Expected #1 | Min score | Min confidence | Result |
|---|---|---|---|---|
| Chill Lofi Study | Library Rain | 4.0 | 0.00 | Pass |
| Weekend Pop Vibes | Sunrise City | 3.5 | 0.15 | Pass |
| Deep Rock Intensity | Storm Runner | 3.5 | 0.35 | Pass |
| Classical Serenity | Moonlit Sonata | 4.0 | 0.50 | Pass |
| Hip-Hop Gym | Corner Store Flex | 3.0 | 0.10 | Pass |
| Relaxed Jazz Sunday | Coffee Shop Stories | 3.5 | 0.50 | Pass |
| Metal Aggression | Iron Collapse | 3.5 | 0.50 | Pass |
| Late-Night Study (Advanced) | Midnight Coding | 5.0 | 0.01 | Pass |
| Unknown genre (guardrail) | Rooftop Lights | 0.5 | 0.00 | Pass |

---

## 8. AI Collaboration — Helpful and Flawed Suggestions

### One helpful AI suggestion

**Prompt caching with `cache_control: ephemeral` on the static system blocks.**

When building the multi-turn agent, AI suggested placing the `cache_control` marker on both the core system prompt and the knowledge base block. The reasoning: these blocks are identical across every API call in a session, so they are ideal cache candidates. Without the suggestion, I would have sent the full ~2,000-token system context on every turn. With it, turns 2+ get a cache hit that reduces input token count by approximately 90% and reduces latency by 30–50%. This was non-obvious because prompt caching requires the Anthropic beta header and specific message structure — it is not the default behavior.

### One flawed AI suggestion

**Test assertion that diversity penalty eliminates duplicate artists.**

The initial test `test_diversity_penalty_no_repeat_artist` was written with AI assistance. It asserted `len(artists) == len(set(artists))` — meaning zero duplicate artists in the top-5 results after applying the diversity penalty.

This was wrong. The diversity penalty is a **soft** multiplier (`score × 0.75`), not a hard exclusion. When LoRoom has two high-scoring lofi songs and no other artist has a comparable third song, LoRoom's second song still appears even at 75% score — it simply has no better-scoring competition. The test passed against a naively constructed example but failed on the real catalog.

The fix required understanding the actual contract: diversity must not *increase* repetition (comparing with and without penalty), and must push a repeat artist's second song *later* in the ranking — not eliminate it entirely. Both invariants held and now have separate named tests.

**Lesson:** AI-generated test assertions often assume the *intended* behavior of an algorithm rather than its *actual* behavior. Always run the tests against the real system before trusting that the assertion is correct.

---

## 9. Intended Use and Non-Intended Use

**Intended use:**
- Educational demonstration of RAG, agentic tool-use, and LLM-powered recommendation
- Classroom study of how prompt engineering, few-shot examples, and tool routing affect LLM output
- Reference implementation showing how deterministic retrieval and LLM generation separate concerns

**Non-intended use:**
- Not suitable for production music apps — the 18-song catalog is far too small
- Not a substitute for behavior-based recommenders (Spotify, Last.fm) that learn from listening history
- Should not be used to make decisions for users without showing the score breakdown — the system can produce confident results that do not match a user's actual preferences
- Not designed for multi-genre or multi-mood preference profiles simultaneously

---

## 10. Ethical Considerations

**Filter bubble:** Genre dominance means users whose preferred genre has few catalog entries will repeatedly see the same 1–2 songs at the top. A production system would need either a larger catalog or explicit diversity enforcement.

**Guardrail limitations:** Off-topic deflection relies on Claude following prompt instructions. An adversarial user could potentially bypass these instructions. A production system would enforce guardrails at the API level (content filtering, tool restrictions), not only in the prompt.

**Misuse surface — preference harvesting:** A deployed version that stores user preference profiles could be used to infer sensitive attributes (mood states, stress levels, sleep patterns) from listening behavior. VibeFinder intentionally does not persist user data between sessions.

**Scoring weight choices are value judgments:** The decision to weight genre at +2.00 and mood at +1.00 encodes a theory about what music preference means. A different weighting (e.g., energy-first) produces different rankings with equal mathematical justification. Users who do not read the model card would never know which theory was applied.

---

## 11. Ideas for Improvement

- **Expand the catalog** — connecting to a music API (Spotify, MusicBrainz) would expose whether the scoring signals hold at scale and reduce the sparsity problem
- **Soft / partial genre matching** — grouping genres into families (rock/metal/punk as "high-energy guitar") so near-miss genres earn partial credit
- **Confidence signaling** — ask Claude to rate how confident it is about the extracted profile, and surface low-confidence extractions so users know when to rephrase
- **Tempo as a scored feature** — BPM is in the dataset but unused; a workout profile asking for 140+ BPM should rank differently from a study session at 80 BPM
- **Streaming replies** — Streamlit supports streamed tokens; adding this would make long explanations feel more responsive
- **Evaluation with real users** — record 20 real conversations, annotate whether recommendations "felt right", and report a human accuracy rate alongside the automated test pass counts
