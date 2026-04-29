"""VibeFinder AI — LLM agent with RAG, agentic planning, and few-shot specialization.

Powered by Google Gemini API (gemini-2.0-flash).

Stretch features implemented here
──────────────────────────────────
RAG Enhancement (+2)
  A second knowledge-base document (data/knowledge_base.md) is loaded at startup
  and appended to the system prompt alongside the song catalog.
  It contains genre profiles, mood psychology, activity→music mappings, and signal
  reference tables. Gemini references it to give richer explanations for genres and
  moods that the catalog labels alone cannot fully capture.

Agentic Workflow Enhancement (+2)
  A `plan_search` tool is added. Gemini calls it *before* get_recommendations when
  the query is ambiguous or complex. Each call is logged and its input is stored in
  `reasoning_steps` alongside every other tool call, making the full decision chain
  observable. The Streamlit UI renders these steps in an expandable panel.

Fine-Tuning / Specialization (+2)
  Three few-shot examples are appended to the system prompt. They constrain
  Gemini to: always cite numeric scores, format each song as "🎵 Title by Artist
  (score/max) — signals", and close with a follow-up offer.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from google import genai
from google.genai import types

from .recommender import apply_diversity_penalty, load_songs, recommend_songs, score_song

logger = logging.getLogger("vibefinder.agent")

# ──────────────────────────────────────────────────────────────────────────────
# Catalog-valid values (used in system prompt and guardrails)
# ──────────────────────────────────────────────────────────────────────────────
_VALID_GENRES = (
    "pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b, "
    "metal, country, reggae, blues, funk, classical"
)
_VALID_MOODS = (
    "happy, chill, intense, relaxed, focused, moody, peaceful, confident, "
    "romantic, angry, nostalgic, uplifting, melancholic, groovy"
)

# ──────────────────────────────────────────────────────────────────────────────
# Static system prompt
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_CORE = """\
You are VibeFinder AI, a music recommendation assistant backed by an 18-song catalog \
and a rule-based scoring engine.

YOUR WORKFLOW
1. When the user's request is ambiguous or uses activity language, call plan_search first \
   to state your interpretation and planned parameters. This is visible to the user.
2. Call get_recommendations to retrieve scored results from the catalog.
3. After receiving results, compose a reply that cites numeric scores and specific signals.
4. Close every reply with a one-sentence offer to refine (different genre, era, diversity mode, etc.).

RESPONSE FORMAT (always follow this pattern):
  Brief opener (1 sentence setting the vibe).
  For each song:  🎵 **Title** by Artist (score X.XX/Y.YY) — [signals that fired]
  Closing offer to refine.

GUARDRAILS
• Only discuss music and recommendations. Politely redirect off-topic queries.
• Never invent songs, artists, or scores — all data must come from tool results.
• If a genre or mood is missing from the catalog, acknowledge it and suggest the closest alternative.

CATALOG VALID VALUES
  Genres : {genres}
  Moods  : {moods}
  Decades: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s
""".format(genres=_VALID_GENRES, moods=_VALID_MOODS)

# ──────────────────────────────────────────────────────────────────────────────
# Few-shot examples — Specialization/Fine-tuning (+2)
# ──────────────────────────────────────────────────────────────────────────────
_FEW_SHOT = """
─── FEW-SHOT EXAMPLES (follow this exact voice and format) ───────────────────

Example 1
User: something for studying
VibeFinder: Your late-night study playlist, ranked by fit:

🎵 **Library Rain** by Paper Lanterns (4.47/4.50) — genre match (lofi) ✓, mood match (chill) ✓, energy great fit (0.35), acoustic warmth (0.86).
🎵 **Midnight Coding** by LoRoom (4.46/4.50) — genre match (lofi) ✓, mood match (chill) ✓, energy great fit (0.42), acoustic warmth (0.71).
🎵 **Focus Flow** by LoRoom (3.48/4.50) — genre match (lofi) ✓, energy great fit (0.40), acoustic warmth (0.78). Mood is "focused" not "chill" so mood bonus didn't fire — but still a great study track.

Want instrumentals only, a 2020s era filter, or diversity mode so the same artist doesn't repeat?

─────────────────────────────────────────────────────────────────────────────

Example 2
User: why did that first song score so high?
VibeFinder: [after explain_song call]
**Library Rain** earned 4.47/4.50 — nearly perfect. Here's the breakdown:
- **Genre** (+2.0): lofi ✓
- **Mood** (+1.0): chill ✓
- **Energy** (+0.97): its energy (0.35) vs your target (0.38) — gap of only 0.03
- **Acoustic warmth** (+0.50): acousticness of 0.86 — very warm, unplugged feel

The only "missing" 0.03 points come from the tiny energy gap. It's about as close to perfect as the catalog gets.

Want me to search for something with even lower energy, or add decade or tag filters?

─────────────────────────────────────────────────────────────────────────────

Example 3
User: what's the weather like?
VibeFinder: I'm a music recommendation assistant, so weather forecasts are outside my lane! 🎵 But if you want something that *sounds* like a rainy day, I can find moody lofi or blues tracks that match that vibe. Just say the word.

─────────────────────────────────────────────────────────────────────────────
"""


def _build_catalog_snapshot(songs: list[dict]) -> str:
    lines = ["\nCATALOG SNAPSHOT (18 songs — used by the scoring engine):"]
    for s in songs:
        tags = ", ".join(s["mood_tags"]) if s["mood_tags"] else "—"
        lines.append(
            f"  • {s['title']} | {s['artist']} | genre={s['genre']} | mood={s['mood']}"
            f" | energy={s['energy']} | decade={s['release_decade']} | tags=[{tags}]"
        )
    return "\n".join(lines)


def _build_gemini_tools() -> types.Tool:
    """Build Gemini FunctionDeclaration tool definitions."""
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="plan_search",
            description=(
                "Call this FIRST when the user's request is ambiguous, uses activity language "
                "(e.g. 'gym', 'studying', 'road trip'), or mixes signals that need interpretation. "
                "State how you read the request and which catalog values you plan to use before "
                "calling get_recommendations. This step is observable to the user."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "interpretation": types.Schema(
                        type=types.Type.STRING,
                        description="How you read the user's request in plain English.",
                    ),
                    "planned_genre": types.Schema(type=types.Type.STRING),
                    "planned_mood": types.Schema(type=types.Type.STRING),
                    "planned_energy": types.Schema(
                        type=types.Type.NUMBER,
                        description="Target energy 0.0-1.0 you will use.",
                    ),
                    "reasoning": types.Schema(
                        type=types.Type.STRING,
                        description="Why you chose these parameters (reference knowledge base if helpful).",
                    ),
                },
                required=["interpretation", "planned_genre", "planned_mood", "planned_energy", "reasoning"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_recommendations",
            description=(
                "Query the VibeFinder scoring engine and return top song recommendations. "
                "Call plan_search first if the query needed interpretation. "
                "Genre, mood, energy, and likes_acoustic are required."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "genre": types.Schema(type=types.Type.STRING, description="Must be a catalog genre."),
                    "mood": types.Schema(type=types.Type.STRING, description="Must be a catalog mood."),
                    "energy": types.Schema(type=types.Type.NUMBER, description="Target energy 0.0-1.0."),
                    "likes_acoustic": types.Schema(type=types.Type.BOOLEAN),
                    "preferred_decade": types.Schema(type=types.Type.STRING),
                    "desired_mood_tags": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                    "popularity_target": types.Schema(
                        type=types.Type.INTEGER,
                        description="Target popularity 0-100; -1 = no preference.",
                    ),
                    "wants_instrumental": types.Schema(type=types.Type.BOOLEAN),
                    "use_diversity": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Apply diversity penalty to avoid same-artist repeats.",
                    ),
                },
                required=["genre", "mood", "energy", "likes_acoustic"],
            ),
        ),
        types.FunctionDeclaration(
            name="explain_song",
            description=(
                "Return a detailed score breakdown for one specific song. "
                "Use when the user asks why a track was recommended or wants more detail."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "song_title": types.Schema(type=types.Type.STRING, description="Title (partial match OK)."),
                    "genre": types.Schema(type=types.Type.STRING),
                    "mood": types.Schema(type=types.Type.STRING),
                    "energy": types.Schema(type=types.Type.NUMBER),
                    "likes_acoustic": types.Schema(type=types.Type.BOOLEAN),
                },
                required=["song_title", "genre", "mood", "energy", "likes_acoustic"],
            ),
        ),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class MusicAgent:
    """Conversational music recommendation agent powered by Google Gemini.

    Stretch features
    ─────────────────
    RAG Enhancement     — knowledge_base.md appended to system prompt.
    Agentic Enhancement — plan_search tool + reasoning_steps property.
    Specialization      — few-shot examples baked into the system prompt.

    Parameters
    ──────────
    songs_path      : path to data/songs.csv
    knowledge_path  : optional path to data/knowledge_base.md (RAG enhancement)
    """

    def __init__(self, songs_path: str, knowledge_path: str | None = None) -> None:
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.songs = load_songs(songs_path)
        self._history: list[types.Content] = []
        self._last_profile: dict | None = None
        self._reasoning_steps: list[dict] = []

        catalog = _build_catalog_snapshot(self.songs)
        self._system_prompt = _SYSTEM_CORE + catalog + _FEW_SHOT

        # RAG Enhancement: append knowledge base to system prompt
        if knowledge_path and os.path.exists(knowledge_path):
            with open(knowledge_path, encoding="utf-8") as f:
                kb_text = f.read()
            self._system_prompt += (
                "\n\nMUSIC KNOWLEDGE BASE (use this to interpret activity language, "
                "mood psychology, and give richer explanations):\n\n" + kb_text
            )
            logger.info("knowledge_base_loaded", extra={"chars": len(kb_text)})

        self._tools = _build_gemini_tools()
        logger.info("agent_initialized", extra={"catalog_size": len(self.songs),
                                                 "kb": knowledge_path is not None})

    # ──────────────────────────────────────────────────────────────────────────
    # Tool execution
    # ──────────────────────────────────────────────────────────────────────────

    def _run_tool(self, name: str, inp: dict) -> str:
        logger.info("tool_called", extra={"tool": name, "input": inp})
        step: dict[str, Any] = {"tool": name, "input": inp, "output": ""}

        if name == "plan_search":
            output = (
                f"Plan: {inp.get('interpretation', '')}\n"
                f"Genre → {inp.get('planned_genre')} | "
                f"Mood → {inp.get('planned_mood')} | "
                f"Energy → {inp.get('planned_energy')}\n"
                f"Reasoning: {inp.get('reasoning', '')}"
            )
            step["output"] = output
            self._reasoning_steps.append(step)
            logger.info("tool_result", extra={"tool": name})
            return output

        if name == "get_recommendations":
            user_prefs: dict[str, Any] = {
                "genre": inp["genre"],
                "mood": inp["mood"],
                "energy": float(inp["energy"]),
                "likes_acoustic": bool(inp.get("likes_acoustic", False)),
                "preferred_decade": inp.get("preferred_decade", ""),
                "desired_mood_tags": list(inp.get("desired_mood_tags") or []),
                "popularity_target": int(inp.get("popularity_target", -1)),
                "wants_instrumental": bool(inp.get("wants_instrumental", False)),
            }
            self._last_profile = user_prefs
            use_div = bool(inp.get("use_diversity", False))
            fetch_k = len(self.songs) if use_div else 5
            results = recommend_songs(user_prefs, self.songs, k=fetch_k)
            if use_div:
                results = apply_diversity_penalty(results, k=5)
            lines = [
                f"#{r}  '{s['title']}' by {s['artist']}  (score {sc:.2f})  — {expl}"
                for r, (s, sc, expl) in enumerate(results, 1)
            ]
            output = "\n".join(lines)
            step["output"] = output
            self._reasoning_steps.append(step)
            logger.info("tool_result", extra={"tool": name, "n_results": len(results)})
            return output

        if name == "explain_song":
            title = inp["song_title"].strip()
            song = next(
                (s for s in self.songs
                 if title.lower() in s["title"].lower()
                 or s["title"].lower() in title.lower()),
                None,
            )
            if song is None:
                output = f"Song '{title}' not found in catalog."
            else:
                user_prefs = {
                    "genre": inp["genre"],
                    "mood": inp["mood"],
                    "energy": float(inp["energy"]),
                    "likes_acoustic": bool(inp.get("likes_acoustic", False)),
                }
                sc, expl = score_song(song, user_prefs)
                output = (
                    f"'{song['title']}' by {song['artist']} | score {sc:.2f} | {expl}\n"
                    f"Attributes — genre={song['genre']}, mood={song['mood']}, "
                    f"energy={song['energy']}, decade={song['release_decade']}, "
                    f"acousticness={song['acousticness']}, "
                    f"instrumentalness={song['instrumentalness']}, "
                    f"popularity={song['popularity']}, tags={song['mood_tags']}"
                )
            step["output"] = output
            self._reasoning_steps.append(step)
            logger.info("tool_result", extra={"tool": name, "song": title})
            return output

        logger.warning("unknown_tool", extra={"tool": name})
        return f"Unknown tool: {name}"

    # ──────────────────────────────────────────────────────────────────────────
    # Agentic loop
    # ──────────────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Process one user turn. Returns the assistant's final text reply."""
        self._history.append(
            types.Content(role="user", parts=[types.Part.from_text(user_message)])
        )
        self._reasoning_steps = []
        logger.info("user_message", extra={"message": user_message[:300]})

        for iteration in range(8):
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=self._history,
                config=types.GenerateContentConfig(
                    tools=[self._tools],
                    system_instruction=self._system_prompt,
                    max_output_tokens=1024,
                ),
            )

            if not response.candidates:
                logger.warning("no_candidates", extra={"iteration": iteration})
                break

            candidate = response.candidates[0]
            content = candidate.content
            self._history.append(content)

            logger.info(
                "llm_turn",
                extra={
                    "iteration": iteration,
                    "finish_reason": str(candidate.finish_reason),
                    "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                    "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else None,
                },
            )

            # Collect any function calls in this response
            function_calls = [
                p for p in (content.parts or [])
                if p.function_call and p.function_call.name
            ]

            if function_calls:
                tool_result_parts = []
                for part in function_calls:
                    fc = part.function_call
                    result = self._run_tool(fc.name, dict(fc.args))
                    tool_result_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )
                    )
                self._history.append(
                    types.Content(role="user", parts=tool_result_parts)
                )
                continue

            # No function calls — extract text and return
            text_parts = [
                p.text for p in (content.parts or [])
                if hasattr(p, "text") and p.text
            ]
            text = "\n".join(text_parts)
            logger.info("assistant_reply", extra={"length": len(text)})
            return text

        return (
            "I'm having trouble completing that request. "
            "Please try rephrasing or ask for something different."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear conversation history for a new session."""
        self._history = []
        self._last_profile = None
        self._reasoning_steps = []
        logger.info("session_reset")

    @property
    def last_profile(self) -> dict | None:
        """Most recently extracted user preference profile, or None."""
        return self._last_profile

    @property
    def reasoning_steps(self) -> list[dict]:
        """Intermediate tool calls from the most recent chat() turn.

        Each element: {"tool": str, "input": dict, "output": str}
        Exposed for the Streamlit UI and the evaluation harness.
        """
        return list(self._reasoning_steps)
