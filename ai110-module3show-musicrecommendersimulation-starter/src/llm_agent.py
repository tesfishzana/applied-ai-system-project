"""VibeFinder AI — LLM agent with RAG and agentic tool-use workflow.

Architecture
------------
1. RAG:  User query → Claude extracts a UserProfile from natural language → the
         VibeFinder scoring engine retrieves matching songs → Claude generates an
         explanation grounded in the actual retrieved scores.

2. Agentic loop: Claude may call tools across multiple turns:
     • get_recommendations — run the scoring engine with extracted preferences
     • explain_song        — deep-dive score breakdown for a specific track
   The loop terminates when Claude produces a final text reply (stop_reason="end_turn").

3. Prompt caching: the static system prompt + full catalog snapshot are cached at the
   Anthropic server (cache_control="ephemeral") to reduce latency and token cost on
   multi-turn conversations.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from .recommender import apply_diversity_penalty, load_songs, recommend_songs, score_song

logger = logging.getLogger("vibefinder.agent")

# ---------------------------------------------------------------------------
# Catalog-valid values used by the guardrail hint in the system prompt
# ---------------------------------------------------------------------------
_VALID_GENRES = (
    "pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, r&b, "
    "metal, country, reggae, blues, funk, classical"
)
_VALID_MOODS = (
    "happy, chill, intense, relaxed, focused, moody, peaceful, confident, "
    "romantic, angry, nostalgic, uplifting, melancholic, groovy"
)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
_TOOLS: list[dict] = [
    {
        "name": "get_recommendations",
        "description": (
            "Query the VibeFinder scoring engine and return the top song recommendations "
            "for a given set of user preferences. Call this whenever you have enough "
            "information about what the user wants to listen to. Genre, mood, energy, "
            "and likes_acoustic are required; all other fields are optional refinements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Preferred genre — must be one of the catalog genres.",
                },
                "mood": {
                    "type": "string",
                    "description": "Desired mood — must be one of the catalog moods.",
                },
                "energy": {
                    "type": "number",
                    "description": "Target energy level 0.0 (very calm) to 1.0 (very intense).",
                },
                "likes_acoustic": {
                    "type": "boolean",
                    "description": "True if the user prefers acoustic-sounding music.",
                },
                "preferred_decade": {
                    "type": "string",
                    "description": "Optional preferred release decade, e.g. '2020s' or '1990s'.",
                },
                "desired_mood_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional mood tags like 'dreamy', 'driving', 'peaceful'.",
                },
                "popularity_target": {
                    "type": "integer",
                    "description": "Optional target popularity 0-100. Pass -1 for no preference.",
                },
                "wants_instrumental": {
                    "type": "boolean",
                    "description": "True if the user wants instrumental-heavy tracks.",
                },
                "use_diversity": {
                    "type": "boolean",
                    "description": (
                        "Apply the diversity penalty to avoid recommending the same "
                        "artist or genre twice. Defaults to false."
                    ),
                },
            },
            "required": ["genre", "mood", "energy", "likes_acoustic"],
        },
    },
    {
        "name": "explain_song",
        "description": (
            "Return a detailed score breakdown for one specific song from the catalog. "
            "Use when the user asks why a particular track was recommended or requests "
            "more detail about a specific song."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title": {
                    "type": "string",
                    "description": "Title of the song (partial match is fine).",
                },
                "genre": {"type": "string"},
                "mood": {"type": "string"},
                "energy": {"type": "number"},
                "likes_acoustic": {"type": "boolean"},
            },
            "required": ["song_title", "genre", "mood", "energy", "likes_acoustic"],
        },
    },
]

# ---------------------------------------------------------------------------
# Static system prompt (cached at the server)
# ---------------------------------------------------------------------------
_SYSTEM_STATIC = """\
You are VibeFinder AI, a music recommendation assistant backed by a 18-song catalog \
and a rule-based scoring engine.

YOUR ROLE
• Listen carefully to what the user wants to listen to.
• Use the get_recommendations tool whenever you have enough preference information.
• Use the explain_song tool when the user asks about a specific track.
• After receiving tool results, craft a warm, concise reply that explains WHY each \
  song fits — reference the actual scores and signals (genre match, mood, energy, \
  acoustic warmth, era, tags) rather than generic descriptions.
• Offer to refine results if the user isn't satisfied.

GUARDRAILS
• Only discuss music and recommendations. If asked about unrelated topics, politely \
  redirect to music.
• Never invent songs, artists, or scores — all data must come from tool results.
• If a requested genre or mood is not in the catalog, acknowledge the gap and suggest \
  the closest available alternative.

CATALOG VALID VALUES
  Genres : {genres}
  Moods  : {moods}
  Decades: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s
""".format(genres=_VALID_GENRES, moods=_VALID_MOODS)


def _build_catalog_snapshot(songs: list[dict]) -> str:
    lines = ["\nCATALOG SNAPSHOT (18 songs — used by the scoring engine):"]
    for s in songs:
        tags = ", ".join(s["mood_tags"]) if s["mood_tags"] else "—"
        lines.append(
            f"  • {s['title']} | {s['artist']} | genre={s['genre']} | mood={s['mood']}"
            f" | energy={s['energy']} | decade={s['release_decade']} | tags=[{tags}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MusicAgent:
    """Conversational music recommendation agent powered by Claude + VibeFinder scoring.

    Usage
    -----
    agent = MusicAgent("data/songs.csv")
    reply = agent.chat("I want something chill to study to")
    print(reply)
    agent.reset()   # clear history for a new conversation
    """

    def __init__(self, songs_path: str) -> None:
        self._client = anthropic.Anthropic()
        self.songs = load_songs(songs_path)
        self._history: list[dict] = []
        self._last_profile: dict | None = None

        # Combine static prompt + catalog into one cacheable block
        catalog = _build_catalog_snapshot(self.songs)
        self._system_blocks: list[dict] = [
            {
                "type": "text",
                "text": _SYSTEM_STATIC + catalog,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        logger.info("agent_initialized", extra={"catalog_size": len(self.songs)})

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _run_tool(self, name: str, inp: dict) -> str:
        logger.info("tool_called", extra={"tool": name, "input": inp})

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

            # Retrieve — fetch full pool when diversity penalty is on
            fetch_k = len(self.songs) if use_div else 5
            results = recommend_songs(user_prefs, self.songs, k=fetch_k)
            if use_div:
                results = apply_diversity_penalty(results, k=5)

            lines = []
            for rank, (song, sc, expl) in enumerate(results, 1):
                lines.append(
                    f"#{rank}  '{song['title']}' by {song['artist']}"
                    f"  (score {sc:.2f})  — {expl}"
                )
            output = "\n".join(lines)
            logger.info("tool_result", extra={"tool": name, "n_results": len(results)})
            return output

        if name == "explain_song":
            title = inp["song_title"].strip()
            # Partial, case-insensitive match
            song = next(
                (
                    s for s in self.songs
                    if title.lower() in s["title"].lower()
                    or s["title"].lower() in title.lower()
                ),
                None,
            )
            if song is None:
                return f"Song '{title}' was not found in the catalog."
            user_prefs = {
                "genre": inp["genre"],
                "mood": inp["mood"],
                "energy": float(inp["energy"]),
                "likes_acoustic": bool(inp.get("likes_acoustic", False)),
            }
            sc, expl = score_song(song, user_prefs)
            output = (
                f"'{song['title']}' by {song['artist']}"
                f" | score {sc:.2f} | {expl}"
                f"\nFull attributes — genre={song['genre']}, mood={song['mood']},"
                f" energy={song['energy']}, decade={song['release_decade']},"
                f" acousticness={song['acousticness']}, instrumentalness={song['instrumentalness']},"
                f" popularity={song['popularity']}, tags={song['mood_tags']}"
            )
            logger.info("tool_result", extra={"tool": name, "song": title, "score": sc})
            return output

        logger.warning("unknown_tool", extra={"tool": name})
        return f"Unknown tool: {name}"

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process one user turn through the RAG + agentic loop.

        Returns the assistant's final text reply.
        """
        self._history.append({"role": "user", "content": user_message})
        logger.info("user_message", extra={"message": user_message[:300]})

        for iteration in range(6):  # safety cap prevents runaway tool loops
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=self._system_blocks,
                tools=_TOOLS,
                messages=self._history,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            logger.info(
                "llm_turn",
                extra={
                    "iteration": iteration,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

            if response.stop_reason == "end_turn":
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                self._history.append(
                    {"role": "assistant", "content": response.content}
                )
                logger.info("assistant_reply", extra={"length": len(text)})
                return text

            if response.stop_reason == "tool_use":
                self._history.append(
                    {"role": "assistant", "content": response.content}
                )
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._run_tool(block.name, block.input),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ]
                self._history.append({"role": "user", "content": tool_results})
                continue

            logger.warning(
                "unexpected_stop", extra={"stop_reason": response.stop_reason}
            )
            break

        return (
            "I'm having trouble completing that request. "
            "Please try rephrasing or ask for something different."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history for a new session."""
        self._history = []
        self._last_profile = None
        logger.info("session_reset")

    @property
    def last_profile(self) -> dict | None:
        """The most recently extracted user preference profile, or None."""
        return self._last_profile
