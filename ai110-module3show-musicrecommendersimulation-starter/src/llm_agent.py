"""VibeFinder AI — LLM agent with RAG, agentic planning, and few-shot specialization.

Powered by Google Gemini via OpenAI-compatible endpoint (openai package, Python 3.8+).

Stretch features implemented here
──────────────────────────────────
RAG Enhancement (+2)
  knowledge_base.md appended to system prompt for richer explanations.

Agentic Workflow Enhancement (+2)
  plan_search tool + reasoning_steps property for observable decision chain.

Fine-Tuning / Specialization (+2)
  Three few-shot examples constrain voice, score-citation format, and reply structure.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from .recommender import apply_diversity_penalty, load_songs, recommend_songs, score_song

logger = logging.getLogger("vibefinder.agent")

# ──────────────────────────────────────────────────────────────────────────────
# Catalog-valid values
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
# Tool definitions (OpenAI function-calling format)
# ──────────────────────────────────────────────────────────────────────────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_search",
            "description": (
                "Call this FIRST when the user's request is ambiguous, uses activity language "
                "(e.g. 'gym', 'studying', 'road trip'), or mixes signals that need interpretation. "
                "State how you read the request and which catalog values you plan to use before "
                "calling get_recommendations. This step is observable to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interpretation": {"type": "string", "description": "How you read the user's request in plain English."},
                    "planned_genre":  {"type": "string"},
                    "planned_mood":   {"type": "string"},
                    "planned_energy": {"type": "number", "description": "Target energy 0.0-1.0 you will use."},
                    "reasoning":      {"type": "string", "description": "Why you chose these parameters."},
                },
                "required": ["interpretation", "planned_genre", "planned_mood", "planned_energy", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Query the VibeFinder scoring engine and return top song recommendations. "
                "Call plan_search first if the query needed interpretation. "
                "genre, mood, energy, and likes_acoustic are required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genre":              {"type": "string",  "description": "Must be a catalog genre."},
                    "mood":               {"type": "string",  "description": "Must be a catalog mood."},
                    "energy":             {"type": "number",  "description": "Target energy 0.0-1.0."},
                    "likes_acoustic":     {"type": "boolean"},
                    "preferred_decade":   {"type": "string"},
                    "desired_mood_tags":  {"type": "array", "items": {"type": "string"}},
                    "popularity_target":  {"type": "integer", "description": "Target popularity 0-100; -1 = no preference."},
                    "wants_instrumental": {"type": "boolean"},
                    "use_diversity":      {"type": "boolean", "description": "Apply diversity penalty to avoid same-artist repeats."},
                },
                "required": ["genre", "mood", "energy", "likes_acoustic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_song",
            "description": (
                "Return a detailed score breakdown for one specific song. "
                "Use when the user asks why a track was recommended or wants more detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "song_title":     {"type": "string",  "description": "Title (partial match OK)."},
                    "genre":          {"type": "string"},
                    "mood":           {"type": "string"},
                    "energy":         {"type": "number"},
                    "likes_acoustic": {"type": "boolean"},
                },
                "required": ["song_title", "genre", "mood", "energy", "likes_acoustic"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
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
- Only discuss music and recommendations. Politely redirect off-topic queries.
- Never invent songs, artists, or scores — all data must come from tool results.
- If a genre or mood is missing from the catalog, acknowledge it and suggest the closest alternative.

CATALOG VALID VALUES
  Genres : {genres}
  Moods  : {moods}
  Decades: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s
""".format(genres=_VALID_GENRES, moods=_VALID_MOODS)

_FEW_SHOT = """
--- FEW-SHOT EXAMPLES (follow this exact voice and format) ---

Example 1
User: something for studying
VibeFinder: Your late-night study playlist, ranked by fit:

🎵 **Library Rain** by Paper Lanterns (4.47/4.50) — genre match (lofi), mood match (chill), energy great fit (0.35), acoustic warmth (0.86).
🎵 **Midnight Coding** by LoRoom (4.46/4.50) — genre match (lofi), mood match (chill), energy great fit (0.42), acoustic warmth (0.71).
🎵 **Focus Flow** by LoRoom (3.48/4.50) — genre match (lofi), energy great fit (0.40), acoustic warmth (0.78).

Want instrumentals only, a 2020s era filter, or diversity mode so the same artist doesn't repeat?

---

Example 2
User: why did that first song score so high?
VibeFinder: **Library Rain** earned 4.47/4.50 — nearly perfect. Here's the breakdown:
- **Genre** (+2.0): lofi
- **Mood** (+1.0): chill
- **Energy** (+0.97): its energy (0.35) vs your target (0.38) — gap of only 0.03
- **Acoustic warmth** (+0.50): acousticness of 0.86

Want me to search for something with even lower energy, or add decade or tag filters?

---

Example 3
User: what's the weather like?
VibeFinder: I'm a music recommendation assistant, so weather forecasts are outside my lane! But if you want something that sounds like a rainy day, I can find moody lofi or blues tracks. Just say the word.

---
"""


def _build_catalog_snapshot(songs):
    lines = ["\nCATALOG SNAPSHOT (18 songs — used by the scoring engine):"]
    for s in songs:
        tags = ", ".join(s["mood_tags"]) if s["mood_tags"] else "-"
        lines.append(
            f"  * {s['title']} | {s['artist']} | genre={s['genre']} | mood={s['mood']}"
            f" | energy={s['energy']} | decade={s['release_decade']} | tags=[{tags}]"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class MusicAgent:
    """Conversational music recommendation agent using Gemini via OpenAI-compatible API."""

    def __init__(self, songs_path, knowledge_path=None):
        self._client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.songs = load_songs(songs_path)
        self._history = []          # list of message dicts (user/assistant/tool)
        self._last_profile = None
        self._reasoning_steps = []

        catalog = _build_catalog_snapshot(self.songs)
        self._system_prompt = _SYSTEM_CORE + catalog + _FEW_SHOT

        if knowledge_path and os.path.exists(knowledge_path):
            with open(knowledge_path, encoding="utf-8") as f:
                kb_text = f.read()
            self._system_prompt += (
                "\n\nMUSIC KNOWLEDGE BASE (use this to interpret activity language, "
                "mood psychology, and give richer explanations):\n\n" + kb_text
            )
            logger.info("knowledge_base_loaded", extra={"chars": len(kb_text)})

        logger.info("agent_initialized", extra={"catalog_size": len(self.songs),
                                                 "kb": knowledge_path is not None})

    # ──────────────────────────────────────────────────────────────────────────
    # Tool execution
    # ──────────────────────────────────────────────────────────────────────────

    def _run_tool(self, name, inp):
        logger.info("tool_called", extra={"tool": name, "input": inp})
        step = {"tool": name, "input": inp, "output": ""}

        if name == "plan_search":
            output = (
                "Plan: {}\nGenre -> {} | Mood -> {} | Energy -> {}\nReasoning: {}".format(
                    inp.get("interpretation", ""),
                    inp.get("planned_genre"),
                    inp.get("planned_mood"),
                    inp.get("planned_energy"),
                    inp.get("reasoning", ""),
                )
            )
            step["output"] = output
            self._reasoning_steps.append(step)
            return output

        if name == "get_recommendations":
            user_prefs = {
                "genre":             inp["genre"],
                "mood":              inp["mood"],
                "energy":            float(inp["energy"]),
                "likes_acoustic":    bool(inp.get("likes_acoustic", False)),
                "preferred_decade":  inp.get("preferred_decade", ""),
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
                "#{} '{}' by {}  (score {:.2f})  — {}".format(r, s["title"], s["artist"], sc, expl)
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
                output = "Song '{}' not found in catalog.".format(title)
            else:
                user_prefs = {
                    "genre":          inp["genre"],
                    "mood":           inp["mood"],
                    "energy":         float(inp["energy"]),
                    "likes_acoustic": bool(inp.get("likes_acoustic", False)),
                }
                sc, expl = score_song(song, user_prefs)
                output = (
                    "'{}' by {} | score {:.2f} | {}\n"
                    "Attributes — genre={}, mood={}, energy={}, decade={}, "
                    "acousticness={}, instrumentalness={}, popularity={}, tags={}".format(
                        song["title"], song["artist"], sc, expl,
                        song["genre"], song["mood"], song["energy"], song["release_decade"],
                        song["acousticness"], song["instrumentalness"],
                        song["popularity"], song["mood_tags"],
                    )
                )
            step["output"] = output
            self._reasoning_steps.append(step)
            logger.info("tool_result", extra={"tool": name, "song": title})
            return output

        logger.warning("unknown_tool", extra={"tool": name})
        return "Unknown tool: {}".format(name)

    # ──────────────────────────────────────────────────────────────────────────
    # Agentic loop
    # ──────────────────────────────────────────────────────────────────────────

    def chat(self, user_message):
        """Process one user turn. Returns the assistant's final text reply."""
        self._history.append({"role": "user", "content": user_message})
        self._reasoning_steps = []
        logger.info("user_message", extra={"user_msg": user_message[:300]})

        messages = [{"role": "system", "content": self._system_prompt}] + self._history

        for iteration in range(8):
            response = self._client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                tools=_TOOLS,
                max_tokens=1024,
            )

            choice = response.choices[0]
            msg = choice.message

            logger.info("llm_turn", extra={
                "iteration": iteration,
                "finish_reason": choice.finish_reason,
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "output_tokens": response.usage.completion_tokens if response.usage else None,
            })

            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                # Add assistant message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
                # Execute each tool and add results
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self._run_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            # End turn — extract text reply
            text = msg.content or ""
            self._history.append({"role": "assistant", "content": text})
            logger.info("assistant_reply", extra={"length": len(text)})
            return text

        return (
            "I'm having trouble completing that request. "
            "Please try rephrasing or ask for something different."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        self._history = []
        self._last_profile = None
        self._reasoning_steps = []
        logger.info("session_reset")

    @property
    def last_profile(self):
        return self._last_profile

    @property
    def reasoning_steps(self):
        return list(self._reasoning_steps)
