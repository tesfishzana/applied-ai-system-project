"""VibeFinder AI — Streamlit chat interface.

Run from the project directory:
    streamlit run app.py

Requires:
    ANTHROPIC_API_KEY environment variable (or a .env file in the same directory).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger import setup_logging
from src.llm_agent import MusicAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
_SONGS_PATH = os.path.join(_DIR, "data", "songs.csv")
_KB_PATH = os.path.join(_DIR, "data", "knowledge_base.md")
_LOGS_DIR = os.path.join(_DIR, "logs")

# ---------------------------------------------------------------------------
# Streamlit page config (must be the first st call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VibeFinder AI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging (idempotent — safe across Streamlit hot-reloads)
# ---------------------------------------------------------------------------

setup_logging(_LOGS_DIR)
_log = logging.getLogger("vibefinder.app")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    kb = _KB_PATH if os.path.exists(_KB_PATH) else None
    st.session_state.agent = MusicAgent(_SONGS_PATH, knowledge_path=kb)
    _log.info("streamlit_session_start", extra={"kb_loaded": kb is not None})

if "messages" not in st.session_state:
    # Each entry: {"role": str, "content": str, "steps": list|None}
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎵 VibeFinder AI")
    st.caption("Claude-powered music recommendations")

    st.divider()

    st.subheader("Conversation")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.messages = []
        _log.info("streamlit_new_conversation")
        st.rerun()

    st.divider()

    last = st.session_state.agent.last_profile
    if last:
        st.subheader("Detected profile")
        display = {k: v for k, v in last.items() if v not in (None, "", [], -1, False)}
        st.json(display, expanded=True)
    else:
        st.info("No profile detected yet.\nAsk for a recommendation to see your preferences here.")

    st.divider()

    with st.expander("How it works"):
        st.markdown(
            """
**RAG pipeline**
1. Your query → Claude interprets it using the music knowledge base.
2. Claude calls `get_recommendations` with extracted preferences.
3. The VibeFinder scoring engine scores all 18 catalog songs.
4. Claude explains results using actual scores — no hallucinated data.

**Agentic planning**
For activity-based queries ("gym", "studying"), Claude calls `plan_search`
first to show its reasoning, then searches the catalog.

**Multi-turn refinement**
Say *"make it more energetic"*, *"explain that first song"*, or
*"try with diversity on"* — Claude refines across turns.

**Scoring signals**
Genre (+2.0) · Mood (+1.0) · Energy fit (+0–1.0) · Acoustic (+0.5)
Era (+0.25) · Tags (+0.10 each) · Popularity (+0.20) · Instrumental (+0.25)
            """
        )

    with st.expander("Log file"):
        log_path = os.path.join(_LOGS_DIR, "vibefinder.log")
        st.code(os.path.abspath(log_path))
        if os.path.exists(log_path):
            size_kb = os.path.getsize(log_path) / 1024
            st.caption(f"{size_kb:.1f} KB written")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🎵 VibeFinder AI")
st.caption(
    "Describe the music you're in the mood for — genre, vibe, activity, or feeling. "
    "I'll find the best tracks and explain exactly why they fit."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "**GEMINI_API_KEY is not set.**\n\n"
        "```\nexport GEMINI_API_KEY=AIza...\nstreamlit run app.py\n```\n"
        "Or create a `.env` file:\n```\nGEMINI_API_KEY=AIza...\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Render chat history (with optional reasoning steps expanders)
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show agentic reasoning steps stored with this message
        steps = msg.get("steps") or []
        if steps:
            with st.expander(f"🔍 Reasoning steps ({len(steps)} tool call{'s' if len(steps) != 1 else ''})"):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i} — `{step['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(step["input"], expanded=False)
                    with col2:
                        st.caption("Output")
                        st.text(step["output"][:400] + ("…" if len(step["output"]) > 400 else ""))
                    if i < len(steps):
                        st.divider()

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hey! I'm VibeFinder AI. Tell me what kind of music you're in the mood "
            "for — a genre, a vibe, an activity, a feeling — and I'll find the best "
            "tracks from our catalog for you. 🎧"
        )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("e.g. Something chill to study to late at night…"):
    st.session_state.messages.append({"role": "user", "content": prompt, "steps": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding your vibe…"):
            try:
                reply = st.session_state.agent.chat(prompt)
                steps = st.session_state.agent.reasoning_steps
            except Exception:
                _log.error("agent_error", exc_info=True)
                reply = (
                    "Sorry, something went wrong. "
                    "Please check that your `ANTHROPIC_API_KEY` is valid and try again."
                )
                steps = []

        st.markdown(reply)

        # Render reasoning steps inline after the reply
        if steps:
            with st.expander(f"🔍 Reasoning steps ({len(steps)} tool call{'s' if len(steps) != 1 else ''})"):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i} — `{step['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(step["input"], expanded=False)
                    with col2:
                        st.caption("Output")
                        st.text(step["output"][:400] + ("…" if len(step["output"]) > 400 else ""))
                    if i < len(steps):
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": reply, "steps": steps})
    st.rerun()
