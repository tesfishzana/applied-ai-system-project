"""VibeFinder AI — Streamlit chat interface.

Run from the project directory:
    streamlit run app.py

Requires:
    ANTHROPIC_API_KEY environment variable (or a .env file in the same directory).
"""

from __future__ import annotations

import logging
import os
import sys

import streamlit as st

# Allow `from src.X import ...` when launched from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env if present (optional convenience — never required in production)
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
    st.session_state.agent = MusicAgent(_SONGS_PATH)
    _log.info("streamlit_session_start")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

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

    # Show the most recently extracted preference profile
    last = st.session_state.agent.last_profile
    if last:
        st.subheader("Detected profile")
        display = {
            k: v
            for k, v in last.items()
            if v not in (None, "", [], -1, False)
        }
        st.json(display, expanded=True)
    else:
        st.info("No profile detected yet.\nAsk for a recommendation to see your preferences here.")

    st.divider()

    with st.expander("How it works"):
        st.markdown(
            """
**RAG pipeline**
1. You describe what you want to hear in plain English.
2. Claude calls the `get_recommendations` tool with extracted preferences.
3. The VibeFinder scoring engine scores all 18 catalog songs.
4. Claude explains the top results using the actual scores as grounding.

**Agentic refinement**
Say *"make it more energetic"*, *"I don't like that artist"*, or
*"explain the first song"* — Claude will call the tools again and
refine the results across multiple turns.

**Scoring signals**
- Genre match (+2.0) · Mood match (+1.0)
- Energy fit (+0–1.0) · Acoustic warmth (+0.5)
- Era / decade (+0.25) · Mood tags (+0.10 each)
- Popularity fit (+0.20) · Instrumental (+0.25)
            """
        )

    with st.expander("Log file"):
        log_path = os.path.join(_LOGS_DIR, "vibefinder.log")
        st.code(os.path.abspath(log_path))
        if os.path.exists(log_path):
            size_kb = os.path.getsize(log_path) / 1024
            st.caption(f"{size_kb:.1f} KB written")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.title("🎵 VibeFinder AI")
st.caption(
    "Describe the music you're in the mood for and I'll find the perfect tracks "
    "from the catalog. You can refine results across multiple turns."
)

# ---------------------------------------------------------------------------
# API key guard
# ---------------------------------------------------------------------------

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "**ANTHROPIC_API_KEY is not set.**\n\n"
        "Add it to your environment before running:\n"
        "```\nexport ANTHROPIC_API_KEY=sk-ant-...\nstreamlit run app.py\n```\n"
        "Or create a `.env` file in the project directory containing:\n"
        "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show a welcome message when the conversation is empty
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
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the agent and render its reply
    with st.chat_message("assistant"):
        with st.spinner("Finding your vibe…"):
            try:
                reply = st.session_state.agent.chat(prompt)
            except Exception:
                _log.error("agent_error", exc_info=True)
                reply = (
                    "Sorry, something went wrong while contacting the AI service. "
                    "Please check that your `ANTHROPIC_API_KEY` is valid and try again."
                )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
