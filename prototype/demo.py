# --------------------------------------------------------------------------- #
#
#   demo.py
#
#   Streamlit chat application for mī lyte System 1.
#   Provides a user-facing interface for RAG-based mindfulness guidance.
#
#   This file imports configuration and streaming logic from src/ modules,
#   keeping the demo self-contained and easy to run.
#
#   Usage:
#       cd prototype
#       streamlit run demo.py
#
#   Simone J. Skeen x Claude Code (07-11-2026)
#   WIP - NOT FOR DISTRIBUTION
#
# --------------------------------------------------------------------------- #

import os
import sys
import time
from pathlib import Path

# === FIX OPENMP CONFLICT ON MACOS === #
# FAISS and other libraries may link multiple OpenMP runtimes.
# This environment variable must be set before importing those libraries.

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st

# === ADD PROJECT ROOT TO PATH === #
# This allows importing from src/ modules when running from prototype/.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === IMPORT FROM SRC MODULES === #
# Centralized configuration and streaming logic.
# - config.py: LLM parameters, retriever settings, prompt template
# - dialogue_stream.py: Unified streaming function with mode parameter

from src.config import LLM_PARAMS, EMBEDDING_MODEL, RETRIEVER_PARAMS, PROMPT_TEMPLATE, OLLAMA_EMBED_HOST
from src.dialogue_stream import query_and_stream


# --------------------------------------------------------------------------- #
#   INITIALIZATION
# --------------------------------------------------------------------------- #

# === LOAD VECTOR INDEX + RETRIEVER === #
# The FAISS index is pre-built from knowledge base PDFs.
# See diagnostic.ipynb for the build process.

embedding = OllamaEmbeddings(
    model = EMBEDDING_MODEL,
    base_url = OLLAMA_EMBED_HOST,  # Embeddings run locally
)

db = FAISS.load_local(
    '../src/faiss_index',
    embeddings = embedding,
    allow_dangerous_deserialization = True,  # Required for loading saved FAISS index
)

retriever = db.as_retriever(**RETRIEVER_PARAMS)

# === CONFIGURE LLM === #
# Uses Ollama with DeepSeek-R1 for local inference.
# Parameters defined in src/config.py.

llm = OllamaLLM(**LLM_PARAMS)

# === PROMPT TEMPLATE === #
# Imported from config - includes system prompt with tone/style guidance.

prompt_template = PROMPT_TEMPLATE


# --------------------------------------------------------------------------- #
#   STREAMLIT UI CONFIGURATION
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title = "mī lyte",
    page_icon = "🍂",
    layout = 'centered',
)

# === WELCOME MESSAGE === #
# Disclaimer about the tool's limitations and scope.

st.write(
    "**mī lyte** is a tool to provide knowledge on stress and resilience, "
    "recommending evidence-based informal mindfulness practices to build into your everyday life."
    "\n\n"
    "**mī lyte** is _not_ a replacement for in-person therapy. "
    "It does not provide medical or psychiatric advice. "
)

# === SIDEBAR BRANDING === #

with st.sidebar:
    st.title("🍂mī lyte")
    st.subheader('Accessible mindfulness skills for everyday life')
    st.write("\n\n")
    st.write("\n\n")

    # mHEAL logo from GitHub
    logo_url = "https://github.com/sskeen/mi_lyte/blob/main/images/mheal_logo.png?raw=true"
    st.image(logo_url, output_format = "PNG", width = 120)

    st.caption(
        "A prototype developed by "
        "[mHEAL: the Mindfulness for Health Equity Lab](https://sites.brown.edu/mheal/) "
        "at Brown University, grounded in _Mindfulness-Based Queer Resilience_ © Dr. Shufang Sun. "
    )


# --------------------------------------------------------------------------- #
#   CHAT HISTORY MANAGEMENT
# --------------------------------------------------------------------------- #

# === INITIALIZE SESSION STATE === #
# Streamlit session state persists across reruns within a session.

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# === QUERY INPUT === #

user_input = st.chat_input("I could use some help with...")

# === DISPLAY PREVIOUS EXCHANGES === #
# Render all prior conversation turns from session state.

for user_q, assistant_r in st.session_state.chat_history:
    with st.chat_message('user'):
        st.markdown(user_q)
    with st.chat_message('assistant'):
        st.markdown(assistant_r)


# --------------------------------------------------------------------------- #
#   QUERY PROCESSING
# --------------------------------------------------------------------------- #

if user_input:

    # === DISPLAY USER MESSAGE === #

    with st.chat_message("user"):
        st.markdown(user_input)

    # === STREAM ASSISTANT RESPONSE === #

    with st.chat_message("assistant"):

        # Placeholder for streaming updates
        response_container = st.empty()

        # Track visible text and hidden state
        visible_text = ""
        hidden = False  # True when inside <think>...</think> block

        # === STREAM FROM RAG PIPELINE === #
        # Use mode='ui' to get masked output with sentinel events.
        # The streaming function yields:
        #   - "__HIDDEN_ON__"  when entering <think> block
        #   - "__HIDDEN_OFF__" when exiting </think> block
        #   - text tokens otherwise

        for part in query_and_stream(
            llm,
            retriever,
            user_input,
            prompt_template,
            mode = 'ui',
        ):

            # --- HANDLE SENTINEL EVENTS --- #

            if part == "__HIDDEN_ON__":
                # Model entered reasoning block - show "Generating..." indicator
                hidden = True
                response_container.markdown(visible_text + " _Generating..._")
                continue

            if part == "__HIDDEN_OFF__":
                # Model exited reasoning block - resume normal streaming
                hidden = False
                response_container.markdown(visible_text + "▍")
                continue

            # --- HANDLE NORMAL TOKENS --- #

            visible_text += part

            if not hidden:
                # Visible streaming - show cursor animation
                response_container.markdown(visible_text + "▍")
                time.sleep(0.005)  # Small delay for visual streaming effect
            else:
                # Still in hidden mode - maintain "Generating..." indicator
                response_container.markdown(visible_text + " _Generating..._")

        # === FINALIZE RESPONSE === #
        # Remove cursor after stream completes.

        response_container.markdown(visible_text)

    # === STORE IN CHAT HISTORY === #
    # Append the exchange for multi-turn conversation support.

    st.session_state.chat_history.append((user_input, visible_text))
