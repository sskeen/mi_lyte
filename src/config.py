# ------------------------------------------------------ #
#
#   config.py
#
#   Shared configuration for mī lyte System 1:
#   LLM parameters, retriever settings, and prompt template.
#
#   Simone J. Skeen x Claude Code (07-11-2026)
#
# ------------------------------------------------------ #

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from .system_prompt import SYSTEM_PROMPT

# === LOAD ENVIRONMENT VARIABLES === #
# Load .env from project root (parent of src/)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / '.env')

# === OLLAMA HOST === #
# Set OLLAMA_HOST in .env for remote GPU server, e.g.:
#   OLLAMA_HOST=http://192.168.1.100:11434
# Defaults to localhost if not specified.

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# === LLM PARAMETERS === #

### local Ollama models
# 'deepseek-r1:14b'
# 'nomic-embed-text:latest' ### OLLAMA_EMBED_HOST only
# 'qwen3:30b'
# 'qwen3-coder:30b'

### GPU Ollama models
# 'qwen3:30b'
# 'qwen3.5-heretic-35b-q4km'
# 'deepseek-r1:32b'
# 'deepseek-r1:70b'
# 'qwen3-coder:30b'
# 'qwen2.5:32b'

LLM_PARAMS = {
    'model': 'deepseek-r1:14b', 
    'base_url': OLLAMA_HOST,
    'temperature': 0.6,
    'top_p': 0.9,
    'top_k': 40,
    'num_ctx': 2048,
    'num_gpu': -1,
    'num_predict': 768,
    'repeat_last_n': 64,
    'stop': None,
}

# === EMBEDDING HOST === #
# Embeddings run locally (nomic-embed-text not on remote GPU).
# Override with OLLAMA_EMBED_HOST in .env if needed.

OLLAMA_EMBED_HOST = os.environ.get('OLLAMA_EMBED_HOST', 'http://localhost:11434')

# === EMBEDDING MODEL === #

EMBEDDING_MODEL = 'nomic-embed-text'

# === RETRIEVER PARAMETERS === #

RETRIEVER_PARAMS = {
    'search_type': 'similarity',
    'search_kwargs': {'k': 4},
}

# === PROMPT TEMPLATE === #

PROMPT_TEMPLATE = PromptTemplate(
    input_variables = ['context', 'question'],
    template = '''
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        '''.strip(),
).partial(system_prompt = SYSTEM_PROMPT)
