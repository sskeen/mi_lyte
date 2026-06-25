# ------------------------------------------------------ #
#
#   config.py
#
#   Shared configuration for mī lyte System 1:
#   LLM parameters, retriever settings, and prompt template.
#
#   Simone J. Skeen x Claude Code (02-02-2026)
#
# ------------------------------------------------------ #

from langchain_core.prompts import PromptTemplate
from .system_prompt import SYSTEM_PROMPT

# LLM parameters (Ollama)

LLM_PARAMS = {
    'model': 'deepseek-r1:14b',
    'base_url': 'http://localhost:11434',
    'temperature': 0.6,
    'top_p': 0.9,
    'top_k': 40,
    'num_ctx': 2048,
    'num_gpu': -1,
    'num_predict': 768,
    'repeat_last_n': 64,
    'stop': None,
    }

# embedding model

EMBEDDING_MODEL = 'nomic-embed-text'

# retriever parameters

RETRIEVER_PARAMS = {
    'search_type': 'similarity',
    'search_kwargs': {'k': 4},
    }

# prompt template

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
