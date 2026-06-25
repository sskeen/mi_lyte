# ------------------------------------------------------ #
#
#   config.py
#
#   Pipeline configuration for crisis query simulation.
#   Configures LLM parameters, generation settings,
#   and file paths.
#
#   Simone J. Skeen x Claude Code (06-23-2026)
#
# ------------------------------------------------------ #

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Pipeline settings
PIPELINE_CONFIG = {
    'num_queries': 10,              # Queries to generate per persona/seed/model combo
    'random_seed': 56,              # Random seed for reproducibility
}

# Rate limiting (OpenAI API)
RATE_LIMIT_CONFIG = {
    'delay_seconds': 1.0,           # Delay between OpenAI API calls
}

# OpenAI configuration
# NOTE: max_tokens for reasoning models includes BOTH reasoning + output tokens
OPENAI_CONFIG = {
    'model': 'gpt-5.4-mini-2026-03-17',
    #'temperature': 0.8,    ### Not supported on GPT reasoning models
    'max_tokens': 4000,               # Must be high enough for reasoning + output
    'reasoning_effort': 'medium',     ### low, medium, high
    'seed': PIPELINE_CONFIG['random_seed'],
}

# Ollama configuration
OLLAMA_CONFIG = {
    'model': 'qwen3:30b',
    'base_url': 'http://localhost:11434',
    'temperature': 0.6,
    'num_predict': 4000,
    'think': True,                  # Enable Qwen3 thinking mode
    'seed': PIPELINE_CONFIG['random_seed'],
}

# File paths
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

INPUT_FILES = {
    'personas': DATA_DIR / 'personas.tsv',
    'seed_phrases': DATA_DIR / 'seed_phrases.tsv',
    'persona_context': DATA_DIR / 'persona_context.txt',
}

OUTPUT_FILES = {
    'generated_queries': OUTPUT_DIR / 'generated_queries.tsv',
    'checkpoint': OUTPUT_DIR / '.checkpoint.json',
}

# Cost estimation (OpenAI pricing per 1M tokens, approximate)
OPENAI_PRICING = {
    'input_per_million': 0.15,      # $ per 1M input tokens
    'output_per_million': 0.60,     # $ per 1M output tokens
}
