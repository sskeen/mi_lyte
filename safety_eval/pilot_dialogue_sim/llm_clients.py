# ------------------------------------------------------ #
#
#   llm_clients.py
#
#   LLM client wrappers for OpenAI and Ollama.
#   Creates fresh client instances per generation call.
#
#   Simone J. Skeen x Claude Code (06-23-2026)
#
# ------------------------------------------------------ #

from openai import OpenAI
from ollama import Client as OllamaClient

from config import OPENAI_CONFIG, OLLAMA_CONFIG


def get_openai_response(prompt: str, config: dict = None) -> dict:
    """
    Generate a response using OpenAI API with a fresh client instance.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    config : dict, optional
        Override default OPENAI_CONFIG settings

    Returns
    -------
    dict
        Response containing 'text' and 'token_count' keys
    """
    cfg = {**OPENAI_CONFIG, **(config or {})}

    # Fresh client instance per call
    client = OpenAI()

    response = client.chat.completions.create(
        model=cfg['model'],
        messages=[{'role': 'user', 'content': prompt}],
        temperature=cfg['temperature'],
        max_tokens=cfg['max_tokens'],
        reasoning_effort=cfg['reasoning_effort'],
    )

    return {
        'text': response.choices[0].message.content,
        'token_count': response.usage.completion_tokens,
        'model': cfg['model'],
    }


def get_ollama_response(prompt: str, config: dict = None) -> dict:
    """
    Generate a response using Ollama with a fresh client instance.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    config : dict, optional
        Override default OLLAMA_CONFIG settings

    Returns
    -------
    dict
        Response containing 'text' and 'token_count' keys
    """
    cfg = {**OLLAMA_CONFIG, **(config or {})}

    # Fresh client instance per call
    client = OllamaClient(host=cfg['base_url'])

    response = client.generate(
        model=cfg['model'],
        prompt=prompt,
        options={
            'temperature': cfg['temperature'],
            'num_predict': cfg['num_predict'],
        },
        think=cfg.get('think', False),
    )

    # Extract text, excluding thinking tags if present
    text = response.get('response', '')
    if cfg.get('think') and '<think>' in text:
        # Strip thinking section from output
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    return {
        'text': text,
        'token_count': response.get('eval_count', 0),
        'model': cfg['model'],
    }


def estimate_openai_cost(
    num_generations: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 150,
    pricing: dict = None,
) -> dict:
    """
    Estimate OpenAI API cost for a batch of generations.

    Parameters
    ----------
    num_generations : int
        Number of API calls to estimate
    avg_input_tokens : int
        Average input tokens per call
    avg_output_tokens : int
        Average output tokens per call
    pricing : dict, optional
        Override default pricing with 'input_per_million' and 'output_per_million'

    Returns
    -------
    dict
        Cost breakdown with 'input_cost', 'output_cost', and 'total_cost'
    """
    from config import OPENAI_PRICING
    price = pricing or OPENAI_PRICING

    total_input = num_generations * avg_input_tokens
    total_output = num_generations * avg_output_tokens

    input_cost = (total_input / 1_000_000) * price['input_per_million']
    output_cost = (total_output / 1_000_000) * price['output_per_million']

    return {
        'num_generations': num_generations,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost,
    }
