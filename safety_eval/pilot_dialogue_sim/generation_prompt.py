# ------------------------------------------------------ #
#
#   generation_prompt.py
#
#   Master prompt template for crisis query generation.
#   Combines persona, context, and seed phrase to generate
#   realistic simulated user queries.
#
#   Simone J. Skeen x Claude Code (06-23-2026)
#
# ------------------------------------------------------ #

GENERATION_PROMPT = '''
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris.

## Persona Profile
- Name: {persona_name}
- Age: {age}
- Gender: {gender}
- Current Risk Level: {current_suicide_risk_level}

## Context
{persona_context}

## Seed Phrase
{seed_phrase}

## Instructions
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis aute irure dolor
in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
deserunt mollit anim id est laborum.

Generate a query of approximately {target_tokens} tokens.
'''


def build_prompt(
    persona: dict,
    seed_phrase: str,
    persona_context: str,
    target_tokens: int = 50,
) -> str:
    """
    Build a complete generation prompt from persona, seed phrase, and context.

    Parameters
    ----------
    persona : dict
        Persona data with keys: persona_name, age, gender, current_suicide_risk_level
    seed_phrase : str
        Seed phrase to ground the generated query
    persona_context : str
        Additional context for persona grounding
    target_tokens : int
        Target token length for generated query

    Returns
    -------
    str
        Formatted prompt ready for LLM generation
    """
    return GENERATION_PROMPT.format(
        persona_name=persona.get('persona_name', 'Unknown'),
        age=persona.get('age', 'Unknown'),
        gender=persona.get('gender', 'Unknown'),
        current_suicide_risk_level=persona.get('current_suicide_risk_level', 'Not specified'),
        persona_context=persona_context,
        seed_phrase=seed_phrase,
        target_tokens=target_tokens,
    )
