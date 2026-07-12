# ------------------------------------------------------ #
#
#   judge_prompt.py
#
#   Prompt template for LLM-as-judge evaluation of
#   RAG chatbot responses.
#
#   Simone J. Skeen x Claude Code (07-12-2026)
#
# ------------------------------------------------------ #

JUDGE_PROMPT = '''
# Lorem Ipsum Placeholder

[Replace this section with your specific evaluation criteria and instructions for the judge model.]

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

## Response to Evaluate
{demo_response}

## Rating Instructions
Evaluate the response above and provide:
1. A rating of 1, 2, or 3
2. A three-sentence explanation for your rating

## Output Format
You MUST respond in exactly this format:

Rating: [1, 2, or 3]
Rationale: [Three sentence explanation for your rating decision.]
'''


def build_judge_prompt(demo_response: str) -> str:
    """
    Build judge prompt from demo response.

    Parameters
    ----------
    demo_response : str
        The chatbot response to evaluate

    Returns
    -------
    str
        Formatted prompt ready for judge LLM
    """
    return JUDGE_PROMPT.format(demo_response=demo_response)
