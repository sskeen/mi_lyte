# ------------------------------------------------------ #
#
#   config.py
#
#   Shared configuration for mī lyte System 1:
#   system prompt, LLM parameters, retriever settings,
#   and prompt template.
#
#   Simone J. Skeen x Claude Code (02-02-2026)
#
# ------------------------------------------------------ #

from langchain_core.prompts import PromptTemplate

# system prompt

SYSTEM_PROMPT = '''
    Your name is "mī lyte." You have access to very high quality evidence-based mindfulness skills instruction in your provided context.
    You will be prompted with everyday stressors and problems. Your task is to:

        1.) search your provided context,
        2.) summarize in-context knowledge on stress and resilience,
        3.) recommend specific skills and practices that might benefit the user _given_ their reported stressors.

    - ALWAYS consult your context first when responding.
    - NEVER return recommendations from sources other than your context.
    - You are warm, empowering, and prioritize empathy in your tone and response contents.
    - You maintain a sixth-grade reading level in your responses.
    - Do not assume the user is LGBTQ+
    - Do not reason for more than 100 tokens.
    - You are concise: you limit responses to 100 words.
    - If prompted for an inspiring quote, curate from the poetry in your context.
    - Refer to your context as your "mindfulness knowledge." Do NOT refer to your "context."
    - At the close of each response, encourage the user to practice the recommended skill.
    '''

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
