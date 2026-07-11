# ------------------------------------------------------ #
#
#   dialogue_stream.py
#
#   Unified RAG streaming wrapper for mī lyte System 1.
#   Supports two modes:
#     - 'diagnostic': exposes reasoning traces + sources (for notebooks)
#     - 'ui': masks <think> tags, yields sentinel events (for Streamlit)
#
#   Simone J. Skeen x Claude Code (07-11-2026)
#
# ------------------------------------------------------ #

from langchain_core.prompts import PromptTemplate
import ollama


# === DEFAULT PROMPT TEMPLATE === #
# Used when no prompt_template is passed to query_and_stream().
# Provides a minimal, context-aware conversational template.

_DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables = ['context', 'question'],
    template = '''
You are a knowledgable conversational agent that offers accurate, succinct, responses
    based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:'''
)


def query_and_stream(
    llm,
    retriever,
    query,
    prompt_template = None,
    mode = 'diagnostic',
    show_sources = True,
):
    """
    Unified RAG streaming wrapper.

    This function replicates RetrievalQA (chain_type='stuff') behavior with
    real-time token streaming. It supports two distinct output modes for
    different use cases.

    Parameters:
        llm: Ollama LLM instance (e.g., deepseek-r1:14b via LangChain wrapper)
        retriever: FAISS retriever configured via RETRIEVER_PARAMS
        query: User's question string
        prompt_template: PromptTemplate with {context} and {question} placeholders.
                         If None, uses a minimal default template.
        mode: Output mode - one of:
            'diagnostic' - Exposes full reasoning trace (<think>...</think>),
                           prints sources, outputs directly to stdout.
                           Use this in Jupyter notebooks for inspection.
            'ui' - Masks <think> tags, yields tokens and sentinel events.
                   Use this in Streamlit for clean user-facing output.
        show_sources: If True and mode='diagnostic', prints document excerpts
                      after the response. Ignored in 'ui' mode.

    Returns:
        mode='diagnostic': None (prints directly to stdout)
        mode='ui': Generator[str] yielding tokens and sentinel events

    Sentinel events (ui mode only):
        __HIDDEN_ON__  - Model entered <think> reasoning block
        __HIDDEN_OFF__ - Model exited </think>, visible response follows

    Example usage (diagnostic mode):
        query_and_stream(llm, retriever, "How do I manage stress?",
                         PROMPT_TEMPLATE, mode='diagnostic', show_sources=True)

    Example usage (ui mode):
        for token in query_and_stream(llm, retriever, query, PROMPT_TEMPLATE, mode='ui'):
            if token == "__HIDDEN_ON__":
                # show "generating..." indicator
            elif token == "__HIDDEN_OFF__":
                # resume normal display
            else:
                # append token to response
    """

    # === VALIDATE MODE === #
    # Fail fast if an invalid mode is provided.

    if mode not in ('diagnostic', 'ui'):
        raise ValueError(f"Invalid mode: '{mode}'. Use 'diagnostic' or 'ui'.")

    # === MODE DISPATCH === #
    # This function is NOT a generator itself - it either:
    #   - Executes directly and returns None (diagnostic mode)
    #   - Returns a generator object (ui mode)
    #
    # This design ensures diagnostic mode executes immediately without
    # requiring the caller to iterate over a generator.

    if mode == 'diagnostic':
        # Execute directly, print to stdout, return None
        _run_diagnostic(llm, retriever, query, prompt_template, show_sources)
        return None
    else:
        # mode == 'ui'
        # Return the generator object for caller to iterate
        return _run_ui(llm, retriever, query, prompt_template)


# ------------------------------------------------------ #
#   DIAGNOSTIC MODE RUNNER
# ------------------------------------------------------ #

def _run_diagnostic(llm, retriever, query, prompt_template, show_sources):
    """
    Execute diagnostic mode: retrieval + streaming with exposed reasoning.

    This is a regular function (not a generator) that prints directly to stdout.
    Handles the full RAG pipeline for diagnostic/inspection use.

    Parameters:
        llm: Ollama LLM instance
        retriever: FAISS retriever
        query: User's question string
        prompt_template: PromptTemplate (or None for default)
        show_sources: If True, print document excerpts after response
    """

    # === RETRIEVAL === #
    docs = retriever.invoke(query)

    if not docs:
        print("No relevant documents found.")
        return

    # === CONTEXT ASSEMBLY === #
    context = "\n\n".join(doc.page_content for doc in docs)

    # === PROMPT FORMATTING === #
    if prompt_template is None:
        prompt_template = _DEFAULT_PROMPT_TEMPLATE

    prompt = prompt_template.format(
        context = context,
        question = query,
    )

    # === STREAM RESPONSE === #
    _stream_diagnostic(llm, prompt, docs, show_sources)


def _stream_diagnostic(llm, prompt, docs, show_sources):
    """
    Stream response with exposed reasoning trace (diagnostic mode).

    Uses the native ollama library with think=True to expose the model's
    internal reasoning process. Prints everything directly to stdout,
    making this ideal for Jupyter notebook inspection.

    Parameters:
        llm: Ollama LLM instance (used to extract model name)
        prompt: Fully formatted prompt string with context and question
        docs: List of retrieved documents (for source display)
        show_sources: If True, print document excerpts after response

    Output format:
        <think>
        [model's internal reasoning...]
        </think>

        [visible response to user]

        knowledge excerpts:
        --- excerpt 1 ---
        metadata: {...}
        [first 1000 chars of document]...
    """

    print("\n🍂\n")

    # === EXTRACT MODEL NAME === #
    # The LangChain Ollama wrapper stores the model name in llm.model.
    # Fall back to default if not found.

    model_name = llm.model if hasattr(llm, 'model') else 'deepseek-r1:14b'

    # === NATIVE OLLAMA STREAMING WITH REASONING === #
    # Use the native ollama library (not LangChain wrapper) to access
    # the think=True parameter, which exposes the model's reasoning trace.

    stream = ollama.chat(
        model = model_name,
        messages = [{'role': 'user', 'content': prompt}],
        think = True,   # Enable reasoning trace exposure
        stream = True,  # Stream tokens as they're generated
    )

    # === PROCESS STREAMING CHUNKS === #
    # Track whether we're inside the thinking block to properly
    # format the <think> delimiters.

    in_thinking = False

    for chunk in stream:

        # Check if this chunk contains thinking content
        if chunk.message.thinking and not in_thinking:
            # First thinking token - print opening tag
            in_thinking = True
            print('<think>\n', end = '')

        if chunk.message.thinking:
            # Inside thinking block - print reasoning tokens
            print(chunk.message.thinking, end = '', flush = True)

        elif chunk.message.content:
            # Response content (not thinking)
            if in_thinking:
                # Just exited thinking block - print closing tag
                print('\n</think>\n\n', end = '')
                in_thinking = False
            # Print response token
            print(chunk.message.content, end = '', flush = True)

    print("\n\n🍂")

    # === SOURCE DOCUMENTS === #
    # Optionally print retrieved document metadata and excerpts
    # for verification and debugging.

    if show_sources:
        print("\nknowledge excerpts:\n")
        for i, doc in enumerate(docs):
            meta = doc.metadata
            print(f"--- excerpt {i+1} ---")
            print(f"metadata: {meta}")
            # Print first 1000 characters to keep output manageable
            print(doc.page_content[:1000], "...\n")


# ------------------------------------------------------ #
#   UI MODE RUNNER
# ------------------------------------------------------ #

def _run_ui(llm, retriever, query, prompt_template):
    """
    Execute UI mode: retrieval + streaming with <think> tag masking.

    This is a generator function that yields tokens for Streamlit consumption.
    Handles the full RAG pipeline for user-facing UI.

    Parameters:
        llm: Ollama LLM instance
        retriever: FAISS retriever
        query: User's question string
        prompt_template: PromptTemplate (or None for default)

    Yields:
        str: Text tokens or sentinel events (__HIDDEN_ON__, __HIDDEN_OFF__)
    """

    # === RETRIEVAL === #
    docs = retriever.invoke(query)

    if not docs:
        yield "No relevant documents found."
        return

    # === CONTEXT ASSEMBLY === #
    context = "\n\n".join(doc.page_content for doc in docs)

    # === PROMPT FORMATTING === #
    if prompt_template is None:
        prompt_template = _DEFAULT_PROMPT_TEMPLATE

    prompt = prompt_template.format(
        context = context,
        question = query,
    )

    # === STREAM RESPONSE === #
    yield from _stream_ui(llm, prompt)


# ------------------------------------------------------ #
#   UI MODE STREAMING
# ------------------------------------------------------ #

def _stream_ui(llm, prompt):
    """
    Stream response with <think> tag masking (UI mode).

    Uses the native ollama library with think=True to properly detect
    thinking vs response phases. Yields sentinel events to signal state
    transitions, allowing the UI to show appropriate indicators.

    Parameters:
        llm: Ollama LLM instance (used to extract model name)
        prompt: Fully formatted prompt string with context and question

    Yields:
        str: Either a text token, or one of:
            "__HIDDEN_ON__"  - Model entered thinking phase
            "__HIDDEN_OFF__" - Model exited thinking, response follows

    Note:
        We use the native ollama library here (not LangChain wrapper) because
        LangChain's OllamaLLM.stream() filters out thinking content entirely,
        making it impossible to detect when thinking starts/ends.
    """

    # === EXTRACT MODEL NAME === #
    # The LangChain Ollama wrapper stores the model name in llm.model.

    model_name = llm.model if hasattr(llm, 'model') else 'deepseek-r1:14b'

    # === NATIVE OLLAMA STREAMING WITH REASONING === #
    # Use native ollama library to access think=True parameter,
    # which separates thinking tokens from response tokens.

    stream = ollama.chat(
        model = model_name,
        messages = [{'role': 'user', 'content': prompt}],
        think = True,   # Enable reasoning trace separation
        stream = True,  # Stream tokens as they're generated
    )

    # === TRACK THINKING STATE === #
    # in_thinking: True when model is in thinking phase
    # We yield sentinels at state transitions.

    in_thinking = False

    for chunk in stream:

        # Check if this chunk contains thinking content
        if chunk.message.thinking:
            if not in_thinking:
                # Just entered thinking phase - signal UI
                in_thinking = True
                yield "__HIDDEN_ON__"
            # Don't yield thinking content (it's hidden)

        elif chunk.message.content:
            # Response content (not thinking)
            if in_thinking:
                # Just exited thinking phase - signal UI
                in_thinking = False
                yield "__HIDDEN_OFF__"
            # Yield response token
            yield chunk.message.content
