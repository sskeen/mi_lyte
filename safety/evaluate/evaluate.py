"""
evaluate.py

Evaluates RAG chatbot responses to simulated crisis queries.
Captures reasoning traces and responses for safety analysis.

Usage:
    cd safety/evaluate
    python evaluate.py

Output:
    demo_responses.tsv - Input data plus condition, demo_reasoning, demo_response, and guardrailed columns
                         (5 rows per query, one for each experimental condition)
    demo_evaluation.tsv - Above plus model_judge, llm_rating, and llm_rationale columns

Simone J. Skeen x Claude Code (07-11-2026)
"""

import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# === FIX OPENMP CONFLICT ON MACOS === #
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === ADD PROJECT ROOT TO PATH === #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === ADD SIMULATE DIR TO PATH (for normalize_text) === #
SIMULATE_DIR = Path(__file__).resolve().parent.parent / "simulate"
sys.path.insert(0, str(SIMULATE_DIR))

import ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from llm_clients import normalize_text
from judge_prompt import build_judge_prompt

from langchain_core.prompts import PromptTemplate

from src.config import (
    LLM_PARAMS,
    EMBEDDING_MODEL,
    RETRIEVER_PARAMS,
    OLLAMA_HOST,
    OLLAMA_EMBED_HOST,
)
from src.system_prompt import _BASE_PROMPT
from src.guardrail_backstop import GUARDRAIL_BACKSTOP, BACKSTOP_MARKER
from src.guardrail_b import GUARDRAIL_B
from src.guardrail_c import GUARDRAIL_C
from src.guardrail_d import GUARDRAIL_D
from src.si_parser import _SI_PARSER


# ------------------------------------------------------ #
#   EXPERIMENTAL CONDITIONS
# ------------------------------------------------------ #

CONDITIONS = {
    1: _BASE_PROMPT,
    2: _BASE_PROMPT + GUARDRAIL_BACKSTOP,
    3: _BASE_PROMPT + _SI_PARSER + GUARDRAIL_B + GUARDRAIL_BACKSTOP,
    4: _BASE_PROMPT + _SI_PARSER + GUARDRAIL_C + GUARDRAIL_BACKSTOP,
    5: _BASE_PROMPT + _SI_PARSER + GUARDRAIL_D + GUARDRAIL_BACKSTOP,
}


def build_prompt_template(condition: int) -> PromptTemplate:
    """
    Build PROMPT_TEMPLATE for a specific experimental condition.

    Parameters:
        condition: Condition number (1-5)

    Returns:
        PromptTemplate with the appropriate system prompt
    """
    return PromptTemplate(
        input_variables=['context', 'question'],
        template='''
        {system_prompt}

        Context:
        {context}

        Question:
        {question}
        '''.strip(),
    ).partial(system_prompt=CONDITIONS[condition])

# === PATHS === #
INPUT_FILE = Path(__file__).parent.parent / "simulate" / "output" / "generated_queries.tsv"
OUTPUT_FILE = Path(__file__).parent / "demo_responses.tsv"
CHECKPOINT_FILE = Path(__file__).parent / ".checkpoint.json"
FAISS_INDEX = PROJECT_ROOT / "src" / "faiss_index"

# === JUDGE CONFIG === #
JUDGE_MODEL = 'glm-4.7-flash:latest'
JUDGE_OUTPUT_FILE = Path(__file__).parent / "demo_evaluation.tsv"
JUDGE_CHECKPOINT_FILE = Path(__file__).parent / ".judge_checkpoint.json"


# ------------------------------------------------------ #
#   GUARDRAIL DETECTION
# ------------------------------------------------------ #

def detect_guardrail(response: str) -> int:
    """
    Detect if response contains the guardrail backstop scripted response.

    Uses BACKSTOP_MARKER from src/guardrail_backstop.py for single-source configuration.

    Returns:
        1 if guardrail detected, 0 otherwise
    """
    import re
    if re.search(BACKSTOP_MARKER, response, re.IGNORECASE):
        return 1
    return 0


# ------------------------------------------------------ #
#   DATA LOADING
# ------------------------------------------------------ #

def load_queries(filepath: Path) -> list[dict]:
    """
    Load generated queries from TSV file.

    Returns list of dicts, one per row, with all original columns.
    """
    queries = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            queries.append(row)
    return queries


# ------------------------------------------------------ #
#   RAG COMPONENTS
# ------------------------------------------------------ #

def init_rag_components():
    """
    Initialize FAISS index, LLM, and retriever.

    Returns:
        tuple: (llm, retriever, ollama_client)
    """
    print(f"Loading FAISS index from {FAISS_INDEX}...")

    # Embeddings use local Ollama (OLLAMA_EMBED_HOST)
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_EMBED_HOST,
    )

    db = FAISS.load_local(
        str(FAISS_INDEX),
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(**RETRIEVER_PARAMS)

    # LLM uses configured host (local or remote GPU)
    llm = OllamaLLM(**LLM_PARAMS)

    # Native ollama client for think=True support
    ollama_client = ollama.Client(host=OLLAMA_HOST)

    print(f"LLM: {LLM_PARAMS['model']} @ {OLLAMA_HOST}")
    print(f"Embeddings: {EMBEDDING_MODEL} @ {OLLAMA_EMBED_HOST}")

    return llm, retriever, ollama_client


def evaluate_query(
    query: str,
    llm,
    retriever,
    prompt_template,
    ollama_client,
) -> tuple[str, str]:
    """
    Run query through RAG pipeline, return (reasoning, response).

    Uses native ollama with think=True to separate reasoning from response.
    Mirrors the logic in src/dialogue_stream.py but collects instead of streams.

    Parameters:
        query: The user's crisis query text
        llm: LangChain OllamaLLM instance (for model name extraction)
        retriever: FAISS retriever
        prompt_template: PromptTemplate with {context} and {question}
        ollama_client: Native ollama.Client for think=True support

    Returns:
        tuple: (reasoning_text, response_text)
    """
    # === RETRIEVAL === #
    docs = retriever.invoke(query)

    if not docs:
        return ("", "No relevant documents found.")

    # === CONTEXT ASSEMBLY === #
    context = "\n\n".join(doc.page_content for doc in docs)

    # === PROMPT FORMATTING === #
    prompt = prompt_template.format(
        context=context,
        question=query,
    )

    # === LLM INFERENCE WITH REASONING === #
    model_name = llm.model if hasattr(llm, 'model') else LLM_PARAMS['model']

    # Use non-streaming call to collect full response
    response = ollama_client.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        think=True,  # Separate reasoning from response
        stream=False,  # Collect full response at once
    )

    # Extract reasoning and response from ollama response object
    reasoning = response.message.thinking or ""
    content = response.message.content or ""

    return (reasoning, content)


# ------------------------------------------------------ #
#   CHECKPOINTING
# ------------------------------------------------------ #

def load_checkpoint() -> set:
    """
    Load completed (query_idx, condition) pairs from checkpoint file.

    Returns:
        set: String keys like "0_1" for (query_idx 0, condition 1)
    """
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))
    return set()


def save_checkpoint(completed: set):
    """
    Save completed (query_idx, condition) pairs to checkpoint file.
    """
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': list(completed)}, f)


def checkpoint_key(query_idx: int, condition: int) -> str:
    """Generate checkpoint key for a (query_idx, condition) pair."""
    return f"{query_idx}_{condition}"


# ------------------------------------------------------ #
#   OUTPUT WRITING
# ------------------------------------------------------ #

def init_output_file(filepath: Path, fieldnames: list):
    """
    Initialize output TSV with headers.

    Adds condition, demo_reasoning, demo_response, and guardrailed columns to existing fieldnames.
    """
    all_fields = list(fieldnames) + ['condition', 'demo_reasoning', 'demo_response', 'guardrailed']

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, delimiter='\t')
        writer.writeheader()


def append_result(filepath: Path, row: dict):
    """
    Append a single evaluation result to the output TSV.
    """
    # Get fieldnames from existing file
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writerow(row)


# ------------------------------------------------------ #
#   JUDGE EVALUATION
# ------------------------------------------------------ #

def parse_judge_output(text: str) -> tuple[str, str]:
    """
    Parse judge LLM output into rating and rationale.

    Expected format:
        Rating: [1, 2, or 3]
        Rationale: [Three sentence explanation]

    Returns:
        tuple: (rating, rationale) - rating as string "1", "2", or "3"
               Returns ("", text) if parsing fails
    """
    import re

    rating = ""
    rationale = ""

    # Try to extract rating
    rating_match = re.search(r'Rating:\s*([123])', text, re.IGNORECASE)
    if rating_match:
        rating = rating_match.group(1)

    # Try to extract rationale
    rationale_match = re.search(r'Rationale:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    # If parsing failed, store raw text in rationale for manual review
    if not rating:
        rationale = f"[PARSE_ERROR] {text}"

    return (rating, rationale)


def judge_response(
    generated_query: str,
    demo_response: str,
    ollama_client,
    model: str = JUDGE_MODEL,
) -> tuple[str, str, str]:
    """
    Evaluate a demo response using the judge LLM.

    Parameters:
        generated_query: The simulated user query
        demo_response: The chatbot response to evaluate
        ollama_client: Native ollama.Client instance
        model: Judge model name

    Returns:
        tuple: (model_judge, llm_rating, llm_rationale)
    """
    prompt = build_judge_prompt(generated_query, demo_response)

    response = ollama_client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False,
    )

    content = response.message.content or ""
    rating, rationale = parse_judge_output(content)

    return (model, rating, rationale)


# ------------------------------------------------------ #
#   JUDGE CHECKPOINTING
# ------------------------------------------------------ #

def load_judge_checkpoint() -> set:
    """Load completed row indices from judge checkpoint file."""
    if JUDGE_CHECKPOINT_FILE.exists():
        with open(JUDGE_CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))
    return set()


def save_judge_checkpoint(completed: set):
    """Save completed row indices to judge checkpoint file."""
    with open(JUDGE_CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': list(completed)}, f)


# ------------------------------------------------------ #
#   JUDGE OUTPUT WRITING
# ------------------------------------------------------ #

def init_judge_output_file(filepath: Path, fieldnames: list):
    """
    Initialize judge output TSV with headers.

    Adds model_judge, llm_rating, and llm_rationale columns.
    """
    all_fields = list(fieldnames) + ['model_judge', 'llm_rating', 'llm_rationale']

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, delimiter='\t')
        writer.writeheader()


def append_judge_result(filepath: Path, row: dict):
    """Append a single judge evaluation result to the output TSV."""
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writerow(row)


# ------------------------------------------------------ #
#   JUDGE LOOP
# ------------------------------------------------------ #

def run_judge_evaluation(ollama_client):
    """
    Run LLM-as-judge evaluation on demo responses.

    Reads demo_responses.tsv and evaluates each row,
    outputting to demo_evaluation.tsv with rating and rationale.
    """
    print("\n" + "=" * 60)
    print("LLM-as-Judge Evaluation")
    print("=" * 60)

    # Check that input file exists
    if not OUTPUT_FILE.exists():
        print(f"\nSkipping judge evaluation: {OUTPUT_FILE} does not exist yet.")
        print("Run the demo response generation first.")
        return

    # Load demo responses
    print(f"\nLoading responses from {OUTPUT_FILE}...")
    responses = load_queries(OUTPUT_FILE)
    print(f"Loaded {len(responses)} responses to evaluate")
    print(f"Judge model: {JUDGE_MODEL} @ {OLLAMA_HOST}")

    # Load checkpoint
    completed = load_judge_checkpoint()
    if completed:
        print(f"\nResuming from checkpoint: {len(completed)} already completed")

    # Initialize output file with headers if it doesn't exist
    if not JUDGE_OUTPUT_FILE.exists():
        fieldnames = list(responses[0].keys())
        init_judge_output_file(JUDGE_OUTPUT_FILE, fieldnames)
        print(f"\nCreated output file: {JUDGE_OUTPUT_FILE}")

    # Process responses
    print(f"\nEvaluating {len(responses)} responses...\n")
    start_time = datetime.now()

    for i, row in enumerate(responses):
        if i in completed:
            print(f"[{i+1}/{len(responses)}] Skipping (already completed)")
            continue

        seed_id = row.get('seed_id', 'unknown')
        persona_id = row.get('persona_id', 'unknown')
        print(f"[{i+1}/{len(responses)}] Judging {persona_id}/{seed_id}...", end=' ', flush=True)

        query_start = datetime.now()

        try:
            model_judge, llm_rating, llm_rationale = judge_response(
                row['generated_query'],
                row['demo_response'],
                ollama_client,
            )

            # Add new columns to row
            row['model_judge'] = model_judge
            row['llm_rating'] = llm_rating
            row['llm_rationale'] = llm_rationale

            # Append to output
            append_judge_result(JUDGE_OUTPUT_FILE, row)

            # Update checkpoint
            completed.add(i)
            save_judge_checkpoint(completed)

            elapsed = (datetime.now() - query_start).total_seconds()
            print(f"done - rating: {llm_rating} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Completed {len(completed)}/{len(responses)} judge evaluations")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Output: {JUDGE_OUTPUT_FILE}")
    print("=" * 60)


# ------------------------------------------------------ #
#   MAIN
# ------------------------------------------------------ #

def main():
    """
    Main evaluation loop with checkpointing.

    Processes each query through the RAG pipeline under 5 experimental conditions,
    capturing reasoning and response. Saves progress after each query×condition
    for resumability.
    """
    print("=" * 60)
    print("mī lyte Safety Evaluation")
    print("=" * 60)

    # Load input queries
    print(f"\nLoading queries from {INPUT_FILE}...")
    queries = load_queries(INPUT_FILE)
    print(f"Loaded {len(queries)} queries")

    # Initialize RAG components
    print("\nInitializing RAG components...")
    llm, retriever, ollama_client = init_rag_components()

    # Load checkpoint
    completed = load_checkpoint()
    if completed:
        print(f"\nResuming from checkpoint: {len(completed)} already completed")

    # Initialize output file with headers if it doesn't exist
    if not OUTPUT_FILE.exists():
        fieldnames = list(queries[0].keys())
        init_output_file(OUTPUT_FILE, fieldnames)
        print(f"\nCreated output file: {OUTPUT_FILE}")

    # Total iterations = queries × conditions
    total_iterations = len(queries) * len(CONDITIONS)
    print(f"\nProcessing {len(queries)} queries × {len(CONDITIONS)} conditions = {total_iterations} iterations...\n")
    start_time = datetime.now()

    iteration = 0
    for i, row in enumerate(queries):
        seed_id = row.get('seed_id', 'unknown')
        persona_id = row.get('persona_id', 'unknown')

        for condition in CONDITIONS.keys():
            iteration += 1
            key = checkpoint_key(i, condition)

            if key in completed:
                print(f"[{iteration}/{total_iterations}] Skipping {persona_id}/{seed_id} C{condition} (already completed)")
                continue

            print(f"[{iteration}/{total_iterations}] {persona_id}/{seed_id} C{condition}...", end=' ', flush=True)

            query_start = datetime.now()

            try:
                # Build prompt template for this condition
                prompt_template = build_prompt_template(condition)

                reasoning, response = evaluate_query(
                    row['generated_query'],
                    llm,
                    retriever,
                    prompt_template,
                    ollama_client,
                )

                # Copy row to avoid mutating original
                result_row = dict(row)

                # Add new columns (normalize to fix mojibake)
                normalized_response = normalize_text(response)
                result_row['condition'] = condition
                result_row['demo_reasoning'] = reasoning
                result_row['demo_response'] = normalized_response
                result_row['guardrailed'] = detect_guardrail(normalized_response)

                # Append to output
                append_result(OUTPUT_FILE, result_row)

                # Update checkpoint
                completed.add(key)
                save_checkpoint(completed)

                elapsed = (datetime.now() - query_start).total_seconds()
                print(f"done ({elapsed:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                # Continue to next iteration on error
                continue

    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Completed {len(completed)}/{total_iterations} evaluations")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    # Run judge evaluation
    run_judge_evaluation(ollama_client)


if __name__ == "__main__":
    main()
