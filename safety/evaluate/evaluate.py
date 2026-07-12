"""
evaluate.py

Evaluates RAG chatbot responses to simulated crisis queries.
Captures reasoning traces and responses for safety analysis.

Usage:
    cd safety/evaluate
    python evaluate.py

Output:
    demo_responses.tsv - Input data plus demo_reasoning and demo_response columns

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

from src.config import (
    LLM_PARAMS,
    EMBEDDING_MODEL,
    RETRIEVER_PARAMS,
    PROMPT_TEMPLATE,
    OLLAMA_HOST,
    OLLAMA_EMBED_HOST,
)

# === PATHS === #
INPUT_FILE = Path(__file__).parent.parent / "simulate" / "output" / "generated_queries.tsv"
OUTPUT_FILE = Path(__file__).parent / "demo_responses.tsv"
CHECKPOINT_FILE = Path(__file__).parent / ".checkpoint.json"
FAISS_INDEX = PROJECT_ROOT / "src" / "faiss_index"


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
    Load completed row indices from checkpoint file.

    Returns:
        set: Indices of already-completed rows
    """
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))
    return set()


def save_checkpoint(completed: set):
    """
    Save completed row indices to checkpoint file.
    """
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': list(completed)}, f)


# ------------------------------------------------------ #
#   OUTPUT WRITING
# ------------------------------------------------------ #

def init_output_file(filepath: Path, fieldnames: list):
    """
    Initialize output TSV with headers.

    Adds demo_reasoning and demo_response columns to existing fieldnames.
    """
    all_fields = list(fieldnames) + ['demo_reasoning', 'demo_response']

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
#   MAIN
# ------------------------------------------------------ #

def main():
    """
    Main evaluation loop with checkpointing.

    Processes each query through the RAG pipeline, capturing reasoning
    and response. Saves progress after each query for resumability.
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

    # Initialize output file with headers if new run
    if not completed:
        fieldnames = list(queries[0].keys())
        init_output_file(OUTPUT_FILE, fieldnames)
        print(f"\nCreated output file: {OUTPUT_FILE}")

    # Process queries
    print(f"\nProcessing {len(queries)} queries...\n")
    start_time = datetime.now()

    for i, row in enumerate(queries):
        if i in completed:
            print(f"[{i+1}/{len(queries)}] Skipping (already completed)")
            continue

        seed_id = row.get('seed_id', 'unknown')
        persona_id = row.get('persona_id', 'unknown')
        print(f"[{i+1}/{len(queries)}] {persona_id}/{seed_id}...", end=' ', flush=True)

        query_start = datetime.now()

        try:
            reasoning, response = evaluate_query(
                row['generated_query'],
                llm,
                retriever,
                PROMPT_TEMPLATE,
                ollama_client,
            )

            # Add new columns to row (normalize to fix mojibake)
            row['demo_reasoning'] = reasoning
            row['demo_response'] = normalize_text(response)

            # Append to output
            append_result(OUTPUT_FILE, row)

            # Update checkpoint
            completed.add(i)
            save_checkpoint(completed)

            elapsed = (datetime.now() - query_start).total_seconds()
            print(f"done ({elapsed:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            # Continue to next query on error
            continue

    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Completed {len(completed)}/{len(queries)} evaluations")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
