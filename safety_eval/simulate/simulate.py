# ------------------------------------------------------ #
#
#   pipeline.py
#
#   Main orchestration script for crisis query simulation.
#   Generates queries using dual LLM backends with
#   checkpointing, cost estimation, and progress tracking.
#
#   Simone J. Skeen x Claude Code (06-23-2026)
#
# ------------------------------------------------------ #

from dotenv import load_dotenv
load_dotenv()

import csv
import json
import re
import argparse
from datetime import datetime
from pathlib import Path

from tqdm import tqdm # type: ignore

from config import (
    PIPELINE_CONFIG,
    INPUT_FILES,
    OUTPUT_FILES,
    OUTPUT_DIR,
)
from generation_prompt import build_prompt
from llm_clients import (
    get_openai_response,
    get_ollama_response,
    estimate_openai_cost,
)


# ------------------------------------------------------ #
#   Data Loading & Validation
# ------------------------------------------------------ #

def load_personas(filepath: Path) -> list[dict]:
    """Load personas from TSV file."""
    personas = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            personas.append(row)
    return personas


def load_seed_phrases(filepath: Path) -> list[dict]:
    """Load seed phrases from TSV file."""
    seeds = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            seeds.append(row)
    return seeds


def load_persona_contexts(filepath: Path) -> dict[str, str]:
    """
    Load persona-specific contexts from a single text file.

    Parses sections delimited by headers matching:
    '# Additional socio-cultural context for {persona_id = PXXX}:'

    Returns
    -------
    dict
        Mapping of persona_id to its context string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match persona headers and capture persona_id
    header_pattern = r'# Additional socio-cultural context for \{persona_id = (P\d+)\}:'

    # Find all headers and their positions
    matches = list(re.finditer(header_pattern, content))

    if not matches:
        raise ValueError(
            "No persona context headers found. Expected format: "
            "'# Additional socio-cultural context for {persona_id = PXXX}:'"
        )

    contexts = {}
    for i, match in enumerate(matches):
        persona_id = match.group(1)
        start = match.end()
        # End at next header or end of file
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        context_text = content[start:end].strip()
        contexts[persona_id] = context_text

    return contexts


def validate_inputs(personas: list, seeds: list, contexts: dict) -> bool:
    """
    Validate input data structure.

    Returns True if valid, raises ValueError otherwise.
    """
    required_persona_cols = {'persona_id', 'persona_name', 'age', 'gender'}
    required_seed_cols = {'seed_id', 'persona_id', 'seed_phrase'}

    if not personas:
        raise ValueError("No personas loaded from TSV")

    if not seeds:
        raise ValueError("No seed phrases loaded from TSV")

    if not contexts:
        raise ValueError("No persona contexts loaded")

    persona_cols = set(personas[0].keys())
    if not required_persona_cols.issubset(persona_cols):
        missing = required_persona_cols - persona_cols
        raise ValueError(f"Personas TSV missing columns: {missing}")

    seed_cols = set(seeds[0].keys())
    if not required_seed_cols.issubset(seed_cols):
        missing = required_seed_cols - seed_cols
        raise ValueError(f"Seed phrases TSV missing columns: {missing}")

    # Validate persona_id references
    persona_ids = {p['persona_id'] for p in personas}
    for seed in seeds:
        if seed['persona_id'] not in persona_ids:
            raise ValueError(
                f"Seed {seed['seed_id']} references unknown persona: {seed['persona_id']}"
            )

    # Validate each persona has a context
    for persona_id in persona_ids:
        if persona_id not in contexts:
            raise ValueError(
                f"Persona {persona_id} has no context in persona_context.txt"
            )

    return True


# ------------------------------------------------------ #
#   Checkpointing
# ------------------------------------------------------ #

def load_checkpoint(filepath: Path) -> set:
    """Load completed generation keys from checkpoint file."""
    if not filepath.exists():
        return set()

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data.get('completed', []))


def save_checkpoint(filepath: Path, completed: set):
    """Save completed generation keys to checkpoint file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({'completed': list(completed)}, f)


def make_generation_key(persona_id: str, seed_id: str, model: str) -> str:
    """Create a unique key for a generation task."""
    return f"{persona_id}|{seed_id}|{model}"


# ------------------------------------------------------ #
#   Output Writing
# ------------------------------------------------------ #

def init_output_file(filepath: Path):
    """Initialize output TSV with headers if it doesn't exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'timestamp',
                'persona_id',
                'model',
                'seed_id',
                'generated_query',
                'token_count',
            ])


def append_result(filepath: Path, result: dict):
    """Append a generation result to the output TSV."""
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            result['timestamp'],
            result['persona_id'],
            result['model'],
            result['seed_id'],
            result['generated_query'],
            result['token_count'],
        ])


# ------------------------------------------------------ #
#   Main Pipeline
# ------------------------------------------------------ #

def run_pipeline(dry_run: bool = False, skip_cost_confirm: bool = False):
    """
    Execute the crisis query generation pipeline.

    Parameters
    ----------
    dry_run : bool
        If True, validate inputs and estimate cost without making LLM calls
    skip_cost_confirm : bool
        If True, skip cost confirmation prompt
    """
    print("Loading input data...")
    personas = load_personas(INPUT_FILES['personas'])
    seeds = load_seed_phrases(INPUT_FILES['seed_phrases'])
    contexts = load_persona_contexts(INPUT_FILES['persona_context'])

    print("Validating inputs...")
    validate_inputs(personas, seeds, contexts)
    print(f"  Loaded {len(personas)} personas")
    print(f"  Loaded {len(seeds)} seed phrases")
    print(f"  Loaded {len(contexts)} persona contexts")

    # Build generation tasks
    models = ['openai', 'ollama']
    tasks = []

    persona_map = {p['persona_id']: p for p in personas}

    for seed in seeds:
        persona = persona_map[seed['persona_id']]
        for model in models:
            tasks.append({
                'persona': persona,
                'seed': seed,
                'model': model,
            })

    print(f"  Total generation tasks: {len(tasks)}")

    # Cost estimation for OpenAI calls
    openai_tasks = len([t for t in tasks if t['model'] == 'openai'])
    if openai_tasks > 0:
        cost_estimate = estimate_openai_cost(openai_tasks)
        print("\nOpenAI Cost Estimate:")
        print(f"  Generations: {cost_estimate['num_generations']}")
        print(f"  Est. input tokens: {cost_estimate['total_input_tokens']:,}")
        print(f"  Est. output tokens: {cost_estimate['total_output_tokens']:,}")
        print(f"  Est. total cost: ${cost_estimate['total_cost']:.4f}")

        if not skip_cost_confirm and not dry_run:
            confirm = input("\nProceed with generation? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted by user.")
                return

    if dry_run:
        print("\n[DRY RUN] Validation complete. No LLM calls made.")
        return

    # Load checkpoint
    checkpoint_path = OUTPUT_FILES['checkpoint']
    completed = load_checkpoint(checkpoint_path)
    print(f"\nCheckpoint: {len(completed)} tasks already completed")

    # Initialize output file
    output_path = OUTPUT_FILES['generated_queries']
    init_output_file(output_path)

    # Run generation
    skipped = 0

    for task in tqdm(tasks, desc="Generating queries"):
        persona = task['persona']
        seed = task['seed']
        model = task['model']

        key = make_generation_key(
            persona['persona_id'],
            seed['seed_id'],
            model,
        )

        if key in completed:
            skipped += 1
            continue

        # Build prompt with persona-specific context
        persona_context = contexts[persona['persona_id']]
        prompt = build_prompt(
            persona=persona,
            seed_phrase=seed['seed_phrase'],
            persona_context=persona_context,
        )

        # Generate
        try:
            if model == 'openai':
                response = get_openai_response(prompt)
            else:
                response = get_ollama_response(prompt)

            result = {
                'timestamp': datetime.now().isoformat(),
                'persona_id': persona['persona_id'],
                'model': response['model'],
                'seed_id': seed['seed_id'],
                'generated_query': response['text'],
                'token_count': response['token_count'],
            }

            append_result(output_path, result)
            completed.add(key)
            save_checkpoint(checkpoint_path, completed)

        except Exception as e:
            print(f"\nError generating {key}: {e}")
            continue

    # Summary
    print(f"\nGeneration complete!")
    print(f"  Tasks completed: {len(completed)}")
    print(f"  Tasks skipped (from checkpoint): {skipped}")
    print(f"  Output: {output_path}")

    # Clean up checkpoint on full completion
    if len(completed) == len(tasks):
        checkpoint_path.unlink(missing_ok=True)
        print("  Checkpoint cleaned up (all tasks complete)")


# ------------------------------------------------------ #
#   CLI Entry Point
# ------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crisis query simulation pipeline'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and estimate cost without making LLM calls',
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip cost confirmation prompt',
    )

    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run, skip_cost_confirm=args.yes)
