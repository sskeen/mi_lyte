# safety/evaluate/

Evaluates RAG chatbot responses to simulated crisis queries, capturing reasoning traces for safety analysis.

> [!NOTE]
> This is a preliminary evaluation script for pilot testing.

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Main evaluation script. Processes queries through RAG pipeline, captures reasoning and responses. |
| `pilot_evaluation.tsv` | Output file with original columns plus `demo_reasoning` and `demo_response`. |
| `.checkpoint.json` | Progress tracker for resumable runs (gitignored). |

## Usage

```bash
cd safety/evaluate
python evaluate.py
```

The script will:
1. Load queries from `../simulate/output/generated_queries.tsv`
2. Process each query through the RAG pipeline
3. Capture model reasoning (`think=True`) and final response
4. Append results to `pilot_evaluation.tsv`
5. Save checkpoint after each query

## Checkpointing

Progress is saved after each query to `.checkpoint.json`. If interrupted, simply re-run the script to resume from where it left off.

To start fresh, delete both `.checkpoint.json` and `pilot_evaluation.tsv`.

## Output Columns

| Column | Description |
|--------|-------------|
| `demo_reasoning` | Model's internal reasoning trace (from `<think>` tags) |
| `demo_response` | Final user-facing response |

All original columns from the input TSV are preserved.

## Configuration

Uses settings from `src/config.py`:
- `LLM_PARAMS` — Model and generation settings
- `RETRIEVER_PARAMS` — FAISS retriever settings
- `PROMPT_TEMPLATE` — System prompt with context injection
- `OLLAMA_HOST` / `OLLAMA_EMBED_HOST` — Ollama endpoints

---

_Last updated: 07-11-2026_
