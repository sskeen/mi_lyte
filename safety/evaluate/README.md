# safety/evaluate/

Evaluates RAG chatbot responses to simulated crisis queries, capturing reasoning traces for safety analysis. Includes LLM-as-judge evaluation for response quality rating.

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Main evaluation script. Runs demo response generation and LLM-as-judge evaluation. |
| `judge_prompt.py` | Prompt template for the LLM judge. |
| `demo_responses.tsv` | Output from demo response generation (input columns + reasoning/response). |
| `demo_evaluation.tsv` | Output from judge evaluation (all columns + rating/rationale). |
| `.checkpoint.json` | Progress tracker for demo response loop (gitignored). |
| `.judge_checkpoint.json` | Progress tracker for judge evaluation loop (gitignored). |

## Usage

```bash
cd safety/evaluate
python evaluate.py
```

The script runs two sequential loops:

### Loop 1: Demo Response Generation
1. Load queries from `../simulate/output/generated_queries.tsv`
2. Process each query through the RAG pipeline
3. Capture model reasoning (`think=True`) and final response
4. Append results to `demo_responses.tsv`

### Loop 2: LLM-as-Judge Evaluation
1. Load responses from `demo_responses.tsv`
2. Evaluate each response using the judge model (GLM-4.7)
3. Parse rating (1-3) and rationale from judge output
4. Append results to `demo_evaluation.tsv`

## Checkpointing

Progress is saved after each row to separate checkpoint files. If interrupted, re-run the script to resume from where it left off.

To start fresh:
```bash
rm .checkpoint.json .judge_checkpoint.json demo_responses.tsv demo_evaluation.tsv
```

## Output Columns

### demo_responses.tsv

| Column | Description |
|--------|-------------|
| `demo_reasoning` | Model's internal reasoning trace (from `<think>` tags) |
| `demo_response` | Final user-facing response |

### demo_evaluation.tsv

All columns from `demo_responses.tsv` plus:

| Column | Description |
|--------|-------------|
| `model_judge` | Judge model tag (e.g., `glm-4.7-flash:latest`) |
| `llm_rating` | Quality rating: 1, 2, or 3 |
| `llm_rationale` | Three-sentence explanation for the rating |

## Configuration

**Demo response generation** uses settings from `src/config.py`:
- `LLM_PARAMS` — Model and generation settings
- `RETRIEVER_PARAMS` — FAISS retriever settings
- `PROMPT_TEMPLATE` — System prompt with context injection
- `OLLAMA_HOST` / `OLLAMA_EMBED_HOST` — Ollama endpoints

**Judge evaluation** is configured at the top of `evaluate.py`:
- `JUDGE_MODEL` — Judge model tag (default: `glm-4.7-flash:latest`)

Judge prompt template is in `judge_prompt.py`.

---

_Last updated: 07-12-2026_
