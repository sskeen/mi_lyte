# pilot_dialogue_sim/

Crisis query simulation pipeline for chatbot safety evaluation.

| File | Description |
|------|-------------|
| `config.py` | Pipeline configuration: LLM parameters (OpenAI + Ollama), token limits, file paths, and cost estimation settings. |
| `generation_prompt.py` | Master prompt template for query generation. Combines persona, context, and seed phrase into formatted prompts. |
| `llm_clients.py` | LLM client wrappers with fresh instance per call. Includes OpenAI and Ollama generators plus cost estimation. |
| `pipeline.py` | Main orchestration script. Handles data loading, validation, checkpointing, progress tracking, and output generation. |
| `data/` | Input files: personas, seed phrases, and persona context. |
| `output/` | Generated query results and checkpoint files. |

---

## Prerequisites

1. **OpenAI API key** — Create a `.env` file in this directory:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your key:
   ```
   OPENAI_API_KEY=sk-...
   ```
   The key is loaded automatically at runtime and never logged or committed (`.env` is gitignored).

2. **Ollama** running locally with Qwen3:30B:
   ```bash
   ollama pull qwen3:30b
   ollama serve
   ```

3. **Python dependencies**:
   ```bash
   pip install openai ollama tqdm python-dotenv
   ```

---

## Usage

### Dry run (validate inputs + estimate cost)
```bash
python pipeline.py --dry-run
```
Validates input file schemas, checks persona-context mappings, and displays estimated OpenAI API cost without making any LLM calls.

### Run with cost confirmation
```bash
python pipeline.py
```
Displays cost estimate and prompts for confirmation before generating queries.

### Run without confirmation
```bash
python pipeline.py -y
```
Skips the cost confirmation prompt and proceeds directly to generation.

---

## Configuration

Edit `config.py` to adjust:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `num_queries` | `PIPELINE_CONFIG` | Queries per persona/seed/model combination |
| `avg_query_length_tokens` | `PIPELINE_CONFIG` | Target token length for generated queries |
| `random_seed` | `PIPELINE_CONFIG` | Seed for reproducibility (default: 56) |
| `delay_seconds` | `RATE_LIMIT_CONFIG` | Delay between OpenAI API calls (default: 1.0s) |
| `temperature` | `OPENAI_CONFIG` / `OLLAMA_CONFIG` | Sampling temperature |
| `reasoning_effort` | `OPENAI_CONFIG` | OpenAI reasoning effort (low/medium/high) |
| `think` | `OLLAMA_CONFIG` | Enable Qwen3 thinking mode |

---

## Input files

Place in `data/`:

- **`personas.tsv`** — One row per persona with columns: `persona_id`, `persona_name`, `age`, `gender`, `current_suicide_risk_level`
- **`seed_phrases.tsv`** — Seed phrases with columns: `seed_id`, `persona_id`, `seed_phrase`
- **`persona_context.txt`** — Per-persona context blocks delimited by headers:
  ```
  # Additional socio-cultural context for {persona_id = P001}:
  ...context for P001...

  # Additional socio-cultural context for {persona_id = P002}:
  ...context for P002...
  ```

---

## Output

Results are written to `output/generated_queries.tsv` with columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO timestamp of generation |
| `persona_id` | Source persona |
| `stressor` | Risk level from persona definition |
| `model` | Model used (gpt-5.4-mini or qwen3:30b) |
| `seed_id` | Source seed phrase ID |
| `generated_query` | The generated crisis query |
| `token_count` | Output token count |

---

## Checkpointing

The pipeline saves progress to `output/.checkpoint.json` after each generation. If interrupted, re-running the pipeline resumes from the last completed task. The checkpoint is automatically deleted upon full completion.
