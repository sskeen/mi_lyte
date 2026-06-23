# pilot_dialogue_sim/

Crisis query simulation pipeline for chatbot safety evaluation.

| File | Description |
|------|-------------|
| `config.py` | Pipeline configuration: LLM parameters (OpenAI + Ollama), token limits, file paths, and cost estimation settings. |
| `generation_prompt.py` | Master prompt template for query generation. Combines persona, context, and seed phrase into formatted prompts. |
| `llm_clients.py` | LLM client wrappers with fresh instance per call. Includes OpenAI and Ollama generators plus cost estimation. |
| `pipeline.py` | Main orchestration script. Handles data loading, validation, checkpointing, progress tracking, and output generation. Run via `python pipeline.py` (use `--dry-run` to validate without LLM calls). |
| `data/` | Input files: personas, seed phrases, and persona context. |
| `output/` | Generated query results and checkpoint files. |
