# src/

Source code and prompts for mī lyte System 1.

| File | Description |
|------|-------------|
| `config.py` | Shared configuration: system prompt, LLM parameters, retriever settings, and prompt template. Single source of truth imported by both the notebook and the prototype. |
| `dialogue_stream.py` | Standalone `query_and_stream()` function that replicates LangChain's `RetrievalQA` (stuff chain) with token-by-token streaming and optional source metadata display. Used by `mi_lyte_sys_1_toy.ipynb`. |
| `system_prompt.py` | System prompt, imported via `config.py`. Retains single source of truth, facilitates iterative tweaking. |

Guardrail variants
> [!WARNING]
> These guardrails are in active development and remain _untested_ for reliable safeguarding in production. 

| File | Description |
|------|-------------|
|`guardrail_a.py`|A. BASE. xx|
|`guardrail_b.py`|B. CONTEXTUAL. xx|
|`xx`|xx|
|`xx`|xx|
|`si_parser.py`| Prepends experimental Guardrails B, C, and D, prompting mi lyte to distinguish passive versus active expressions of suicidal ideation before adhering to experimental or base Guardrail A procedure. |