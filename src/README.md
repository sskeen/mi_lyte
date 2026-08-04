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
|`guardrail_backstop.py`|A. BACKSTOP. A simple instruction to recommend the user call or access 988 in response to active suicidal ideation (SI). Appends to system prompt by default. Appends to experimental Guardrails B, C, and D for safeguarding audit. |
|`guardrail_b.py`|B. CONTEXTUAL. Instructs mī lyte to search retrieved context and issue mindfulness recommendations in line with expected behavior. |
|`guardrail_c.py`|C. ZERO-SHOT MINDFUL. Maps SI drivers to mindfulness assets grounded in theory and (weak) evidence; instructs mī lyte to respond accordingly. |
|`guardrail_d.py`|D. FEW-SHOT MINDFUL. Maps SI drivers to mindfulness assets grounded in theory and (weak) evidence; provides mī lyte examples of appropriate responses. |
|`si_parser.py`| Prepends experimental Guardrails B, C, and D, prompting mī lyte to distinguish passive versus active expressions of suicidal ideation before adhering to experimental or base Guardrail A procedure. |